"""Sequential subprocess supervision with sampled process-tree RSS guards."""

import os
import subprocess
import sys
import time
from pathlib import Path

import psutil

from hm_recsys.contracts import Config, read_json, write_json

GIB = 1024**3
THREAD_ENV = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
)


def constrained_env(config: Config) -> dict[str, str]:
    """Set library limits before child interpreters import numeric packages."""
    env = os.environ.copy()
    env.update({key: str(config.limit("threads")) for key in THREAD_ENV})
    env.update(PYTHONHASHSEED="42", TOKENIZERS_PARALLELISM="false")
    return env


def memory_violation(rss: int, available: int, config: Config) -> str | None:
    """Return a reason to stop, using available memory rather than installed RAM."""
    if rss >= config.values["rss_abort_gib"] * GIB:
        return "process_tree_rss_at_or_above_abort_threshold"
    if available < config.values["available_floor_gib"] * GIB:
        return "system_available_memory_below_floor"
    return None


def tree_rss(process: psutil.Process) -> int:
    """Sum resident bytes of the supervisor and its current descendants."""
    total = 0
    for member in [process, *process.children(recursive=True)]:
        try:
            total += member.memory_info().rss
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            continue
    return total


def stop_owned(child: subprocess.Popen) -> None:
    """Terminate only this spawned worker and its descendants."""
    try:
        parent = psutil.Process(child.pid)
        owned = parent.children(recursive=True)
    except psutil.NoSuchProcess:
        return
    for member in owned:
        try:
            member.terminate()
        except psutil.NoSuchProcess:
            pass
    # Popen must reap its own child. psutil.wait_procs(parent) can consume the
    # exit status first, causing Popen to incorrectly report a zero exit code.
    try:
        child.terminate()
        child.wait(timeout=3)
    except subprocess.TimeoutExpired:
        child.kill()
        child.wait()
    except ProcessLookupError:
        child.wait()
    _, alive = psutil.wait_procs(owned, timeout=3)
    for member in alive:
        try:
            member.kill()
        except psutil.NoSuchProcess:
            pass
    child.wait()


def run_stage(stage: str, run: Path, config: Config) -> None:
    """Run one owned stage and persist monitoring evidence even on failure.

    Sampling is not an OS-enforced cap. Shared pages can be double-counted in
    summed RSS; peaks between observations can be missed.
    """
    log = run / "resource_usage.json"
    usage = read_json(log) if log.exists() else {"stages": []}
    supervisor = psutil.Process()
    available = psutil.virtual_memory().available
    try:
        reason = memory_violation(tree_rss(supervisor), available, config)
    except (PermissionError, psutil.AccessDenied) as error:
        usage["stages"].append(
            {
                "stage": stage,
                "status": "blocked",
                "reason": "process_tree_monitor_permission_denied",
            }
        )
        write_json(log, usage)
        raise RuntimeError("Process-tree monitoring permission denied") from error
    if reason:
        usage["stages"].append({"stage": stage, "status": "blocked", "reason": reason})
        write_json(log, usage)
        raise RuntimeError(reason)
    command = [sys.executable, "-m", "hm_recsys.worker", stage, str(run)]
    with (run / "commands.log").open("a") as handle:
        handle.write(" ".join(command) + "\n")
    start = time.monotonic()
    peak = 0
    minimum = available
    observations = 0
    with (run / f"{stage}.log").open("x") as output:
        child = subprocess.Popen(
            command,
            env=constrained_env(config),
            stdout=output,
            stderr=subprocess.STDOUT,
        )
        try:
            while child.poll() is None:
                rss = tree_rss(supervisor)
                available = psutil.virtual_memory().available
                peak = max(peak, rss)
                minimum = min(minimum, available)
                observations += 1
                reason = memory_violation(rss, available, config)
                if reason:
                    stop_owned(child)
                    break
                time.sleep(config.values["monitor_seconds"])
        finally:
            if child.poll() is None:
                stop_owned(child)
    usage["stages"].append(
        {
            "stage": stage,
            "seconds": time.monotonic() - start,
            "exit_code": child.returncode,
            "termination_reason": reason,
            "sampled_peak_tree_rss_bytes": peak,
            "minimum_system_available_bytes": minimum,
            "target_exceeded": peak > config.values["rss_target_gib"] * GIB,
            "observations": observations,
            "sampling_interval_seconds": config.values["monitor_seconds"],
            "includes_supervisor": True,
            "os_hard_cap": False,
        }
    )
    write_json(log, usage)
    if child.returncode or reason:
        raise RuntimeError(f"Stage {stage} failed; inspect {stage}.log")
