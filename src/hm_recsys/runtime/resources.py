"""Bound and observe independent native-library worker processes.

Consumes a numerical phase, private run view and resource policy; emits streamed
logs, progress and measured process-tree RSS. Depends on psutil and subprocess.
Memory or child failures terminate the stage and prevent publication of a completed
artifact.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

import psutil

from hm_recsys.core.contracts import Config, read_json, write_json

# Relay flushed nested-worker events while retaining private logs and RSS guards.
from hm_recsys.core.progress import Lines, emit

GIB = 1024**3
THREAD_ENV = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
)


# Set library limits before child interpreters import numeric packages.
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


# Sum resident bytes of the supervisor and its current descendants.
def tree_rss(process: psutil.Process) -> int:
    """Sum resident bytes of the supervisor and its current descendants."""
    total = 0
    for member in [process, *process.children(recursive=True)]:
        # Protect this operation with explicit failure/cleanup handling so partial
        # work is never reported as complete.
        try:
            total += member.memory_info().rss
        # Handle the declared exception type here and preserve the surrounding
        # failure/recovery contract.
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            # Skip this explicitly inapplicable/empty record while retaining the
            # order of all remaining valid inputs.
            continue
    return total


# Terminate only this spawned worker and its descendants.
def stop_owned(child: subprocess.Popen) -> None:
    """Terminate only this spawned worker and its descendants."""
    # Protect this operation with explicit failure/cleanup handling so partial work
    # is never reported as complete.
    try:
        parent = psutil.Process(child.pid)
        owned = parent.children(recursive=True)
    # Handle the declared exception type here and preserve the surrounding
    # failure/recovery contract.
    except psutil.NoSuchProcess:
        return
    for member in owned:
        # Protect this operation with explicit failure/cleanup handling so partial
        # work is never reported as complete.
        try:
            member.terminate()
        # Handle the declared exception type here and preserve the surrounding
        # failure/recovery contract.
        except psutil.NoSuchProcess:
            # Preserve this intentionally empty handler; the surrounding branch
            # defines which failure is optional.
            pass
    # Popen must reap its own child. psutil.wait_procs(parent) can consume the
    # exit status first, causing Popen to incorrectly report a zero exit code.
    try:
        child.terminate()
        child.wait(timeout=3)
    # Handle the declared exception type here and preserve the surrounding
    # failure/recovery contract.
    except subprocess.TimeoutExpired:
        child.kill()
        child.wait()
    # Handle the declared exception type here and preserve the surrounding
    # failure/recovery contract.
    except ProcessLookupError:
        child.wait()
    _, alive = psutil.wait_procs(owned, timeout=3)
    for member in alive:
        # Protect this operation with explicit failure/cleanup handling so partial
        # work is never reported as complete.
        try:
            member.kill()
        # Handle the declared exception type here and preserve the surrounding
        # failure/recovery contract.
        except psutil.NoSuchProcess:
            # Preserve this intentionally empty handler; the surrounding branch
            # defines which failure is optional.
            pass
    child.wait()


# Run one owned stage and persist monitoring evidence even on failure.
def run_stage(
    stage: str,
    run: Path,
    config: Config,
    *,
    module: str = "hm_recsys.runtime.worker",
    cwd: Path | None = None,
) -> None:
    """Run one owned stage and persist monitoring evidence even on failure.

    Sampling is not an OS-enforced cap. Shared pages can be double-counted in
    summed RSS; peaks between observations can be missed.
    """
    log = run / "resource_usage.json"
    usage = read_json(log) if log.exists() else {"stages": []}
    supervisor = psutil.Process()
    available = psutil.virtual_memory().available
    # Protect this operation with explicit failure/cleanup handling so partial work
    # is never reported as complete.
    try:
        reason = memory_violation(tree_rss(supervisor), available, config)
    # Handle the declared exception type here and preserve the surrounding
    # failure/recovery contract.
    except (PermissionError, psutil.AccessDenied) as error:
        usage["stages"].append(
            {
                "stage": stage,
                "status": "blocked",
                "reason": "process_tree_monitor_permission_denied",
            }
        )
        # Write measured state/lineage to the owned run path without mutating source
        # manifests.
        write_json(log, usage)
        raise RuntimeError("Process-tree monitoring permission denied") from error
    if reason:
        usage["stages"].append({"stage": stage, "status": "blocked", "reason": reason})
        # Write measured state/lineage to the owned run path without mutating source
        # manifests.
        write_json(log, usage)
        raise RuntimeError(reason)
    command = [sys.executable, "-m", module, stage, str(run)]
    with (run / "commands.log").open("a") as handle:
        # Serialize the explicitly selected metadata/records to this operation's
        # output; raw source inputs are not overwritten.
        handle.write(" ".join(command) + "\n")
    start = time.monotonic()
    peak = 0
    minimum = available
    observations = 0
    with (run / f"{stage}.log").open("x") as output:
        child = subprocess.Popen(
            command,
            env={
                **constrained_env(config),
                "PYTHONUNBUFFERED": "1",
                "HM_PROGRESS_FILE": str(run / "progress" / f"{stage}.jsonl"),
                "HM_PROGRESS_STAGE": stage,
            },
            stdout=output,
            stderr=subprocess.STDOUT,
            cwd=cwd,
        )
        reader = (run / f"{stage}.log").open(encoding="utf-8", errors="replace")
        # Partial native-library lines remain bounded until their newline arrives.
        relay = Lines()
        # Emit a grounded start with an unknown worker-total denominator.
        emit("worker", 0, None, status="started", stage=stage, force=True)
        # Protect this operation with explicit failure/cleanup handling so partial
        # work is never reported as complete.
        try:
            # Continue this owned loop only under its stated condition, respecting
            # the consumed-offset/resource boundary.
            while child.poll() is None:
                # Forward nested records now, not after the outer CLI finishes.
                for line in relay.feed(reader.read(65536)):
                    # Original events retain their prefix for the Notebook parser.
                    print(line, flush=True)
                rss = tree_rss(supervisor)
                available = psutil.virtual_memory().available
                peak = max(peak, rss)
                minimum = min(minimum, available)
                observations += 1
                reason = memory_violation(rss, available, config)
                if reason:
                    stop_owned(child)
                    # Stop at this declared completion/limit boundary; no extra rows
                    # or optimization steps are consumed.
                    break
                if observations % 75 == 0:
                    # Unknown total means this is a resource heartbeat, not completed
                    # training.
                    emit(
                        "resources",
                        observations,
                        None,
                        unit="samples",
                        stage=stage,
                        metrics={
                            "process_tree_rss_bytes": rss,
                            "available_bytes": available,
                        },
                    )
                # Retain the existing RSS monitor sampling interval while the child
                # works.
                time.sleep(config.values["monitor_seconds"])
        finally:
            if child.poll() is None:
                stop_owned(child)
        # Drain traceback/final events emitted immediately before worker exit.
        for block in iter(lambda: reader.read(65536), ""):
            # Each drained block remains bounded even after a verbose worker failure.
            for line in relay.feed(block):
                # The full log stays private; the UI applies identifier redaction.
                print(line, flush=True)
        # Surface even a final unterminated error message with bounded memory.
        if relay.pending:
            # No completion is inferred from a log line.
            print(relay.pending, flush=True)
        # Close the tail after the writer process has stopped.
        reader.close()
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
    # Write measured state/lineage to the owned run path without mutating source
    # manifests.
    write_json(log, usage)
    if child.returncode or reason:
        # Keep the failed computation separate from verified stage publication.
        emit(
            "worker",
            0,
            None,
            status="failed",
            stage=stage,
            message=reason or "Worker exited nonzero; see inner traceback",
            force=True,
        )
        raise RuntimeError(f"Stage {stage} failed; inspect {stage}.log")
