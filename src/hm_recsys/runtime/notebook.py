"""Live worker event rendering and explicit Notebook command execution.

Consumes argument vectors and structured progress records; emits bounded
interactive status, resource tables and actual child exit results. Uses subprocess
readers and optional IPython display. Progress is observed work, never an invented
completion counter.
"""

# Stdlib tailing works before numerical packages or display extensions are available.
import json

# Inherit and restore process context or manage the owned subprocess environment.
import os

# Redact private H&M identifiers and possible purchase rows before display.
import re

# Stop only the child process group owned by the running Notebook command.
import signal
import subprocess

# Measure monotonic delivery/elapsed intervals independently of model RNG.
import time

# Type the event callback used for live Notebook display.
from collections.abc import Callable
from pathlib import Path

from hm_recsys.core.progress import PREFIX, Lines, describe


# Customer/article coordinates are not suitable for public progress displays.
def safe_text(text: str) -> str:
    """Redact H&M identifiers and potential raw purchase rows in displayed logs."""
    # Raw purchase rows must remain in the private diagnostic file only.
    if re.search(r"\d{4}-\d{2}-\d{2}.*[,\t]", text):
        # Retain the error context without exposing a source transaction row.
        return "[potential purchase row suppressed; inspect private log]"
    # H&M customer IDs are long hexadecimal strings.
    text = re.sub(r"\b[0-9a-fA-F]{64}\b", "[customer ID]", text)
    # Article IDs remain private audit keys rather than progress metrics.
    return re.sub(r"\b\d{10}\b", "[article ID]", text)


# A CLI leader can exit while a descendant remains; always stop the owned group.
def _stop_group(child: subprocess.Popen) -> None:
    """Terminate owned descendants even when the original CLI has already exited."""
    # The helper created this process group; unrelated user processes are excluded.
    try:
        # Offer the existing worker signal handlers their normal termination path.
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        # An already vanished group needs no cleanup.
        pass
    # Reap the leader with a bounded grace period.
    try:
        # A dead leader does not imply all its descendants stopped.
        child.wait(timeout=2)
    except subprocess.TimeoutExpired:
        # Surviving processes are forcibly stopped below; no final checkpoint is
        # promised.
        pass
    # macOS can briefly deny KILL while a TERM-killed orphan is being reaped.
    deadline = time.monotonic() + 0.5
    # Retry only this transient failure; a persistent denial must remain visible.
    while True:
        # The group ID is still the fresh group owned by this invocation.
        try:
            # Surviving workers cannot continue after the owning command failed.
            os.killpg(child.pid, signal.SIGKILL)
        # A fully reaped group has already satisfied cleanup.
        except ProcessLookupError:
            # No unrelated process is probed or signalled as a replacement.
            break
        # Do not silently treat a denied signal as successful termination.
        except PermissionError:
            # Bound exceptional retry latency without changing normal training.
            if time.monotonic() >= deadline:
                # A still-protected group requires explicit operator diagnosis.
                raise
            # Allow the kernel reaper to finish before retrying this same owned group.
            time.sleep(0.02)
        else:
            # A successful forced stop requires no further signals.
            break
    # Reap the process owned by Popen, without psutil consuming its exit status.
    child.wait()


# Display actual file output as it arrives instead of waiting for command completion.
def command(
    argv: list[str],
    project: Path,
    log: Path,
    *,
    poll_seconds: float = 0.1,
    heartbeat_seconds: float = 15,
    on_event: Callable[[dict], None] | None = None,
    run_id: str | None = None,
    profile: str | None = None,
) -> None:
    """Run a logged child and stream flushed events, text and bounded failure tails.

    The child starts a new process group, so interruption terminates nested
    workers too. on_event receives real events before exit; rendering errors or
    child failures are propagated. Full private logs remain on local disk.
    """
    # Reject busy-loop/negative timing options before starting an owned process.
    if poll_seconds <= 0 or heartbeat_seconds <= 0:
        # Timing options affect display only, not training configuration.
        raise ValueError("Progress polling and heartbeat intervals must be positive")
    # Never discard earlier invocation logs used for resume diagnostics.
    log.parent.mkdir(parents=True, exist_ok=True)
    # Show the actual no-shell argv so operators can understand and reproduce execution.
    print(f"Command: {argv!r}\nLive/private log: {log}", flush=True)
    # Start reading at the new invocation rather than replaying older completed events.
    offset = log.stat().st_size if log.exists() else 0
    # Split partial flushed lines without buffering an unbounded library message.
    lines = Lines()
    # Keep only a small displayed failure tail; full logs stay on disk.
    tail: list[str] = []
    # Preserve bounded last-work observations independently of a long failure traceback.
    work_events: dict[tuple[str, str], dict] = {}
    # Fast per-file milestones remain in the private log without flooding cell output.
    last_file_display = 0.0
    # Flush Python output from every descendant using the inherited environment.
    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "HM_PROGRESS_FILE": str(log.with_suffix(".events.jsonl")),
        "HM_PROGRESS_STAGE": "setup",
        "HM_PROGRESS_RUN_ID": argv[argv.index("--run-id") + 1]
        if "--run-id" in argv
        else "setup",
        "HM_PROGRESS_PROFILE": argv[argv.index("--profile") + 1]
        if "--profile" in argv
        else profile or os.environ.get("HM_PROGRESS_PROFILE", "setup"),
    }
    # Keep the output handle open throughout the child lifetime.
    with log.open("a", encoding="utf-8") as output:
        # One owned group covers the outer CLI and all numerical workers.
        child = subprocess.Popen(
            argv,
            cwd=project,
            env=env,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        # Heartbeats report silence, never fictional percentages.
        last = started = time.monotonic()
        # File tailing handles nested progress forwarded by the resource supervisor.
        try:
            with log.open(encoding="utf-8", errors="replace") as reader:
                # Skip prior invocations without interpreting their events again.
                reader.seek(offset)
                # Process exit is not enough; output written just before exit must
                # drain.
                while True:
                    # Bound each poll's memory even if a dependency prints a huge log.
                    block = reader.read(65536)
                    # Drain an unterminated final line, including top-level stderr
                    # errors.
                    if not block and child.poll() is not None and lines.pending:
                        # The synthetic newline only completes framing, never a
                        # progress count.
                        block = "\n"
                    # Handle every complete event/text line emitted so far.
                    for line in lines.feed(block):
                        # Protocol lines have a bounded JSON record payload.
                        if line.startswith(PREFIX):
                            # Malformed protocol output is an explicit error, not
                            # silently accepted progress.
                            event = json.loads(line[len(PREFIX) :])
                            # Keep real work counters separate from failed lifecycle
                            # markers.
                            if event["task"] not in ("stage", "verification", "worker"):
                                # A bounded task map retains last training loss even
                                # after traceback output.
                                work_events[(event["stage"], event["task"])] = event
                                # Drop oldest task kinds if an extension emits an
                                # excessive task vocabulary.
                                if len(work_events) > 64:
                                    # Model/data counters themselves are never
                                    # modified by display eviction.
                                    work_events.pop(next(iter(work_events)))
                            # UI callbacks can update a lightweight stage table
                            # during execution.
                            if on_event is not None:
                                # Never defer callbacks until process completion.
                                on_event(event)
                            # File verification can finish many small files per second.
                            if (
                                event["task"] in ("file_copy", "file_verify")
                                and event["status"] != "failed"
                            ):
                                # Keep every event/callback, throttle only duplicate
                                # human text.
                                if time.monotonic() - last_file_display < 1:
                                    # Terminal stage/input summaries still print
                                    # independently.
                                    continue
                                # The next file update can be shown after a second of
                                # real elapsed time.
                                last_file_display = time.monotonic()
                            # Text is always available outside IPython and in
                            # captured logs.
                            text = describe(event)
                        else:
                            # Keep arbitrary dependency text bounded and redact
                            # identifiers.
                            text = safe_text(line)
                        # Empty lines do not provide useful operator information.
                        if text:
                            # Preserve a compact tail for failures rather than
                            # reading the entire log.
                            tail = (tail + [text[-2000:]])[-12:]
                            # Flush immediately so Colab displays current work.
                            print(text, flush=True)
                            # Record the last received observation for silence
                            # heartbeats.
                            last = time.monotonic()
                    # A final empty read after exit means all flushed records were
                    # consumed.
                    if not block and child.poll() is not None:
                        # Leave the loop with the real child exit status intact.
                        break
                    # Report no-news duration separately from the latest real worker
                    # event.
                    if time.monotonic() - last >= heartbeat_seconds:
                        # No total or ETA is invented during an opaque SQL/library call.
                        print(
                            f"Heartbeat: elapsed={time.monotonic() - started:.1f}s; "
                            f"no new event for {heartbeat_seconds:g}s; log={log}",
                            flush=True,
                        )
                        # Rate-limit heartbeats while continuing to poll.
                        last = time.monotonic()
                    # Avoid spinning while a child is performing numerical or I/O work.
                    if not block:
                        # Display polling never consumes model RNG or changes batch
                        # boundaries.
                        time.sleep(poll_seconds)
        except BaseException:
            # Cleanup also handles an exited leader and the poll/kill race.
            _stop_group(child)
            # The Notebook must not treat interruption as success.
            raise
    # Preserve the actual nonzero exit and show the relevant forwarded inner-log tail.
    if child.returncode:
        # A failed leader must not leave any owned numerical descendant running.
        _stop_group(child)
        # Retain the latest real counters/losses even if a traceback displaced the
        # text tail.
        for event in list(work_events.values())[-8:]:
            # These observations remain explicitly last observed, not completed stages.
            print("Last real progress: " + describe(event), flush=True)
        # Tail includes real inner-stage traceback, last event and failure reason.
        print("FAILED: last observed output\n" + "\n".join(tail), flush=True)
        # Standard callers can inspect the exact argv and return code.
        raise subprocess.CalledProcessError(child.returncode, argv)
    # A successful command is distinct from the verified stage-completion events it
    # emitted.
    print(f"Exit 0; elapsed={time.monotonic() - started:.1f}s; log={log}", flush=True)
