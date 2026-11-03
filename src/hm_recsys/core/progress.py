"""Local append-only worker progress; counters never drive model execution."""

# Keep telemetry independent of numerical libraries and their RNG/device state.
import contextlib

# Encode/decode the structured aggregate event protocol.
import json

# Reject nonfinite counters rather than emitting invalid JSON percentages.
import math

# Inherit and restore process context or manage the owned subprocess environment.
import os

# Measure monotonic delivery/elapsed intervals independently of model RNG.
import time

# Describe the scoped context generator without importing numerical runtimes.
from collections.abc import Iterator

# Stamp event chronology in UTC while elapsed time uses a monotonic clock.
from datetime import UTC, datetime

# Resolve owned local event/log paths independently of file-name spelling.
from pathlib import Path

# Prefix distinguishes events from arbitrary library stdout without parsing logs as
# data.
PREFIX = "HM_PROGRESS "
# Each process owns its rate/clock state; no worker shares a DuckDB connection here.
_CLOCKS: dict[tuple, tuple[float, float, float]] = {}


# Scope event destinations to an owned local run or stage attempt.
@contextlib.contextmanager
def scope(path: Path, *, run_id: str, profile: str, stage: str) -> Iterator[None]:
    """Set inherited event context and restore the caller's environment afterward."""
    # Preserve prior context for nested supervisors and opt-in parity tests.
    values = {
        "HM_PROGRESS_FILE": str(path),
        "HM_PROGRESS_RUN_ID": run_id,
        "HM_PROGRESS_PROFILE": profile,
        "HM_PROGRESS_STAGE": stage,
    }
    # Environment restoration prevents a later unrelated CLI from using this run.
    previous = {key: os.environ.get(key) for key in values}
    # Descendant workers inherit the same run identity with their own local file.
    os.environ.update(values)
    # Always restore context even when validation or fitting raises.
    try:
        # Yield control without performing model/data work.
        yield
    finally:
        # Restore only the telemetry keys owned by this context.
        for key, value in previous.items():
            # Absent original keys should remain absent after this invocation.
            if value is None:
                # Do not leave stale output destinations in a reused kernel.
                os.environ.pop(key, None)
            else:
                # Retain the enclosing stage context for nested supervision.
                os.environ[key] = value


# The event function observes completed work; it never samples, trains or
# synchronizes CUDA.
def emit(
    task: str,
    completed: int | float = 0,
    total: int | float | None = None,
    unit: str = "rows",
    status: str = "running",
    message: str = "",
    metrics: dict | None = None,
    force: bool = False,
    *,
    stage: str | None = None,
) -> None:
    """Append and flush a bounded aggregate event when telemetry is enabled.

    Totals describe the named task's unit. None means indeterminate, including
    a running opaque library call. Rates/ETA use observed deltas, not file-size
    guesses. A started event resets a reused task (for example a new file).
    HM_PROGRESS_DISABLED=1 provides an instrumentation-off parity reference.
    """
    # Disabled instrumentation must be a no-op, including no new files or RNG work.
    if os.environ.get("HM_PROGRESS_DISABLED") == "1":
        # Leave reference model execution unchanged.
        return
    # A destination or explicit stdout opt-in is required outside the shared runner.
    destination = os.environ.get("HM_PROGRESS_FILE")
    # Normal import/unit use remains silent and creates no runtime artifacts.
    if not destination and os.environ.get("HM_PROGRESS_STDOUT") != "1":
        # Training helpers can still be called without a Notebook.
        return
    # Invalid counters are implementation defects rather than invented progress.
    if (
        completed < 0
        or not math.isfinite(completed)
        or (
            total is not None
            and (total < completed or total < 0 or not math.isfinite(total))
        )
    ):
        # Fail at the faulty counter boundary instead of displaying a false percentage.
        raise ValueError("Invalid progress counter or denominator")
    # Use a monotonic clock for durations and UTC only for display chronology.
    now = time.monotonic()
    # Separate stage/task clocks even when two file/epoch counters have identical names.
    label = stage or os.environ.get("HM_PROGRESS_STAGE", "setup")
    # The local destination also distinguishes fresh stage attempts in one process.
    key = (destination, label, task)
    # Reused task names explicitly start a new denominator/clock.
    first, last, previous = _CLOCKS.get(key, (now, now, completed))
    # A new file/epoch must not inherit the preceding task's rate.
    if status == "started":
        # Begin this task with no extrapolated rate.
        first, last, previous = now, now, completed
    # Milestones and final/failure events bypass throttling.
    terminal = status != "running" or force
    # Bound event volume to at most one periodic event per second per task.
    if key in _CLOCKS and not terminal and now - last < 1:
        # Counters still advance in the real loop; a later event reports the latest
        # count.
        return
    # Compute recent throughput only from monotonic actual work increments.
    rate = (
        (completed - previous) / (now - last)
        if now > last and completed > previous
        else None
    )
    # Persist the emission boundary rather than each skipped loop iteration.
    _CLOCKS[key] = (first, now, completed)
    # A known zero-work task is completed without inventing a throughput estimate.
    percent = 100 * completed / total if total and status != "failed" else None
    # ETA is an estimate only while meaningful work and a denominator remain.
    eta = (
        (total - completed) / rate
        if total is not None and rate and status == "running"
        else None
    )
    # Only aggregate caller-supplied metrics belong in this public display contract.
    event = {
        "version": 1,
        "timestamp": datetime.now(UTC).isoformat(),
        "run_id": os.environ.get("HM_PROGRESS_RUN_ID", "unknown"),
        "profile": os.environ.get("HM_PROGRESS_PROFILE", "unknown"),
        "stage": label,
        "task": task,
        "completed": completed,
        "total": total,
        "unit": unit,
        "elapsed_seconds": now - first,
        "percentage": percent,
        "rate_per_second": rate,
        "eta_seconds": eta,
        "message": message,
        "metrics": metrics or {},
        "status": status,
    }
    # Reject NaN/Inf in telemetry rather than creating nonstandard JSON artifacts.
    line = json.dumps(event, allow_nan=False, separators=(",", ":"))
    # Bound individual records so every downstream tailer can retain bounded memory.
    if len(line.encode("utf-8")) > 8192:
        # Metrics must remain compact aggregates, never per-customer rows.
        raise ValueError("Progress event exceeds 8 KiB")
    # Files are always local; verified stage publication handles later Drive transfer.
    if destination:
        # Create only this stage's dedicated progress directory.
        path = Path(destination)
        # Nested workers use distinct files, avoiding upstream symlink mutation.
        path.parent.mkdir(parents=True, exist_ok=True)
        # Append one flushed event without rewriting the full history.
        with path.open("a", encoding="utf-8") as handle:
            # An append preserves previous completed observations after interruption.
            handle.write(line + "\n")
    # Flush stdout so nested supervisors can forward before the worker exits.
    print(PREFIX + line, flush=True)


# Readers retain partial lines and cap arbitrary library output memory.
class Lines:
    """Split bounded file chunks without leaking huge unstructured lines to output."""

    # One tail reader owns the unfinished suffix for a single file.
    def __init__(self) -> None:
        # No entire log is ever buffered in memory.
        self.pending = ""

    # Feed at most a fixed file read block from the caller.
    def feed(self, value: str) -> list[str]:
        # Complete newline-delimited records before retaining the suffix.
        parts = (self.pending + value).split("\n")
        # Bound unterminated third-party text while preserving valid small JSON events.
        self.pending = parts.pop()[-16384:]
        # Bound long completed library lines too; structured events fit this ceiling.
        return [line[:16384] for line in parts]


# Convert one protocol event into a compact truthful text fallback.
def describe(event: dict) -> str:
    """Render actual task counts with nullable rates/ETA and labelled metrics."""
    # Unknown totals remain indeterminate rather than borrowing stage percentages.
    total = event["total"]
    # A known unit denominator is the only source of a displayed percentage.
    amount = (
        f"{event['completed']}/{total}"
        if total is not None
        else f"{event['completed']}/unknown"
    )
    # Missing throughput cannot support an ETA.
    percent = (
        f"{event['percentage']:.1f}%"
        if event.get("percentage") is not None
        else "indeterminate"
    )
    # Label extrapolation explicitly so it is not a promised finish time.
    eta = (
        f"estimated ETA {event['eta_seconds']:.1f}s"
        if event.get("eta_seconds") is not None
        else "ETA unknown"
    )
    # Metrics are already bounded aggregates validated by the emitter.
    metrics = json.dumps(event.get("metrics", {}), separators=(",", ":"))
    # Keep stage/task identity and the most recent real observation visible.
    return (
        f"[{event['stage']}] {event['task']} {event['status'].upper()} "
        f"{amount} {event['unit']} {percent} elapsed={event['elapsed_seconds']:.1f}s "
        f"rate={event.get('rate_per_second')} {eta} "
        f"{event.get('message', '')} {metrics}"
    )
