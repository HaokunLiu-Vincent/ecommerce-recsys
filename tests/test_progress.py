"""Real nested-process latency, failure, unknown totals and model parity evidence."""

# Tests inspect actual worker output rather than mocking its progress counters.
import json

# Inherit and restore process context or manage the owned subprocess environment.
import os

# Distinguish the graceful stop from the exceptional forced-stop retry.
import signal

# Assert actual process exit errors rather than accepting unrelated failures.
import subprocess

# Use the actual current interpreter for child-process contract tests.
import sys

# Use only short bounded waits to detect a wrongly orphaned process.
import time

# Resolve owned local event/log paths independently of file-name spelling.
from pathlib import Path

# Reuse the existing bounded test contracts and explicit expected failures.
import pytest

# Use the real repository root for bounded subprocess contract tests.
from hm_recsys.core.contracts import ROOT

# Exercise the production framing, emission, scope and text-display contract.
from hm_recsys.core.progress import Lines, describe, emit, scope

# Exercise the production Notebook reader and protocol, including the real supervisor.
from hm_recsys.runtime.notebook import _stop_group, command


# Require timely records while a real child remains alive between updates.
def test_nested_worker_events_arrive_before_exit(tmp_path: Path, monkeypatch) -> None:
    """Observe true nested-worker counters before the child exits."""
    # The test worker is generated privately and never substitutes for a model stage.
    worker = tmp_path / "delayed_worker.py"
    # Use actual delayed file/iteration-like observations from a nested process.
    worker.write_text(
        """import sys,time
from pathlib import Path
from hm_recsys.core.progress import emit
stage,run=sys.argv[1:]
for i in range(3):
    emit('delayed',i+1,3,unit='batches',force=True)
    time.sleep(.25)
Path(run,'alive_after_events').touch()
time.sleep(.4)
""",
        encoding="utf-8",
    )
    # The outer child runs the genuine RSS-protected isolated-worker supervisor.
    outer = tmp_path / "outer.py"
    # No pipe/log shortcut bypasses resources.run_stage in this test.
    outer.write_text(
        """import sys
from pathlib import Path
from hm_recsys.core.contracts import Config
from hm_recsys.runtime.resources import run_stage
run_stage('delay',Path(sys.argv[1]),Config.load(),module='delayed_worker')
""",
        encoding="utf-8",
    )
    # All worker imports resolve to this checkout and owned fixture helper.
    monkeypatch.setenv("PYTHONPATH", str(ROOT / "src") + os.pathsep + str(tmp_path))
    # Record wall-clock delivery to distinguish live streaming from post-exit replay.
    observed = []

    # A callback can see the real filesystem while the worker is still running.
    def receive(event: dict) -> None:
        # Keep only the delayed task's actual three events for this latency assertion.
        if event["task"] == "delayed":
            # A marker written after all events proves whether we saw early updates.
            observed.append((event, (tmp_path / "alive_after_events").exists()))

    # Use short display polling without changing the worker's computation schedule.
    command(
        [sys.executable, str(outer), str(tmp_path)],
        ROOT,
        tmp_path / "outer.log",
        poll_seconds=0.02,
        heartbeat_seconds=0.1,
        on_event=receive,
    )
    # Exact counters must survive both layers, and first delivery must precede exit.
    assert [e[0]["completed"] for e in observed] == [1, 2, 3]
    # Delayed worker was still in its first event interval, not already finished.
    assert not observed[0][1]
    # The genuine worker's local JSONL is available as normal stage evidence.
    assert (tmp_path / "progress/delay.jsonl").is_file()


# Unknown totals, partial writes and failure tails must never imply success.
def test_unknown_totals_failure_and_bounded_lines(tmp_path: Path, capsys) -> None:
    """Preserve indeterminate status and surface the last actual inner error."""
    # Emit one no-total event into an owned local file and stdout.
    with scope(
        tmp_path / "events.jsonl", run_id="fixture", profile="smoke", stage="sql"
    ):
        # SQL progress has no inferred row denominator.
        emit("query", 7, None, force=True)
    # Read the exact event persisted before any completion claim.
    event = json.loads((tmp_path / "events.jsonl").read_text())
    # Null total/rate cannot create a percentage or ETA.
    assert event["percentage"] is None and event["eta_seconds"] is None
    # The text fallback must describe this as unknown, not 100%.
    assert "indeterminate" in describe(event) and "ETA unknown" in describe(event)
    # Pathological unstructured output must not grow retained reader state.
    reader = Lines()
    # Feed more than a display window without a terminating newline.
    reader.feed("x" * 100000)
    # The retained incomplete line has a strict fixed upper bound.
    assert len(reader.pending) <= 16384
    # A failed child produces a real exit code and a bounded, visible error tail.
    script = (
        "print('z'*100000); print('INNER_FAILURE_REASON',flush=True); "
        "raise SystemExit(7)"
    )
    # The helper propagates failure instead of marking the stage completed.
    with pytest.raises(subprocess.CalledProcessError) as failure:
        # Execute an actual subprocess, including a huge prior unstructured line.
        command(
            [sys.executable, "-c", script],
            ROOT,
            tmp_path / "failed.log",
            poll_seconds=0.02,
        )
    # Inspect the actual subprocess return code rather than an invented stage state.
    assert getattr(failure.value, "returncode", None) == 7
    # The reason remains visible after bounded truncation of earlier noise.
    assert "INNER_FAILURE_REASON" in capsys.readouterr().out


# A Notebook interrupt must stop its owned worker process group and propagate.
def test_reader_interrupt_propagates(tmp_path: Path) -> None:
    """Do not complete a command when a live UI callback interrupts it."""
    # A real event arrives before the worker's long sleep.
    script = (
        "from hm_recsys.core.progress import emit; import time; "
        "emit('wait',0,None,force=True); time.sleep(30)"
    )

    # The UI callback models the operator stopping a live cell.
    def interrupt(event: dict) -> None:
        # Interrupt only after observing actual child output.
        raise KeyboardInterrupt

    # The helper must stop its child group and re-raise rather than return success.
    with pytest.raises(KeyboardInterrupt):
        # A bounded callback-triggered stop avoids waiting for the long worker sleep.
        command(
            [sys.executable, "-c", script],
            ROOT,
            tmp_path / "interrupt.log",
            on_event=interrupt,
        )


# The fix for compacted feature shards must bridge Arrow's old 32-row reference batches.
def test_reload_expected_rows_cross_batch_boundary(tmp_path: Path) -> None:
    """Compare complete groups >32 without depending on Parquet batch boundaries."""
    # Keep native LightGBM/PyArrow imports local to this focused test.
    import pyarrow as pa

    # Construct a physical Parquet reference with uneven row groups for reload tests.
    import pyarrow.parquet as pq

    # Exercise the real reload reader, not a hand-copied implementation.
    from hm_recsys.ranking.pipeline import ExpectedRows

    # A small typed table crosses several physical row groups.
    path = tmp_path / "top12.parquet"
    # These are row coordinates only, never user purchase histories.
    pq.write_table(pa.table({"row": list(range(100))}), path, row_group_size=17)
    # Actual reload batches can contain up to512 whole query records.
    reader = ExpectedRows(pq.ParquetFile(path))
    # Read uneven requested group counts across the old 32-row boundary.
    assert len(reader.take(65)) == 65
    # The next block must begin at the immediately following reference row.
    assert reader.take(34)[0]["row"] == 65
    # EOF yields only remaining rows, allowing existing parity checks to reject
    # shortages.
    assert len(reader.take(5)) == 1 and reader.take(1) == []


# A dead CLI leader cannot leave its owned numerical descendant working in the
# background.
def test_failed_leader_stops_descendants_and_drains_partial_line(
    tmp_path: Path, capsys
) -> None:
    """Preserve unterminated failure reason and terminate the entire owned group."""
    # Child completion would leave a marker if group cleanup failed.
    marker = tmp_path / "orphan_completed"
    # This real descendant is intentionally longer-lived than its failed leader.
    descendant = (
        "import time; from pathlib import Path; time.sleep(1); Path("
        + repr(str(marker))
        + ").touch()"
    )
    # The leader exits after launching the descendant and writing a non-newline error.
    script = (
        "import subprocess,sys; subprocess.Popen([sys.executable,'-c',"
        + repr(descendant)
        + "]); sys.stderr.write('PARTIAL_FAILURE_REASON'); "
        "sys.stderr.flush(); sys.exit(9)"
    )
    # The Notebook helper must propagate the actual leader failure.
    with pytest.raises(subprocess.CalledProcessError) as failure:
        # The owned group should be stopped even though its leader is already dead.
        command(
            [sys.executable, "-c", script],
            ROOT,
            tmp_path / "orphan.log",
            poll_seconds=0.02,
        )
    # Allow the descendant's intended completion deadline to pass, without long sleeps.
    time.sleep(1.2)
    # Only the intended real leader exit qualifies this failure-path test.
    assert failure.value.returncode == 9
    # No background descendant may perform the scheduled side effect.
    assert not marker.exists()
    # The final stderr bytes remain visible despite lacking a terminating newline.
    assert "PARTIAL_FAILURE_REASON" in capsys.readouterr().out


# A transient kernel-reaper denial may clear, while persistent denial stays explicit.
@pytest.mark.parametrize("persistent", [False, True])
def test_owned_group_permission_retry(monkeypatch, persistent: bool) -> None:
    """Retry a bounded transient denial without hiding a real cleanup failure."""
    # Use a real already-reaped child so the fixture never leaves a live descendant.
    child = subprocess.Popen([sys.executable, "-c", "pass"], start_new_session=True)
    # Reap before mocking signals so no real process can receive a fixture signal.
    child.wait(timeout=5)
    # Record forced-stop attempts independently of the graceful TERM path.
    attempts = []

    # Simulate only the observed kernel signal boundary, never model progress/results.
    def killpg(pid: int, signum: int) -> None:
        # Ensure every retry targets the original owned group rather than another PID.
        assert pid == child.pid
        # The initial graceful termination remains a normal successful call.
        if signum == signal.SIGTERM:
            # This fixture has no live process to terminate.
            return
        # Only KILL is eligible for the bounded permission-denial retry.
        assert signum == signal.SIGKILL
        # Retain the exact number of attempted forced stops.
        attempts.append(signum)
        # A permanently denied group must raise rather than pretending completion.
        if persistent or len(attempts) < 3:
            # No private process metadata is needed for this controlled exception.
            raise PermissionError("synthetic cleanup permission denial")
        # Reaping removes the group after the transient denial clears.
        raise ProcessLookupError("synthetic group already reaped")

    # Supply a deterministic deadline instead of sleeping through an OS race.
    ticks = iter([0.0, 1.0] if persistent else [0.0, 0.01, 0.02])
    # Patch only the signal operation exercised by the production cleanup function.
    monkeypatch.setattr(os, "killpg", killpg)
    # The real child was already reaped, so child.wait returns immediately.
    monkeypatch.setattr(time, "monotonic", lambda: next(ticks))
    # Deadline behavior is tested without introducing a wall-time assertion.
    monkeypatch.setattr(time, "sleep", lambda _: None)
    # Persistent denial remains an actionable failure rather than a swallowed warning.
    if persistent:
        # A verified completion cannot be claimed when the process cannot be signalled.
        with pytest.raises(PermissionError, match="synthetic cleanup"):
            # Use the actual helper called after a failed Notebook subprocess.
            _stop_group(child)
        # The expired deadline permits exactly one forced-stop attempt.
        assert len(attempts) == 1
    else:
        # A transient denial is resolved only after the owned group is known gone.
        _stop_group(child)
        # Two denied signals precede the kernel's already-reaped response.
        assert len(attempts) == 3
