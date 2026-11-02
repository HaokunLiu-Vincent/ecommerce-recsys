"""Small stdlib command adapter for the thin Colab notebook."""

import os
import signal
import subprocess
import time
from pathlib import Path


def command(argv: list[str], project: Path, log: Path) -> None:
    """Run one explicit child, retaining logs and terminating its group on interrupt.

    Args:
        argv: Executable and individually quoted arguments, without a shell.
        project: Verified checkout used as the working directory.
        log: Local run log. Existing logs are appended, never discarded.

    Raises:
        CalledProcessError: The command exits unsuccessfully.
    """
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        handle.write(f"\nCommand: {argv!r}\n")
        handle.flush()
        child = subprocess.Popen(
            argv,
            cwd=project,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        started = time.monotonic()
        try:
            while child.poll() is None:
                time.sleep(2)
                if int(time.monotonic() - started) % 30 < 2:
                    print(
                        f"Running ({time.monotonic() - started:.0f}s); log: {log}",
                        flush=True,
                    )
        except BaseException:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            raise
    print(f"Exit {child.returncode}; log: {log}")
    if child.returncode:
        # Bound displayed output; the full log stays local to this run.
        with log.open("rb") as reader:
            reader.seek(max(0, log.stat().st_size - 6000))
            print(reader.read().decode("utf-8", errors="replace"))
        raise subprocess.CalledProcessError(child.returncode, argv)
