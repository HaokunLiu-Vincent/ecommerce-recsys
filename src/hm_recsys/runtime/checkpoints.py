"""Complete, atomically published Two-Tower training checkpoints.

Consumes model/optimizer/scheduler/RNG and target progress state plus immutable
identity; emits verified local snapshots and optional remote generations. Depends
on file integrity helpers and PyTorch; retains the last valid published checkpoint
on failure.
"""

import os
import random
import re
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from hm_recsys.core.contracts import digest, read_json

# Publish aggregate work counters without exposing source rows or changing RNG.
from hm_recsys.core.progress import emit
from hm_recsys.runtime.files import atomic_json, canonical_hash, verified_copy


# Publish a complete checkpoint; weights alone never imply continuation.
def save_checkpoint(
    folder: Path,
    payload: dict,
    identity: dict,
    *,
    persistent: Path | None = None,
    mount_root: Path | None = None,
    retain_latest_only: bool = False,
    measurements=None,
) -> Path:
    """Publish a complete checkpoint; weights alone never imply continuation."""
    # Create only the required owned directory; immutable inputs and other runs
    # remain untouched.
    folder.mkdir(parents=True, exist_ok=True)
    previous = read_previous(folder) if retain_latest_only else None
    previous_remote = (
        read_previous(persistent) if retain_latest_only and persistent else None
    )
    step = payload["global_step"]
    # Local snapshot serialization has no truthful byte denominator until it ends.
    emit(
        "checkpoint_local",
        0,
        None,
        unit="snapshots",
        status="started",
        message="Saving complete model, optimizer, scheduler, RNG and data offset",
        metrics={"global_step": step},
        force=True,
    )
    suffix = payload.get("checkpoint_suffix", "")
    if suffix and not re.fullmatch(r"[a-zA-Z0-9_-]+", suffix):
        raise ValueError("Unsafe checkpoint suffix")
    name = f"step-{step:08d}{'-' + suffix if suffix else ''}.pt"
    path = folder / name
    temporary = folder / f"{name}.partial"
    state = dict(payload)
    numpy_state = np.random.get_state()
    state.update(
        identity=identity,
        identity_sha256=canonical_hash(identity),
        python_rng=random.getstate(),
        torch_rng=torch.get_rng_state(),
        numpy_rng=[numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
        cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        scaler=None,
        precision="float32_no_amp",
    )
    # Time the named operation here; host dispatch and asynchronous CUDA duration
    # remain distinct measurements.
    with (
        measurements.span("checkpoint_serialization_local_write")
        if measurements
        else nullcontext()
    ):
        torch.save(state, temporary)
        with temporary.open("rb") as handle:
            # Request durable local bytes before the completion pointer can advance.
            os.fsync(handle.fileno())
        # Publish/move the owned filesystem entry at this explicit boundary after its
        # required validation.
        temporary.replace(path)
    # Time the named operation here; host dispatch and asynchronous CUDA duration
    # remain distinct measurements.
    with measurements.span("checkpoint_hash") if measurements else nullcontext():
        sha = digest(path)
    record = {
        "file": name,
        "sha256": sha,
        "bytes": path.stat().st_size,
        "identity_sha256": canonical_hash(identity),
        "global_step": step,
        "next_target_offset": payload["next_target_offset"],
        "boundary": "complete_batch",
        "torch_version": str(torch.__version__),
    }
    atomic_json(folder / "latest.json", record)
    # The local latest pointer follows complete serialization and digest generation.
    emit(
        "checkpoint_local",
        1,
        1,
        unit="snapshots",
        status="completed",
        message="Complete local checkpoint saved and hashed",
        metrics={
            "global_step": step,
            "checkpoint_bytes": record["bytes"],
            "next_target_offset": payload["next_target_offset"],
        },
        force=True,
    )
    if persistent is not None:
        # Preserve the previous verified remote snapshot if any publication step fails.
        try:
            # Publishing remotely is independent from completing a local save.
            emit(
                "checkpoint_remote",
                0,
                None,
                unit="snapshots",
                status="started",
                message="Publishing verified checkpoint; previous remote remains valid",
                metrics={"global_step": step, "checkpoint_bytes": record["bytes"]},
                force=True,
            )
            if mount_root is not None and not os.path.ismount(mount_root):
                raise OSError("Checkpoint persistence mount is unavailable")
            if not persistent.parent.is_dir():
                raise FileNotFoundError(
                    "Checkpoint persistence destination unavailable"
                )
            # Create only the required owned directory; immutable inputs and other runs
            # remain untouched.
            persistent.mkdir(exist_ok=True)
            # Transfer through a temporary file and verify both ends before publishing
            # completion.
            verified_copy(
                path,
                persistent / name,
                expected_sha256=record["sha256"],
                measurements=measurements,
            )
            # Time the named operation here; host dispatch and asynchronous CUDA
            # duration
            # remain distinct measurements.
            with (
                measurements.span("checkpoint_remote_pointer")
                if measurements
                else nullcontext()
            ):
                atomic_json(persistent / "latest.json", record)
            # The remote checkpoint becomes durable evidence only after pointer
            # publication.
            emit(
                "checkpoint_remote",
                1,
                1,
                unit="snapshots",
                status="completed",
                message="Remote snapshot verified and latest pointer published",
                metrics={"global_step": step, "checkpoint_bytes": record["bytes"]},
                force=True,
            )
        # Surface the remote failure without rewriting either latest pointer.
        except BaseException as error:
            # Keep exception details aggregate-only; the caller still receives the
            # error.
            emit(
                "checkpoint_remote",
                0,
                None,
                unit="snapshots",
                status="failed",
                message="Remote publication failed; prior verified snapshot retained",
                metrics={"global_step": step, "error_type": type(error).__name__},
                force=True,
            )
            # Restore the original failure behavior instead of continuing silently.
            raise
    # Full retains a complete continuation checkpoint including its best model.
    # Retire only this run's previously verified checkpoint, after publication.
    for location, old in ((folder, previous), (persistent, previous_remote)):
        if (
            location is not None
            and old
            and old["file"] != name
            and old["identity_sha256"] == canonical_hash(identity)
            and Path(old["file"]).name == old["file"]
        ):
            old_path = location / old["file"]
            if old_path.is_file() and digest(old_path) == old["sha256"]:
                # Remove only the identified owned temporary/lease file, never a
                # source dataset or old run.
                old_path.unlink()
    return path


def read_previous(folder: Path) -> dict | None:
    """Read the latest owned checkpoint record for bounded full-run retention."""
    path = folder / "latest.json"
    return read_json(path) if path.is_file() else None


# Restore full same-runtime optimizer/RNG/data position or reject reuse.
def load_checkpoint(folder: Path, identity: dict) -> dict:
    """Restore full same-runtime optimizer/RNG/data position or reject reuse."""
    record = read_json(folder / "latest.json")
    if record["identity_sha256"] != canonical_hash(identity):
        raise ValueError("Checkpoint configuration/data/order identity mismatch")
    if record["torch_version"] != str(torch.__version__):
        raise ValueError("Checkpoint torch runtime mismatch")
    path = folder / record["file"]
    if path.stat().st_size != record["bytes"] or digest(path) != record["sha256"]:
        raise ValueError("Incomplete or corrupt checkpoint")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state["identity_sha256"] != canonical_hash(identity):
        raise ValueError("Checkpoint payload identity mismatch")
    if any(state[key] != record[key] for key in ("global_step", "next_target_offset")):
        raise ValueError("Checkpoint pointer progress mismatch")
    random.setstate(state["python_rng"])
    npstate = state["numpy_rng"]
    np.random.set_state(
        (npstate[0], np.array(npstate[1], dtype=np.uint32), *npstate[2:])
    )
    torch.set_rng_state(state["torch_rng"])
    # Handle CUDA only under the explicit device condition; input placement cannot
    # silently move the model to CPU.
    if state["cuda_rng"]:
        if not torch.cuda.is_available():
            raise ValueError("CUDA checkpoint cannot silently continue on CPU")
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    # Checkpoint reuse restores progress; it must never be labelled training done now.
    emit(
        "checkpoint_restore",
        1,
        1,
        unit="snapshots",
        status="restored",
        message="Verified complete-batch state and RNG restored",
        metrics={
            "global_step": state["global_step"],
            "next_target_offset": state["next_target_offset"],
        },
        force=True,
    )
    return state
