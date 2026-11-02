"""Complete-batch tower state, with verified publication and explicit identity."""

import os
import random
import re
from pathlib import Path

import numpy as np
import torch

from hm_recsys.colab.files import atomic_json, canonical_hash, verified_copy
from hm_recsys.contracts import digest, read_json


def save_checkpoint(
    folder: Path,
    payload: dict,
    identity: dict,
    *,
    persistent: Path | None = None,
    mount_root: Path | None = None,
) -> Path:
    """Publish a complete checkpoint; weights alone never imply continuation."""
    folder.mkdir(parents=True, exist_ok=True)
    step = payload["global_step"]
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
    torch.save(state, temporary)
    temporary.replace(path)
    record = {
        "file": name,
        "sha256": digest(path),
        "bytes": path.stat().st_size,
        "identity_sha256": canonical_hash(identity),
        "global_step": step,
        "next_target_offset": payload["next_target_offset"],
        "boundary": "complete_batch",
        "torch_version": str(torch.__version__),
    }
    atomic_json(folder / "latest.json", record)
    if persistent is not None:
        if mount_root is not None and not os.path.ismount(mount_root):
            raise OSError("Checkpoint persistence mount is unavailable")
        if not persistent.parent.is_dir():
            raise FileNotFoundError("Checkpoint persistence destination unavailable")
        persistent.mkdir(exist_ok=True)
        verified_copy(path, persistent / name, expected_sha256=record["sha256"])
        atomic_json(persistent / "latest.json", record)
    return path


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
    random.setstate(state["python_rng"])
    npstate = state["numpy_rng"]
    np.random.set_state(
        (npstate[0], np.array(npstate[1], dtype=np.uint32), *npstate[2:])
    )
    torch.set_rng_state(state["torch_rng"])
    if state["cuda_rng"]:
        if not torch.cuda.is_available():
            raise ValueError("CUDA checkpoint cannot silently continue on CPU")
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    return state
