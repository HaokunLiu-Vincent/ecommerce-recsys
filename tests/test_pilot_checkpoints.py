"""Retain checkpoint rejection coverage after removal of the parallel pilot trainer.

The historical filename remains for test continuity; every operation below uses
canonical runtime checkpoints and a genuine Two-Tower state. Pointer tampering
and payload corruption must fail rather than silently resuming stale progress.
Real multi-epoch continuation is covered by test_full_checkpoints.py.
"""

from pathlib import Path

import pytest
from test_colab_checkpoints import trained_state

from hm_recsys.core.contracts import read_json, write_json
from hm_recsys.runtime.checkpoints import load_checkpoint, save_checkpoint


def test_canonical_checkpoint_rejects_pointer_and_payload_corruption(
    tmp_path: Path,
) -> None:
    """Reject forged offsets and corrupt snapshots without weakening validation."""
    _, payload, _ = trained_state()
    identity = {"fixture": "canonical-complete-state"}
    save_checkpoint(tmp_path, payload, identity)
    record = read_json(tmp_path / "latest.json")
    write_json(tmp_path / "latest.json", {**record, "next_target_offset": 999})
    with pytest.raises(ValueError, match="progress"):
        load_checkpoint(tmp_path, identity)
    write_json(tmp_path / "latest.json", record)
    path = tmp_path / record["file"]
    with path.open("r+b") as handle:
        handle.seek(-1, 2)
        handle.write(b"X")
    with pytest.raises(ValueError, match="corrupt"):
        load_checkpoint(tmp_path, identity)
