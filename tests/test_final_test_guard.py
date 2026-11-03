"""Verify the optional final-stage guard using generated synthetic labels only.

This test creates a tiny authenticated input, trains the unchanged full profile
on that tiny input and exercises the explicit Test-only branch after the frozen
Validation lock. It never accesses real final-Test labels or cloud services.
"""

from pathlib import Path

import pytest

from hm_recsys.core.contracts import ROOT, read_json
from hm_recsys.runtime.files import stage_inputs
from hm_recsys.runtime.modular import resolved_profile, run_one
from hm_recsys.runtime.testing import create_full_source, execute_full_fixture


@pytest.mark.integration
def test_synthetic_final_stage_lock_and_no_refitting(tmp_path: Path) -> None:
    """Reject early/unapproved scoring, then score once with the locked model."""
    source = tmp_path / "source"
    create_full_source(source, include_test=True)
    data = tmp_path / "data"
    stage_inputs(source, data)
    local, persistent = tmp_path / "local", tmp_path / "remote"
    profile = resolved_profile(ROOT, "full", device="cpu")

    def final_stage(
        *,
        run_final_test: bool = False,
        approve_final_test: bool = False,
        locked_signature: str | None = None,
    ) -> dict:
        return run_one(
            ROOT,
            data,
            local,
            persistent,
            profile=profile,
            stage="final_test",
            synthetic_fixture=True,
            run_final_test=run_final_test,
            approve_final_test=approve_final_test,
            locked_signature=locked_signature,
        )

    with pytest.raises(ValueError, match="signature"):
        final_stage(
            run_final_test=True,
            approve_final_test=True,
            locked_signature="0" * 64,
        )
    result = execute_full_fixture(
        ROOT, data, local, persistent, allow_full=True, performance="optimized"
    )
    assert (
        result["complete"] and not result["stages"]["final_test"]["persisted_complete"]
    )
    artifacts = read_json(local / "artifacts.json")
    assert not any(name.startswith("labels/Test") for name in artifacts)
    lock = read_json(local / "validation_lock.json")
    with pytest.raises(PermissionError):
        final_stage(run_final_test=True)
    outcome = final_stage(
        run_final_test=True,
        approve_final_test=True,
        locked_signature=lock["signature"],
    )
    assert outcome["status"] == "completed"
    assert read_json(local / "validation_lock.json") == lock
    final_artifacts = read_json(local / "artifacts.json")
    final = read_json(local / final_artifacts["final_test_results.json"])
    assert final["metrics"]["eligible_queries"] == 24
    assert final["refit"] is False
    record_path = persistent / "stages/ranker/complete.json"
    model_record = read_json(record_path)
    again = final_stage(
        run_final_test=True,
        approve_final_test=True,
        locked_signature=lock["signature"],
    )
    assert again["reused"]
    assert read_json(local / "validation_lock.json") == lock
    # A second invocation must reuse the verified stage rather than select again.
    assert {k: v for k, v in outcome.items() if k != "reused"} == {
        k: v for k, v in again.items() if k != "reused"
    }
    assert read_json(record_path) == model_record
