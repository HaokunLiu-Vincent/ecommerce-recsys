"""Stage isolation, dependency identities and default future-label safety.

These offline fixtures exercise real StageStore publication with tiny text
artifacts and actual CLI argument validation. They never read production inputs
or authorize training on the development host.
"""

from pathlib import Path

import pytest

from hm_recsys.core.contracts import ROOT
from hm_recsys.runtime.files import atomic_json
from hm_recsys.runtime.modular import (
    GRAPH,
    ORDER,
    prerequisites,
    resolved_profile,
    run_one,
)
from hm_recsys.runtime.profiles import execution_gate
from hm_recsys.runtime.state import StageStore


def test_sibling_cells_have_no_hidden_training_dependencies() -> None:
    """Rule recall and WALS depend on preparation, never on other model cells."""
    for name in ("trend", "repurchase", "sibling", "itemcf", "wals", "twotower"):
        assert prerequisites(name) == ("prepare",)
    assert set(prerequisites("fusion")) == set(ORDER[:7])
    assert "final_test" not in ORDER


def test_independent_stage_publication_and_restore(tmp_path: Path) -> None:
    """Restore only the declared dependency, rejecting changed and corrupt bytes."""
    store = StageStore(
        tmp_path / "local",
        tmp_path / "remote",
        {"fixture": 1},
        stages=ORDER,
        dependency_graph=GRAPH,
    )

    def work(path: Path) -> None:
        atomic_json(
            path / "report.json", {"complete": True, "attempt": path.parent.name}
        )

    record = store.execute("prepare", {}, {}, work, lambda p: None)
    dependency = {"prepare": record["manifest_sha256"]}
    second = store.execute(
        "wals", dependency, store.view([record]), work, lambda p: None
    )
    assert second["dependencies"] == dependency
    assert store.status()["stages"]["wals"]["persisted_complete"]
    assert not store.status()["stages"]["twotower"]["persisted_complete"]
    with pytest.raises(ValueError, match="Upstream"):
        store.restore("wals", {"prepare": "wrong"})
    with pytest.raises(ValueError, match="identity"):
        StageStore(
            tmp_path / "local",
            tmp_path / "remote",
            {"fixture": 2},
            stages=ORDER,
            dependency_graph=GRAPH,
        )


def test_default_test_denial_precedes_any_input_read(tmp_path: Path) -> None:
    """A nonexistent source proves an unauthorized Test cannot inspect its labels."""
    for name in ("smoke", "full"):
        profile = resolved_profile(ROOT, name)
        with pytest.raises(PermissionError, match="Final Test"):
            run_one(
                ROOT,
                tmp_path / "missing",
                tmp_path / "local",
                tmp_path / "remote",
                profile=profile,
                stage="final_test",
            )
    with pytest.raises(ValueError, match="allow-full"):
        execution_gate(
            resolved_profile(ROOT, "full"), allow_full=False, final_test=False
        )


def test_only_two_active_profiles_and_null_full_limits() -> None:
    """Full means uncapped legal data, never inherited smoke/pilot limits."""
    profile = resolved_profile(ROOT, "full")
    assert profile["training_subset"] is None
    for key in (
        "max_users",
        "max_catalog",
        "max_queries_per_fold",
        "tt_max_targets",
        "tt_max_steps",
    ):
        assert profile["compute"][key] is None
    with pytest.raises(ValueError, match="Only smoke and full"):
        resolved_profile(ROOT, "pilot")
