"""Exercise the canonical full stage graph on generated, bounded inputs only.

Null population/target limits, official fold boundaries, complete Validation,
resource/full authorization, one ranker and fresh-runtime artifact restoration
are checked without real H&M training. The synthetic input is smaller; full
compute settings and production algorithms are retained. Test labels stay off.
"""

import shutil
from pathlib import Path

import numpy as np
import pytest

from hm_recsys.core.contracts import ROOT, read_json
from hm_recsys.data.history import History, canonical, rosters
from hm_recsys.data.streaming import Partition, TargetStream
from hm_recsys.runtime.files import stage_inputs
from hm_recsys.runtime.profiles import execution_gate, load_profile, stages_for
from hm_recsys.runtime.testing import create_full_source, execute_full_fixture


def test_full_none_population_targets_and_truth(tmp_path: Path) -> None:
    """Null means all legal data; streamed event order/Q match a bounded reference."""
    source, run = (tmp_path / "source", tmp_path / "run")
    create_full_source(source)
    run.mkdir()
    profile = load_profile(ROOT, "full")
    canonical(run, source, profile)
    rosters(run, profile)
    history = History(run)
    try:
        stream = TargetStream(history, None)
        reference = history.con.execute(
            """SELECT DISTINCT customer_id,t_dat,article_id
            FROM transactions JOIN cohort USING(customer_id)
            WHERE t_dat>=DATE '2020-05-19' AND t_dat<DATE '2020-08-05'
            ORDER BY sha256('42:' || customer_id || ':' ||
            cast(t_dat AS VARCHAR) || ':' || article_id),
            customer_id,t_dat,article_id"""
        ).fetchall()
        actual = [row for batch in stream.batches(0, 512) for row in batch]
        assert actual == reference and len(actual) == stream.count
        assert [r for b in stream.batches(513, 512) for r in b] == reference[513:]
        from collections import Counter

        counts = Counter((i for _, _, i in actual))
        assert stream.probabilities == {i: n / len(actual) for i, n in counts.items()}
        assert read_json(run / "cohort_manifest.json")["count"] == 24
        assert read_json(run / "population_manifest.json")["base_selection"] is None
        for name in profile["folds"]:
            part = Partition(run, name)
            assert part.count == 24
            users = [u for batch in part.batches(7) for u in batch]
            assert len(users) == len(set(users)) == 24
            assert all(part.truth(users).values())
            part.close()
        validation = Partition(run, "Validation")
        users = next(validation.batches())
        active = history.active(__import__("datetime").date(2020, 9, 9))
        assert any(not truth & active for truth in validation.truth(users).values())
        validation.close()
    finally:
        history.close()


def test_full_authorization_and_resource_rejection(tmp_path: Path, monkeypatch) -> None:
    """Full has real opt-in and host gates; resource rejection never samples."""
    from hm_recsys.runtime.capacity import check
    from hm_recsys.runtime.runner import execute

    profile = load_profile(ROOT, "full")
    stages = stages_for(profile)
    assert stages == (
        "prepare",
        "trend",
        "repurchase",
        "sibling",
        "itemcf",
        "wals",
        "twotower",
        "fusion",
        "features",
        "ranker",
        "validation",
        "reload",
    )
    assert stages[-1] == "reload" and "prepare" in stages
    with pytest.raises(ValueError, match="explicit"):
        execution_gate(profile, allow_full=False, final_test=False)
    with pytest.raises(ValueError, match="development"):
        execute(
            ROOT,
            tmp_path,
            tmp_path / "run",
            tmp_path / "drive",
            profile_name="full",
            allow_full=True,
        )
    with pytest.raises(ValueError, match="authorization"):
        execute_full_fixture(ROOT, tmp_path, tmp_path / "run", tmp_path / "drive")
    monkeypatch.setattr("hm_recsys.runtime.profiles.is_colab_runtime", lambda: True)
    execution_gate(profile, allow_full=True, final_test=False)
    with pytest.raises(ValueError, match="Test"):
        execution_gate(profile, allow_full=True, final_test=True)
    with pytest.raises(RuntimeError, match="blocked_resource"):
        check(tmp_path, "fixture_rejection", memory=10**20, disk=0, details={})
    assert (
        read_json(tmp_path / "capacity/fixture_rejection.json")[
            "estimated_memory_bytes"
        ]
        == 10**20
    )


def test_full_actual_training_restore_and_complete_validation(tmp_path: Path) -> None:
    """Train four folds with one ranker, restore a lost runtime, and reload scores."""
    source = tmp_path / "source"
    create_full_source(source)
    local_data = tmp_path / "runtime/data"
    stage_inputs(source, local_data)
    persistent = tmp_path / "drive/full"
    persistent.parent.mkdir()
    execute_full_fixture(
        ROOT,
        local_data,
        local_data / "runs/full",
        persistent,
        until="features",
        allow_full=True,
    )
    completed = read_json(persistent / "stages/features/complete.json")
    shutil.rmtree(local_data)
    replacement = tmp_path / "replacement/data"
    stage_inputs(source, replacement)
    result = execute_full_fixture(
        ROOT, replacement, replacement / "runs/full", persistent, allow_full=True
    )
    assert result["complete"]
    assert read_json(persistent / "stages/features/complete.json") == completed
    report = read_json(persistent / "run_summary.json")
    assert report["profile"] == "full" and (not report["is_smoke"])
    assert report["twotower"]["epochs"] == 10
    assert report["twotower"]["steps"] > 10
    assert report["wals"]["iterations"] == 15
    assert report["wals"]["factors"] == 64
    assert report["validation"]["eligible_queries"] == 24
    assert report["validation"]["zero_hit_queries"] >= 1
    assert report["reload"]["status"] == "passed"
    assert report["reload"]["max_score_absolute_difference"] == 0
    local = replacement / "runs/full"
    paths = read_json(local / "artifacts.json")
    sampling = read_json(local / paths["ranker_sampling.json"])
    assert set(sampling["fold_groups"]) == {"F1", "F2", "F3", "F4"}
    assert all(sampling["fold_groups"].values())
    assert sampling["injection"] is False and sampling["negative_cap"] == 20
    resolved = read_json(local / paths["profile.resolved.json"])
    original = load_profile(ROOT, "full")
    assert resolved["compute"] == original["compute"]
    assert resolved["training_subset"] is None
    assert resolved["device"] == "cpu"
    assert read_json(local / paths["feature_schema.json"])["cluster_count"] == 128
    assert len(list((persistent / "checkpoints/towers").glob("*.pt"))) == 1
    assert report["ranker"]["split_count"] > 0
    assert all(
        read_json(local / paths[f"channels/F1/{channel}.json"])["rows"] > 0
        for channel in ("trend", "repurchase", "sibling", "itemcf", "wals", "twotower")
    )
    assert "capacity/single_lambdamart.json" in paths
    import pyarrow.parquet as pq

    predictions = pq.read_table(local / paths["predictions/reloaded_top12.parquet"])
    assert len(predictions) == 24
    assert all(
        len(v) == len(set(v)) <= 12 for v in predictions["article_ids"].to_pylist()
    )


def test_parquet_resume_crosses_row_groups(tmp_path: Path) -> None:
    """Bounded offset iteration retains exact order across physical row groups."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from hm_recsys.data.streaming import rows

    path = tmp_path / "rows.parquet"
    pq.write_table(pa.table({"position": np.arange(17001)}), path, row_group_size=8192)
    batches = list(rows(path, 512, 8177))
    assert max(map(len, batches)) <= 512
    assert [r["position"] for b in batches for r in b] == list(range(8177, 17001))


def test_notebook_full_command_contract(tmp_path: Path) -> None:
    """Parse actual root-Notebook Full argv against a clean source-only Git fixture."""
    import json
    import subprocess
    import sys

    from test_github_notebook import git_project
    from test_modular_notebook import _cell, _settings

    with pytest.raises(ValueError, match="explicit"):
        _settings(RUN_PROFILE="full", DEVICE="cuda")
    namespace = _settings(RUN_PROFILE="full", DEVICE="cuda", ALLOW_FULL_RUN=True)
    project = tmp_path / "git_project"
    commit = git_project(project)
    calls = []
    namespace.update(
        PYTHON=sys.executable,
        CLI=[sys.executable, str(project / "scripts/run_pipeline.py")],
        CHECKOUT_COMMIT=commit,
        LOCAL_DATA_ROOT=tmp_path / "data",
        PERSISTENT_ROOT=tmp_path / "drive",
        LOG_ROOT=tmp_path,
        run_command=lambda argv, log: calls.append(argv),
    )
    exec(compile(_cell("execution-plan"), "actual root Full plan", "exec"), namespace)
    argv = calls[0]
    assert "--allow-full" in argv and "--allow-pilot" not in argv
    assert argv[argv.index("--git-commit") + 1] == commit
    assert argv[argv.index("--device") + 1] == "cuda"
    result = subprocess.run(argv, capture_output=True, text=True, check=True)
    plan = json.loads(result.stdout)
    assert plan["stages"] == [
        "prepare",
        "trend",
        "repurchase",
        "sibling",
        "itemcf",
        "wals",
        "twotower",
        "fusion",
        "features",
        "ranker",
        "validation",
        "reload",
    ]
    assert plan["profile"]["compute"]["tt_max_steps"] is None
    assert "final_test" not in plan["stages"]
