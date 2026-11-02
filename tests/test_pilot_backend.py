"""Bounded tests of real pilot operations, never real local pilot training."""

from datetime import date
from pathlib import Path

import numpy as np
import pytest

from hm_recsys.colab.files import stage_inputs
from hm_recsys.contracts import ROOT, Config, read_json
from hm_recsys.data import Transaction
from hm_recsys.learned import WALS, confidence, fold_user
from hm_recsys.pilot.data import (
    History,
    canonical,
)
from hm_recsys.pilot.snapshot import GlobalSnapshot, build_snapshot
from hm_recsys.pilot.testing import create_source, execute_fixture, fixture_profile
from hm_recsys.pilot.vectors import BlockIndex, block_topk
from hm_recsys.recall import Snapshot


@pytest.fixture
def canonical_fixture(tmp_path: Path):
    """Construct tiny global canonical data, separate from the training cohort."""
    source = tmp_path / "source"
    create_source(source)
    run = tmp_path / "run"
    run.mkdir()
    (run / "models").mkdir()
    (run / "tmp").mkdir()
    profile = fixture_profile(ROOT)
    canonical(run, source, profile)
    yield source, run, profile


def test_global_snapshot_and_bounded_histories(canonical_fixture) -> None:
    """SQL global counts/ItemCF agree with the raw-line reference, not a cohort."""
    source, run, _ = canonical_fixture
    history = History(run)
    try:
        users = [
            r[0]
            for r in history.con.execute(
                "SELECT customer_id FROM customers ORDER BY customer_id"
            ).fetchall()
        ]
        before = date(2020, 8, 12)
        histories = history.histories(users, before)
        sample = history.sample(users)
        sample.transactions = [r for rows in histories.values() for r in rows]
        sample.manifest = {
            "sampling_scope": "smoke",
            "validated": True,
            "history_end_exclusive": "2020-09-16",
            "users": len(users),
            "catalog": len(sample.catalog),
            "transactions": len(sample.transactions),
        }
        reference = Snapshot(sample, cutoff=date(2020, 8, 11), config=Config.load())
        build_snapshot(run, "F1", date(2020, 8, 11))
        actual = GlobalSnapshot(history, run, "F1", date(2020, 8, 11))
        actual.set_users(users)
        assert actual.active == reference.active
        assert actual.sales == reference.sales
        assert actual.bucket_ranks == reference.bucket_ranks
        for bucket, values in reference.pop.items():
            for item, score in values.items():
                assert actual.pop[bucket][item] == pytest.approx(score, rel=1e-12)
        for user in users:
            assert actual.repurchase(user) == reference.repurchase(user)
            assert actual.sibling(user) == reference.sibling(user)
        for item, neighbors in actual.itemcf.items():
            expected = dict(reference.itemcf[item])
            for other, score in neighbors:
                assert score == pytest.approx(expected[other], rel=1e-12)
        assert read_json(run / "population_manifest.json")["base_users"] == 12
        assert len(users) == 24
        # The same FeatureBuilder consumes raw reference rows or SQL aggregates.
        from hm_recsys.contracts import write_json
        from hm_recsys.features import CATEGORICAL, FEATURES, FeatureBuilder
        from hm_recsys.pilot.data import MappedContent, adapt_content
        from hm_recsys.recall import Candidate, Hit

        adapt_content(run, source)
        content = MappedContent(run)
        valid = np.flatnonzero(content.masks["image"])[:8]
        np.save(run / "models/visual_centers.npy", content.vectors["image"][valid])
        maps = {
            key: {
                value: j
                for j, value in enumerate(
                    sorted({str(meta[key]) for meta in sample.articles.values()})
                )
            }
            for key in ("product_type_no", "index_group_no")
        }
        write_json(
            run / "feature_schema.json",
            {
                "names": list(FEATURES),
                "cluster_count": 8,
                "category_maps": maps,
                "categorical": list(CATEGORICAL),
            },
        )
        builder = FeatureBuilder(content, run)
        for user in users[:3]:
            candidates = [
                Candidate(i, {"trend": Hit(float(j + 1), j + 1)})
                for j, i in enumerate(sorted(actual.active)[:8])
            ]
            np.testing.assert_allclose(
                builder.build(reference, user, candidates),
                builder.build(actual, user, candidates),
                rtol=1e-6,
                atol=1e-6,
                equal_nan=True,
            )

        events = history.event_histories(
            [(users[0], date(2020, 6, 15)), (users[0], date(2020, 8, 1))]
        )
        from hm_recsys.towers import event_history

        for actual_rows, day in zip(
            events, (date(2020, 6, 15), date(2020, 8, 1)), strict=True
        ):
            assert actual_rows == event_history(histories[users[0]], before=day)
        actual.close()
    finally:
        history.close()


def test_large_exact_index_and_cached_wals(tmp_path: Path) -> None:
    """Keep FAISS out of the pytest process that also exercises PyTorch."""
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            "-c",
            "import runpy,sys; from pathlib import Path; "
            "runpy.run_path(sys.argv[1])['_verify_large_index_and_cached_wals']"
            "(Path(sys.argv[2]))",
            str(Path(__file__).resolve()),
            str(tmp_path),
        ],
        check=True,
        timeout=60,
    )


def _verify_large_index_and_cached_wals(tmp_path: Path) -> None:
    """Exercise 6,001 IDs, distant eligible ties, and a cached fixed-factor solve."""
    import faiss

    faiss.omp_set_num_threads(2)
    ids = [f"{j:010d}" for j in range(6001)]
    rng = np.random.default_rng(42)
    vectors = rng.normal(size=(len(ids), 8)).astype(np.float32)
    vectors[5500:] = vectors[0]
    active = set(ids[5500:])
    query = np.ones((1, 8), dtype=np.float32)
    index = BlockIndex(ids, vectors)
    expected = sorted(
        ((i, float(vectors[j] @ query[0])) for j, i in enumerate(ids) if i in active),
        key=lambda x: (-x[1], x[0]),
    )[:50]
    result = block_topk(ids, vectors, query, active, 50)
    assert [i for i, _ in result[0]] == [i for i, _ in expected]
    np.testing.assert_allclose(
        [v for _, v in result[0]], [v for _, v in expected], rtol=1e-6, atol=1e-6
    )
    index.save(tmp_path, "learned")
    assert (
        BlockIndex.load(tmp_path, "learned").search(query[0], active, 50) == result[0]
    )
    model = WALS(index)
    rows = [
        Transaction("u", ids[3], date(2020, 8, 1), 0.1, 1),
        Transaction("u", ids[3], date(2020, 8, 1), 0.1, 1),
        Transaction("u", ids[9], date(2020, 8, 8), 0.1, 2),
    ]
    obs = confidence(rows, cutoff=date(2020, 8, 11), vocabulary=model.mapping)
    np.testing.assert_allclose(
        fold_user(vectors, obs),
        fold_user(vectors, obs, gram=model.gram),
        rtol=1e-6,
        atol=1e-7,
    )


def test_pilot_staged_fixture_and_restore(tmp_path: Path) -> None:
    """Real pilot WALS/two epochs/ranker through F1/Validation and fresh reload."""
    source = tmp_path / "source"
    create_source(source)
    data = tmp_path / "local/data"
    stage_inputs(source, data)
    persistent = tmp_path / "drive/pilot"
    persistent.parent.mkdir()
    first = execute_fixture(
        ROOT, data, data / "runs/pilot", persistent, until="index_towers"
    )
    assert first["stages"]["index_towers"]["persisted_complete"]
    replacement = tmp_path / "replacement/data"
    stage_inputs(source, replacement)
    import shutil

    shutil.rmtree(data)
    final = execute_fixture(ROOT, replacement, replacement / "runs/pilot", persistent)
    assert final["complete"]
    assert "features_F2" not in final["stages"]
    report = read_json(persistent / "run_summary.json")
    assert report["profile"] == "pilot" and not report["is_smoke"]
    assert report["twotower"]["epochs"] == 2
    assert report["twotower"]["steps"] > 0
    assert report["sample"]["catalog_items"] == 42
    assert report["ranker"]["split_count"] > 0
    assert report["reload"]["status"] == "passed"
    assert all(report["partitions"]["F1"]["channel_rows"].values())
    import zipfile

    bundle = read_json(persistent / "review_manifest.json")
    with zipfile.ZipFile(persistent / bundle["archive"]) as archive:
        payload = archive.read("pilot_review.json").decode()
    # Descriptive sampling strings may name the customer_id field; values may not.
    users = read_json(
        replacement
        / "runs/pilot"
        / read_json(replacement / "runs/pilot/artifacts.json")[
            "queries/Validation.json"
        ]
    )["users"]
    assert all(user not in payload for user in users)
    assert '"query_order"' not in payload and '"users"' not in payload

    # Change only a new synthetic copy to exercise zero-candidate Validation.
    # Frozen real fixture models and F1 groups are reused, without fake scores.
    import duckdb

    from hm_recsys.resources import run_stage

    local = replacement / "runs/pilot"
    empty = tmp_path / "empty_validation"
    for name, relative in read_json(local / "artifacts.json").items():
        if name.startswith(("features/Validation/", "candidates/Validation/")):
            continue
        if name.endswith(".log"):
            continue
        destination = empty / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local / relative, destination)
    with duckdb.connect(str(empty / "snapshots/Validation.duckdb")) as connection:
        connection.execute("DELETE FROM active")
    for stage in ("features_Validation", "ranker"):
        run_stage(stage, empty, Config.load(), module="hm_recsys.colab.worker")
    zero = read_json(empty / "pilot_results.json")
    assert zero["partitions"]["Validation"]["zero_candidate_queries"] == 24
    assert zero["partitions"]["Validation"]["zero_retrieved_positive_queries"] == 24
    assert zero["validation"]["MAP@12"] == 0
    assert zero["validation"]["NDCG@12"] == 0
    assert len(read_json(empty / "predictions/reference_top12.json")) == 24


def test_notebook_probe_uses_actual_parser(tmp_path: Path) -> None:
    """Execute the actual probe cell with a recorder; no mount/install/CUDA occurs."""
    import importlib.util

    import nbformat

    spec = importlib.util.spec_from_file_location(
        "probe_entry", ROOT / "scripts/probe_device.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    notebook = nbformat.read(ROOT / "notebooks/01_Colab_Training.ipynb", as_version=4)
    source = next(
        c.source
        for c in notebook.cells
        if c.cell_type == "code" and "if RUN_CUDA_PROBE:" in c.source
    )
    calls = []
    namespace = {
        "RUN_CUDA_PROBE": True,
        "LOCAL_DATA_ROOT": tmp_path / "data",
        "RUN_ID": "fixture",
        "PYTHON": "/explicit/project/python",
        "PROJECT_ROOT": ROOT,
        "LOG_ROOT": tmp_path / "logs",
        "CLI": ["python", "pipeline.py"],
        "RUN_ACTION": "run",
        "OPTIONS": [],
        "command": lambda argv, *_: calls.append(argv),
    }
    exec(compile(source, "actual notebook probe cell", "exec"), namespace)
    first = calls[0]
    assert first[:2] == [
        "/explicit/project/python",
        str(ROOT / "scripts/probe_device.py"),
    ]
    parsed = module.build_parser().parse_args(first[2:])
    assert parsed.device == "cuda" and parsed.output.is_relative_to(
        tmp_path / "data/runs/colab"
    )
    assert "--run-id" not in first
    exec(compile(source, "actual notebook probe cell", "exec"), namespace)
    assert module.build_parser().parse_args(calls[2][2:]).output != parsed.output
    assert not (tmp_path / "data").exists()


def test_pilot_guards_and_complete_truth(
    canonical_fixture, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No local real fitting, no fake fixture marker, no hidden resource sampling."""
    from hm_recsys.colab.profiles import execution_gate, load_profile, stages_for
    from hm_recsys.colab.runner import execute
    from hm_recsys.pilot.data import allocation, rosters
    from hm_recsys.pilot.testing import assert_fixture

    source, run, profile = canonical_fixture
    with pytest.raises(ValueError, match="development"):
        execute(
            ROOT,
            source,
            tmp_path / "local",
            tmp_path / "remote",
            profile_name="pilot",
            allow_pilot=True,
        )
    with pytest.raises(ValueError, match="explicit"):
        execution_gate(load_profile(ROOT, "pilot"), allow_full=False, final_test=False)
    with pytest.raises(ValueError, match="Test"):
        execution_gate(
            load_profile(ROOT, "pilot"),
            allow_full=False,
            final_test=True,
            allow_pilot=True,
        )
    with pytest.raises(RuntimeError, match="blocked_resource"):
        allocation(10**20, "oversized fixture")
    stages = stages_for(profile)
    assert "features_F1" in stages and "features_Validation" in stages
    assert not {"features_F2", "features_F3", "features_F4"} & set(stages)
    monkeypatch.setattr("hm_recsys.colab.profiles.is_colab_runtime", lambda: True)
    execution_gate(
        load_profile(ROOT, "pilot"),
        allow_full=False,
        final_test=False,
        allow_pilot=True,
    )
    with pytest.raises(RuntimeError, match="blocked_implementation"):
        execution_gate(load_profile(ROOT, "full"), allow_full=True, final_test=False)
    rosters(run, profile)
    history = History(run)
    active = history.active(date(2020, 8, 12))
    truth = read_json(run / "labels/F1.json")
    assert "0000000042" not in active
    assert any("0000000042" in v for v in truth.values())
    assert len(read_json(run / "queries/Validation.json")["users"]) == 24
    history.close()
    (source / "original data/customers.csv").write_text(
        "self-declared real data", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="exact generated"):
        assert_fixture(source)


def test_cuda_request_does_not_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """A parsed CUDA profile is not evidence of CUDA computation."""
    from hm_recsys.pilot.towers import device_for

    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    with pytest.raises(RuntimeError, match="no CPU fallback"):
        device_for({"device": "cuda"})
    assert str(device_for({"device": "cpu"})) == "cpu"


def test_cuda_probe_explicitly_not_run(tmp_path: Path) -> None:
    """CUDA verification is opt-in and separately reported from CPU tests."""
    import os

    if os.environ.get("HM_RUN_CUDA_TESTS") != "1":
        pytest.skip("CUDA not requested; actual new-release CUDA execution is not run")
    from hm_recsys.colab.device import probe

    assert probe(tmp_path / "cuda", "cuda")["status"] == "passed"


def test_future_partition_cannot_change_earlier_snapshot(canonical_fixture) -> None:
    """Future Parquet rows cannot change earlier cutoff aggregates."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    _, run, _ = canonical_fixture
    build_snapshot(run, "before", date(2020, 8, 11))
    history = History(run)
    original = GlobalSnapshot(history, run, "before", date(2020, 8, 11))
    sales = {day: dict(counts) for day, counts in original.sales.items()}
    active = set(original.active)
    original.close()
    history.close()
    future = pa.table(
        {
            "customer_id": ["fixture-user-000"] * 3,
            "article_id": ["0000000041"] * 3,
            "t_dat": [date(2020, 8, 12)] * 3,
            "price": [999.0] * 3,
            "sales_channel_id": pa.array([2] * 3, type=pa.int32()),
        }
    )
    pq.write_table(
        future, run / "canonical/transactions/month=2020-08/future_fixture.parquet"
    )
    build_snapshot(run, "after", date(2020, 8, 11))
    history = History(run)
    after = GlobalSnapshot(history, run, "after", date(2020, 8, 11))
    assert after.active == active and after.sales == sales
    assert history.active(date(2020, 8, 13)) >= active | {"0000000041"}
    assert all(
        row.day < date(2020, 8, 12)
        for row in history.histories(["fixture-user-000"], date(2020, 8, 12))[
            "fixture-user-000"
        ]
    )
    after.close()
    history.close()
