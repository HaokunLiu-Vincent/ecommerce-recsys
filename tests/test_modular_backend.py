"""Validate shared SQL retrieval, genuine smoke stages and portable recovery.

Tiny synthetic sources exercise the same canonical backend as full. Independent
scalar references check global statistics and directed ItemCF; no retired model
implementation or archived source is imported. CUDA remains opt-in and real H&M
pilot/full training is never launched on the development host.
"""

import csv
import math
import shutil
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest

from hm_recsys.core.contracts import ROOT, read_json, write_json
from hm_recsys.data.history import History, allocation, canonical, rosters
from hm_recsys.data.sample import Transaction, age_bucket
from hm_recsys.data.streaming import Partition
from hm_recsys.retrieval.snapshot import GlobalSnapshot, build_snapshot
from hm_recsys.retrieval.vectors import BlockIndex, block_topk
from hm_recsys.retrieval.wals.model import WALS, confidence, fold_user
from hm_recsys.runtime.files import stage_inputs
from hm_recsys.runtime.profiles import load_profile
from hm_recsys.runtime.testing import create_source


@pytest.fixture
def canonical_fixture(tmp_path: Path):
    """Keep complete legal statistics while a synthetic gradient cohort is smaller."""
    source, run = (tmp_path / "source", tmp_path / "run")
    create_source(source)
    run.mkdir()
    (run / "models").mkdir()
    (run / "tmp").mkdir()
    profile = load_profile(ROOT, "full")
    canonical(run, source, profile)
    import duckdb

    with duckdb.connect(str(run / "canonical/history.duckdb")) as con:
        con.execute(
            "DELETE FROM cohort WHERE customer_id NOT IN "
            "(SELECT customer_id FROM cohort ORDER BY customer_id LIMIT 12)"
        )
    yield (source, run, profile)


def test_global_snapshot_and_bounded_histories(canonical_fixture) -> None:
    """Global SQL statistics match raw purchase lines independently of fit cohort."""
    source, run, profile = canonical_fixture
    history = History(run)
    cutoff = date(2020, 8, 11)
    before = cutoff + timedelta(days=1)
    users = [
        r[0]
        for r in history.con.execute(
            "SELECT customer_id FROM customers ORDER BY customer_id"
        ).fetchall()
    ]
    assert len(users) == 24
    assert history.con.execute("SELECT count(*) FROM cohort").fetchone() == (12,)
    with (source / "original data/transactions_train.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        raw = [r for r in csv.DictReader(handle) if r["t_dat"] < str(before)]
    build_snapshot(run, "F1", cutoff, optimized=True, profile=profile)
    actual = GlobalSnapshot(history, run, "F1", cutoff)
    actual.set_users(users)
    try:
        sales = {
            days: Counter(
                r["article_id"]
                for r in raw
                if r["t_dat"] >= str(before - timedelta(days=days))
            )
            for days in (3, 7, 14, 28)
        }
        assert actual.sales == sales
        assert actual.active == {item for item, n in sales[28].items() if n >= 3}
        customers = history.customers(users)
        expected_pop: dict[str, dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        bucket_counts = defaultdict(Counter)
        edges = defaultdict(dict)
        for row in raw:
            day = date.fromisoformat(row["t_dat"])
            user = row["customer_id"]
            item = row["article_id"]
            if day >= before - timedelta(days=28):
                edges[user][item] = max(day, edges[user].get(item, day))
            if day >= before - timedelta(days=7):
                bucket = age_bucket(customers[user]["age"])
                weight = math.exp(-(cutoff - day).days / 3)
                expected_pop[bucket][item] += weight
                expected_pop["GLOBAL"][item] += weight
                bucket_counts[bucket][item] += 1
        for bucket, values in expected_pop.items():
            for item, score in values.items():
                assert actual.pop[bucket][item] == pytest.approx(score, rel=1e-12)
        expected_ranks = {
            bucket: {
                item: j + 1
                for j, (item, _) in enumerate(
                    sorted(
                        ((i, n) for i, n in counts.items() if i in actual.active),
                        key=lambda p: (-p[1], p[0]),
                    )
                )
            }
            for bucket, counts in bucket_counts.items()
        }
        assert dict(actual.bucket_ranks) == expected_ranks
        for seed, neighbors in actual.itemcf.items():
            ui = {u for u, e in edges.items() if seed in e}
            for other, score in neighbors:
                uj = {u for u, e in edges.items() if other in e}
                expected = sum(
                    math.exp(-abs((edges[u][seed] - edges[u][other]).days) / 7)
                    / math.log1p(len(edges[u]))
                    for u in ui & uj
                ) / math.sqrt(len(ui) * len(uj) ** 0.75)
                assert score == pytest.approx(expected, rel=1e-12)
        from hm_recsys.features.builder import CATEGORICAL, FEATURES, FeatureBuilder
        from hm_recsys.features.content import MappedContent, adapt_content
        from hm_recsys.fusion.native import Candidate
        from hm_recsys.retrieval.base import Hit

        adapt_content(run, source)
        content = MappedContent(run)
        valid = np.flatnonzero(content.masks["image"])[:8]
        np.save(run / "models/visual_centers.npy", content.vectors["image"][valid])
        maps = {
            key: {
                value: j
                for j, value in enumerate(
                    sorted({str(meta[key]) for meta in history.articles.values()})
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
                Candidate(item, {"trend": Hit(float(j + 1), j + 1)})
                for j, item in enumerate(sorted(actual.active)[:8])
            ]
            values = builder.build(actual, user, candidates)
            assert values.shape == (len(candidates), 49)
            for j, candidate in enumerate(candidates):
                assert (
                    values[j, FEATURES.index("item_sales_28d")]
                    == sales[28][candidate.article]
                )
                assert values[j, FEATURES.index("rrf_fused_score")] == pytest.approx(
                    1 / (61 + j)
                )
                assert np.isnan(values[j, FEATURES.index("wals_score")])
        from hm_recsys.retrieval.two_tower.inputs import event_history

        histories = history.histories(users, before)
        days = (date(2020, 6, 15), date(2020, 8, 1))
        events = history.event_histories([(users[0], day) for day in days])
        for rows, day in zip(events, days, strict=True):
            assert rows == event_history(histories[users[0]], before=day)
    finally:
        actual.close()
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
            "runpy.run_path(sys.argv[1])"
            "['_verify_large_index_and_cached_wals'](Path(sys.argv[2]))",
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
        [v for _, v in result[0]], [v for _, v in expected], rtol=1e-06, atol=1e-06
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
        rtol=1e-06,
        atol=1e-07,
    )


def test_modular_staged_fixture_restore_and_empty_validation(tmp_path: Path) -> None:
    """Resume genuine smoke stages in a fresh directory and preserve failed queries."""
    from hm_recsys.runtime.modular import resolved_profile, run_all

    source = tmp_path / "source"
    create_source(source)
    data = tmp_path / "local/data"
    stage_inputs(source, data)
    persistent = tmp_path / "drive/smoke"
    persistent.parent.mkdir()
    profile = resolved_profile(ROOT, "smoke")
    first = run_all(
        ROOT,
        data,
        data / "runs/smoke",
        persistent,
        profile=profile,
        until="twotower",
        synthetic_fixture=True,
    )
    assert first["stages"]["twotower"]["persisted_complete"]
    replacement = tmp_path / "replacement/data"
    stage_inputs(source, replacement)
    shutil.rmtree(data)
    local = replacement / "runs/smoke"
    final = run_all(
        ROOT, replacement, local, persistent, profile=profile, synthetic_fixture=True
    )
    assert final["complete"]
    report = read_json(persistent / "run_summary.json")
    assert report["profile"] == "smoke" and report["is_smoke"]
    assert report["twotower"]["epochs"] == 1 and report["twotower"]["steps"] > 0
    assert (
        report["ranker"]["split_count"] > 0 and report["reload"]["status"] == "passed"
    )
    assert all(report["partitions"]["F1"]["channel_rows"].values())
    paths = read_json(local / "artifacts.json")
    payload = "\n".join(p.read_text() for p in (local / "reports").glob("*.json"))
    roster_path = local / paths["queries/Validation.parquet"]
    import pyarrow.parquet as pq

    users = pq.read_table(roster_path)["customer_id"].to_pylist()
    assert all(user not in payload for user in users)
    assert '"query_order"' not in payload and '"users"' not in payload
    empty = tmp_path / "empty_validation"
    for name, relative in paths.items():
        if (
            name.startswith(
                (
                    "features/Validation/",
                    "channels/Validation/",
                    "fusion/Validation/",
                    "predictions/",
                    "reports/",
                    "phase_reports/",
                )
            )
            or name == "ranker_groups.jsonl"
            or name.endswith(".log")
        ):
            continue
        target = empty / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local / relative, target)
    import duckdb

    with duckdb.connect(str(empty / "snapshots/Validation.duckdb")) as con:
        con.execute("DELETE FROM active")
    from hm_recsys.core.contracts import CHANNELS, Config
    from hm_recsys.features.pipeline import generate
    from hm_recsys.fusion.pipeline import fuse
    from hm_recsys.retrieval.pipeline import run_channel
    from hm_recsys.runtime.resources import run_stage

    (empty / "predictions").mkdir(exist_ok=True)
    for channel in CHANNELS:
        run_channel(empty, profile, channel, partitions=["Validation"])
    fuse(empty, profile, partitions=["Validation"])
    generate(empty, profile, "Validation")
    write_json(empty / "profile.resolved.json", profile)
    write_json(
        empty / "colab_stage.json",
        {"synthetic_fixture": True, "data_root": str(replacement)},
    )
    # Keep LightGBM fitting out of the parent process that tested Torch/FAISS.
    run_stage("ranker_fit", empty, Config.load())
    zero = read_json(empty / "smoke_results.json")
    assert zero["partitions"]["Validation"]["zero_candidate_queries"] == len(users)
    assert zero["validation"]["zero_hit_queries"] == len(users)
    assert zero["validation"]["MAP@12"] == zero["validation"]["NDCG@12"] == 0
    predictions = pq.read_table(empty / "predictions/reference_top12.parquet")
    assert len(predictions) == len(users) and all(
        not row for row in predictions["article_ids"].to_pylist()
    )


def test_probe_cli_uses_actual_parser(tmp_path: Path) -> None:
    """The explicit device-probe CLI has no invented run-id argument."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "probe_entry", ROOT / "scripts/probe_device.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    parsed = module.build_parser().parse_args(
        ["--device", "cuda", "--output", str(tmp_path / "probe")]
    )
    assert parsed.device == "cuda" and parsed.output == tmp_path / "probe"
    with pytest.raises(SystemExit):
        module.build_parser().parse_args(["--run-id", "invented"])
    assert not (tmp_path / "probe").exists()


def test_full_guards_and_complete_truth(
    canonical_fixture, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject unauthorized full/Test work and retain all legal future positives."""
    from hm_recsys.runtime.modular import ORDER
    from hm_recsys.runtime.profiles import execution_gate
    from hm_recsys.runtime.runner import execute
    from hm_recsys.runtime.testing import assert_fixture

    source, run, profile = canonical_fixture
    with pytest.raises(ValueError, match="development"):
        execute(
            ROOT,
            source,
            tmp_path / "local",
            tmp_path / "remote",
            profile_name="full",
            allow_full=True,
        )
    with pytest.raises(ValueError, match="explicit"):
        execution_gate(load_profile(ROOT, "full"), allow_full=False, final_test=False)
    with pytest.raises(ValueError, match="Test"):
        execution_gate(load_profile(ROOT, "full"), allow_full=True, final_test=True)
    with pytest.raises(RuntimeError, match="blocked_resource"):
        allocation(10**20, "oversized fixture")
    assert {"prepare", "wals", "twotower", "features", "ranker", "reload"} <= set(ORDER)
    monkeypatch.setattr("hm_recsys.runtime.profiles.is_colab_runtime", lambda: True)
    execution_gate(load_profile(ROOT, "full"), allow_full=True, final_test=False)
    rosters(run, profile)
    history = History(run)
    part = Partition(run, "F1")
    try:
        active = history.active(date(2020, 8, 12))
        users = [u for b in part.batches() for u in b]
        assert "0000000042" not in active
        assert any("0000000042" in v for v in part.truth(users).values())
        assert Partition(run, "Validation", labels=False).count == 24
    finally:
        part.close()
        history.close()
    (source / "original data/customers.csv").write_text(
        "self-declared real data", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="exact generated"):
        assert_fixture(source)


def test_cuda_request_does_not_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """A parsed CUDA profile is not evidence of CUDA computation."""
    from hm_recsys.retrieval.two_tower.training import device_for

    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    with pytest.raises(RuntimeError, match="no CPU fallback"):
        device_for({"device": "cuda"})
    assert str(device_for({"device": "cpu"})) == "cpu"


def test_cuda_probe_explicitly_not_run(tmp_path: Path) -> None:
    """CUDA verification is opt-in and separately reported from CPU tests."""
    import os

    if os.environ.get("HM_RUN_CUDA_TESTS") != "1":
        pytest.skip("CUDA not requested; actual new-release CUDA execution is not run")
    from hm_recsys.runtime.device import probe

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
