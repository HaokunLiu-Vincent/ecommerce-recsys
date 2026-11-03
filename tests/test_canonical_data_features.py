"""Verify shared canonical data, feature and metric contracts on tiny sources.

All inputs and derived artifacts are synthetic and owned by tmp_path. The tests
exercise real SQL/Parquet preparation, frozen visual preprocessing and feature
construction without fitting retrieval/ranking models or reading real H&M data.
"""

import copy
import csv
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from hm_recsys.core.contracts import ROOT, read_json
from hm_recsys.core.synthetic import create_fixture
from hm_recsys.data.history import History, canonical, prepare_test_roster, rosters
from hm_recsys.data.streaming import Partition
from hm_recsys.evaluation.metrics import MetricAccumulator, metrics, recall_at_k
from hm_recsys.features.builder import FEATURES, FeatureBuilder
from hm_recsys.features.content import MappedContent, adapt_content
from hm_recsys.features.preprocessing import preprocessing
from hm_recsys.fusion.native import Candidate
from hm_recsys.retrieval.base import Hit
from hm_recsys.retrieval.snapshot import GlobalSnapshot, build_statistics
from hm_recsys.runtime.profiles import load_profile


def _smoke_profile() -> dict:
    profile = copy.deepcopy(load_profile(ROOT, "smoke"))
    profile["compute"].update(
        max_users=3,
        max_catalog=7,
        max_transactions=20000,
        max_queries_per_fold=2,
        cluster_count=3,
    )
    return profile


def _source_run(tmp_path: Path) -> tuple[Path, Path, dict]:
    source, run = tmp_path / "source", tmp_path / "run"
    create_fixture(source)
    run.mkdir()
    profile = _smoke_profile()
    canonical(run, source, profile)
    return source, run, profile


def test_smoke_sql_keeps_truth_separate_from_candidate_cap(tmp_path: Path) -> None:
    """The selected population is bounded without filtering eligible future truth."""
    source, run, profile = _source_run(tmp_path)
    rosters(run, profile)
    history = History(run)
    try:
        manifest = read_json(run / "population_manifest.json")
        assert manifest["base_users"] == 3
        assert len(history.candidate_catalog) == 7
        assert history.catalog == sorted(history.articles)
        assert manifest["global_history_rows"] <= 20000
        assert manifest["metadata_items"] > len(history.candidate_catalog)
        assert history.con.execute(
            "SELECT count(*) FROM transactions WHERE t_dat>=DATE '2020-09-16'"
        ).fetchone() == (0,)
        rows = history.con.execute(
            "SELECT count(*)-count(DISTINCT "
            "(customer_id,article_id,t_dat,price,sales_channel_id)) "
            "FROM transactions"
        ).fetchone()
        assert rows is not None and rows[0] > 0
        with (source / "original data/transactions_train.csv").open(
            encoding="utf-8", newline=""
        ) as handle:
            original = list(csv.DictReader(handle))
        for name in profile["folds"]:
            part = Partition(run, name)
            try:
                users = [u for batch in part.batches() for u in batch]
                truth = part.truth(users)
                expected = {
                    u: {
                        row["article_id"]
                        for row in original
                        if row["customer_id"] == u
                        and part.meta["start"] <= row["t_dat"] < part.meta["end"]
                    }
                    for u in users
                }
                assert truth == expected
                assert len(users) == 2
                assert any(truth[u] - set(history.candidate_catalog) for u in users)
            finally:
                part.close()
        assert not (run / "labels/Test.parquet").exists()
    finally:
        history.close()


def test_smoke_row_budget_rejects_complete_cohort(tmp_path: Path) -> None:
    """An oversized cohort stops instead of silently cutting purchase lines."""
    source, run = tmp_path / "source", tmp_path / "run"
    create_fixture(source)
    run.mkdir()
    profile = _smoke_profile()
    profile["compute"]["max_transactions"] = 1
    with pytest.raises(ValueError, match="exceeds transaction cap"):
        canonical(run, source, profile)
    assert not (run / "population_manifest.json").exists()
    assert not (run / "canonical/transactions").exists()


def test_test_truth_has_an_explicit_gate_and_exact_window(tmp_path: Path) -> None:
    """Default preparation creates no Test labels; authorization reads one week."""
    source, run, profile = _source_run(tmp_path)
    rosters(run, profile)
    with pytest.raises(PermissionError, match="authorization"):
        prepare_test_roster(run, source, authorized=False)
    raw = source / "original data/transactions_train.csv"
    with raw.open(encoding="utf-8", newline="") as handle:
        fields = next(csv.reader(handle))
    with raw.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writerow(
            {
                "customer_id": "fixture-user-000",
                "article_id": "0000000001",
                "t_dat": "2020-09-16",
                "price": "0.1",
                "sales_channel_id": "1",
            }
        )
        writer.writerow(
            {
                "customer_id": "fixture-user-000",
                "article_id": "0000000002",
                "t_dat": "2020-09-23",
                "price": "0.1",
                "sales_channel_id": "1",
            }
        )
    prepare_test_roster(run, source, authorized=True)
    part = Partition(run, "Test")
    try:
        assert part.meta["start"] == "2020-09-16"
        assert part.meta["end"] == "2020-09-23"
        assert part.truth(["fixture-user-000"]) == {"fixture-user-000": {"0000000001"}}
    finally:
        part.close()
    with pytest.raises(FileExistsError, match="refusing replacement"):
        prepare_test_roster(run, source, authorized=True)


def test_one_columnar_builder_is_cutoff_safe_and_keeps_schema(tmp_path: Path) -> None:
    """Real frozen preprocessing and SQL histories feed the shared 49 columns."""
    source, run, profile = _source_run(tmp_path)
    (run / "models").mkdir()
    rosters(run, profile)
    adapt_content(run, source)
    preprocessing(run, profile)
    cutoff = date(2020, 9, 8)
    build_statistics(run, "Validation", cutoff)
    history = History(run)
    snapshot = GlobalSnapshot(history, run, "Validation", cutoff)
    roster = Partition(run, "Validation", labels=False)
    try:
        users = next(roster.batches())
        snapshot.set_users(users)
        item = sorted(snapshot.active)[0]
        candidate = Candidate(item, {"trend": Hit(0.7, 1, False)})
        builder = FeatureBuilder(MappedContent(run), run)
        first = builder.build(snapshot, users[0], [candidate])
        second = builder.build(snapshot, users[0], [candidate])
        np.testing.assert_array_equal(first, second)
        assert first.shape == (1, 49)
        assert read_json(run / "feature_schema.json")["names"] == list(FEATURES)
        assert first[0, FEATURES.index("hit_trend")] == 1
        assert first[0, FEATURES.index("hit_wals")] == 0
        assert np.isnan(first[0, FEATURES.index("wals_score")])
        assert first[0, FEATURES.index("rrf_fused_score")] == pytest.approx(1 / 61)
        assert all(row.day <= cutoff for row in snapshot.by_user[users[0]])
    finally:
        roster.close()
        snapshot.close()
        history.close()


def test_streamed_metrics_keep_complete_denominators() -> None:
    """Batching preserves failed-query and out-of-pool-positive metric semantics."""
    predictions = {"a": ["one", "other"], "b": [], "c": ["miss"]}
    labels = {"a": {"one", "outside"}, "b": {"two"}, "c": {"three"}, "empty": set()}
    accumulator = MetricAccumulator()
    for user in labels:
        accumulator.add({user: predictions.get(user, [])}, {user: labels[user]})
    assert accumulator.result() == metrics(predictions, labels)
    assert accumulator.result()["eligible_queries"] == 3
    assert accumulator.result()["zero_hit_queries"] == 2
    assert recall_at_k(["one", "one"], labels["a"]) == 0.5
    with pytest.raises(ValueError, match="Duplicate"):
        metrics({"a": ["one", "one"]}, {"a": labels["a"]})


def test_smoke_candidate_cap_does_not_remove_tower_input_rows(tmp_path: Path) -> None:
    """Legal targets and raw history remain indexable outside the candidate cap."""
    from hm_recsys.core.contracts import write_json
    from hm_recsys.data.streaming import TargetStream
    from hm_recsys.retrieval.two_tower.inputs import SqlEncoder
    from hm_recsys.retrieval.two_tower.prepared import Prepared, prepare

    source, run, _ = _source_run(tmp_path)
    adapt_content(run, source)
    write_json(
        run / "run_manifest.json", {"migration_signature": "bounded-scope-fixture"}
    )
    history = History(run)
    try:
        encoder = SqlEncoder(history, MappedContent(run))
        targets = TargetStream(history, 128)
        events = [row for batch in targets.batches(0, 32) for row in batch]
        assert any(item not in history.candidate_catalog for _, _, item in events)
        prepare(run, encoder, targets, include_internal=False)
        inputs = Prepared(run / "prepared/tower")
        assert inputs.catalog == history.catalog
        assert set(item for _, _, item in events) <= set(inputs.catalog)
        contexts = np.asarray(inputs.arrays["target_context"][:32])
        profile, codes, weights = inputs.users(contexts)
        assert profile.shape[0] == codes.shape[0] == weights.shape[0] == 32
    finally:
        history.close()
