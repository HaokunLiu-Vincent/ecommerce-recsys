"""Offline contracts, independent references, and bounded model integration."""

import copy
import math
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
import torch

from hm_recsys.cli import execute
from hm_recsys.content import ContentBank, ContentSource
from hm_recsys.contracts import (
    BASE_END,
    CHANNELS,
    EXTRACT_END,
    Config,
    article_id,
    digest,
    read_json,
    split_manifest,
    validate_splits,
    write_json,
)
from hm_recsys.data import Sample, Transaction, extract_sample, query_users
from hm_recsys.evaluation import metrics, union_recall
from hm_recsys.features import FEATURES, FeatureBuilder
from hm_recsys.learned import ExactIndex, confidence, fold_user
from hm_recsys.pipeline import build_query_features, catalog_ids
from hm_recsys.ranking import matrix, select_training_rows
from hm_recsys.recall import Candidate, Hit, Snapshot
from hm_recsys.resources import GIB, memory_violation
from hm_recsys.synthetic import create_fixture
from hm_recsys.towers import (
    TowerEncoder,
    TwoTower,
    age_bucket,
    event_history,
    logq_loss,
    positive_mask,
    sample_targets,
)


@pytest.fixture(scope="module")
def fixture_data(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, Path, Sample, ContentBank]:
    """Create source-shaped fixture data once, without fitting a model."""
    root = tmp_path_factory.mktemp("smoke sources with spaces")
    source = root / "inputs"
    run = root / "extract"
    run.mkdir()
    create_fixture(source)
    config = Config.load()
    extract_sample(run, source, config)
    sample = Sample.load(run, config)
    content = ContentBank(source, list(sample.articles), catalog_ids(source))
    return source, run, sample, content


def test_ids_splits_and_age_boundaries() -> None:
    """Reject malformed keys/legacy anchors and keep exact weekly/age edges."""
    assert article_id("0000000001") == "0000000001"
    for value in (1, "123", " 000000001", "000000000x"):
        with pytest.raises(ValueError):
            article_id(value)
    splits = split_manifest()
    validate_splits(splits)
    days = []
    for partition in splits:
        start = date.fromisoformat(partition["start"])
        end = date.fromisoformat(partition["end"])
        assert end - start == timedelta(days=7)
        assert date.fromisoformat(partition["cutoff"]) == start - timedelta(days=1)
        days.extend(start + timedelta(days=i) for i in range(7))
    assert len(days) == len(set(days)) == 42
    assert min(days) == BASE_END and max(days) == date(2020, 9, 22)
    splits[1]["start"] = "2020-08-15"
    with pytest.raises(ValueError):
        validate_splits(splits)
    assert [age_bucket(v) for v in (None, 24, 25, 34, 35, 50, 51)] == [
        "UNKNOWN",
        "UNDER25",
        "25TO34",
        "25TO34",
        "35TO50",
        "35TO50",
        "OVER50",
    ]


def test_content_mapping_readonly_and_missing(fixture_data, tmp_path: Path) -> None:
    """Different shuffled maps join by ID; reads never mutate source arrays."""
    source, _, sample, content = fixture_data
    before = {str(p): digest(p) for p in source.rglob("*") if p.is_file()}
    text = ContentSource(source, "text", catalog_ids(source))
    image = ContentSource(source, "image", catalog_ids(source))
    assert text.array.shape[1] == 12 and image.array.shape[1] == 16
    assert not text.array.flags.writeable
    assert text.mapping != image.mapping
    ids = list(sample.articles)[:10]
    vectors, masks = text.fetch(ids)
    for i, item in enumerate(ids):
        np.testing.assert_allclose(
            vectors[i], text.array[text.mapping[item]], atol=1e-6
        )
    assert masks.all()
    assert content.vector("0000000001", "image") is None
    assert content.vector("0000000001", "text") is not None
    assert before == {str(p): digest(p) for p in source.rglob("*") if p.is_file()}
    broken = tmp_path / "bad input"
    create_fixture(broken)
    path = broken / "text_features/text_features_manifest.json"
    manifest = read_json(path)
    manifest["embedding_dim"] = 999
    write_json(path, manifest)
    with pytest.raises(ValueError, match="dimension"):
        ContentSource(broken, "text", catalog_ids(broken))


def test_content_invalid_consumed_vector(tmp_path: Path) -> None:
    """Numeric corruption blocks its consumer rather than creating fake content."""
    source = tmp_path / "bad vector"
    create_fixture(source)
    path = source / "text_features/text_embeddings.npy"
    vectors = np.load(path)
    vectors[0] = 0
    np.save(path, vectors)
    content = ContentSource(source, "text", catalog_ids(source))
    item = next(i for i, row in content.mapping.items() if row == 0)
    with pytest.raises(ValueError, match="zero"):
        content.fetch([item])


def test_sample_preserves_lines_labels_and_model_boundaries(fixture_data) -> None:
    """Future-only positives remain labels but do not create frozen vocab/Q."""
    _, _, sample, content = fixture_data
    assert max(t.day for t in sample.transactions) < EXTRACT_END
    raw_keys = [
        (t.customer, t.article, t.day, t.price, t.channel) for t in sample.transactions
    ]
    assert len(raw_keys) > len(set(raw_keys))
    future = "0000000042"
    assert future not in sample.catalog
    users, labels = query_users(sample, start=date(2020, 9, 9), cap=50)
    assert users and any(future in labels[user] for user in users)
    encoder = TowerEncoder(sample, content)
    assert future not in encoder.maps["article_id"]
    targets, q = sample_targets(sample, Config.load())
    assert all(date(2020, 5, 19) <= day < date(2020, 8, 5) for _, day, _ in targets)
    assert future not in q and sum(q.values()) == pytest.approx(1)
    earlier = replace(
        sample,
        transactions=[t for t in sample.transactions if t.day < date(2020, 8, 5)],
    )
    assert TowerEncoder(earlier, content).maps == encoder.maps
    assert sample_targets(earlier, Config.load()) == (targets, q)


def test_windows_itemcf_and_sibling_formula(fixture_data) -> None:
    """Compare directional ItemCF and lookbacks against independent small sums."""
    _, _, sample, _ = fixture_data
    snapshot = Snapshot(sample, cutoff=date(2020, 8, 11), config=Config.load())
    user = next(iter(sample.customers))
    assert all(
        date(2020, 8, 5) <= r.day < BASE_END
        for r in snapshot.window(snapshot.history, 7)
    )
    assert all(snapshot.sales[28][i] >= 3 for i in snapshot.active)
    seed = next(i for i, pairs in snapshot.itemcf.items() if pairs)
    other, actual = snapshot.itemcf[seed][0]
    users_i = {u for u, edges in snapshot.edges.items() if seed in edges}
    users_j = {u for u, edges in snapshot.edges.items() if other in edges}
    expected = sum(
        math.exp(-abs((snapshot.edges[u][seed] - snapshot.edges[u][other]).days) / 7)
        / math.log1p(len(snapshot.edges[u]))
        for u in users_i & users_j
    )
    expected /= math.sqrt(len(users_i) * len(users_j) ** 0.75)
    assert actual == pytest.approx(expected)
    scores, days = snapshot.sibling(user)
    for item, score in scores.items():
        legal_seeds = [
            r
            for r in snapshot.window(snapshot.by_user[user], 30)
            if r.article != item
            and sample.articles[r.article]["product_code"]
            == sample.articles[item]["product_code"]
        ]
        expected = max(
            math.exp(-(snapshot.cutoff - r.day).days / 14)
            * math.log1p(snapshot.sales[7][item])
            for r in legal_seeds
        )
        assert score == pytest.approx(expected)
        assert 0 <= days[item] < 30
    hit = Candidate(seed, {"trend": Hit(2.0, 1), "sibling": Hit(1.0, 4)})
    assert hit.rrf == pytest.approx(1 / 61 + 1 / 64)


def test_wals_confidence_and_reference_solve() -> None:
    """Validate one confidence transform and dense reference only on a tiny case."""
    rows = [Transaction("u", "0000000001", date(2020, 8, 11), 0.1, 1)] * 2
    rows += [Transaction("u", "0000000002", date(2020, 8, 12), 0.1, 1)]
    values = confidence(
        rows, cutoff=date(2020, 8, 11), vocabulary={"0000000001": 0, "0000000002": 1}
    )
    assert values == {0: pytest.approx(1 + 20 * np.log(3))}
    factors = np.array([[0.3, 0.4], [0.8, -0.1], [0.2, 0.7]], dtype=np.float32)
    c = np.diag([values[0], 1, 1])
    preference = np.array([1, 0, 0])
    reference = np.linalg.solve(
        factors.T @ c @ factors + 0.05 * np.eye(2), factors.T @ c @ preference
    )
    np.testing.assert_allclose(fold_user(factors, values), reference, atol=1e-6)
    assert np.array_equal(fold_user(factors, {}), np.zeros(2))


def test_two_tower_loss_history_dropout_and_padding(fixture_data) -> None:
    """Same-day exclusion, multi-positive masks, padding, and inference determinism."""
    _, _, sample, content = fixture_data
    encoder = TowerEncoder(sample, content)
    user = next(iter(sample.customers))
    day = date(2020, 8, 4)
    history = event_history(sample.transactions, before=day)
    assert all(t.day < day for t in history) and len(history) <= 20
    profile, codes, weights = encoder.users([("unknown-user", day)])
    assert weights.count_nonzero() == 0
    model = TwoTower(encoder.maps, 12, 16).eval()
    with torch.no_grad():
        result = model.user_vectors(profile, codes, weights)
        codes[:] = 1
        torch.testing.assert_close(result, model.user_vectors(profile, codes, weights))
        batch = encoder.items(sample.catalog[:3])
        first = model.item_vectors(*batch)
        torch.testing.assert_close(first, model.item_vectors(*batch))
        assert first.shape == (3, 128)
        torch.testing.assert_close(first.norm(dim=1), torch.ones(3))
    positives = {(user, day): {"a", "b"}, ("second", day): {"c"}}
    mask = positive_mask([(user, day), ("second", day)], ["a", "b", "c"], positives)
    assert mask.tolist() == [[True, True, False], [False, False, True]]
    u = torch.tensor([[0.2, 0.7], [0.8, 0.3]], requires_grad=True)
    i = torch.tensor([[0.1, 0.2], [0.3, 0.2], [0.8, 0.4]])
    probability = torch.tensor([0.25, 0.25, 0.5])
    loss = logq_loss(u, i, probability, mask)
    logits = u @ i.T / 0.07 - probability.log()
    expected = (
        -(
            torch.log(torch.exp(logits[0, :2]).sum() / torch.exp(logits[0]).sum())
            + torch.log(torch.exp(logits[1, 2]) / torch.exp(logits[1]).sum())
        )
        / 2
    )
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert u.grad is not None and torch.isfinite(u.grad).all()
    assert torch.all(TwoTower.shared_id_mask(32, False) == 1)


def test_sampling_metrics_and_denominators() -> None:
    """No injections, exact hard/random caps, full truth, and zero-candidate queries."""
    ids = [f"{i:010d}" for i in range(100)]
    truth = {ids[0], "9999999999"}
    rrf = np.arange(100, dtype=float)
    selected = select_training_rows(ids, truth, rrf, query="F1:user")
    assert len(selected) == 21 and 0 in selected
    assert set(range(90, 100)) <= set(selected)
    assert selected == select_training_rows(ids, truth, rrf, query="F1:user")
    assert select_training_rows(ids, {"outside"}, rrf, query="F1:user") == []
    assert select_training_rows(ids, set(ids), rrf, query="F1:user") == []
    labels = {"a": {"x", "z", "outside"}, "b": {"missing"}, "c": set()}
    result = metrics({"a": ["x", "other", "z"]}, labels)
    assert result["MAP@12"] == pytest.approx((1 + 2 / 3) / 3 / 2)
    assert result["Recall@12"] == pytest.approx(1 / 3)
    assert result["NDCG@12"] == pytest.approx(
        (1 + 0.5) / (1 + 1 / math.log2(3) + 0.5) / 2
    )
    assert result["eligible_queries"] == 2 and result["zero_candidate_queries"] == 1
    assert metrics({}, {"u": {"x"}})["MAP@12"] == 0
    assert union_recall({"a": ["x"]}, labels) == pytest.approx(1 / 6)
    with pytest.raises(ValueError, match="Duplicate"):
        metrics({"a": ["x", "x"]}, labels)
    assert not {"label", "customer_id", "article_id", "was_injected"} & set(FEATURES)


def test_limits_reject_before_training(fixture_data, tmp_path: Path) -> None:
    """Hard caps do not depend on a mutable full-scale default or actual total RAM."""
    _, _, sample, _ = fixture_data
    config = Config.load()
    bad = dict(sample.manifest, transactions=200001)
    with pytest.raises(ValueError, match="limit"):
        config.check_sample(bad)
    for key, value in (("max_users", 501), ("tt_batch_size", 2048), ("device", "mps")):
        values = dict(config.values, **{key: value})
        path = tmp_path / "invalid.json"
        write_json(path, values)
        with pytest.raises(ValueError):
            Config.load(path)
    assert memory_violation(5 * GIB, 8 * GIB, config)
    assert memory_violation(GIB, GIB, config)
    assert memory_violation(3 * GIB, 3 * GIB, config) is None
    tiny = dict(config.values, max_itemcf_pairs=1)
    path = tmp_path / "small.json"
    write_json(path, tiny)
    with pytest.raises(ValueError, match="pair budget"):
        Snapshot(sample, cutoff=date(2020, 8, 11), config=Config.load(path))


def test_exact_faiss_scores_ties_and_reload(tmp_path: Path) -> None:
    """Compare true inner products and deterministic ties, preserving scale."""
    vectors = np.array([[2, 1], [2, 1], [4, 0]], dtype=np.float32)
    ids = ["0000000002", "0000000001", "0000000003"]
    index = ExactIndex(ids, vectors)
    user = np.array([1, 1], dtype=np.float32)
    expected = [("0000000001", 3.0), ("0000000002", 3.0)]
    assert index.search(user, set(ids[:2]), 2) == expected
    index.save(tmp_path, "own_learned_index")
    loaded = ExactIndex.load(tmp_path, "own_learned_index")
    assert loaded.search(user, set(ids[:2]), 2) == expected
    assert loaded.index.ntotal == 3


def test_guarded_integration_and_future_perturbation(tmp_path: Path) -> None:
    """Fit all models once; check real updates, future invariance, and fresh reload."""
    run = tmp_path / "integration"
    run.mkdir()
    config = Config.load()
    source = run / "synthetic_inputs"
    execute("synthetic", run, config, source)
    result = read_json(run / "smoke_results.json")
    assert result["status"] == "passed"
    assert result["ranker"]["split_count"] > 0
    assert result["twotower"]["steps"] > 0
    assert result["twotower"]["projection_max_parameter_change"] > 0
    assert result["wals"]["iterations"] == 2
    for partition in result["partitions"].values():
        assert all(partition["channel_rows"][c] > 0 for c in CHANNELS)
        assert partition["feature_rows"] <= partition["queries"] * 270
    reload = read_json(run / "reload_results.json")
    assert reload["status"] == "passed" and reload["labels_read"] is False
    sample = Sample.load(run, config)
    content = ContentBank(source, list(sample.articles), catalog_ids(source))
    snapshot = Snapshot(sample, cutoff=date(2020, 8, 11), config=config)
    queries = read_json(run / "queries/F1.json")["users"][:2]
    before, provenance, _ = build_query_features(
        sample, content, run, config, cutoff=date(2020, 8, 11), users=queries
    )
    changed = copy.copy(sample)
    changed.transactions = [
        replace(row, article="0000000042", price=99.0) if row.day >= BASE_END else row
        for row in sample.transactions
    ]
    after, new_provenance, _ = build_query_features(
        changed, content, run, config, cutoff=date(2020, 8, 11), users=queries
    )
    np.testing.assert_allclose(
        matrix(before), matrix(after), atol=0, rtol=0, equal_nan=True
    )
    assert provenance == new_provenance
    assert TowerEncoder(sample, content).maps == TowerEncoder(changed, content).maps
    assert confidence(
        sample.transactions,
        cutoff=date(2020, 8, 11),
        vocabulary={i: j for j, i in enumerate(sample.catalog)},
    ) == confidence(
        changed.transactions,
        cutoff=date(2020, 8, 11),
        vocabulary={i: j for j, i in enumerate(sample.catalog)},
    )
    # Explicit missing history has missing cross features and a zero mask.
    candidate = Candidate(sample.catalog[0], {"trend": Hit(1, 1)})
    features = FeatureBuilder(content, run).build(snapshot, "unknown", [candidate])[0]
    assert features[FEATURES.index("has_user_image_history")] == 0
    assert np.isnan(features[FEATURES.index("user_item_img_cosine_sim")])
    assert sum(result["ranker"]["group_sizes"]) == result["ranker"]["training_rows"]
    validation = pq.read_table(run / "features/Validation.parquet")
    assert len(validation) == result["partitions"]["Validation"]["feature_rows"]
    assert result["partitions"]["Validation"]["zero_retrieved_positive_queries"] >= 1
    for path in (run / "models").iterdir():
        assert path.is_relative_to(run)


def test_supervisor_stops_only_owned_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise a simulated threshold crossing without allocating unsafe memory."""
    from hm_recsys import resources

    config = Config.load()
    write_json(tmp_path / "config.resolved.json", config.values)
    write_json(
        tmp_path / "run_manifest.json", {"source_root": str(tmp_path / "source")}
    )
    observations = 0

    def simulated_limit(rss: int, available: int, supplied: Config) -> str | None:
        nonlocal observations
        observations += 1
        return "fixture_threshold_crossing" if observations > 1 else None

    monkeypatch.setattr(resources, "memory_violation", simulated_limit)
    with pytest.raises(RuntimeError, match="failed"):
        resources.run_stage("fixture", tmp_path, config)
    stage = read_json(tmp_path / "resource_usage.json")["stages"][0]
    assert stage["termination_reason"] == "fixture_threshold_crossing"
    assert stage["exit_code"] != 0


def test_training_rejects_misdeclared_sample(fixture_data, tmp_path: Path) -> None:
    """Every fitting entry point rejects incorrect counts before its fit call."""
    from hm_recsys.features import fit_preprocessing
    from hm_recsys.learned import WALS
    from hm_recsys.towers import fit_towers

    _, _, sample, content = fixture_data
    bad = replace(sample, transactions=sample.transactions * 2)
    for fit in (
        lambda: WALS.fit(bad, Config.load(), tmp_path),
        lambda: fit_towers(bad, content, Config.load(), tmp_path),
        lambda: fit_preprocessing(bad, content, Config.load(), tmp_path),
    ):
        with pytest.raises(ValueError, match="counts"):
            fit()
