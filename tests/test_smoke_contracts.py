"""Exercise canonical temporal, numerical, resource and reload contracts.

All sources are generated fixtures. Independent scalar equations remain only in
these tests; real SQL snapshots, the shared builder and genuine trained models
are used in execution checks. No archived implementation is imported.
"""

import csv
import math
import shutil
from contextlib import contextmanager
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest
import torch

from hm_recsys.core.contracts import (
    BASE_END,
    CHANNELS,
    EXTRACT_END,
    ROOT,
    Config,
    article_id,
    digest,
    read_json,
    split_manifest,
    validate_splits,
    write_json,
)
from hm_recsys.core.synthetic import create_fixture
from hm_recsys.data.history import History, canonical, rosters
from hm_recsys.data.sample import Sample, Transaction, age_bucket
from hm_recsys.data.streaming import Partition, TargetStream
from hm_recsys.evaluation.metrics import metrics, union_recall
from hm_recsys.features.builder import FEATURES, FeatureBuilder
from hm_recsys.features.content import (
    ContentBank,
    ContentSource,
    MappedContent,
    adapt_content,
)
from hm_recsys.fusion.native import Candidate
from hm_recsys.fusion.storage import CandidateReader
from hm_recsys.ranking.model import select_training_rows
from hm_recsys.retrieval.base import Hit
from hm_recsys.retrieval.snapshot import GlobalSnapshot, build_snapshot
from hm_recsys.retrieval.two_tower.inputs import SqlEncoder, TowerEncoder, event_history
from hm_recsys.retrieval.two_tower.model import TwoTower, logq_loss, positive_mask
from hm_recsys.retrieval.vectors import BlockIndex
from hm_recsys.retrieval.wals.model import confidence, fold_user
from hm_recsys.runtime.acceptance import catalog_ids
from hm_recsys.runtime.profiles import load_profile
from hm_recsys.runtime.resources import GIB, memory_violation


@pytest.fixture(scope="module")
def fixture_data(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, Path, Sample, ContentBank]:
    """Prepare canonical SQL once and expose only a tiny unit-test tensor adapter."""
    root = tmp_path_factory.mktemp("smoke sources with spaces")
    source, run = (root / "inputs", root / "canonical_run")
    run.mkdir()
    create_fixture(source)
    canonical(run, source, load_profile(ROOT, "smoke"))
    rosters(run, load_profile(ROOT, "smoke"))
    adapt_content(run, source)
    history = History(run)
    try:
        users = [
            r[0]
            for r in history.con.execute(
                "SELECT customer_id FROM customers ORDER BY customer_id"
            ).fetchall()
        ]
        transactions = [
            r for rows in history.histories(users, EXTRACT_END).values() for r in rows
        ]
        manifest = {
            "sampling_scope": "smoke",
            "validated": True,
            "users": len(users),
            "catalog": len(history.candidate_catalog),
            "transactions": len(transactions),
            "history_end_exclusive": str(EXTRACT_END),
        }
        sample = Sample(
            transactions,
            history.articles,
            history.customers(users),
            history.candidate_catalog,
            manifest,
        )
    finally:
        history.close()
    return (source, run, sample, MappedContent(run))


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
            vectors[i], text.array[text.mapping[item]], atol=1e-06
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
    item = next((i for i, row in content.mapping.items() if row == 0))
    with pytest.raises(ValueError, match="zero"):
        content.fetch([item])


def test_sample_preserves_lines_labels_and_model_boundaries(fixture_data) -> None:
    """Future-only positives remain labels but do not create frozen vocab/Q."""
    _, run, sample, content = fixture_data
    assert max(t.day for t in sample.transactions) < EXTRACT_END
    keys = [
        (t.customer, t.article, t.day, t.price, t.channel) for t in sample.transactions
    ]
    assert len(keys) > len(set(keys))
    future = "0000000042"
    assert future not in sample.catalog
    part = Partition(run, "Validation")
    users = [u for b in part.batches() for u in b]
    labels = part.truth(users)
    part.close()
    assert users and any(future in labels[u] for u in users)
    encoder = TowerEncoder(sample, content)
    assert future not in encoder.maps["article_id"]
    history = History(run)
    try:
        stream = TargetStream(history, Config.load().limit("tt_max_targets"))
        targets = [row for batch in stream.batches(0, 32) for row in batch]
        q = stream.probabilities
    finally:
        history.close()
    assert all((date(2020, 5, 19) <= day < date(2020, 8, 5) for _, day, _ in targets))
    assert future not in q and sum(q.values()) == pytest.approx(1)
    earlier = replace(
        sample,
        transactions=[r for r in sample.transactions if r.day < date(2020, 8, 5)],
    )
    assert TowerEncoder(earlier, content).maps == encoder.maps
    eligible = {
        (r.customer, r.day, r.article)
        for r in earlier.transactions
        if date(2020, 5, 19) <= r.day < date(2020, 8, 5)
    }
    assert set(targets) == eligible
    from collections import Counter

    counts = Counter((item for _, _, item in targets))
    assert q == {item: n / len(targets) for item, n in counts.items()}


def test_windows_itemcf_and_sibling_formula(fixture_data) -> None:
    """Compare directional ItemCF and lookbacks against independent small sums."""
    _, run, sample, _ = fixture_data
    users = list(sample.customers)
    with _snapshot(run, "contract", date(2020, 8, 11), users) as snapshot:
        user = users[0]
        assert all(
            date(2020, 8, 5) <= r.day < BASE_END
            for r in snapshot.window(sample.transactions, 7)
        )
        assert all(snapshot.sales[28][i] >= 3 for i in snapshot.active)
        seed = next((i for i, pairs in snapshot.itemcf.items() if pairs))
        other, actual = snapshot.itemcf[seed][0]
        ui = {u for u, e in snapshot.edges.items() if seed in e}
        uj = {u for u, e in snapshot.edges.items() if other in e}
        expected = sum(
            math.exp(
                -abs((snapshot.edges[u][seed] - snapshot.edges[u][other]).days) / 7
            )
            / math.log1p(len(snapshot.edges[u]))
            for u in ui & uj
        )
        expected /= math.sqrt(len(ui) * len(uj) ** 0.75)
        assert actual == pytest.approx(expected)
        scores, days = snapshot.sibling(user)
        for item, score in scores.items():
            seeds = [
                r
                for r in snapshot.window(snapshot.by_user[user], 30)
                if r.article != item
                and sample.articles[r.article]["product_code"]
                == sample.articles[item]["product_code"]
            ]
            expected = max(
                math.exp(-(snapshot.cutoff - r.day).days / 14)
                * math.log1p(snapshot.sales[7][item])
                for r in seeds
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
    np.testing.assert_allclose(fold_user(factors, values), reference, atol=1e-06)
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
    """Hard caps are checked against real selected rows/pairs, not total RAM."""
    _, run, sample, _ = fixture_data
    config = Config.load()
    with pytest.raises(ValueError, match="limit"):
        config.check_sample(dict(sample.manifest, transactions=200001))
    for key, value in (("max_users", 501), ("tt_batch_size", 2048), ("device", "mps")):
        path = tmp_path / "invalid.json"
        write_json(path, dict(config.values, **{key: value}))
        with pytest.raises(ValueError):
            Config.load(path)
    assert memory_violation(5 * GIB, 8 * GIB, config)
    assert memory_violation(GIB, GIB, config)
    assert memory_violation(3 * GIB, 3 * GIB, config) is None
    profile = load_profile(ROOT, "smoke")
    profile["compute"]["max_itemcf_pairs"] = 1
    with pytest.raises(RuntimeError, match="pair"):
        build_snapshot(
            run, "pair_budget", date(2020, 8, 11), optimized=True, profile=profile
        )


def test_exact_faiss_scores_ties_and_reload(tmp_path: Path) -> None:
    """Compare exact inner products/ties and a genuine learned index reload."""
    vectors = np.array([[2, 1], [2, 1], [4, 0]], dtype=np.float32)
    ids = ["0000000002", "0000000001", "0000000003"]
    index = BlockIndex(ids, vectors)
    user = np.array([1, 1], dtype=np.float32)
    expected = [("0000000001", 3.0), ("0000000002", 3.0)]
    assert index.search(user, set(ids[:2]), 2) == expected
    index.save(tmp_path, "own_learned_index")
    loaded = BlockIndex.load(tmp_path, "own_learned_index")
    assert loaded.search(user, set(ids[:2]), 2) == expected
    import faiss

    assert faiss.read_index(str(tmp_path / "own_learned_index.faiss")).ntotal == 3


def test_guarded_integration_and_future_perturbation(tmp_path: Path) -> None:
    """Fit canonical models once, reload in workers, and perturb only future rows."""
    from hm_recsys.runtime.files import stage_inputs
    from hm_recsys.runtime.modular import resolved_profile, run_all
    from hm_recsys.runtime.testing import create_source

    source, local, remote = (
        tmp_path / "source",
        tmp_path / "local",
        tmp_path / "remote",
    )
    create_source(source)
    staged = tmp_path / "staged_data"
    stage_inputs(source, staged)
    profile = resolved_profile(ROOT, "smoke")
    status = run_all(
        ROOT, staged, local, remote, profile=profile, synthetic_fixture=True
    )
    assert status["complete"]
    result = read_json(local / "run_summary.json")
    assert result["status"] == "passed" and result["ranker"]["split_count"] > 0
    assert result["twotower"]["steps"] > 0 and result["wals"]["iterations"] == 2
    for part in result["partitions"].values():
        assert all(part["channel_rows"][c] > 0 for c in CHANNELS)
        assert part["feature_rows"] <= part["queries"] * 270
    view = tmp_path / "immutable_view"
    for name, relative in read_json(local / "artifacts.json").items():
        target = view / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local / relative, target)
    assert (
        read_json(view / "models/twotower.json")["projection_max_parameter_change"] > 0
    )
    assert read_json(view / "reload_results.json")["status"] == "passed"
    assert read_json(view / "reload_results.json")["labels_read"] is False
    changed_source = tmp_path / "changed_source"
    shutil.copytree(source, changed_source)
    path = changed_source / "original data/transactions_train.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        rows = list(reader)
    assert fields is not None
    for row in rows:
        if row["t_dat"] >= str(BASE_END):
            row["article_id"], row["price"] = ("0000000042", "99.0")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    changed = tmp_path / "changed"
    changed.mkdir()
    canonical(changed, changed_source, profile)
    adapt_content(changed, changed_source)
    (changed / "models").mkdir()
    shutil.copyfile(
        view / "models/visual_centers.npy", changed / "models/visual_centers.npy"
    )
    shutil.copyfile(view / "feature_schema.json", changed / "feature_schema.json")
    cutoff = date(2020, 8, 11)
    build_snapshot(changed, "F1", cutoff, optimized=True, profile=profile)
    roster = Partition(view, "F1", labels=False)
    queries = next(roster.batches(2))
    roster.close()
    original_history, new_history = (History(view), History(changed))
    before = GlobalSnapshot(original_history, view, "F1", cutoff)
    after = GlobalSnapshot(new_history, changed, "F1", cutoff)
    native = CandidateReader(view, "F1")
    try:
        before.set_users(queries)
        after.set_users(queries)
        candidates = native.read(queries)
        original_builder = FeatureBuilder(MappedContent(view), view)
        changed_builder = FeatureBuilder(MappedContent(changed), changed)
        for user in queries:
            learned = {
                channel: [
                    (c.article, c.hits[channel].score)
                    for c in sorted(
                        (c for c in candidates[user] if channel in c.hits),
                        key=lambda c: c.hits[channel].rank,
                    )
                ]
                for channel in ("wals", "twotower")
            }
            rebuilt, _ = after.retrieve(user, None, None, learned=learned)
            assert candidates[user] == rebuilt
            np.testing.assert_allclose(
                original_builder.build(before, user, candidates[user]),
                changed_builder.build(after, user, rebuilt),
                atol=0,
                rtol=0,
                equal_nan=True,
            )
            vocabulary = {item: j for j, item in enumerate(original_history.catalog)}
            assert confidence(
                before.by_user[user], cutoff=cutoff, vocabulary=vocabulary
            ) == confidence(after.by_user[user], cutoff=cutoff, vocabulary=vocabulary)
        assert (
            SqlEncoder(original_history, MappedContent(view)).maps
            == SqlEncoder(new_history, MappedContent(changed)).maps
        )
        before.set_users(["unknown"])
        candidate = Candidate(sorted(before.active)[0], {"trend": Hit(1, 1)})
        features = original_builder.build(before, "unknown", [candidate])[0]
        assert features[FEATURES.index("has_user_image_history")] == 0
        assert np.isnan(features[FEATURES.index("user_item_img_cosine_sim")])
    finally:
        native.close()
        before.close()
        after.close()
        original_history.close()
        new_history.close()
    sampling = read_json(view / "ranker_sampling.json")
    assert sum(sampling["group_sizes"]) == result["ranker"]["training_rows"]
    from hm_recsys.ranking.pipeline import feature_manifest

    manifest = feature_manifest(view, "Validation")
    assert manifest["rows"] == result["partitions"]["Validation"]["feature_rows"]
    validation = Partition(view, "Validation")
    union = CandidateReader(view, "Validation")
    missing = 0
    try:
        for users in validation.batches():
            candidates = union.read(users)
            truth = validation.truth(users)
            missing += sum(
                not {c.article for c in candidates[u]} & truth[u] for u in users
            )
    finally:
        union.close()
        validation.close()
    assert missing >= 1
    assert all(path.is_relative_to(view) for path in (view / "models").iterdir())


def test_supervisor_stops_only_owned_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise a real owned child and simulated threshold without allocating RAM."""
    from hm_recsys.runtime import resources

    config = Config.load()
    module = tmp_path / "owned_fixture_worker.py"
    module.write_text("import time\ntime.sleep(30)\n", encoding="utf-8")
    observations = 0

    def simulated_limit(rss: int, available: int, supplied: Config) -> str | None:
        nonlocal observations
        observations += 1
        return "fixture_threshold_crossing" if observations > 1 else None

    monkeypatch.setattr(resources, "memory_violation", simulated_limit)
    with pytest.raises(RuntimeError, match="failed"):
        resources.run_stage(
            "fixture", tmp_path, config, module="owned_fixture_worker", cwd=tmp_path
        )
    stage = read_json(tmp_path / "resource_usage.json")["stages"][0]
    assert stage["termination_reason"] == "fixture_threshold_crossing"
    assert stage["exit_code"] != 0


def test_training_rejects_misdeclared_sample(fixture_data) -> None:
    """The bounded tensor fixture adapter rejects false counts before consumption.

    Production fitters no longer accept an in-memory Sample; canonical SQL caps
    are separately tested against actual source rows in canonical data tests.
    """
    _, _, sample, _ = fixture_data
    bad = replace(sample, transactions=sample.transactions * 2)
    with pytest.raises(ValueError, match="counts"):
        bad.validate_training(Config.load())


@contextmanager
def _snapshot(
    run: Path, name: str, cutoff: date, users: list[str], *, profile: dict | None = None
):
    """Open actual bounded SQL statistics and ItemCF for one owned test view."""
    build_snapshot(run, name, cutoff, optimized=True, profile=profile)
    history = History(run)
    snapshot = GlobalSnapshot(history, run, name, cutoff)
    snapshot.set_users(users)
    try:
        yield snapshot
    finally:
        snapshot.close()
        history.close()
