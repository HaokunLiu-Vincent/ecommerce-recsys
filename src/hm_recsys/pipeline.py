"""Sequential smoke stages and label-free fresh-process inference."""

import csv
import importlib.metadata
import shutil
from datetime import date
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from hm_recsys.content import ContentBank, ContentSource
from hm_recsys.contracts import (
    CHANNELS,
    QUOTAS,
    Config,
    article_id,
    digest,
    read_json,
    source_identity,
    split_manifest,
    validate_splits,
    write_json,
)
from hm_recsys.data import (
    ARTICLE_FIELDS,
    CUSTOMER_FIELDS,
    Sample,
    inspect_csv,
    query_users,
)
from hm_recsys.evaluation import metrics, union_recall
from hm_recsys.features import FEATURES, FeatureBuilder


def catalog_ids(root: Path) -> set[str]:
    """Read static catalog identity only, retaining leading zeros."""
    with (root / "original data/articles.csv").open(newline="") as handle:
        ids = [article_id(row["article_id"]) for row in csv.DictReader(handle)]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate catalog IDs")
    return set(ids)


def preflight(run: Path, source: Path, config: Config) -> None:
    """Inspect input headers/maps and CPU dependencies without training models."""
    validate_splits(split_manifest())
    report: dict = {"status": "accepted_headers_and_maps", "csv": {}, "content": {}}
    for name, required in (
        ("articles.csv", {"article_id", *ARTICLE_FIELDS}),
        ("customers.csv", {"customer_id", *CUSTOMER_FIELDS}),
        (
            "transactions_train.csv",
            {"customer_id", "article_id", "t_dat", "price", "sales_channel_id"},
        ),
    ):
        report["csv"][name] = inspect_csv(source / "original data" / name, required)
    catalog = catalog_ids(source)
    for modality in ("text", "image"):
        content = ContentSource(source, modality, catalog)
        report["content"][modality] = content.report
    report["versions"] = {
        name: importlib.metadata.version(name)
        for name in (
            "duckdb",
            "numpy",
            "pyarrow",
            "scipy",
            "scikit-learn",
            "implicit",
            "torch",
            "faiss-cpu",
            "lightgbm",
            "psutil",
        )
    }
    report["inputs"] = {
        str(p.relative_to(source)): source_identity(p)
        for folder in ("original data", "text_features", "image_features")
        for p in (source / folder).iterdir()
        if p.is_file()
    }
    report["config"] = config.values
    write_json(run / "input_acceptance.json", report)
    write_json(
        run / "split_manifest.json", {"splits": split_manifest(), "test_enabled": False}
    )


def load_content(run: Path, sample: Sample, source: Path) -> ContentBank:
    """Validate every used vector in a bounded catalog/history subset."""
    content = ContentBank(source, list(sample.articles), catalog_ids(source))
    report = read_json(run / "input_acceptance.json")
    # Only numeric validation scope advances; no supplied manifests are edited.
    report["content"] = content.reports
    write_json(run / "input_acceptance.json", report)
    return content


def feature_table(
    users: list[str], ids: list[str], blocks: list[np.ndarray]
) -> pa.Table:
    """Construct a typed bounded table with separate audit keys and features."""
    values = (
        np.concatenate(blocks)
        if blocks
        else np.empty((0, len(FEATURES)), dtype=np.float32)
    )
    arrays = {
        "customer_id": pa.array(users, type=pa.string()),
        "article_id": pa.array(ids, type=pa.string()),
    }
    arrays.update(
        {
            name: pa.array(values[:, i], type=pa.float32())
            for i, name in enumerate(FEATURES)
        }
    )
    return pa.table(arrays)


def build_query_features(
    sample: Sample,
    content: ContentBank,
    run: Path,
    config: Config,
    *,
    cutoff: date,
    users: list[str],
    reloaded: bool = False,
) -> tuple[pa.Table, list[dict], dict]:
    """Shared native retrieval/features path, with no label argument."""
    if len(users) > config.limit("max_queries_per_fold"):
        raise ValueError("Query cap exceeded before candidate generation")
    from hm_recsys.learned import WALS, ExactIndex, TowerRetriever
    from hm_recsys.recall import Snapshot

    snapshot = Snapshot(sample, cutoff=cutoff, config=config)
    wals = WALS(ExactIndex.load(run / "indexes", "wals"))
    towers = TowerRetriever(run, reloaded=reloaded)
    builder = FeatureBuilder(content, run)
    row_users, ids, blocks, provenance = [], [], [], []
    output: dict = {
        "pool": sorted(snapshot.active),
        "channels": {},
        "union": {},
        "itemcf_estimated_pairs": snapshot.pair_estimate,
    }
    for user in sorted(users):
        candidates, channels = snapshot.retrieve(user, wals, towers)
        blocks.append(builder.build(snapshot, user, candidates))
        row_users.extend([user] * len(candidates))
        ids.extend(candidate.article for candidate in candidates)
        output["channels"][user] = {
            c: [i for i, _ in pairs] for c, pairs in channels.items()
        }
        output["union"][user] = [candidate.article for candidate in candidates]
        for candidate in candidates:
            for channel, hit in candidate.hits.items():
                provenance.append(
                    {
                        "customer_id": user,
                        "cutoff_date": str(cutoff),
                        "article_id": candidate.article,
                        "channel": channel,
                        "raw_score": hit.score,
                        "source_rank": hit.rank,
                        "trend_global_backfill": hit.fallback,
                        "retriever_version": "v0.3.2-smoke-1",
                    }
                )
    return feature_table(row_users, ids, blocks), provenance, output


def candidates_and_features(
    run: Path,
    source: Path,
    config: Config,
    *,
    partitions: list[str] | None = None,
) -> None:
    """Build four native Train folds and unfiltered sampled Validation."""
    sample = Sample.load(run, config)
    content = load_content(run, sample, source)
    validate_splits(read_json(run / "split_manifest.json")["splits"])
    for folder in ("queries", "labels", "features", "candidates", "predictions"):
        (run / folder).mkdir(exist_ok=True)
    report: dict = (
        read_json(run / "candidate_results.json")
        if partitions is not None and (run / "candidate_results.json").exists()
        else {"partitions": {}}
    )
    total_rows = sum(value["feature_rows"] for value in report["partitions"].values())
    for partition in split_manifest()[:5]:
        name = partition["partition"]
        if partitions is not None and name not in partitions:
            continue
        if name in report["partitions"]:
            raise ValueError("Refusing to overwrite an already completed fold")
        start = date.fromisoformat(partition["start"])
        cutoff = date.fromisoformat(partition["cutoff"])
        users, labels = query_users(
            sample, start=start, cap=config.limit("max_queries_per_fold")
        )
        if not users:
            raise ValueError(f"Empty required query partition: {name}")
        if total_rows + len(users) * sum(QUOTAS.values()) > config.limit(
            "max_feature_rows"
        ):
            raise ValueError("Feature row upper bound exceeds run budget")
        # Query membership may depend on purchase activity, never recall hits.
        write_json(
            run / "queries" / f"{name}.json",
            {
                **partition,
                "users": users,
                "rule": "purchase-active then SHA256(42:start:user)",
            },
        )
        table, provenance, output = build_query_features(
            sample, content, run, config, cutoff=cutoff, users=users
        )
        total_rows += len(table)
        pq.write_table(table, run / "features" / f"{name}.parquet")
        if provenance:
            pq.write_table(
                pa.Table.from_pylist(provenance), run / "candidates" / f"{name}.parquet"
            )
        write_json(
            run / "candidates" / f"{name}_pool.json",
            {"articles": output["pool"], "cutoff": str(cutoff)},
        )
        # Only now attach labels for supervised consumers and diagnostics.
        write_json(
            run / "labels" / f"{name}.json",
            {u: sorted(items) for u, items in labels.items()},
        )
        counts = {
            c: sum(len(output["channels"][u][c]) for u in users) for c in CHANNELS
        }
        report["partitions"][name] = {
            "queries": len(users),
            "feature_rows": len(table),
            "active_items": len(output["pool"]),
            "channel_rows": counts,
            "union_recall": union_recall(output["union"], labels),
            "zero_candidate_queries": sum(not output["union"][u] for u in users),
            "zero_retrieved_positive_queries": sum(
                not (set(output["union"][u]) & labels[u]) for u in users
            ),
            "active_pool_coverage": sum(
                len(set(output["pool"]) & labels[u]) / len(labels[u]) for u in users
            )
            / len(users),
            "out_of_pool_positive_count": sum(
                len(labels[u] - set(output["pool"])) for u in users
            ),
            "channel_recall": {
                c: {
                    str(k): metrics(
                        {u: output["channels"][u][c] for u in users}, labels, k
                    )[f"Recall@{k}"]
                    for k in sorted({12, QUOTAS[c]})
                }
                for c in CHANNELS
            },
            "itemcf_estimated_pairs": output["itemcf_estimated_pairs"],
        }
    report["total_feature_rows"] = total_rows
    write_json(run / "candidate_results.json", report)
    if partitions is None or "Validation" in partitions:
        prepare_inference_input(run)


def prepare_inference_input(run: Path) -> None:
    """Persist a label-free Validation inference history ending before Sep 9."""
    folder = run / "inference_input/sample"
    folder.mkdir(parents=True, exist_ok=False)
    for name in ("articles.parquet", "customers.parquet", "catalog.json"):
        shutil.copyfile(run / "sample" / name, folder / name)
    table = pq.read_table(
        run / "sample/transactions.parquet", filters=[("t_dat", "<", date(2020, 9, 9))]
    )
    pq.write_table(table, folder / "transactions.parquet")
    manifest = read_json(run / "sample_manifest.json")
    manifest["transactions"] = len(table)
    manifest["inference_history_end_exclusive"] = "2020-09-09"
    manifest["file_sha256"] = {p.name: digest(p) for p in folder.iterdir()}
    write_json(run / "inference_input/sample_manifest.json", manifest)


def seal_run(run: Path, source: Path) -> None:
    """Freeze model/schema identities and verify cheap source immutability."""
    manifest = read_json(run / "run_manifest.json")
    manifest["artifact_sha256"] = {
        str(p.relative_to(run)): digest(p)
        for folder in ("models", "indexes")
        for p in (run / folder).iterdir()
        if p.is_file()
    }
    for name in ("feature_schema.json", "config.resolved.json", "split_manifest.json"):
        manifest["artifact_sha256"][name] = digest(run / name)
    manifest["content_references"] = {
        modality: {
            "manifest_sha256": digest(
                source / f"{modality}_features/{modality}_features_manifest.json"
            ),
            "array_identity": source_identity(
                source / f"{modality}_features/{modality}_embeddings.npy"
            ),
        }
        for modality in ("text", "image")
    }
    inputs = read_json(run / "input_acceptance.json")["inputs"]
    portable = manifest.get("portable_inputs", False)
    if any(
        not compatible_source(source_identity(source / name), identity, portable)
        for name, identity in inputs.items()
    ):
        raise ValueError("Source file identity changed during smoke execution")
    manifest["source_stat_identity_unchanged"] = True
    write_json(run / "run_manifest.json", manifest)


def compatible_source(current: dict, previous: dict, portable: bool) -> bool:
    """Allow root relocation only for inputs verified by the staging ledger."""
    keys = ("bytes", "mtime_ns") if portable else tuple(previous)
    return all(current.get(key) == previous.get(key) for key in keys)


def verify_frozen(run: Path, source: Path) -> None:
    """Verify saved model/schema identities before any reload consumer starts."""
    manifest = read_json(run / "run_manifest.json")
    for name, expected in manifest["artifact_sha256"].items():
        if digest(run / name) != expected:
            raise ValueError(f"Frozen artifact checksum mismatch: {name}")
    for modality, reference in manifest["content_references"].items():
        folder = source / f"{modality}_features"
        if (
            digest(folder / f"{modality}_features_manifest.json")
            != reference["manifest_sha256"]
        ):
            raise ValueError("Source content manifest changed")
        if not compatible_source(
            source_identity(folder / f"{modality}_embeddings.npy"),
            reference["array_identity"],
            manifest.get("portable_inputs", False),
        ):
            raise ValueError("Source content array identity changed")


def reload_features(run: Path, source: Path, config: Config) -> None:
    """Rebuild native features in a fresh FAISS process without future labels."""
    verify_frozen(run, source)
    sample = Sample.load(run / "inference_input", config)
    content = ContentBank(source, list(sample.articles), catalog_ids(source))
    queries = read_json(run / "queries/Validation.json")
    table, _, _ = build_query_features(
        sample,
        content,
        run,
        config,
        cutoff=date.fromisoformat(queries["cutoff"]),
        users=queries["users"],
        reloaded=True,
    )
    pq.write_table(table, run / "predictions/reloaded_features.parquet")


def reload_inference(run: Path, source: Path, config: Config) -> None:
    """Load LambdaMART in a fresh process, compare features/scores and Top-12.

    FAISS and PyTorch computations have finished in preceding owned processes;
    this process reads only label-free inference features and saved references.
    """
    import lightgbm as lgb

    from hm_recsys.ranking import matrix, ordered_predictions

    verify_frozen(run, source)
    queries = read_json(run / "queries/Validation.json")
    table = pq.read_table(run / "predictions/reloaded_features.parquet")
    model = lgb.Booster(model_file=str(run / "models/lambdamart.txt"))
    if model.feature_name() != list(FEATURES):
        raise ValueError("Ranker feature names do not match shared schema")
    scores = (
        np.asarray(
            model.predict(matrix(table), num_threads=config.limit("threads")),
            dtype=np.float64,
        )
        if len(table)
        else np.empty(0, dtype=np.float64)
    )
    predictions = ordered_predictions(table, scores, queries["users"])
    reference = pq.read_table(run / "features/Validation.parquet")
    for name in ("customer_id", "article_id"):
        if table[name].to_pylist() != reference[name].to_pylist():
            raise ValueError("Fresh-process native candidate identity/order mismatch")
    np.testing.assert_allclose(
        matrix(table), matrix(reference), rtol=1e-5, atol=1e-6, equal_nan=True
    )
    old_scores = np.load(run / "predictions/reference_scores.npy", allow_pickle=False)
    np.testing.assert_allclose(scores, old_scores, rtol=1e-6, atol=1e-7)
    if predictions != read_json(run / "predictions/reference_top12.json"):
        raise ValueError("Fresh-process Top-12 mismatch")
    np.save(run / "predictions/reloaded_scores.npy", scores, allow_pickle=False)
    write_json(run / "predictions/reloaded_top12.json", predictions)
    write_json(
        run / "reload_results.json",
        {
            "status": "passed",
            "fresh_process": True,
            "labels_read": False,
            "refit": False,
            "queries": len(predictions),
            "candidate_rows": len(table),
            "feature_rtol": 1e-5,
            "feature_atol": 1e-6,
            "score_rtol": 1e-6,
            "score_atol": 1e-7,
            "max_score_absolute_difference": float(np.max(np.abs(scores - old_scores)))
            if len(scores)
            else 0,
            "unique_top12": True,
            "short_lists": sum(len(items) < 12 for items in predictions.values()),
            "runtime_isolation": (
                "PyTorch encoding, FAISS/features, LightGBM scoring "
                "in sequential fresh processes"
            ),
        },
    )
