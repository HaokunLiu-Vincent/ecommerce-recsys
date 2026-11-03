"""Train one grouped LambdaMART and verify fresh-process ranking parity.

Native feature shards and independent query/label rosters feed the shared
train-only sampler. All folds jointly form one contiguous grouped fit; complete
Validation is unsampled and scored in bounded blocks with full-truth metrics.
Resource estimates precede numeric memmaps and LightGBM allocation. Reload and
final authorized Test use frozen weights/schema and never sample or refit.
"""

import json
from pathlib import Path
from time import perf_counter

import lightgbm as lgb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from hm_recsys.core.contracts import digest, read_json, write_json
from hm_recsys.core.progress import emit
from hm_recsys.data.streaming import Partition, load_manifest
from hm_recsys.evaluation.metrics import MetricAccumulator
from hm_recsys.features.builder import CATEGORICAL, FEATURES
from hm_recsys.ranking.model import (
    BoostProgress,
    matrix,
    ordered_predictions,
    select_training_rows,
)
from hm_recsys.runtime.capacity import check


def read_shard(run: Path, entry: dict) -> pa.Table:
    """Validate complete query groups once, without repeated all-row user scans."""
    path = run / entry["file"]
    if digest(path) != entry["sha256"]:
        raise ValueError("Feature shard checksum mismatch")
    table = pq.read_table(path)
    if (
        table.column_names != ["customer_id", "article_id", *FEATURES]
        or len(table) != entry["rows"]
    ):
        raise ValueError("Feature shard/schema mismatch")
    offset = 0
    for group in entry["groups"]:
        n = group["rows"]
        block = table.slice(offset, n)
        users = block["customer_id"].to_pylist()
        ids = block["article_id"].to_pylist()
        if any(u != group["user"] for u in users) or len(ids) != len(set(ids)):
            raise ValueError("Noncontiguous or duplicate query groups")
        offset += n
    if offset != len(table):
        raise ValueError("Group sizes do not sum to feature rows")
    return table


def feature_manifest(run: Path, name: str) -> dict:
    """Verify the frozen schema identity before any array allocation."""
    manifest = load_manifest(run / "features" / name / "manifest.json")
    if manifest["schema_sha256"] != digest(run / "feature_schema.json"):
        raise ValueError("Feature schema identity mismatch")
    return manifest


def train(run: Path, profile: dict) -> None:
    """Sample native Train groups, estimate actual matrices, then fit exactly once."""
    for folder in ("tmp", "models", "predictions"):
        (run / folder).mkdir(parents=True, exist_ok=True)
    stage_start = perf_counter()
    timings = {}
    query_upper = sum(Partition(run, n, labels=False).count for n in profile["folds"])
    check(
        run,
        "ranker_group_metadata",
        memory=query_upper * 256 + 256 * 1024**2,
        disk=0,
        details={"query_upper_bound": query_upper},
    )
    raw_x, raw_y = (run / "tmp/train_x.raw", run / "tmp/train_y.raw")
    raw_x.touch(exist_ok=False)
    raw_y.touch(exist_ok=False)
    group_sizes: list[int] = []
    total = positive = dropped = 0
    fold_counts = {}
    audit = (run / "ranker_groups.jsonl").open("x", encoding="utf-8")
    with raw_x.open("ab") as handle_x, raw_y.open("ab") as handle_y:
        for fold in profile["folds"]:
            if fold == "Validation":
                continue
            partition = Partition(run, fold)
            fold_counts[fold] = 0
            for shard in feature_manifest(run, fold)["shards"]:
                table = read_shard(run, shard)
                truth = partition.truth([g["user"] for g in shard["groups"]])
                full = matrix(table)
                indices: list[int] = []
                labels: list[int] = []
                offset = 0
                for group in shard["groups"]:
                    u, n = (group["user"], group["rows"])
                    ids = table["article_id"].slice(offset, n).to_pylist()
                    selected = select_training_rows(
                        ids,
                        set(truth[u]),
                        full[offset : offset + n, FEATURES.index("rrf_fused_score")],
                        query=f"{fold}:{u}",
                    )
                    if selected:
                        group_sizes.append(len(selected))
                        fold_counts[fold] += 1
                        audit.write(
                            json.dumps({"fold": fold, "user": u, "rows": len(selected)})
                            + "\n"
                        )
                        indices.extend(offset + j for j in selected)
                        labels.extend(int(ids[j] in truth[u]) for j in selected)
                    else:
                        dropped += 1
                    offset += n
                if indices:
                    np.asarray(full[indices], dtype=np.float32).tofile(handle_x)
                    np.asarray(labels, dtype=np.int32).tofile(handle_y)
                    total += len(indices)
                    positive += sum(labels)
            partition.close()
    audit.close()
    timings["negative_sampling_seconds"] = perf_counter() - stage_start
    dataset_start = perf_counter()
    valid = feature_manifest(run, "Validation")
    valid_rows = valid["rows"]
    estimated = (total + valid_rows) * (len(FEATURES) * 4 + 4) * 10 + 512 * 1024**2
    estimate = {
        "training_rows": total,
        "validation_rows": valid_rows,
        "features": len(FEATURES),
        "estimated_bytes": estimated,
        "assumption": "10x numeric payload plus 512 MiB, not an out-of-core guarantee",
    }
    write_json(run / "ranker_memory_estimate.json", estimate)
    check(
        run,
        "single_lambdamart",
        memory=estimated + len(group_sizes) * 128,
        disk=(total + valid_rows) * (len(FEATURES) * 4 + 4) * 3,
        details=estimate,
    )
    if not total or sum(group_sizes) != total:
        raise ValueError("No valid native training preference groups")
    x = np.memmap(raw_x, mode="r", dtype=np.float32, shape=(total, len(FEATURES)))
    y = np.memmap(raw_y, mode="r", dtype=np.int32, shape=(total,))
    vx = (
        np.lib.format.open_memmap(
            run / "tmp/valid_x.npy",
            mode="w+",
            dtype=np.float32,
            shape=(valid_rows, len(FEATURES)),
        )
        if valid_rows
        else np.empty((0, len(FEATURES)), dtype=np.float32)
    )
    vy = np.empty(valid_rows, dtype=np.int32)
    validation = Partition(run, "Validation")
    valid_sizes = []
    offset = 0
    for entry in valid["shards"]:
        table = read_shard(run, entry)
        truth = validation.truth([g["user"] for g in entry["groups"]])
        vx[offset : offset + len(table)] = matrix(table)
        vy[offset : offset + len(table)] = [
            int(i in truth[u])
            for u, i in zip(
                table["customer_id"].to_pylist(),
                table["article_id"].to_pylist(),
                strict=True,
            )
        ]
        valid_sizes.extend(g["rows"] for g in entry["groups"] if g["rows"])
        offset += len(table)
    if sum(valid_sizes) != valid_rows:
        raise ValueError("Validation group alignment failure")
    c = profile["compute"]
    model = lgb.LGBMRanker(
        objective="lambdarank",
        n_estimators=c["ranker_rounds"],
        learning_rate=0.03,
        num_leaves=c["ranker_num_leaves"],
        max_depth=c["ranker_max_depth"],
        min_child_samples=c["ranker_min_child_samples"],
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=5.0,
        random_state=42,
        n_jobs=12,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    )
    emit(
        "ranker_dataset",
        total,
        total,
        "sampled_train_rows",
        status="complete",
        metrics={
            "train_groups": len(group_sizes),
            "positive_rows": positive,
            "validation_rows": valid_rows,
            "validation_groups": len(valid_sizes),
            "feature_columns": len(FEATURES),
        },
        force=True,
    )
    reporter = BoostProgress(c["ranker_rounds"])
    options: dict = {"callbacks": [reporter]}
    if valid_rows:
        options = {
            "eval_set": [(vx, vy)],
            "eval_group": [valid_sizes],
            "eval_at": (12,),
            "callbacks": [reporter],
        }
        if c.get("ranker_early_stopping") is not None:
            options["callbacks"].append(
                lgb.early_stopping(c["ranker_early_stopping"], verbose=False)
            )
    timings["dataset_construction_seconds"] = perf_counter() - dataset_start
    fit_start = perf_counter()
    model.fit(
        x,
        y,
        group=group_sizes,
        feature_name=list(FEATURES),
        categorical_feature=[FEATURES.index(n) for n in CATEGORICAL],
        **options,
    )
    emit(
        "lambdamart_boosting",
        reporter.last_iteration,
        reporter.total,
        "iterations",
        status="complete" if reporter.last_iteration == reporter.total else "stopped",
        metrics={
            "best_iteration": model.best_iteration_,
            "retained_trees": model.booster_.current_iteration(),
        },
        force=True,
    )
    timings["fit_seconds"] = perf_counter() - fit_start
    scoring_start = perf_counter()
    model.booster_.save_model(str(run / "models/lambdamart.txt"))
    scored_rows = 0
    metric_accumulator = MetricAccumulator()
    prediction_schema = pa.schema(
        [("customer_id", pa.string()), ("article_ids", pa.list_(pa.string()))]
    )
    with pq.ParquetWriter(
        run / "predictions/reference_top12.parquet", prediction_schema
    ) as writer:
        for j, entry in enumerate(valid["shards"]):
            table = read_shard(run, entry)
            scores = (
                np.asarray(model.predict(matrix(table)), dtype=np.float64)
                if len(table)
                else np.empty(0)
            )
            np.save(
                run / "predictions" / f"reference-{j:05d}.npy",
                scores,
                allow_pickle=False,
            )
            users = [g["user"] for g in entry["groups"]]
            batch_predictions = ordered_predictions(table, scores, users)
            truth = validation.truth(users)
            metric_accumulator.add(batch_predictions, truth)
            writer.write_table(
                pa.Table.from_pylist(
                    [
                        {"customer_id": u, "article_ids": batch_predictions[u]}
                        for u in users
                    ],
                    schema=prediction_schema,
                )
            )
            scored_rows += len(table)
            emit(
                "validation_scoring",
                scored_rows,
                valid_rows,
                "candidate_rows",
                metrics={
                    "eligible_queries_processed": metric_accumulator.result()[
                        "eligible_queries"
                    ]
                },
            )
    emit(
        "validation_scoring",
        scored_rows,
        valid_rows,
        "candidate_rows",
        status="complete",
        force=True,
    )
    metric_totals = metric_accumulator.result()
    if metric_totals["eligible_queries"] != validation.count:
        raise ValueError("Validation query denominator changed")
    validation.close()
    sampling = {
        "query_order_file": "ranker_groups.jsonl",
        "fold_groups": fold_counts,
        "group_sizes": group_sizes,
        "rows": total,
        "positive_rows": positive,
        "dropped_no_pair_groups": dropped,
        "negative_cap": 20,
        "hard_fraction": 0.5,
        "seed": 42,
        "injection": False,
    }
    write_json(run / "ranker_sampling.json", sampling)
    results = read_json(run / "candidate_results.json")
    results.update(
        status="trained_pending_fresh_process_reload",
        profile=profile["profile"],
        is_smoke=profile["profile"] == "smoke",
        is_benchmark=False,
        validation=metric_totals,
        final_test="not_read_or_evaluated",
        ranker={
            "requested_rounds": c["ranker_rounds"],
            "actual_rounds": model.booster_.current_iteration(),
            "training_rows": total,
            "training_groups": len(group_sizes),
            "positive_rows": positive,
            "split_count": sum(
                t["num_leaves"] - 1 for t in model.booster_.dump_model()["tree_info"]
            ),
            "early_stopping": "candidate NDCG@12 diagnostic, not end-to-end MAP",
            "early_stopping_rounds": c.get("ranker_early_stopping"),
            "candidate_ndcg_diagnostic": model.evals_result_,
        },
    )
    write_json(run / f"{profile['profile']}_results.json", results)
    timings.update(
        validation_scoring_seconds=perf_counter() - scoring_start,
        total_seconds=perf_counter() - stage_start,
        training_rows=total,
        validation_rows=valid_rows,
    )
    write_json(run / "ranker_performance.json", timings)


class ExpectedRows:
    """Consume up to 512 expected query records across 32-row Arrow batch boundaries."""

    def __init__(self, reference: pq.ParquetFile) -> None:
        self.batches = iter(reference.iter_batches(batch_size=32))
        self.pending = []

    def take(self, count: int) -> list[dict]:
        """Return bounded expected rows, including empty-candidate query records."""
        if not 0 <= count <= 512:
            raise ValueError("Expected prediction batch exceeds 512 queries")
        result = []
        while len(result) < count:
            if not self.pending:
                batch = next(self.batches, None)
                if batch is None:
                    break
                self.pending = batch.to_pylist()
            take = min(count - len(result), len(self.pending))
            result.extend(self.pending[:take])
            self.pending = self.pending[take:]
        return result


def reload(run: Path) -> None:
    """Score freshly rebuilt shards in a new process; no label reads or refits."""
    for folder in ("predictions",):
        (run / folder).mkdir(parents=True, exist_ok=True)
    model = lgb.Booster(model_file=str(run / "models/lambdamart.txt"))
    if model.feature_name() != list(FEATURES):
        raise ValueError("Reload model schema mismatch")
    original = feature_manifest(run, "Validation")
    fresh = load_manifest(run / "predictions/reloaded_features/manifest.json")
    if (
        len(original["shards"]) != len(fresh["shards"])
        or original["rows"] != fresh["rows"]
    ):
        raise ValueError("Reload shard/group mismatch")
    partition = Partition(run, "Validation", labels=False)
    reference = pq.ParquetFile(run / "predictions/reference_top12.parquet")
    expected_rows = ExpectedRows(reference)
    writer = pq.ParquetWriter(
        run / "predictions/reloaded_top12.parquet", reference.schema_arrow
    )
    sum_scores = 0
    row_count = 0
    maximum = 0.0
    for j, (old, new) in enumerate(
        zip(original["shards"], fresh["shards"], strict=True)
    ):
        before = read_shard(run, old)
        after = read_shard(run, new)
        for key in ("customer_id", "article_id"):
            if before[key].to_pylist() != after[key].to_pylist():
                raise ValueError("Reload candidate order changed")
        np.testing.assert_allclose(
            matrix(before), matrix(after), rtol=1e-05, atol=1e-06, equal_nan=True
        )
        scores = (
            np.asarray(model.predict(matrix(after), num_threads=2), dtype=np.float64)
            if len(after)
            else np.empty(0)
        )
        expected = np.load(
            run / "predictions" / f"reference-{j:05d}.npy", allow_pickle=False
        )
        np.testing.assert_allclose(scores, expected, rtol=1e-06, atol=1e-07)
        if len(scores):
            maximum = max(maximum, float(np.max(np.abs(scores - expected))))
        current = ordered_predictions(after, scores, [g["user"] for g in new["groups"]])
        expected_batch = expected_rows.take(len(current))
        if [
            {"customer_id": u, "article_ids": values} for u, values in current.items()
        ] != expected_batch:
            raise ValueError("Reload Top-12 mismatch")
        writer.write_table(
            pa.Table.from_pylist(
                [
                    {"customer_id": u, "article_ids": values}
                    for u, values in current.items()
                ],
                schema=reference.schema_arrow,
            )
        )
        sum_scores += len(after)
        row_count += len(current)
        emit(
            "reload_queries",
            row_count,
            partition.count,
            "queries",
            metrics={
                "candidate_rows_rescored": sum_scores,
                "max_score_absolute_difference": maximum,
            },
        )
    emit(
        "reload_queries",
        row_count,
        partition.count,
        "queries",
        status="complete",
        metrics={
            "candidate_rows_rescored": sum_scores,
            "max_score_absolute_difference": maximum,
        },
        force=True,
    )
    writer.close()
    if row_count != partition.count or expected_rows.take(1):
        raise ValueError("Reload query count mismatch")
    write_json(
        run / "predictions/reloaded_top12_manifest.json",
        {
            "verified_against": "predictions/reference_top12.parquet",
            "file": "predictions/reloaded_top12.parquet",
            "sha256": digest(run / "predictions/reloaded_top12.parquet"),
            "queries": row_count,
            "exact_match": True,
        },
    )
    write_json(
        run / "reload_results.json",
        {
            "status": "passed",
            "fresh_process": True,
            "labels_read": False,
            "refit": False,
            "queries": row_count,
            "candidate_rows": fresh["rows"],
            "max_score_absolute_difference": maximum,
            "feature_rtol": 1e-05,
            "feature_atol": 1e-06,
            "score_rtol": 1e-06,
            "score_atol": 1e-07,
            "unique_top12": True,
        },
    )


def predict_partition(run: Path, profile: dict, partition: str = "Test") -> dict:
    """Score an explicitly prepared final partition with the frozen ranker only.

    The runtime owns double authorization, configuration locking and single-use
    Test execution. This adapter independently requires full mode and preserves
    the complete query denominator, including zero-candidate queries. It does
    not invoke training, sampling, feature construction or retrieval.
    """
    if profile["profile"] != "full" or partition != "Test":
        raise PermissionError("Final Test inference requires the full Test path")
    manifest = feature_manifest(run, partition)
    model_path = run / "models/lambdamart.txt"
    booster = lgb.Booster(model_file=str(model_path))
    roster = Partition(run, partition)
    output = run / "final_test"
    # Query vectors already live in this isolated stage scope. Refuse only actual
    # prediction/report replacement; unrelated prepared inputs are prerequisites.
    output.mkdir(parents=True, exist_ok=True)
    if (output / "predictions.parquet").exists() or (
        run / "final_test_results.json"
    ).exists():
        raise FileExistsError("Final Test predictions already exist in this attempt")
    accumulator = MetricAccumulator()
    processed = candidate_rows = 0
    schema = pa.schema(
        [("customer_id", pa.string()), ("article_ids", pa.list_(pa.string()))]
    )
    started = perf_counter()
    emit("final_test_scoring", 0, roster.count, "queries", status="started", force=True)
    try:
        with pq.ParquetWriter(output / "predictions.parquet", schema) as writer:
            for entry in manifest["shards"]:
                table = read_shard(run, entry)
                users = [group["user"] for group in entry["groups"]]
                scores = (
                    np.asarray(booster.predict(matrix(table)), dtype=np.float64)
                    if len(table)
                    else np.empty(0, dtype=np.float64)
                )
                predictions = ordered_predictions(table, scores, users)
                accumulator.add(predictions, roster.truth(users))
                writer.write_table(
                    pa.Table.from_pylist(
                        [
                            {"customer_id": user, "article_ids": predictions[user]}
                            for user in users
                        ],
                        schema=schema,
                    )
                )
                processed += len(users)
                candidate_rows += len(table)
                emit(
                    "final_test_scoring",
                    processed,
                    roster.count,
                    "queries",
                    metrics={"candidate_rows": candidate_rows},
                )
        measured = accumulator.result()
        if processed != roster.count or measured["eligible_queries"] != roster.count:
            raise ValueError("Final Test query denominator changed")
        if candidate_rows != manifest["rows"]:
            raise ValueError("Final Test candidate row accounting changed")
        result = {
            "status": "completed",
            "partition": partition,
            "metrics": measured,
            "candidate_rows": candidate_rows,
            "model_sha256": digest(model_path),
            "schema_sha256": digest(run / "feature_schema.json"),
            "feature_manifest_sha256": digest(run / "features/Test/manifest.json"),
            "sampling": False,
            "refit": False,
            "seconds": perf_counter() - started,
        }
        write_json(run / "final_test_results.json", result)
        emit(
            "final_test_scoring",
            processed,
            roster.count,
            "queries",
            status="completed",
            metrics={"candidate_rows": candidate_rows},
            force=True,
        )
        return result
    finally:
        roster.close()
