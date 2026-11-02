"""One resource-estimated LambdaMART fit and bounded shard scoring/reload."""

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from hm_recsys.contracts import digest, read_json, write_json
from hm_recsys.evaluation import metrics
from hm_recsys.features import CATEGORICAL, FEATURES
from hm_recsys.pilot.data import allocation
from hm_recsys.ranking import matrix, ordered_predictions, select_training_rows


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
    manifest = read_json(run / "features" / name / "manifest.json")
    if manifest["schema_sha256"] != digest(run / "feature_schema.json"):
        raise ValueError("Feature schema identity mismatch")
    return manifest


def train(run: Path, profile: dict) -> None:
    """Sample native Train groups, estimate actual matrices, then fit exactly once."""
    sampled = run / "tmp/sampled"
    sampled.mkdir(parents=True)
    entries = []
    group_sizes: list[int] = []
    query_order = []
    total = positive = dropped = 0
    for fold in profile["folds"]:
        if fold == "Validation":
            continue
        truth = read_json(run / "labels" / f"{fold}.json")
        for shard in feature_manifest(run, fold)["shards"]:
            table = read_shard(run, shard)
            full = matrix(table)
            indices: list[int] = []
            labels: list[int] = []
            offset = 0
            for group in shard["groups"]:
                u, n = group["user"], group["rows"]
                ids = table["article_id"].slice(offset, n).to_pylist()
                selected = select_training_rows(
                    ids,
                    set(truth[u]),
                    full[offset : offset + n, FEATURES.index("rrf_fused_score")],
                    query=f"{fold}:{u}",
                )
                if selected:
                    group_sizes.append(len(selected))
                    query_order.append(f"{fold}:{u}")
                    indices.extend(offset + j for j in selected)
                    labels.extend(int(ids[j] in truth[u]) for j in selected)
                else:
                    dropped += 1
                offset += n
            if indices:
                data = pa.table(
                    {
                        name: pa.array(full[indices, j], type=pa.float32())
                        for j, name in enumerate(FEATURES)
                    }
                )
                data = data.append_column("label", pa.array(labels, type=pa.int32()))
                path = sampled / f"{len(entries):05d}.parquet"
                pq.write_table(data, path)
                entries.append(path)
                total += len(indices)
                positive += sum(labels)
    valid = feature_manifest(run, "Validation")
    valid_rows = valid["rows"]
    # Includes matrix copies, binning/histograms and model overhead; LightGBM
    # is still an in-memory fit. The monitor remains the final runtime guard.
    estimated = (total + valid_rows) * (len(FEATURES) * 4 + 4) * 10 + 512 * 1024**2
    estimate = {
        "training_rows": total,
        "validation_rows": valid_rows,
        "features": len(FEATURES),
        "estimated_bytes": estimated,
        "assumption": "10x numeric payload plus 512 MiB, not an out-of-core guarantee",
    }
    write_json(run / "ranker_memory_estimate.json", estimate)
    allocation(estimated, "single LambdaMART fit")
    if not total or sum(group_sizes) != total:
        raise ValueError("No valid native training preference groups")
    x = np.lib.format.open_memmap(
        run / "tmp/train_x.npy",
        mode="w+",
        dtype=np.float32,
        shape=(total, len(FEATURES)),
    )
    y = np.empty(total, dtype=np.int32)
    offset = 0
    for path in entries:
        table = pq.read_table(path)
        x[offset : offset + len(table)] = matrix(table)
        y[offset : offset + len(table)] = table["label"].to_numpy()
        offset += len(table)
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
    truth = read_json(run / "labels/Validation.json")
    valid_sizes = []
    offset = 0
    for entry in valid["shards"]:
        table = read_shard(run, entry)
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
        n_jobs=2,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    )
    options = {}
    if valid_rows:
        options = {
            "eval_set": [(vx, vy)],
            "eval_group": [valid_sizes],
            "eval_at": (12,),
            "callbacks": [
                lgb.early_stopping(c["ranker_early_stopping"], verbose=False)
            ],
        }
    model.fit(
        x,
        y,
        group=group_sizes,
        feature_name=list(FEATURES),
        categorical_feature=[FEATURES.index(n) for n in CATEGORICAL],
        **options,
    )
    model.booster_.save_model(str(run / "models/lambdamart.txt"))
    predictions = {u: [] for u in read_json(run / "queries/Validation.json")["users"]}
    for j, entry in enumerate(valid["shards"]):
        table = read_shard(run, entry)
        scores = (
            np.asarray(model.predict(matrix(table)), dtype=np.float64)
            if len(table)
            else np.empty(0)
        )
        np.save(
            run / "predictions" / f"reference-{j:05d}.npy", scores, allow_pickle=False
        )
        predictions.update(
            ordered_predictions(table, scores, [g["user"] for g in entry["groups"]])
        )
    write_json(run / "predictions/reference_top12.json", predictions)
    sampling = {
        "query_order": query_order,
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
        profile="pilot",
        is_smoke=False,
        is_benchmark=False,
        validation=metrics(predictions, {u: set(v) for u, v in truth.items()}),
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
            "candidate_ndcg_diagnostic": model.evals_result_,
        },
    )
    write_json(run / "pilot_results.json", results)


def reload(run: Path) -> None:
    """Score freshly rebuilt shards in a new process; no label reads or refits."""
    model = lgb.Booster(model_file=str(run / "models/lambdamart.txt"))
    if model.feature_name() != list(FEATURES):
        raise ValueError("Reload model schema mismatch")
    original = feature_manifest(run, "Validation")
    fresh = read_json(run / "predictions/reloaded_features/manifest.json")
    if (
        len(original["shards"]) != len(fresh["shards"])
        or original["rows"] != fresh["rows"]
    ):
        raise ValueError("Reload shard/group mismatch")
    predictions = {u: [] for u in read_json(run / "queries/Validation.json")["users"]}
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
            matrix(before), matrix(after), rtol=1e-5, atol=1e-6, equal_nan=True
        )
        scores = (
            np.asarray(model.predict(matrix(after), num_threads=2), dtype=np.float64)
            if len(after)
            else np.empty(0)
        )
        expected = np.load(
            run / "predictions" / f"reference-{j:05d}.npy", allow_pickle=False
        )
        np.testing.assert_allclose(scores, expected, rtol=1e-6, atol=1e-7)
        if len(scores):
            maximum = max(maximum, float(np.max(np.abs(scores - expected))))
        predictions.update(
            ordered_predictions(after, scores, [g["user"] for g in new["groups"]])
        )
    if predictions != read_json(run / "predictions/reference_top12.json"):
        raise ValueError("Reload Top-12 mismatch")
    write_json(run / "predictions/reloaded_top12.json", predictions)
    write_json(
        run / "reload_results.json",
        {
            "status": "passed",
            "fresh_process": True,
            "labels_read": False,
            "refit": False,
            "queries": len(predictions),
            "candidate_rows": fresh["rows"],
            "max_score_absolute_difference": maximum,
            "feature_rtol": 1e-5,
            "feature_atol": 1e-6,
            "score_rtol": 1e-6,
            "score_atol": 1e-7,
            "unique_top12": True,
        },
    )
