"""Train-only negative sampling and a bounded native LambdaMART ranker."""

from pathlib import Path

import lightgbm as lgb
import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.parquet as pq

from hm_recsys.content import FloatArray
from hm_recsys.contracts import Config, read_json, stable_key, write_json
from hm_recsys.evaluation import metrics
from hm_recsys.features import CATEGORICAL, FEATURES


def select_training_rows(
    ids: list[str], truth: set[str], rrf: npt.NDArray, *, query: str
) -> list[int]:
    """Keep all native positives and <=20P hard/random negatives; no injection."""
    positive = [j for j, item in enumerate(ids) if item in truth]
    negative = [j for j, item in enumerate(ids) if item not in truth]
    if not positive or not negative:
        return []
    count = min(len(negative), 20 * len(positive))
    hard = sorted(negative, key=lambda j: (-float(rrf[j]), ids[j]))[: count // 2]
    hard_set = set(hard)
    remainder = [j for j in negative if j not in hard_set]
    rng = np.random.default_rng(int(stable_key(query)[:16], 16))
    random = rng.choice(remainder, size=count - len(hard), replace=False).tolist()
    return sorted([*positive, *hard, *random], key=lambda j: ids[j])


def matrix(table: pa.Table) -> FloatArray:
    """Read the named float32 feature columns in the persisted schema order."""
    return np.column_stack([table[name].to_numpy() for name in FEATURES]).astype(
        np.float32
    )


def ordered_predictions(
    table: pa.Table, scores: npt.NDArray, users: list[str]
) -> dict[str, list[str]]:
    """Return stable unique Top-12, preserving empty query keys."""
    if len(table) != len(scores) or not np.isfinite(scores).all():
        raise ValueError("Invalid ranker score alignment")
    grouped: dict[str, list[tuple[str, float]]] = {u: [] for u in users}
    for user, item, score in zip(
        table["customer_id"].to_pylist(),
        table["article_id"].to_pylist(),
        scores,
        strict=True,
    ):
        grouped[user].append((item, float(score)))
    predictions = {}
    for user, pairs in grouped.items():
        if len(pairs) != len({i for i, _ in pairs}):
            raise ValueError("Duplicate query/article rows")
        predictions[user] = [
            i for i, _ in sorted(pairs, key=lambda pair: (-pair[1], pair[0]))[:12]
        ]
    return predictions


def train_ranker(run: Path, config: Config) -> None:
    """Fit one bounded configuration; external Validation MAP is diagnostic only."""
    config.check_sample(read_json(run / "sample_manifest.json"))
    total = sum(
        pq.ParquetFile(path).metadata.num_rows
        for path in (run / "features").glob("*.parquet")
    )
    if total > config.limit("max_feature_rows"):
        raise ValueError("Feature budget exceeded before ranker allocation")
    x_blocks, y_blocks, sizes = [], [], []
    sampling = {}
    query_order = []
    for partition in ("F1", "F2", "F3", "F4"):
        table = pq.read_table(run / "features" / f"{partition}.parquet")
        full = matrix(table)
        users = table["customer_id"].to_pylist()
        ids = table["article_id"].to_pylist()
        labels = read_json(run / "labels" / f"{partition}.json")
        query_ids = read_json(run / "queries" / f"{partition}.json")["users"]
        kept = dropped = 0
        for user in sorted(query_ids):
            indices = np.array(
                [i for i, u in enumerate(users) if u == user], dtype=np.int64
            )
            group_ids = [ids[i] for i in indices]
            truth = set(labels[user])
            key = f"{partition}:{user}"
            selected = select_training_rows(
                group_ids,
                truth,
                full[indices, FEATURES.index("rrf_fused_score")],
                query=key,
            )
            if not selected:
                dropped += 1
                continue
            selected_ids = [group_ids[i] for i in selected]
            labels_array = np.array(
                [int(item in truth) for item in selected_ids], dtype=np.int32
            )
            if set(labels_array) != {0, 1}:
                raise ValueError("Ranker group lacks a preference pair")
            x_blocks.append(full[indices[selected]])
            y_blocks.append(labels_array)
            sizes.append(len(selected))
            query_order.append(key)
            kept += 1
        sampling[partition] = {
            "kept_groups": kept,
            "dropped_no_pair_groups": dropped,
            "original_queries": len(query_ids),
        }
        if not kept:
            write_json(run / "ranker_sampling.json", sampling)
            raise ValueError(f"No native training pairs in required fold {partition}")
    x_train = np.concatenate(x_blocks)
    y_train = np.concatenate(y_blocks)
    if sum(sizes) != len(x_train) or len(x_train) != len(y_train):
        raise ValueError("Ranker query-group alignment failure")
    validation = pq.read_table(run / "features/Validation.parquet")
    x_valid = matrix(validation)
    labels = read_json(run / "labels/Validation.json")
    users = validation["customer_id"].to_pylist()
    ids = validation["article_id"].to_pylist()
    y_valid = np.array([int(i in labels[u]) for u, i in zip(users, ids, strict=True)])
    valid_sizes = [users.count(u) for u in sorted(set(users))]
    ranker = lgb.LGBMRanker(
        objective="lambdarank",
        n_estimators=config.limit("ranker_rounds"),
        learning_rate=0.03,
        num_leaves=config.limit("ranker_num_leaves"),
        max_depth=config.limit("ranker_max_depth"),
        min_child_samples=config.limit("ranker_min_child_samples"),
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=5.0,
        random_state=42,
        n_jobs=config.limit("threads"),
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    )
    categorical = [FEATURES.index(name) for name in CATEGORICAL]
    if len(validation):
        ranker.fit(
            x_train,
            y_train,
            group=sizes,
            feature_name=list(FEATURES),
            categorical_feature=categorical,
            eval_set=[(x_valid, y_valid)],
            eval_group=[valid_sizes],
            eval_at=(12,),
        )
    else:
        ranker.fit(
            x_train,
            y_train,
            group=sizes,
            feature_name=list(FEATURES),
            categorical_feature=categorical,
        )
    ranker.booster_.save_model(str(run / "models/lambdamart.txt"))
    scores = ranker.predict(x_valid) if len(validation) else np.empty(0)
    scores = np.asarray(scores, dtype=np.float64)
    query_ids = read_json(run / "queries/Validation.json")["users"]
    predictions = ordered_predictions(validation, scores, query_ids)
    np.save(run / "predictions/reference_scores.npy", scores, allow_pickle=False)
    write_json(run / "predictions/reference_top12.json", predictions)
    results = read_json(run / "candidate_results.json")
    results.update(
        {
            "status": "trained_pending_fresh_process_reload",
            "is_smoke": True,
            "is_benchmark": False,
            "final_test": "not_read_or_evaluated",
            "validation": metrics(
                predictions, {u: set(items) for u, items in labels.items()}
            ),
            "ranker": {
                "requested_rounds": config.limit("ranker_rounds"),
                "actual_rounds": ranker.booster_.current_iteration(),
                "split_count": sum(
                    tree["num_leaves"] - 1
                    for tree in ranker.booster_.dump_model()["tree_info"]
                ),
                "training_rows": len(x_train),
                "training_groups": len(sizes),
                "group_sizes": sizes,
                "query_order": query_order,
                "positive_rows": int(y_train.sum()),
                "candidate_ndcg_diagnostic": ranker.evals_result_,
                "selection": "single fixed smoke configuration; no tuning search",
            },
        }
    )
    write_json(
        run / "ranker_sampling.json",
        {
            "folds": sampling,
            "negative_cap": 20,
            "hard_fraction": 0.5,
            "seed": 42,
            "policy": "native-rrf-half-stable-query-seed-v1",
        },
    )
    write_json(run / "smoke_results.json", results)
