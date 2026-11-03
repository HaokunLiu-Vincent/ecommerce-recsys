"""Replay frozen feature artifacts through the canonical builder on bounded runs.

The comparator checks candidate order, all feature columns, train-only sampling,
ranked Top-12 and full-truth metrics using genuine fitted fixture models. Saved
features are the reference, so no second feature algorithm remains in active
source. Read time and rebuild time have different scopes and are not a speedup
comparison. Callers must select an explicitly bounded fixture run.
"""

import sys
from datetime import date
from pathlib import Path
from time import perf_counter

import lightgbm as lgb
import numpy as np

from hm_recsys.core.contracts import write_json
from hm_recsys.data.history import History
from hm_recsys.data.streaming import Partition
from hm_recsys.evaluation.metrics import MetricAccumulator
from hm_recsys.features.builder import FEATURES, FeatureBuilder
from hm_recsys.features.content import MappedContent
from hm_recsys.fusion.storage import CandidateReader
from hm_recsys.ranking.model import matrix, ordered_predictions, select_training_rows
from hm_recsys.ranking.pipeline import feature_manifest, read_shard
from hm_recsys.retrieval.snapshot import GlobalSnapshot


def compare(work: Path, output: Path) -> dict:
    """Check saved-versus-rebuilt parity without another implementation or fit."""
    history = History(work)
    builder = FeatureBuilder(MappedContent(work), work)
    model = lgb.Booster(model_file=str(work / "models/lambdamart.txt"))
    timings = {"saved_read": 0.0, "canonical_rebuild": 0.0}
    maximum = 0.0
    rows = queries = 0
    results = {}
    try:
        for name in ("F1", "Validation"):
            part = Partition(work, name)
            snapshot = GlobalSnapshot(
                history,
                work,
                name,
                date.fromisoformat(part.meta["cutoff"]),
                include_itemcf=False,
            )
            candidate_reader = CandidateReader(work, name)
            left, right = MetricAccumulator(), MetricAccumulator()
            try:
                for entry in feature_manifest(work, name)["shards"]:
                    started = perf_counter()
                    table = read_shard(work, entry)
                    expected = matrix(table)
                    timings["saved_read"] += perf_counter() - started
                    users = [group["user"] for group in entry["groups"]]
                    blocks = []
                    offset = 0
                    for start in range(0, len(users), 32):
                        batch = users[start : start + 32]
                        native = candidate_reader.read(batch)
                        snapshot.set_users(batch)
                        truth = part.truth(batch)
                        for user in batch:
                            candidates = native[user]
                            ids = [candidate.article for candidate in candidates]
                            count = len(ids)
                            if (
                                table["article_id"].slice(offset, count).to_pylist()
                                != ids
                            ):
                                raise ValueError("Saved native candidate order changed")
                            started = perf_counter()
                            actual = builder.build(snapshot, user, candidates)
                            timings["canonical_rebuild"] += perf_counter() - started
                            reference = expected[offset : offset + count]
                            np.testing.assert_allclose(
                                reference, actual, rtol=1e-5, atol=1e-6, equal_nan=True
                            )
                            discrete = [
                                j
                                for j, field in enumerate(FEATURES)
                                if field.startswith(("hit_", "has_"))
                                or field
                                in (
                                    "item_visual_cluster_id",
                                    "product_type_no",
                                    "index_group_no",
                                )
                            ]
                            np.testing.assert_array_equal(
                                reference[:, discrete], actual[:, discrete]
                            )
                            finite = np.isfinite(reference) & np.isfinite(actual)
                            if finite.any():
                                maximum = max(
                                    maximum,
                                    float(
                                        np.max(
                                            np.abs(reference[finite] - actual[finite])
                                        )
                                    ),
                                )
                            if name == "F1":
                                column = FEATURES.index("rrf_fused_score")
                                old = select_training_rows(
                                    ids,
                                    truth[user],
                                    reference[:, column],
                                    query=f"{name}:{user}",
                                )
                                new = select_training_rows(
                                    ids,
                                    truth[user],
                                    actual[:, column],
                                    query=f"{name}:{user}",
                                )
                                if old != new:
                                    raise ValueError(
                                        "Native train-only sampling changed"
                                    )
                            blocks.append(actual)
                            offset += count
                            rows += count
                            queries += 1
                    rebuilt = (
                        np.concatenate(blocks) if blocks else np.empty_like(expected)
                    )
                    predictions = []
                    for values in (expected, rebuilt):
                        scores = (
                            np.asarray(model.predict(values, num_threads=2))
                            if len(values)
                            else np.empty(0)
                        )
                        predictions.append(ordered_predictions(table, scores, users))
                    if predictions[0] != predictions[1]:
                        raise ValueError("Top-12 parity failure")
                    truth = part.truth(users)
                    left.add(predictions[0], truth)
                    right.add(predictions[1], truth)
                if left.result() != right.result():
                    raise ValueError("Metric denominator/value parity failure")
                results[name] = left.result()
            finally:
                candidate_reader.close()
                snapshot.close()
                part.close()
    finally:
        history.close()
    result = {
        "status": "passed",
        "scope": "bounded saved-artifact replay; no speedup claim",
        "queries": queries,
        "candidate_rows": rows,
        "features": len(FEATURES),
        "seconds": timings,
        "max_feature_absolute_difference": maximum,
        "discrete_exact": True,
        "top12_exact": True,
        "metrics": results,
    }
    write_json(output, result)
    return result


if __name__ == "__main__":
    compare(Path(sys.argv[1]), Path(sys.argv[2]))
