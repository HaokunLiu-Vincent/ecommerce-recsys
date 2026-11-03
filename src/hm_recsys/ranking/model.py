"""Define the sole native ranker sampling and prediction contracts.

The query-local sampler retains every native positive and at most twenty
negatives per positive, half hard by RRF and half deterministically random.
Typed feature matrices follow the frozen schema; score sorting returns unique
Top-12 with article-ID ties and preserves empty queries. LightGBM progress is
observational and never changes boosting parameters or evaluation semantics.
"""

import lightgbm as lgb
import numpy as np
import numpy.typing as npt
import pyarrow as pa

from hm_recsys.core.contracts import stable_key
from hm_recsys.core.progress import emit
from hm_recsys.features.builder import FEATURES
from hm_recsys.features.content import FloatArray


class BoostProgress:
    """Expose actual LightGBM iterations and label candidate-list metrics clearly."""

    order = 10
    before_iteration = False

    def __init__(self, total: int) -> None:
        self.total = total
        self.last_iteration = 0

    def __call__(self, environment: lgb.callback.CallbackEnv) -> None:
        self.last_iteration = environment.iteration + 1
        diagnostics = {
            f"{v[0]}/{v[1]}": float(v[2]) if np.isfinite(v[2]) else None
            for v in environment.evaluation_result_list or []
        }
        emit(
            "lambdamart_boosting",
            self.last_iteration,
            self.total,
            "iterations",
            metrics={"candidate_list_diagnostics": diagnostics},
            force=self.last_iteration in (1, self.total),
        )


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
