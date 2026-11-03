"""Fold current users into frozen WALS item factors under one confidence transform.

Observed raw purchase lines are time-decayed and mapped onto the fixed item
vocabulary. A regularized implicit normal equation uses the cached item Gram
matrix; unobserved confidence is one. Outputs preserve unnormalized inner-product
scores and repeat-purchase eligibility. This mathematical module never fits
models, accesses future labels, changes mappings or manages stage persistence.
"""

from collections import defaultdict
from datetime import date

import numpy as np

from hm_recsys.core.contracts import (
    START,
)
from hm_recsys.data.sample import Transaction
from hm_recsys.features.content import FloatArray
from hm_recsys.retrieval.vectors import BlockIndex

REGULARIZATION = 0.05


def confidence(
    history: list[Transaction], *, cutoff: date, vocabulary: dict[str, int]
) -> dict[int, float]:
    """Construct full observed confidence once; unobserved entries have c=1."""
    weights: dict[int, float] = defaultdict(float)
    for row in history:
        if START <= row.day <= cutoff and row.article in vocabulary:
            weights[vocabulary[row.article]] += np.exp(-(cutoff - row.day).days / 30)
    return {i: float(1 + 20 * np.log1p(value)) for i, value in weights.items()}


def fold_user(
    factors: FloatArray, observed: dict[int, float], *, gram: np.ndarray | None = None
) -> FloatArray:
    """Solve regularized implicit least squares with fixed, unnormalized items."""
    if not observed:
        return np.zeros(factors.shape[1], dtype=np.float32)
    positions = list(observed)
    conf = np.array([observed[i] for i in positions], dtype=np.float64)
    selected = factors[positions].astype(np.float64)
    if gram is None:
        y = factors.astype(np.float64)
        gram = y.T @ y
    lhs = gram + selected.T * (conf - 1) @ selected
    lhs += np.eye(factors.shape[1]) * REGULARIZATION
    rhs = selected.T @ conf
    return np.linalg.solve(lhs, rhs).astype(np.float32)


def fold_users_batch(
    factors: FloatArray, observed: list[dict[int, float]], *, gram: np.ndarray
) -> FloatArray:
    """Batch the same float64 fixed-item normal equations; empty users stay zero."""
    if len(observed) > 512:
        raise ValueError("WALS folding batch exceeds 512")
    dimension = factors.shape[1]
    lhs = np.empty((len(observed), dimension, dimension), dtype=np.float64)
    rhs = np.zeros((len(observed), dimension), dtype=np.float64)
    valid = []
    for j, obs in enumerate(observed):
        if not obs:
            continue
        selected = factors[list(obs)].astype(np.float64)
        conf = np.array(list(obs.values()), dtype=np.float64)
        lhs[j] = gram + selected.T * (conf - 1) @ selected
        lhs[j] += np.eye(dimension) * REGULARIZATION
        rhs[j] = selected.T @ conf
        valid.append(j)
    result = np.zeros((len(observed), dimension), dtype=np.float32)
    if valid:
        result[valid] = (
            np.linalg.solve(lhs[valid], rhs[valid, :, None])
            .squeeze(-1)
            .astype(np.float32)
        )
    return result


class WALS:
    """Frozen item factors with genuine current-history user folding-in."""

    def __init__(self, index: BlockIndex) -> None:
        self.index = index
        self.mapping = {item: i for i, item in enumerate(index.ids)}
        self.gram = np.zeros((index.vectors.shape[1], index.vectors.shape[1]))
        for start in range(0, len(index.ids), 4096):
            block = index.vectors[start : start + 4096].astype(np.float64)
            self.gram += block.T @ block

    def retrieve(
        self, history: list[Transaction], *, cutoff: date, active: set[str]
    ) -> list[tuple[str, float]]:
        """Return fixed-factor scores without suppressing previously bought items."""
        obs = confidence(history, cutoff=cutoff, vocabulary=self.mapping)
        if not obs:
            return []
        user = fold_user(self.index.vectors, obs, gram=self.gram)
        return self.index.search(user, active, 50)
