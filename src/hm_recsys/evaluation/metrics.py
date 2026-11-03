"""Compute native recommendation metrics from complete, independent ground truth.

Queries with nonempty catalog-matched truth define the macro denominator, even
when retrieval returns no candidates. Predictions must contain unique articles.
The same AP/Recall/NDCG math serves training diagnostics, independent Validation,
retrieval adapters and bounded streaming accumulation; no active-pool or factor
vocabulary restriction is applied to truth. This module performs no artifact I/O.
"""

import math


def recall_at_k(ranked: list[str], truth: set[str], k: int | None = None) -> float:
    """Return unique-hit recall using the complete truth denominator.

    A null cutoff evaluates the entire native union. Empty truth contributes
    zero to this primitive; macro adapters exclude it from eligible queries.
    """
    if k is not None and k <= 0:
        raise ValueError("Metric cutoff must be positive")
    selected = ranked if k is None else ranked[:k]
    return len(set(selected) & truth) / len(truth) if truth else 0.0


def metrics(
    predictions: dict[str, list[str]],
    labels: dict[str, set[str]],
    k: int = 12,
) -> dict[str, float | int]:
    """Compute full-truth macro AP, Recall and NDCG, retaining failed queries.

    Args:
        predictions: Ordered unique articles per query; absent keys are empty.
        labels: Independent full catalog-matched truth, not candidate-filtered.
        k: Ranking cutoff; the primary external report uses 12.

    Returns:
        Macro metrics and explicit eligible, empty-truth and zero-hit counts.

    Raises:
        ValueError: If predictions contain duplicates or the cutoff is invalid.
    """
    if k <= 0:
        raise ValueError("Metric cutoff must be positive")
    total_ap = total_recall = total_ndcg = 0.0
    eligible = zero_hits = zero_candidates = 0
    for query, truth in labels.items():
        ranked = predictions.get(query, [])
        if len(set(ranked)) != len(ranked):
            raise ValueError("Duplicate prediction IDs are prohibited")
        if not truth:
            continue
        eligible += 1
        hits = 0
        ap = dcg = 0.0
        for rank, item in enumerate(ranked[:k], 1):
            if item in truth:
                hits += 1
                ap += hits / rank
                dcg += 1 / math.log2(rank + 1)
        total_ap += ap / min(k, len(truth))
        total_recall += recall_at_k(ranked, truth, k)
        total_ndcg += dcg / sum(
            1 / math.log2(j + 1) for j in range(1, min(k, len(truth)) + 1)
        )
        zero_hits += hits == 0
        zero_candidates += not ranked
    denominator = max(eligible, 1)
    return {
        f"MAP@{k}": total_ap / denominator,
        f"Recall@{k}": total_recall / denominator,
        f"NDCG@{k}": total_ndcg / denominator,
        "eligible_queries": eligible,
        "zero_hit_queries": zero_hits,
        "zero_candidate_queries": zero_candidates,
        "excluded_empty_truth_queries": len(labels) - eligible,
    }


def union_recall(
    candidates: dict[str, list[str]], labels: dict[str, set[str]]
) -> float:
    """Return macro native-union recall without truncating or filtering truth."""
    values = [
        recall_at_k(candidates.get(query, []), truth)
        for query, truth in labels.items()
        if truth
    ]
    return sum(values) / len(values) if values else 0.0


class MetricAccumulator:
    """Accumulate bounded macro batches with their actual eligible denominators."""

    def __init__(self, k: int = 12) -> None:
        """Initialize a cutoff-specific accumulator without retaining query IDs."""
        if k <= 0:
            raise ValueError("Metric cutoff must be positive")
        self.k = k
        self.totals = {key: 0.0 for key in metrics({}, {}, k)}

    def add(
        self, predictions: dict[str, list[str]], labels: dict[str, set[str]]
    ) -> None:
        """Consume one bounded batch through the authoritative metric formulas."""
        measured = metrics(predictions, labels, self.k)
        eligible = measured["eligible_queries"]
        for key, value in measured.items():
            self.totals[key] += value * eligible if "@" in key else value

    def result(self) -> dict[str, float | int]:
        """Return macro values and exact counts, including an empty population."""
        denominator = max(self.totals["eligible_queries"], 1)
        return {
            key: value / denominator if "@" in key else int(value)
            for key, value in self.totals.items()
        }
