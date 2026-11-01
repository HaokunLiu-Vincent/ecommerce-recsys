"""Independent full-ground-truth query metrics for protocol v2.0."""

import math


def metrics(
    predictions: dict[str, list[str]], labels: dict[str, set[str]], k: int = 12
) -> dict[str, float | int]:
    """Compute macro AP, recall, and NDCG, retaining zero-candidate queries.

    Args:
        predictions: Ordered unique article IDs per query; absent keys are empty.
        labels: Independent full catalog-matched truth; empty sets are ineligible.
        k: Ranking cutoff. The smoke primary report uses 12.

    Raises:
        ValueError: If a prediction has duplicates or k is invalid.
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
        total_recall += hits / len(truth)
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
    """Mean full-truth recall without a ranking truncation or hit filtering."""
    values = [
        len(set(candidates.get(q, [])) & truth) / len(truth)
        for q, truth in labels.items()
        if truth
    ]
    return sum(values) / len(values) if values else 0.0
