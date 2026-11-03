"""Define native source-hit records and the shared eligibility/Top-K contract.

Each hit keeps its channel-local finite score, one-based rank and explicit
fallback flag. Eligibility precedes truncation; article IDs resolve ties.
This low-level module contains no model fit, persistence or orchestration and
is shared by all six retrievers and native candidate fusion."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Hit:
    """One source contribution; raw scores retain their channel-local scale."""

    score: float
    rank: int
    fallback: bool = False


def top(scores: dict[str, float], active: set[str], k: int) -> list[tuple[str, float]]:
    """Apply eligibility first and stable article-ID ties before the quota."""
    return sorted(
        ((i, s) for i, s in scores.items() if i in active),
        key=lambda pair: (-pair[1], pair[0]),
    )[:k]
