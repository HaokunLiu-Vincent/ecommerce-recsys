"""Merge six native lists without labels and compute reciprocal-rank features.

Candidate identity is query/article; every contributing score/rank/fallback is
retained. No positives are injected and no RRF prefilter is introduced. RRF uses
smoothing sixty and the complete union is returned in article-ID order, bounded
by the configured pre-deduplication quota sum of 270."""

import math
from dataclasses import dataclass

from hm_recsys.core.contracts import CHANNELS, QUOTAS
from hm_recsys.retrieval.base import Hit


@dataclass
class Candidate:
    """One native item with every contributing channel, without future labels."""

    article: str
    hits: dict[str, Hit]

    @property
    def rrf(self) -> float:
        """Sum contributions from actual ranks only, without truncating union."""
        return sum(1 / (60 + hit.rank) for hit in self.hits.values())


def merge(
    outputs: dict, active: set[str], fallback: set[str] | None = None
) -> list[Candidate]:
    """Validate channel-local results and preserve their complete native union."""
    union: dict[str, Candidate] = {}
    fallback = fallback or set()
    for channel in CHANNELS:
        values = outputs[channel]
        if len(values) > QUOTAS[channel] or len({item for item, _ in values}) != len(
            values
        ):
            raise ValueError("Channel quota or uniqueness violation")
        for rank, (item, score) in enumerate(values, 1):
            if item not in active or not math.isfinite(score):
                raise ValueError("Invalid channel eligibility or score")
            candidate = union.setdefault(item, Candidate(item, {}))
            candidate.hits[channel] = Hit(
                score, rank, channel == "trend" and item in fallback
            )
    if len(union) > sum(QUOTAS.values()):
        raise ValueError("Native union exceeds budget")
    return [union[item] for item in sorted(union)]
