"""Build R2 repeat-purchase scores from legal sixty-day raw purchase lines.

Every repeated line contributes its existing fourteen-day decay. The bounded
snapshot cache is shared with ranking features and reset when the query batch
changes. Outputs include count and latest date for audit/features; eligibility
and deterministic Top-30 truncation are applied only by retrieve()."""

import math
from collections import Counter, defaultdict
from datetime import date

from hm_recsys.retrieval.base import top


def reference_scores(
    snapshot, user: str
) -> tuple[dict[str, float], Counter, dict[str, date]]:
    """Aggregate 60-day line counts, last dates, and 14-day decays."""
    score: dict[str, float] = defaultdict(float)
    count: Counter = Counter()
    last: dict[str, date] = {}
    for row in snapshot.window(snapshot.by_user.get(user, []), 60):
        score[row.article] += math.exp(-(snapshot.cutoff - row.day).days / 14)
        count[row.article] += 1
        last[row.article] = max(last.get(row.article) or row.day, row.day)
    return (score, count, last)


def scores(snapshot, user: str):
    """Reuse the current bounded user's rule output in shared features."""
    if not getattr(snapshot, "optimized", False):
        return reference_scores(snapshot, user)
    if not hasattr(snapshot, "_rep_cache"):
        snapshot._rep_cache = {}
    if user not in snapshot._rep_cache:
        snapshot._rep_cache[user] = reference_scores(snapshot, user)
    return snapshot._rep_cache[user]


def retrieve(snapshot, user: str) -> list[tuple[str, float]]:
    """Return native eligible repeat purchases, including previously bought items."""
    return top(scores(snapshot, user)[0], snapshot.active, 30)
