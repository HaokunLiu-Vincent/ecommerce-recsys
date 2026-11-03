"""Build R3 variant nominations using the actual product_code relationship.

Latest purchases in the thirty-day window nominate active sibling articles,
excluding the source article. Scores preserve maximum nomination, fourteen-day
decay and seven-day sales; the same cached source ages feed ranking features.
No future labels or article-ID prefix guesses enter this channel."""

import math
from datetime import date

from hm_recsys.retrieval.base import top


def reference_scores(snapshot, user: str) -> tuple[dict[str, float], dict[str, int]]:
    """Use real product-code joins and maximum nomination, excluding self."""
    seeds: dict[str, date] = {}
    for row in snapshot.window(snapshot.by_user.get(user, []), 30):
        seeds[row.article] = max(seeds.get(row.article) or row.day, row.day)
    scores: dict[str, float] = {}
    days: dict[str, int] = {}
    for seed, day in sorted(seeds.items()):
        product = str(snapshot.sample.articles[seed].get("product_code"))
        age = (snapshot.cutoff - day).days
        for item in snapshot.products.get(product, []):
            if item == seed or item not in snapshot.active:
                continue
            score = math.exp(-age / 14) * math.log1p(snapshot.sales[7][item])
            if item not in scores or score > scores[item]:
                scores[item] = score
                days[item] = age
    return (scores, days)


def scores(snapshot, user: str):
    """Reuse sibling nominations with the same cutoff and active pool."""
    if not getattr(snapshot, "optimized", False):
        return reference_scores(snapshot, user)
    if not hasattr(snapshot, "_sibling_cache"):
        snapshot._sibling_cache = {}
    if user not in snapshot._sibling_cache:
        snapshot._sibling_cache[user] = reference_scores(snapshot, user)
    return snapshot._sibling_cache[user]


def retrieve(snapshot, user: str) -> list[tuple[str, float]]:
    """Return eligible Top-30 siblings with deterministic article-ID ties."""
    return top(scores(snapshot, user)[0], snapshot.active, 30)
