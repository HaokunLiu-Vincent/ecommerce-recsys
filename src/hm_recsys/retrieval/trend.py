"""Build R1 seven-day cutoff popularity with age-bucket/global fallback.

Global SQL statistics use legal past purchase lines, never training labels.
Cached cutoff/bucket orders preserve the original decayed scores and active
filter. The global list fills only native bucket shortages to sixty and exposes
fallback article IDs so provenance survives union and feature construction."""

from hm_recsys.data.sample import age_bucket
from hm_recsys.retrieval.base import top


def popularity(snapshot, bucket: str):
    """Cache global/bucket order once per immutable cutoff snapshot."""
    key = bucket if bucket != "UNKNOWN" else "GLOBAL"
    if not getattr(snapshot, "optimized", False):
        return top(snapshot.pop.get(key, {}), snapshot.active, len(snapshot.active))
    if not hasattr(snapshot, "_pop_cache"):
        snapshot._pop_cache = {}
    if key not in snapshot._pop_cache:
        snapshot._pop_cache[key] = top(
            snapshot.pop.get(key, {}), snapshot.active, len(snapshot.active)
        )
    return snapshot._pop_cache[key]


def retrieve(snapshot, user: str) -> tuple[list[tuple[str, float]], set[str]]:
    """Return Top-60 trend entries and their explicit global fallback IDs."""
    bucket = age_bucket(snapshot.sample.customers.get(user, {}).get("age"))
    primary = popularity(snapshot, bucket)[:60]
    primary_ids = {item for item, _ in primary}
    backfill = [
        pair for pair in popularity(snapshot, "GLOBAL") if pair[0] not in primary_ids
    ][: 60 - len(primary)]
    return primary + backfill, {item for item, _ in backfill}
