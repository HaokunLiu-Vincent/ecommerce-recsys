"""Bounded native candidates and shared feature shards with independent rosters."""

from datetime import date
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from hm_recsys.contracts import CHANNELS, QUOTAS, digest, read_json, write_json
from hm_recsys.evaluation import metrics, union_recall
from hm_recsys.features import FeatureBuilder
from hm_recsys.learned import WALS, TowerRetriever, confidence, fold_user
from hm_recsys.pilot.data import History, MappedContent
from hm_recsys.pilot.snapshot import GlobalSnapshot
from hm_recsys.pilot.vectors import BlockIndex, block_topk
from hm_recsys.pipeline import feature_table


class PilotTowerRetriever(TowerRetriever):
    """Load immutable full-catalog vectors without the smoke index constructor."""

    def __init__(self, run: Path, *, reloaded: bool = False) -> None:
        folder = run / ("predictions" if reloaded else "queries")
        name = "reloaded_query_vectors" if reloaded else "query_vectors"
        state = read_json(folder / f"{name}.json")
        if state["model_sha256"] != digest(run / "models/twotower.pt"):
            raise ValueError("Query-vector model mismatch")
        self.mapping = {
            (u, date.fromisoformat(d)): j for j, (u, d) in enumerate(state["queries"])
        }
        self.vectors = np.load(
            folder / f"{name}.npy", mmap_mode="r", allow_pickle=False
        )
        self.has_history = state["has_history"]
        self.index = BlockIndex.load(run / "indexes", "twotower")


def learned_batch(
    snapshot: GlobalSnapshot, users: list[str], wals: WALS, towers: PilotTowerRetriever
) -> dict:
    """Solve each user once, then score bounded query/item blocks together."""
    observed = [
        confidence(
            snapshot.by_user.get(u, []), cutoff=snapshot.cutoff, vocabulary=wals.mapping
        )
        for u in users
    ]
    factors = np.stack(
        [fold_user(wals.index.vectors, obs, gram=wals.gram) for obs in observed]
    )
    wr = block_topk(wals.index.ids, wals.index.vectors, factors, snapshot.active, 50)
    rows = [towers.mapping[(u, snapshot.before)] for u in users]
    tr = block_topk(
        towers.index.ids,
        towers.index.vectors,
        towers.vectors[rows],
        snapshot.active,
        50,
    )
    return {
        u: {
            "wals": wr[j] if observed[j] else [],
            "twotower": tr[j] if towers.has_history[rows[j]] else [],
        }
        for j, u in enumerate(users)
    }


def generate(run: Path, partition: str, profile: dict, *, reload: bool = False) -> None:
    """Write at most 32 complete query groups per shard; never inject positives."""
    roster = read_json(run / "queries" / f"{partition}.json")
    users = roster["users"]
    cutoff = date.fromisoformat(roster["cutoff"])
    history = History(run)
    snapshot = GlobalSnapshot(history, run, partition, cutoff)
    content = MappedContent(run)
    builder = FeatureBuilder(content, run)
    wals = WALS(BlockIndex.load(run / "indexes", "wals"))
    tower = PilotTowerRetriever(run, reloaded=reload)
    labels = {} if reload else read_json(run / "labels" / f"{partition}.json")
    folder = run / (
        "predictions/reloaded_features" if reload else f"features/{partition}"
    )
    folder.mkdir(parents=True, exist_ok=False)
    audit = run / "candidates" / partition
    if not reload:
        audit.mkdir(parents=True, exist_ok=False)
    shards = []
    summary = {
        "queries": len(users),
        "feature_rows": 0,
        "active_items": len(snapshot.active),
        "zero_candidate_queries": 0,
        "zero_retrieved_positive_queries": 0,
        "union_recall": 0.0,
        "active_pool_coverage": 0.0,
        "out_of_pool_positive_count": 0,
        "channel_rows": {c: 0 for c in CHANNELS},
        "channel_recall": {
            c: {str(k): 0.0 for k in sorted({12, QUOTAS[c]})} for c in CHANNELS
        },
    }
    try:
        for start in range(0, len(users), profile["feature_query_batch"]):
            batch = users[start : start + profile["feature_query_batch"]]
            snapshot.set_users(batch)
            learned = learned_batch(snapshot, batch, wals, tower)
            row_users, ids, blocks, provenance, groups = [], [], [], [], []
            union = {}
            channels = {c: {} for c in CHANNELS}
            for u in batch:
                candidates, hits = snapshot.retrieve(u, wals, tower, learned=learned[u])
                blocks.append(builder.build(snapshot, u, candidates))
                item_ids = [c.article for c in candidates]
                row_users.extend([u] * len(candidates))
                ids.extend(item_ids)
                groups.append({"user": u, "rows": len(candidates)})
                union[u] = item_ids
                for channel in CHANNELS:
                    channels[channel][u] = [i for i, _ in hits[channel]]
                for candidate in candidates:
                    for channel, hit in candidate.hits.items():
                        provenance.append(
                            {
                                "customer_id": u,
                                "article_id": candidate.article,
                                "cutoff_date": str(cutoff),
                                "channel": channel,
                                "raw_score": hit.score,
                                "source_rank": hit.rank,
                                "trend_global_backfill": hit.fallback,
                            }
                        )
            table = feature_table(row_users, ids, blocks)
            name = f"part-{start // profile['feature_query_batch']:05d}.parquet"
            pq.write_table(table, folder / name, row_group_size=max(1, len(table)))
            shards.append(
                {
                    "file": str((folder / name).relative_to(run)),
                    "rows": len(table),
                    "groups": groups,
                    "sha256": digest(folder / name),
                }
            )
            if not reload:
                if provenance:
                    pq.write_table(pa.Table.from_pylist(provenance), audit / name)
                truth = {u: set(labels[u]) for u in batch}
                summary["feature_rows"] += len(table)
                summary["union_recall"] += (
                    union_recall(union, truth) * len(batch) / len(users)
                )
                summary["zero_candidate_queries"] += sum(not union[u] for u in batch)
                summary["zero_retrieved_positive_queries"] += sum(
                    not (set(union[u]) & truth[u]) for u in batch
                )
                summary["active_pool_coverage"] += sum(
                    len(snapshot.active & truth[u]) / len(truth[u]) for u in batch
                ) / len(users)
                summary["out_of_pool_positive_count"] += sum(
                    len(truth[u] - snapshot.active) for u in batch
                )
                for c in CHANNELS:
                    summary["channel_rows"][c] += sum(
                        len(v) for v in channels[c].values()
                    )
                    for k in summary["channel_recall"][c]:
                        summary["channel_recall"][c][k] += (
                            metrics(channels[c], truth, int(k))[f"Recall@{k}"]
                            * len(batch)
                            / len(users)
                        )
        write_json(
            folder / "manifest.json",
            {
                "partition": partition,
                "cutoff": str(cutoff),
                "shards": shards,
                "rows": sum(s["rows"] for s in shards),
                "queries": len(users),
                "labels_read": not reload,
                "schema_sha256": digest(run / "feature_schema.json"),
            },
        )
        if not reload:
            path = run / "candidate_results.json"
            report: dict = read_json(path) if path.exists() else {"partitions": {}}
            report["partitions"][partition] = summary
            report["total_feature_rows"] = sum(
                v["feature_rows"] for v in report["partitions"].values()
            )
            write_json(path, report)
            write_json(
                audit / "pool.json",
                {"articles": sorted(snapshot.active), "cutoff": str(cutoff)},
            )
    finally:
        snapshot.close()
        history.close()
