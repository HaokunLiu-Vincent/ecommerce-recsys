"""Generate one explicitly selected native retrieval channel for saved rosters.

Each invocation reads bounded cutoff histories and only its own frozen learned
model, never silently fits prerequisites. Source ranks/scores/fallback flags are
persisted as one ordered Parquet per channel/fold. Independent complete truth is
used only after retrieval for macro diagnostics. Reload writes a separate tree;
all source inputs and completed prior stages remain immutable.
"""

import math
from datetime import date
from pathlib import Path
from time import perf_counter

import pyarrow as pa
import pyarrow.parquet as pq

from hm_recsys.core.contracts import CHANNELS, QUOTAS, digest, write_json
from hm_recsys.core.progress import emit
from hm_recsys.data.history import History
from hm_recsys.data.streaming import Partition
from hm_recsys.evaluation.metrics import recall_at_k
from hm_recsys.retrieval import itemcf, repurchase, sibling, trend
from hm_recsys.retrieval.learned import TowerRetriever, tower_batch, wals_batch
from hm_recsys.retrieval.snapshot import GlobalSnapshot
from hm_recsys.retrieval.vectors import BlockIndex
from hm_recsys.retrieval.wals.model import WALS

SOURCE_SCHEMA = pa.schema(
    [
        ("customer_id", pa.string()),
        ("cutoff_date", pa.date32()),
        ("article_id", pa.string()),
        ("source_rank", pa.int32()),
        ("raw_score", pa.float64()),
        ("trend_global_backfill", pa.bool_()),
    ]
)


def run_channel(
    run: Path,
    profile: dict,
    channel: str,
    partitions: list[str] | None = None,
    *,
    reloaded: bool = False,
) -> dict:
    """Retrieve and validate one channel without running unrelated model stages."""
    if channel not in CHANNELS:
        raise ValueError("Unknown retrieval channel")
    selected = partitions if partitions is not None else profile["folds"]
    model = (
        WALS(BlockIndex.load(run / "indexes", "wals")) if channel == "wals" else None
    )
    tower = (
        TowerRetriever(
            run, reloaded=reloaded, scope="final_test" if selected == ["Test"] else None
        )
        if channel == "twotower"
        else None
    )
    reports = {}
    for partition in selected:
        if (
            channel == "itemcf"
            and not (run / "indexes" / f"itemcf_{partition}.duckdb").is_file()
        ):
            raise ValueError("Missing ItemCF index; run the ItemCF build cell")
        folder = run / ("channels_reload" if reloaded else "channels") / partition
        path = folder / f"{channel}.parquet"
        if path.exists():
            raise ValueError(
                "Channel output exists; reuse its verified completed stage"
            )
        reader = Partition(run, partition, labels=not reloaded)
        history = History(run)
        cutoff = date.fromisoformat(reader.meta["cutoff"])
        snapshot = GlobalSnapshot(
            history, run, partition, cutoff, include_itemcf=channel == "itemcf"
        )
        snapshot.optimized = profile.get("performance", {}).get("backend") == (
            "optimized"
        )
        folder = run / ("channels_reload" if reloaded else "channels") / partition
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{channel}.parquet"
        if path.exists():
            raise ValueError(
                "Channel output exists; reuse its verified completed stage"
            )
        if (
            channel == "itemcf"
            and not (run / "indexes" / f"itemcf_{partition}.duckdb").is_file()
        ):
            raise ValueError("Missing ItemCF index; run the ItemCF build cell")
        if tower is not None:
            layout = tower.layout[partition]
            if layout["roster_sha256"] != digest(run / "queries" / f"{partition}.json"):
                raise ValueError("Encoded query roster changed")
        started = perf_counter()
        processed = rows = nonempty = 0
        recalls = {str(k): 0.0 for k in sorted({12, QUOTAS[channel]})}
        try:
            with pq.ParquetWriter(path, SOURCE_SCHEMA, compression="zstd") as writer:
                for batch in reader.batches(profile["feature_query_batch"]):
                    snapshot.set_users(batch)
                    if model is not None:
                        values = wals_batch(snapshot, batch, model)
                    elif tower is not None:
                        tower.mapping = {
                            (user, snapshot.before): tower.layout[partition][("offset")]
                            + processed
                            + j
                            for j, user in enumerate(batch)
                        }
                        values = tower_batch(snapshot, batch, tower)
                    else:
                        values = {}
                    records = []
                    fallback = {}
                    for user in batch:
                        if channel == "trend":
                            values[user], fallback[user] = trend.retrieve(
                                snapshot, user
                            )
                        elif channel in ("repurchase", "sibling", "itemcf"):
                            adapter = {
                                "repurchase": repurchase,
                                "sibling": sibling,
                                ("itemcf"): itemcf,
                            }[channel]
                            values[user] = adapter.retrieve(snapshot, user)
                        result = values[user]
                        if len(result) > QUOTAS[channel] or len(
                            set(item for item, _ in result)
                        ) != len(result):
                            raise ValueError("Channel quota or uniqueness violation")
                        nonempty += bool(result)
                        for rank, (item, score) in enumerate(result, 1):
                            if item not in snapshot.active or not math.isfinite(score):
                                raise ValueError("Invalid channel eligibility or score")
                            records.append(
                                {
                                    "customer_id": user,
                                    "cutoff_date": cutoff,
                                    ("article_id"): item,
                                    ("source_rank"): rank,
                                    ("raw_score"): score,
                                    ("trend_global_backfill"): item
                                    in fallback.get(user, set()),
                                }
                            )
                    writer.write_table(
                        pa.Table.from_pylist(records, schema=SOURCE_SCHEMA)
                    )
                    if not reloaded:
                        truth = reader.truth(batch)
                        for user in batch:
                            for k in recalls:
                                recalls[k] += recall_at_k(
                                    [item for item, _ in values[user]],
                                    truth[user],
                                    int(k),
                                )
                    processed += len(batch)
                    rows += len(records)
                    emit(
                        f"{channel}_{partition}",
                        processed,
                        reader.count,
                        "queries",
                        metrics={
                            ("candidate_rows"): rows,
                            ("nonempty_queries"): nonempty,
                        },
                    )
            if processed != reader.count:
                raise ValueError("Channel query roster was truncated")
            report = {
                "channel": channel,
                "partition": partition,
                "cutoff": str(cutoff),
                ("queries"): processed,
                ("rows"): rows,
                ("nonempty_queries"): nonempty,
                ("recall"): {k: v / processed for k, v in recalls.items()}
                if not reloaded
                else {},
                ("seconds"): perf_counter() - started,
                ("file"): path.name,
                ("sha256"): digest(path),
                ("labels_read"): not reloaded,
                ("quota"): QUOTAS[channel],
            }
            write_json(folder / f"{channel}.json", report)
            reports[partition] = report
            emit(
                f"{channel}_{partition}",
                processed,
                reader.count,
                "queries",
                status=("complete"),
                metrics=report[("recall")],
                force=True,
            )
        finally:
            snapshot.close()
            history.close()
            reader.close()
    return reports
