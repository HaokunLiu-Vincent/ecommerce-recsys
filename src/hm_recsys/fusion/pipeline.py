"""Persist the complete six-source native union and nonlearned RRF diagnostics.

Inputs are six independently verified source files plus a complete query roster.
Fusion uses no labels to select candidates; truth is read afterward only for
macro union recall, signed leave-one-source-out recall and rule MAP@12. Nested
provenance preserves source ranks/scores and active-pool eligibility. Reload
writes separate files and skips label/metric access entirely.
"""

from datetime import date
from pathlib import Path
from time import perf_counter

import pyarrow as pa
import pyarrow.parquet as pq

from hm_recsys.core.contracts import CHANNELS, digest, read_json, write_json
from hm_recsys.core.progress import emit
from hm_recsys.data.history import connect
from hm_recsys.data.streaming import Partition
from hm_recsys.evaluation.metrics import metrics, recall_at_k
from hm_recsys.fusion.native import merge
from hm_recsys.fusion.storage import CANDIDATE_SCHEMA


def fuse(
    run: Path,
    profile: dict,
    partitions: list[str] | None = None,
    *,
    reloaded: bool = False,
) -> dict:
    """Merge six completed channels without fitting models or shrinking queries."""
    reports = {}
    for partition in partitions if partitions is not None else profile["folds"]:
        reader = Partition(run, partition, labels=not reloaded)
        cutoff = date.fromisoformat(reader.meta["cutoff"])
        source = run / ("channels_reload" if reloaded else "channels") / partition
        for channel in CHANNELS:
            meta = read_json(source / f"{channel}.json")
            path = source / f"{channel}.parquet"
            if (
                meta["queries"] != reader.count
                or meta["cutoff"] != str(cutoff)
                or digest(path) != meta[("sha256")]
            ):
                raise ValueError(
                    f"Missing or incompatible {channel} source; run its retrieval cell"
                )
        folder = run / ("fusion_reload" if reloaded else "fusion") / partition
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / "candidates.parquet"
        if path.exists():
            raise ValueError("Fusion output exists; reuse its verified completed stage")
        con = connect(Path(":memory:"), run, readonly=False)
        started = perf_counter()
        processed = rows = zero = 0
        union_recall = 0.0
        loo = {c: 0.0 for c in CHANNELS}
        baseline = {}
        # The active pool is independently persisted before any channel's Top-K.
        stats = run / "snapshots" / f"{partition}.duckdb"
        quoted = str(stats.resolve()).replace(("'"), ("''"))
        con.execute(f"ATTACH '{quoted}' AS snapshot (READ_ONLY)")
        active = {
            record[0]
            for record in con.execute(
                "SELECT article_id FROM snapshot.active"
            ).fetchall()
        }
        for channel in CHANNELS:
            con.execute(
                f"CREATE TEMP TABLE ch_{channel} AS SELECT * FROM read_parquet(?)",
                [str(source / f"{channel}.parquet")],
            )
        try:
            with pq.ParquetWriter(path, CANDIDATE_SCHEMA, compression="zstd") as writer:
                for batch in reader.batches(profile["feature_query_batch"]):
                    con.register("fusion_users", pa.table({"customer_id": batch}))
                    outputs = {
                        user: {channel: [] for channel in CHANNELS} for user in batch
                    }
                    fallback = {user: set() for user in batch}
                    for channel in CHANNELS:
                        rel = con.execute(
                            (
                                f"""SELECT c.* FROM ch_{channel} c JOIN
                                fusion_users USING(customer_id)
                                WHERE c.customer_id>=? AND c.customer_id<=?
                            ORDER BY customer_id,source_rank"""
                            ),
                            [
                                min(batch),
                                max(batch),
                            ],
                        )
                        to_arrow = getattr(rel, "to_arrow_table", None) or getattr(
                            rel, "fetch_arrow_table"
                        )
                        records = to_arrow().to_pylist()
                        for record in records:
                            user = record["customer_id"]
                            values = outputs[user][channel]
                            if (
                                record["cutoff_date"] != cutoff
                                or record["source_rank"] != len(values) + 1
                            ):
                                raise ValueError("Source rank/cutoff contract changed")
                            values.append((record["article_id"], record["raw_score"]))
                            if record["trend_global_backfill"]:
                                if channel != "trend":
                                    raise ValueError(
                                        "Only Trend can carry "
                                        "global fallback provenance"
                                    )
                                fallback[user].add(record["article_id"])
                    records = []
                    candidates = {}
                    predictions = {}
                    for user in batch:
                        candidates[user] = merge(outputs[user], active, fallback[user])
                        zero += not candidates[user]
                        predictions[user] = [
                            c.article
                            for c in sorted(
                                candidates[user], key=lambda c: (-c.rrf, c.article)
                            )[:12]
                        ]
                        for candidate in candidates[user]:
                            records.append(
                                {
                                    "customer_id": user,
                                    "cutoff_date": cutoff,
                                    ("article_id"): candidate.article,
                                    ("hits"): [
                                        {
                                            ("channel"): channel,
                                            ("raw_score"): hit.score,
                                            ("source_rank"): hit.rank,
                                            ("trend_global_backfill"): hit.fallback,
                                        }
                                        for channel, hit in candidate.hits.items()
                                    ],
                                    ("rrf"): candidate.rrf,
                                }
                            )
                    writer.write_table(
                        pa.Table.from_pylist(records, schema=CANDIDATE_SCHEMA)
                    )
                    if not reloaded:
                        truth = reader.truth(batch)
                        for user in batch:
                            full = recall_at_k(
                                [c.article for c in candidates[user]], truth[user]
                            )
                            union_recall += full
                            for channel in CHANNELS:
                                without = [
                                    c.article
                                    for c in candidates[user]
                                    if any(c2 != channel for c2 in c.hits)
                                ]
                                loo[channel] += full - recall_at_k(without, truth[user])
                        measured = metrics(predictions, truth)
                        for key, value in measured.items():
                            baseline[key] = baseline.get(key, 0) + (
                                value * len(batch) if "@" in key else value
                            )
                    processed += len(batch)
                    rows += len(records)
                    emit(
                        f"fusion_{partition}",
                        processed,
                        reader.count,
                        "queries",
                        metrics={
                            ("candidate_rows"): rows,
                            ("zero_candidate_queries"): zero,
                        },
                    )
            if processed != reader.count:
                raise ValueError("Fusion lost eligible queries")
            for key in baseline:
                if "@" in key:
                    baseline[key] /= processed
            report = {
                "partition": partition,
                "cutoff": str(cutoff),
                "queries": processed,
                ("rows"): rows,
                ("zero_candidate_queries"): zero,
                ("union_recall"): union_recall / processed if not reloaded else None,
                ("signed_leave_one_out_recall"): {
                    c: value / processed for c, value in loo.items()
                }
                if not reloaded
                else {},
                ("rrf_baseline"): baseline,
                ("seconds"): perf_counter() - started,
                ("file"): path.name,
                ("sha256"): digest(path),
                ("labels_read"): not reloaded,
                ("schema"): ("native-six-source-union-v1"),
            }
            write_json(folder / "manifest.json", report)
            reports[partition] = report
            emit(
                f"fusion_{partition}",
                processed,
                reader.count,
                "queries",
                status=("complete"),
                metrics={("candidate_rows"): rows},
                force=True,
            )
        finally:
            con.close()
            reader.close()
    return reports
