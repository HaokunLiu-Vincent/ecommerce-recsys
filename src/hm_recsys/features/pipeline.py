"""Build shared ranking features from persisted native candidate unions.

Each call reads one independent query roster and its already fused candidates.
It never trains a retriever, regenerates a channel, reads future labels, or injects
positives. Cutoff-safe SQL snapshots and frozen content/preprocessing produce
bounded float32 feature blocks. Whole-query Parquet shards and a streaming ledger
preserve empty queries, schema identity and fresh-process reload alignment.
"""

import json
from contextlib import ExitStack
from datetime import date
from pathlib import Path
from time import perf_counter

from hm_recsys.core.contracts import CHANNELS, digest, read_json, write_json
from hm_recsys.core.progress import emit
from hm_recsys.data.history import History
from hm_recsys.data.streaming import Partition
from hm_recsys.features.builder import FeatureBuilder
from hm_recsys.features.content import MappedContent
from hm_recsys.features.storage import FeatureShards, feature_table
from hm_recsys.fusion.storage import CandidateReader
from hm_recsys.retrieval.snapshot import GlobalSnapshot


def generate(
    run: Path, profile: dict, partition: str, *, reloaded: bool = False
) -> None:
    """Write one partition's features using the production path in every mode.

    Args:
        run: Owned stage view containing canonical history and frozen inputs.
        profile: Recorded execution configuration; batches must fit 32 queries.
        partition: F1–F4, Validation, or explicitly authorized Test roster name.
        reloaded: Consume freshly rebuilt candidates for reload parity checks.

    Raises:
        ValueError: If fusion/roster/schema identity or row accounting differs.
    """
    started = perf_counter()
    batch_size = profile["feature_query_batch"]
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("Feature query batch size must be a positive integer")
    with ExitStack() as resources:
        fusion_root = run / ("fusion_reload" if reloaded else "fusion") / partition
        candidate_reader = CandidateReader(run, partition, reloaded=reloaded)
        resources.callback(candidate_reader.close)
        fused = candidate_reader.meta
        roster = Partition(run, partition, labels=False)
        resources.callback(roster.close)
        cutoff = date.fromisoformat(roster.meta["cutoff"])
        if fused.get("queries") != roster.count:
            raise ValueError("Fused candidate query population changed")
        if fused.get("cutoff") != str(cutoff):
            raise ValueError("Fused candidate cutoff changed")
        history = History(run)
        resources.callback(history.close)
        snapshot = GlobalSnapshot(history, run, partition, cutoff, include_itemcf=False)
        resources.callback(snapshot.close)
        builder = FeatureBuilder(MappedContent(run), run)
        folder = run / (
            "predictions/reloaded_features" if reloaded else f"features/{partition}"
        )
        folder.mkdir(parents=True, exist_ok=False)
        writer = FeatureShards(run, folder, None)
        resources.callback(writer.close)
        total_rows = processed = shard_count = 0
        feature_seconds = 0.0
        emit(
            "feature_queries",
            0,
            roster.count,
            "queries",
            status="started",
            message="Reading frozen native unions and building shared features",
            force=True,
        )
        with (folder / "shards.jsonl").open("x", encoding="utf-8") as ledger:
            for users in roster.batches(batch_size):
                candidates = candidate_reader.read(users)
                snapshot.set_users(users)
                row_users, article_ids, blocks, groups = ([], [], [], [])
                feature_started = perf_counter()
                for user in users:
                    native = candidates[user]
                    blocks.append(builder.build(snapshot, user, native))
                    article_ids.extend(candidate.article for candidate in native)
                    row_users.extend([user] * len(native))
                    groups.append({"user": user, "rows": len(native)})
                feature_seconds += perf_counter() - feature_started
                table = feature_table(row_users, article_ids, blocks)
                entry = writer.append(table, groups, None)
                if entry is not None:
                    ledger.write(json.dumps(entry) + "\n")
                    shard_count += 1
                total_rows += len(table)
                processed += len(users)
                emit(
                    "feature_queries",
                    processed,
                    roster.count,
                    "queries",
                    metrics={
                        "feature_rows": total_rows,
                        "feature_columns": len(builder.schema["names"]),
                        "closed_feature_shards": shard_count,
                    },
                )
            entry = writer.finish()
            if entry is not None:
                ledger.write(json.dumps(entry) + "\n")
                shard_count += 1
        if processed != roster.count or total_rows != fused["rows"]:
            raise ValueError(
                "Feature rows or eligible queries differ from native fusion"
            )
        write_json(
            folder / "manifest.json",
            {
                "partition": partition,
                "cutoff": str(cutoff),
                "shards_file": "shards.jsonl",
                "shard_count": shard_count,
                "rows": total_rows,
                "queries": processed,
                "labels_read": False,
                "schema_sha256": digest(run / "feature_schema.json"),
                "fusion_sha256": digest(fusion_root / "manifest.json"),
            },
        )
        write_json(
            run / "feature_performance" / f"{partition}_{reloaded}.json",
            {
                "queries": processed,
                "candidate_rows": total_rows,
                "feature_seconds": feature_seconds,
                "total_seconds": perf_counter() - started,
                "backend": "columnar",
                "retrieval_recomputed": False,
            },
        )
        if not reloaded and partition != "Test":
            channels = {
                channel: read_json(run / "channels" / partition / f"{channel}.json")
                for channel in CHANNELS
            }
            summary_path = run / "candidate_results.json"
            summary: dict = (
                read_json(summary_path)
                if summary_path.exists()
                else {
                    "partitions": {},
                    "sample": read_json(run / "population_manifest.json"),
                }
            )
            summary["partitions"][partition] = {
                "queries": processed,
                "feature_rows": total_rows,
                "union_recall": fused["union_recall"],
                "zero_candidate_queries": fused["zero_candidate_queries"],
                "channel_rows": {c: channels[c]["rows"] for c in CHANNELS},
                "channel_recall": {c: channels[c]["recall"] for c in CHANNELS},
                "rrf_baseline": fused["rrf_baseline"],
                "signed_leave_one_out_recall": fused["signed_leave_one_out_recall"],
            }
            summary["total_feature_rows"] = sum(
                value["feature_rows"] for value in summary["partitions"].values()
            )
            write_json(summary_path, summary)
        emit(
            "feature_queries",
            processed,
            roster.count,
            "queries",
            status="completed",
            metrics={
                "feature_rows": total_rows,
                "closed_feature_shards": shard_count,
                "feature_columns": len(builder.schema["names"]),
            },
            force=True,
        )
