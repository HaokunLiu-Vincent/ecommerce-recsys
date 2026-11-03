"""Stream canonical targets, query rosters and feature ledgers in bounded batches.

Parquet row-group offsets preserve deterministic target ordering across resume.
TargetStream derives LogQ only from selected legal training events. Partition
reads independent query manifests and full truth, with optional label access
explicitly disabled for inference. Both smoke and full use these same readers;
no target, history or all-query feature table becomes an unbounded Python list.
"""

import json
from collections.abc import Iterator
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from hm_recsys.core.contracts import TT_END, TT_START, digest, read_json
from hm_recsys.data.history import connect


def export_query(con, sql: str, parameters: list, path: Path) -> None:
    """Materialize a deterministic SQL result with bounded Parquet row groups."""
    path.parent.mkdir(parents=True, exist_ok=True)
    target = str(path.resolve()).replace("'", "''")
    con.execute(
        f"COPY ({sql}) TO '{target}' (FORMAT PARQUET, COMPRESSION ZSTD, "
        "ROW_GROUP_SIZE 8192)",
        parameters,
    )


def rows(path: Path, size: int = 512, offset: int = 0) -> Iterator[list[dict]]:
    """Read bounded batches from a stable offset, skipping preceding row groups."""
    if size < 1 or offset < 0:
        raise ValueError("Invalid bounded row request")
    parquet = pq.ParquetFile(path)
    groups = []
    skipped = 0
    position = 0
    for j in range(parquet.num_row_groups):
        count = parquet.metadata.row_group(j).num_rows
        position += count
        if position <= offset:
            skipped = position
        else:
            groups.append(j)
    remaining = offset - skipped
    pending = []
    for batch in parquet.iter_batches(batch_size=size, row_groups=groups):
        if remaining:
            take = min(remaining, len(batch))
            batch = batch.slice(take)
            remaining -= take
        pending.extend(batch.to_pylist())
        while len(pending) >= size:
            yield pending[:size]
            pending = pending[size:]
    if pending:
        yield pending


class TargetStream:
    """Stable SHA256 event order, training-only LogQ and bounded resumable reads."""

    def __init__(self, history, limit: int | None) -> None:
        if limit is not None and limit < 1:
            raise ValueError("Target limit must be positive or null")
        self.path = history.run / "tmp/training_targets.parquet"
        sql = (
            "SELECT DISTINCT customer_id,t_dat,article_id\n            FROM "
            "transactions JOIN cohort USING(customer_id)\n            WHERE "
            "t_dat>=? AND t_dat<?\n            ORDER BY sha256('42:' || "
            "customer_id || ':' || cast(t_dat AS VARCHAR)\n                || "
            "':' || article_id), customer_id,t_dat,article_id"
        )
        parameters: list = [TT_START, TT_END]
        if limit is not None:
            sql += " LIMIT ?"
            parameters.append(limit)
        from hm_recsys.runtime.capacity import check

        count = history.con.execute(
            (
                "SELECT count(*) FROM (SELECT DISTINCT "
                "customer_id,t_dat,article_id FROM transactions JOIN cohort "
                "USING(customer_id) WHERE t_dat>=? AND t_dat<?)"
            ),
            [TT_START, TT_END],
        ).fetchall()[0][0]
        selected = count if limit is None else min(count, limit)
        check(
            history.run,
            "tower_targets",
            memory=64 * 1024**2,
            disk=selected * 512,
            details={
                "legal_targets": count,
                "selected_targets": selected,
                "limit": limit,
                "order": "SHA256(42:user:date:item), then user,date,item",
            },
        )
        export_query(history.con, sql, parameters, self.path)
        self.count = pq.ParquetFile(self.path).metadata.num_rows
        if self.count < 2:
            raise ValueError("Insufficient legal targets")
        self.probabilities = {
            i: n / self.count
            for i, n in history.con.execute(
                "SELECT article_id,count(*) FROM read_parquet(?) GROUP BY article_id",
                [str(self.path)],
            ).fetchall()
        }
        self.identity = digest(self.path)

    def batches(self, offset: int, size: int) -> Iterator[list[tuple]]:
        """Preserve the original batch boundaries after a complete-batch checkpoint."""
        for batch in rows(self.path, size, offset):
            yield [(r["customer_id"], r["t_dat"], r["article_id"]) for r in batch]


class Partition:
    """Independent complete query roster and separate bounded label lookups."""

    def __init__(self, run: Path, name: str, *, labels: bool = True) -> None:
        self.run, self.name = (run, name)
        self.meta = read_json(run / "queries" / f"{name}.json")
        self.disk = "file" in self.meta
        self.count = self.meta["count"] if self.disk else len(self.meta["users"])
        self.con = (
            connect(Path(":memory:"), run, readonly=False)
            if self.disk and labels
            else None
        )
        if self.con is not None:
            self.con.execute(
                "CREATE TEMP TABLE partition_labels AS SELECT customer_id,article_id FROM read_parquet(?)",
                [str(self.run / "labels" / f"{self.name}.parquet")],
            )
        self.labels = (
            read_json(run / "labels" / f"{name}.json")
            if labels and (not self.disk)
            else None
        )

    def batches(self, size: int = 32) -> Iterator[list[str]]:
        """Yield all eligible queries, including eventual retrieval failures."""
        if self.disk:
            for batch in rows(self.run / self.meta["file"], size):
                yield [r["customer_id"] for r in batch]
        else:
            users = self.meta["users"]
            for start in range(0, len(users), size):
                yield users[start : start + size]

    def truth(self, users: list[str]) -> dict[str, set[str]]:
        """Return full catalog truth for a bounded query batch."""
        if len(users) > 8192:
            raise ValueError("Label batch exceeds 8192 queries")
        if self.labels is not None:
            return {u: set(self.labels[u]) for u in users}
        if self.con is None:
            raise ValueError("Label access was disabled for inference")
        self.con.register("requested", pa.table({"customer_id": users}))
        result = {u: set() for u in users}
        for user, item in self.con.execute(
            "SELECT l.customer_id,l.article_id FROM partition_labels l JOIN requested USING(customer_id)"
        ).fetchall():
            result[user].add(item)
        if any(not truth for truth in result.values()):
            raise ValueError("Eligible query lost its complete ground truth")
        return result

    def close(self) -> None:
        """Close the optional label reader."""
        if self.con is not None:
            self.con.close()


class Shards:
    """Stream feature manifests without loading all query groups into Python."""

    def __init__(self, path: Path, count: int) -> None:
        self.path, self.count = (path, count)

    def __len__(self) -> int:
        return self.count

    def __iter__(self) -> Iterator[dict]:
        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                yield json.loads(line)


def load_manifest(path: Path) -> dict:
    """Read small metadata and bind a streaming shard index when present."""
    result = read_json(path)
    if "shards_file" in result:
        result["shards"] = Shards(
            path.parent / result["shards_file"], result["shard_count"]
        )
    return result
