"""Read bounded native source/candidate batches with verified columnar contracts.

Readers validate the file hash once per partition and keep one small DuckDB
connection across query batches. They never materialize every user's candidates
or future labels. Nested candidate hits retain original double scores, one-based
ranks and explicit fallback flags. Complete external query rosters own empty
queries; an absent source list therefore remains an empty list rather than a
population filter. Derived files remain beneath the supplied run directory.
"""

import math
from datetime import date
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from hm_recsys.core.contracts import CHANNELS, QUOTAS, digest, read_json
from hm_recsys.data.history import connect
from hm_recsys.fusion.native import Candidate
from hm_recsys.retrieval.base import Hit

HIT_TYPE = pa.struct(
    [
        ("channel", pa.string()),
        ("raw_score", pa.float64()),
        ("source_rank", pa.int32()),
        (("trend_global_backfill"), pa.bool_()),
    ]
)
CANDIDATE_SCHEMA = pa.schema(
    [
        ("customer_id", pa.string()),
        ("cutoff_date", pa.date32()),
        (("article_id"), pa.string()),
        (("hits"), pa.list_(HIT_TYPE)),
        (("rrf"), pa.float64()),
    ]
)


class CandidateReader:
    """Own one hash-verified partition reader and bounded native-candidate lookups."""

    def __init__(self, run: Path, partition: str, *, reloaded: bool = False) -> None:
        folder = run / ("fusion_reload" if reloaded else "fusion") / partition
        self.meta = read_json(folder / "manifest.json")
        roster = read_json(run / "queries" / f"{partition}.json")
        count = roster["count"] if "count" in roster else len(roster["users"])
        if (
            self.meta["partition"] != partition
            or self.meta["cutoff"] != roster["cutoff"]
            or self.meta["queries"] != count
        ):
            raise ValueError("Native fusion roster/cutoff mismatch")
        self.path = folder / self.meta["file"]
        self.cutoff = date.fromisoformat(self.meta["cutoff"])
        if not self.path.is_file() or digest(self.path) != self.meta["sha256"]:
            raise ValueError(
                "Missing or corrupt native fusion file; run candidate union"
            )
        parquet = pq.ParquetFile(self.path)
        if (
            parquet.schema_arrow.remove_metadata() != CANDIDATE_SCHEMA
            or parquet.metadata.num_rows != self.meta["rows"]
        ):
            raise ValueError("Native fusion schema/row count mismatch")
        self.con = connect(Path(":memory:"), run, readonly=False)
        self.con.execute(
            "CREATE TEMP TABLE fused_candidates AS SELECT * FROM read_parquet(?)",
            [str(self.path)],
        )

    def read(self, users: list[str]) -> dict[str, list[Candidate]]:
        """Return complete native candidates in requested-query/article-ID order."""
        if len(users) > 8192 or len(users) != len(set(users)):
            raise ValueError("Candidate lookup requires at most 8192 unique queries")
        result: dict[str, list[Candidate]] = {user: [] for user in users}
        if not users:
            return result
        self.con.register("candidate_users", pa.table({"customer_id": users}))
        rel = self.con.execute(
            (
                """SELECT c.* FROM fused_candidates c JOIN candidate_users
                USING(customer_id) WHERE c.customer_id>=? AND c.customer_id<=?
            ORDER BY customer_id,article_id"""
            ),
            [min(users), max(users)],
        )
        to_arrow = getattr(rel, "to_arrow_table", None) or getattr(
            rel, "fetch_arrow_table"
        )
        rows = to_arrow().to_pylist()
        seen = set()
        for row in rows:
            user, item = row["customer_id"], row["article_id"]
            if row["cutoff_date"] != self.cutoff or (user, item) in seen:
                raise ValueError("Native candidate key/cutoff mismatch")
            seen.add((user, item))
            hits = {}
            for source in row["hits"]:
                channel = source["channel"]
                rank = source["source_rank"]
                score = source[("raw_score")]
                if (
                    channel not in CHANNELS
                    or channel in hits
                    or not 1 <= rank <= QUOTAS[channel]
                    or not math.isfinite(score)
                ):
                    raise ValueError("Invalid native source provenance")
                hits[channel] = Hit(score, rank, source["trend_global_backfill"])
            candidate = Candidate(item, hits)
            if not hits or not math.isclose(
                candidate.rrf, row["rrf"], rel_tol=1e-12, abs_tol=1e-15
            ):
                raise ValueError("Native RRF/source mismatch")
            result[user].append(candidate)
            if len(result[user]) > sum(QUOTAS.values()):
                raise ValueError("Native union exceeds quota sum")
        return result

    def close(self) -> None:
        """Release the owned in-memory reader at the partition boundary."""
        self.con.close()


def load_candidates(
    run: Path, partition: str, users: list[str], *, reloaded: bool = False
) -> dict[str, list[Candidate]]:
    """Read one bounded group; long-lived consumers should reuse CandidateReader."""
    reader = CandidateReader(run, partition, reloaded=reloaded)
    try:
        return reader.read(users)
    finally:
        reader.close()
