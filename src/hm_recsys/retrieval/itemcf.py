"""Build R4 sparse asymmetric neighborhoods and retrieve from recent seed items.

Shared cutoff edges contain one user/item edge with its latest purchase date;
all original raw purchase lines remain upstream. The exact asymmetric formula,
active-before-Top-K policy and top-100 neighbors are unchanged. A separate index
DuckDB file avoids mutating completed shared statistics. Pair work is estimated
before enumeration and explicit resource rejection never truncates degrees.
"""

import math
from collections import defaultdict
from datetime import date
from pathlib import Path

from hm_recsys.core.contracts import read_json, write_json
from hm_recsys.core.progress import emit
from hm_recsys.data.history import connect
from hm_recsys.retrieval.base import top
from hm_recsys.runtime.files import require_space


def build(
    run: Path,
    name: str,
    cutoff: date,
    *,
    optimized: bool = False,
    profile: dict | None = None,
) -> None:
    """Build one immutable ItemCF index using existing legal cutoff statistics."""
    source = run / "snapshots" / f"{name}.duckdb"
    if not source.is_file():
        raise ValueError(
            f"Missing shared cutoff statistics for {name}; run data preparation"
        )
    metadata = read_json(run / "snapshots" / f"{name}.json")
    if metadata["cutoff"] != str(cutoff):
        raise ValueError("ItemCF source cutoff mismatch")
    output = run / "indexes" / f"itemcf_{name}.duckdb"
    if output.exists():
        raise ValueError("ItemCF index exists; reuse its verified completed stage")
    output.parent.mkdir(exist_ok=True, parents=True)
    con = connect(output, run, readonly=False)
    try:
        quoted = str(source.resolve()).replace("'", "''")
        con.execute(f"ATTACH '{quoted}' AS snapshot (READ_ONLY)")
        for table in ("active", "edges", "item_degree", "user_degree"):
            con.execute(f"CREATE TEMP VIEW {table} AS SELECT * FROM snapshot.{table}")
        pairs = int(
            con.execute("SELECT coalesce(sum(n*(n-1)),0) FROM user_degree").fetchall()[
                0
            ][0]
        )
        configured = profile or (
            read_json(run / "profile.resolved.json")
            if (run / ("profile.resolved.json")).is_file()
            else {}
        )
        limit = configured.get("compute", {}).get("max_itemcf_pairs", 1_000_000_000)
        if pairs > limit:
            raise RuntimeError(
                f"blocked_resource: exact ItemCF exceeds {limit} directed pairs; "
                "no truncation applied"
            )
        require_space(run, pairs * 32 + 10 * 1024**3)
        con.execute(
            """CREATE TABLE neighbors(seed VARCHAR, article_id VARCHAR, score
            DOUBLE)"""
        )
        seeds = [
            row[0]
            for row in con.execute(
                "SELECT article_id FROM item_degree ORDER BY article_id"
            ).fetchall()
        ]
        if optimized:
            con.execute(
                """CREATE TABLE seed_edges AS SELECT * FROM edges ORDER BY
                article_id,customer_id"""
            )
            con.execute("CREATE INDEX seed_edge_lookup ON seed_edges(article_id)")
            decay = con.execute(
                "SELECT list(exp(-i/7.0) ORDER BY i) FROM range(28) t(i)"
            ).fetchone()
            if decay is None:
                raise ValueError("Missing ItemCF decay lookup")
            con.execute(
                """CREATE TEMP TABLE decay_values AS SELECT ?::DOUBLE[] AS values""",
                [decay[0]],
            )
        sql = """INSERT INTO neighbors WITH scores AS (
              SELECT a.article_id AS seed,b.article_id,
              sum(exp(-abs(date_diff('day',a.last_day,b.last_day))/7.0)/ln(1+ud.n))
              / sqrt(ni.n * pow(nj.n,0.75)) AS score
              FROM edges a JOIN edges b ON a.customer_id=b.customer_id
              AND a.article_id!=b.article_id
              JOIN active act ON b.article_id=act.article_id
              JOIN user_degree ud ON a.customer_id=ud.customer_id
              JOIN item_degree ni ON a.article_id=ni.article_id
              JOIN item_degree nj ON b.article_id=nj.article_id
              WHERE a.article_id>=? AND a.article_id<=?
              GROUP BY a.article_id,b.article_id,ni.n,nj.n)
              SELECT seed,article_id,score FROM scores
              QUALIFY row_number() OVER(PARTITION BY seed
              ORDER BY score DESC,article_id)<=100"""
        if optimized:
            sql = sql.replace("FROM edges a", "FROM seed_edges a").replace(
                ("exp(-abs(date_diff('day',a.last_day,b.last_day))/7.0)"),
                (
                    "(SELECT values FROM decay_values)[abs(date_d"
                    "iff('day',a.last_day,b.last_day))+1]"
                ),
            )
        emit(
            "itemcf_seeds",
            0,
            len(seeds),
            "seed_items",
            metrics={("estimated_directed_pairs"): pairs},
            force=True,
        )
        for start in range(0, len(seeds), 128):
            batch = seeds[start : start + 128]
            con.execute(sql, [batch[0], batch[-1]])
            emit("itemcf_seeds", start + len(batch), len(seeds), "seed_items")
        if optimized:
            con.execute("CREATE INDEX neighbor_seed ON neighbors(seed)")
            con.execute("DROP TABLE seed_edges")
        counts = {
            table: con.execute(f"SELECT count(*) FROM {table}").fetchall()[0][0]
            for table in (("active"), ("edges"), ("neighbors"))
        }
        write_json(
            run / "indexes" / f"itemcf_{name}.json",
            {
                "cutoff": str(cutoff),
                ("global_history"): True,
                ("directed_pairs"): pairs,
                **counts,
            },
        )
        emit(
            "itemcf_seeds",
            len(seeds),
            len(seeds),
            "seed_items",
            status="complete",
            metrics=counts,
            force=True,
        )
    finally:
        con.close()


def retrieve(snapshot, user: str) -> list[tuple[str, float]]:
    """Score active items from fifteen latest distinct seeds with ten-day decay."""
    scores: dict[str, float] = defaultdict(float)
    seeds = sorted(
        snapshot.edges.get(user, {}).items(),
        key=lambda pair: (-pair[1].toordinal(), pair[0]),
    )[:15]
    for seed, day in seeds:
        for item, value in snapshot.itemcf.get(seed, []):
            scores[item] += math.exp(-(snapshot.cutoff - day).days / 10) * value
    return top(scores, snapshot.active, 50)
