"""Expose bounded cutoff snapshots backed by complete legal SQL statistics.

Shared sales/activity/popularity tables are built separately from ItemCF so a
rule-channel cell cannot silently run an unrelated expensive index build. The
SQL adapter retains one user batch and immutable global aggregates. Histories
use t_dat<cutoff+1; future labels never enter this module. ItemCF neighbors are
loaded from their separately persisted read-only index when available."""

from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path
from time import perf_counter

from hm_recsys.core.contracts import read_json, write_json
from hm_recsys.core.progress import emit
from hm_recsys.data.history import History, allocation, connect, transaction_view
from hm_recsys.data.sample import Transaction
from hm_recsys.retrieval import itemcf, repurchase, sibling, trend


def build_statistics(run: Path, name: str, cutoff: date) -> None:
    """Build only shared past-only statistics; expensive ItemCF stays in R4."""
    folder = run / "snapshots"
    folder.mkdir(exist_ok=True)
    con = connect(folder / f"{name}.duckdb", run, readonly=False)
    before = cutoff + timedelta(days=1)
    emit(
        "cutoff_statistics",
        0,
        None,
        "rows",
        message=("Building shared legal cutoff statistics"),
        force=True,
    )
    try:
        path = str((run / "canonical/history.duckdb").resolve()).replace("'", "''")
        con.execute(f"ATTACH '{path}' AS canonical (READ_ONLY)")
        transaction_view(con, run, "tx")
        con.execute(
            (
                """CREATE TABLE sales AS SELECT article_id,
                count(*) FILTER (WHERE t_dat>=? - INTERVAL 3 DAY) AS d3,
                count(*) FILTER (WHERE t_dat>=? - INTERVAL 7 DAY) AS d7,
                count(*) FILTER (WHERE t_dat>=? - INTERVAL 14 DAY) AS d14,
                count(*) AS d28, avg(price) AS mean_price
                
                FROM tx
                WHERE t_dat>=? - INTERVAL 28 DAY AND t_dat<?
                GROUP BY article_id"""
            ),
            [before] * 5,
        )
        con.execute(
            """CREATE TABLE active AS SELECT article_id FROM sales SEMI JOIN
            canonical.candidate_catalog USING(article_id) WHERE d28>=3"""
        )
        con.execute(
            (
                """CREATE TABLE edges AS
                SELECT customer_id,article_id,max(t_dat) AS last_day,
                count(DISTINCT t_dat) AS days
                FROM tx
                WHERE t_dat>=? - INTERVAL 28 DAY
                AND t_dat<?
                GROUP BY customer_id,article_id
                ORDER BY customer_id,article_id"""
            ),
            [before, before],
        )
        con.execute(
            """CREATE TABLE item_degree AS SELECT article_id,count(*) AS
            n,avg(cast(days>=2 AS DOUBLE)) AS repurchase_ratio FROM edges
            GROUP BY
            article_id"""
        )
        con.execute(
            """CREATE TABLE user_degree AS SELECT customer_id,count(*) AS n FROM
            edges
            GROUP BY customer_id"""
        )
        con.execute(
            (
                """CREATE TABLE popularity AS WITH recent AS (
                SELECT t.article_id, exp(-date_diff('day',t.t_dat,?)/3.0) AS
                w,
                CASE WHEN try_cast(c.age AS DOUBLE) IS NULL OR NOT
                isfinite(try_cast(c.age AS DOUBLE)) OR try_cast(c.age AS
                DOUBLE)<=0 OR try_cast(c.age AS DOUBLE)>120 THEN 'UNKNOWN'
                WHEN try_cast(c.age AS DOUBLE)<25 THEN 'UNDER25'
                WHEN try_cast(c.age AS DOUBLE)<35 THEN '25TO34'
                WHEN try_cast(c.age AS DOUBLE)<=50 THEN '35TO50' ELSE
                'OVER50' END AS bucket
                
                FROM tx t JOIN canonical.customers c USING(customer_id)
                
                WHERE t_dat>=? - INTERVAL 7 DAY AND t_dat<?)
                SELECT bucket,article_id,sum(w) AS score,count(*) AS n
                FROM recent
                
                GROUP BY bucket,article_id
                UNION ALL SELECT 'GLOBAL',article_id,sum(w),count(*)
                FROM recent
                
                GROUP BY article_id"""
            ),
            [cutoff, before, before],
        )
        con.execute(
            """CREATE TABLE bucket_ranks AS SELECT bucket,article_id,
            row_number() OVER(PARTITION BY bucket
            ORDER BY n DESC,article_id) AS rank
            
            FROM popularity JOIN active USING(article_id)
            WHERE bucket!='GLOBAL'"""
        )
        counts = {
            table: con.execute(f"SELECT count(*) FROM {table}").fetchall()[0][0]
            for table in (("active"), ("edges"))
        }
        write_json(
            folder / f"{name}.json",
            {"cutoff": str(cutoff), "global_history": True, **counts},
        )
        emit(
            "cutoff_statistics",
            counts["edges"],
            counts["edges"],
            "unique_edges",
            status=("complete"),
            metrics=counts,
            force=True,
        )
    finally:
        con.close()


def build_snapshot(
    run: Path,
    name: str,
    cutoff: date,
    *,
    optimized: bool = False,
    include_itemcf: bool = True,
    profile: dict | None = None,
) -> None:
    """Build shared statistics and, only when explicitly requested, ItemCF."""
    build_statistics(run, name, cutoff)
    if include_itemcf:
        itemcf.build(run, name, cutoff, optimized=optimized, profile=profile)


class Snapshot:
    """Provide shared cutoff helpers and rule-channel delegation for SQL snapshots."""

    before: date
    cutoff: date
    by_user: dict[str, list[Transaction]]
    active: set[str]
    optimized: bool = True

    def window(self, rows: list[Transaction], days: int) -> list[Transaction]:
        """Return the exact inclusive-T trailing window from past-only rows."""
        lower = self.before - timedelta(days=days)
        return [row for row in rows if lower <= row.day < self.before]

    def record_compute(self, name: str, start: float) -> None:
        """Accumulate cutoff-stage host work without retaining per-query logs."""
        if not hasattr(self, "compute_seconds"):
            self.compute_seconds = {}
        self.compute_seconds[name] = (
            self.compute_seconds.get(name, 0.0) + perf_counter() - start
        )

    def repurchase(self, user):
        return repurchase.scores(self, user)

    def sibling(self, user):
        return sibling.scores(self, user)

    def popularity(self, bucket):
        return trend.popularity(self, bucket)

    def retrieve(self, user, wals, tower, *, learned=None):
        from hm_recsys.fusion.native import merge

        trend_values, fallback = trend.retrieve(self, user)
        if learned is None:
            if wals is None or tower is None:
                raise ValueError("Missing learned model adapters")
            learned = {
                "wals": wals.retrieve(
                    self.by_user.get(user, []), cutoff=self.cutoff, active=self.active
                ),
                ("twotower"): tower.retrieve(
                    user, before=self.before, active=self.active
                ),
            }
        outputs = {
            "trend": trend_values,
            "repurchase": repurchase.retrieve(self, user),
            ("sibling"): sibling.retrieve(self, user),
            ("itemcf"): itemcf.retrieve(self, user),
            **learned,
        }
        return merge(outputs, self.active, fallback), outputs


class GlobalSnapshot(Snapshot):
    """Shared rule retrieval with SQL aggregates and only one user-history batch."""

    def __init__(
        self,
        history: History,
        run: Path,
        name: str,
        cutoff: date,
        *,
        include_itemcf: bool = True,
    ) -> None:
        metadata = read_json(run / "snapshots" / f"{name}.json")
        if metadata["cutoff"] != str(cutoff):
            raise ValueError("Shared statistics cutoff mismatch")
        self.include_itemcf = include_itemcf
        self.history_store = history
        self.name = name
        self.neighbor_con = None
        self.cutoff = cutoff
        self.before = cutoff + timedelta(days=1)
        self.con = connect(run / "snapshots" / f"{name}.duckdb", run)
        count = self.con.execute("SELECT count(*) FROM popularity").fetchall()[0][0]
        allocation(
            count * 200 + len(history.catalog) * 300, ("cutoff aggregate dictionaries")
        )
        self.sales = {d: Counter() for d in (3, 7, 14, 28)}
        self.item_stats = {}
        for item, d3, d7, d14, d28, price, ratio in self.con.execute(
            """SELECT s.*,i.repurchase_ratio FROM sales s JOIN item_degree i
            USING(article_id)"""
        ).fetchall():
            for d, n in zip((3, 7, 14, 28), (d3, d7, d14, d28), strict=True):
                self.sales[d][item] = n
            self.item_stats[item] = (ratio, price)
        self.active = {
            r[0] for r in self.con.execute("SELECT article_id FROM active").fetchall()
        }
        self.pop = defaultdict(dict)
        for b, i, score in self.con.execute(
            "SELECT bucket,article_id,score FROM popularity"
        ).fetchall():
            self.pop[b][i] = score
        self.bucket_ranks = defaultdict(dict)
        for b, i, rank in self.con.execute("SELECT * FROM bucket_ranks").fetchall():
            self.bucket_ranks[b][i] = rank
        self.products = defaultdict(list)
        for i, meta in history.articles.items():
            if meta.get("product_code") is not None:
                self.products[str(meta["product_code"])].append(i)
        self._active_indexes: dict = {}
        self._active_index_key: tuple | None = None
        self.by_item = {}
        self.pair_estimate = 0
        self.set_users([])

    def set_users(self, users: list[str]) -> None:
        """Replace bounded user state while retaining the same global aggregates."""
        self._rep_cache = {}
        self._sibling_cache = {}
        self.sample = self.history_store.sample(users)
        self.by_user = self.history_store.histories(users, self.before) if users else {}
        self.edges = defaultdict(dict)
        seeds = set()
        for user, rows in self.by_user.items():
            for row in self.window(rows, 28):
                self.edges[user][row.article] = max(
                    row.day, self.edges[user].get(row.article, row.day)
                )
            seeds.update(
                (
                    i
                    for i, _ in sorted(
                        self.edges[user].items(),
                        key=lambda p: (-p[1].toordinal(), p[0]),
                    )[:15]
                )
            )
        self.itemcf = defaultdict(list)
        neighbor_path = (
            self.history_store.run / "indexes" / f"itemcf_{self.name}.duckdb"
        )
        if self.include_itemcf and seeds and neighbor_path.is_file():
            if self.neighbor_con is None:
                self.neighbor_con = connect(neighbor_path, self.history_store.run)
            import pyarrow as pa

            self.neighbor_con.register(
                "requested_seeds", pa.table({"seed": sorted(seeds)})
            )
            for seed, i, score in self.neighbor_con.execute(
                """SELECT n.* FROM neighbors n JOIN requested_seeds USING(seed)
                ORDER BY
                seed,score DESC,article_id"""
            ).fetchall():
                self.itemcf[seed].append((i, score))

    def item_statistics(self, item: str) -> tuple[float, float]:
        """Return global distinct-date repurchase ratio and 28-day mean price."""
        return self.item_stats.get(item, (float("nan"), float("nan")))

    def close(self) -> None:
        """Close the immutable cutoff database."""
        self.con.close()
        if self.neighbor_con is not None:
            self.neighbor_con.close()
