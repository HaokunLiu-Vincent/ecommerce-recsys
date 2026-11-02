"""Exact disk-backed global cutoff aggregates and ItemCF neighborhoods."""

from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path

from hm_recsys.contracts import write_json
from hm_recsys.pilot.data import History, allocation, connect, transaction_view
from hm_recsys.recall import Snapshot


def build_snapshot(run: Path, name: str, cutoff: date) -> None:
    """Aggregate all legal users; stream exact ItemCF seed blocks through DuckDB.

    The pair estimate is a resource rejection gate, never degree truncation.
    Each seed retains the original asymmetric formula and top-100 active items.
    """
    folder = run / "snapshots"
    folder.mkdir(exist_ok=True)
    con = connect(folder / f"{name}.duckdb", run, readonly=False)
    before = cutoff + timedelta(days=1)
    try:
        path = str((run / "canonical/history.duckdb").resolve()).replace("'", "''")
        con.execute(f"ATTACH '{path}' AS canonical (READ_ONLY)")
        transaction_view(con, run, "tx")
        con.execute(
            """CREATE TABLE sales AS SELECT article_id,
            count(*) FILTER (WHERE t_dat>=? - INTERVAL 3 DAY) AS d3,
            count(*) FILTER (WHERE t_dat>=? - INTERVAL 7 DAY) AS d7,
            count(*) FILTER (WHERE t_dat>=? - INTERVAL 14 DAY) AS d14,
            count(*) AS d28, avg(price) AS mean_price
           
            FROM tx
            WHERE t_dat>=? - INTERVAL 28 DAY AND t_dat<?
            GROUP BY article_id""",
            [before] * 5,
        )
        con.execute("CREATE TABLE active AS SELECT article_id FROM sales WHERE d28>=3")
        con.execute(
            """CREATE TABLE edges AS
            SELECT customer_id,article_id,max(t_dat) AS last_day,
            count(DISTINCT t_dat) AS days
            FROM tx
            WHERE t_dat>=? - INTERVAL 28 DAY
            AND t_dat<?
            GROUP BY customer_id,article_id
            ORDER BY customer_id,article_id""",
            [before, before],
        )
        con.execute(
            "CREATE TABLE item_degree AS SELECT article_id,count(*) AS "
            "n,avg(cast(days>=2 AS DOUBLE)) AS repurchase_ratio FROM "
            "edges GROUP BY article_id"
        )
        con.execute(
            "CREATE TABLE user_degree AS SELECT customer_id,count(*) AS n"
            " FROM edges GROUP BY customer_id"
        )
        pairs = int(
            con.execute("SELECT coalesce(sum(n*(n-1)),0) FROM user_degree").fetchall()[
                0
            ][0]
        )
        if pairs > 1_000_000_000:
            raise RuntimeError(
                "blocked_resource: exact ItemCF exceeds 1 billion directed "
                "pairs; no truncation applied"
            )
        from hm_recsys.colab.files import require_space

        require_space(run, pairs * 32 + 10 * 1024**3)
        con.execute(
            """CREATE TABLE popularity AS WITH recent AS (
            SELECT t.article_id, exp(-date_diff('day',t.t_dat,?)/3.0) AS w,
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
           
            GROUP BY article_id""",
            [cutoff, before, before],
        )
        con.execute("""CREATE TABLE bucket_ranks AS SELECT bucket,article_id,
            row_number() OVER(PARTITION BY bucket
            ORDER BY n DESC,article_id) AS rank
           
            FROM popularity JOIN active USING(article_id)
            WHERE bucket!='GLOBAL'""")
        con.execute(
            "CREATE TABLE neighbors(seed VARCHAR, article_id VARCHAR, score DOUBLE)"
        )
        seeds = [
            r[0]
            for r in con.execute(
                "SELECT article_id FROM item_degree ORDER BY article_id"
            ).fetchall()
        ]
        for start in range(0, len(seeds), 128):
            batch = seeds[start : start + 128]
            con.execute(
                """INSERT INTO neighbors
                WITH scores AS (
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
                SELECT seed,article_id,score
            FROM scores
                QUALIFY row_number() OVER(PARTITION BY seed
            ORDER BY score
            DESC,article_id)<=100""",
                [batch[0], batch[-1]],
            )
        counts = {
            t: con.execute(f"SELECT count(*) FROM {t}").fetchall()[0][0]
            for t in ("active", "edges", "neighbors")
        }
        write_json(
            folder / f"{name}.json",
            {
                "cutoff": str(cutoff),
                "global_history": True,
                "directed_pairs": pairs,
                **counts,
            },
        )
    finally:
        con.close()


class GlobalSnapshot(Snapshot):
    """Shared rule retrieval with SQL aggregates and only one user-history batch."""

    def __init__(self, history: History, run: Path, name: str, cutoff: date) -> None:
        self.history_store = history
        self.cutoff = cutoff
        self.before = cutoff + timedelta(days=1)
        self.con = connect(run / "snapshots" / f"{name}.duckdb", run)
        count = self.con.execute("SELECT count(*) FROM popularity").fetchall()[0][0]
        allocation(
            count * 200 + len(history.catalog) * 300, "cutoff aggregate dictionaries"
        )
        self.sales = {d: Counter() for d in (3, 7, 14, 28)}
        self.item_stats = {}
        for item, d3, d7, d14, d28, price, ratio in self.con.execute(
            "SELECT s.*,i.repurchase_ratio FROM sales s JOIN item_degree "
            "i USING(article_id)"
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
        self.by_item = {}
        self.pair_estimate = 0
        self.set_users([])

    def set_users(self, users: list[str]) -> None:
        """Replace bounded user state while retaining the same global aggregates."""
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
                i
                for i, _ in sorted(
                    self.edges[user].items(), key=lambda p: (-p[1].toordinal(), p[0])
                )[:15]
            )
        self.itemcf = defaultdict(list)
        if seeds:
            import pyarrow as pa

            self.con.register("requested_seeds", pa.table({"seed": sorted(seeds)}))
            for seed, i, score in self.con.execute(
                "SELECT n.* FROM neighbors n JOIN requested_seeds USING(seed)"
                " ORDER BY seed,score DESC,article_id"
            ).fetchall():
                self.itemcf[seed].append((i, score))

    def item_statistics(self, item: str) -> tuple[float, float]:
        """Return global distinct-date repurchase ratio and 28-day mean price."""
        return self.item_stats.get(item, (float("nan"), float("nan")))

    def close(self) -> None:
        """Close the immutable cutoff database."""
        self.con.close()
