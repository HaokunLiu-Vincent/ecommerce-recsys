"""Disk-backed global history, independent training cohorts and query rosters."""

from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

import duckdb
import numpy as np
import psutil
import pyarrow as pa

from hm_recsys.colab.files import require_space
from hm_recsys.content import ContentBank, ContentSource
from hm_recsys.contracts import TT_END, TT_START, split_manifest, write_json
from hm_recsys.data import Sample, Transaction


def allocation(required: int, purpose: str) -> None:
    """Reject an estimated allocation with headroom; never shrink its population."""
    available = psutil.virtual_memory().available
    if required < 0 or required * 1.25 > available - 2 * 1024**3:
        raise RuntimeError(
            f"blocked_resource: {purpose} needs {required} bytes plus reserve"
        )


def connect(path: Path, run: Path, *, readonly: bool = True):
    """Open bounded DuckDB with local spill space; caller closes the connection."""
    (run / "tmp").mkdir(exist_ok=True, parents=True)
    con = duckdb.connect(str(path), read_only=readonly)
    con.execute("SET threads=2")
    con.execute("SET memory_limit='1GB'")
    con.execute("SET max_temp_directory_size='8GiB'")
    con.execute("SET temp_directory=?", [str(run / "tmp" / path.stem)])
    con.execute("SET preserve_insertion_order=false")
    return con


def transaction_view(con, run: Path, name: str = "transactions") -> None:
    """Bind portable monthly Parquet partitions to a temporary SQL view."""
    if name not in ("transactions", "tx"):
        raise ValueError("Unsupported history relation name")
    folder = run / "canonical/transactions"
    if not list(folder.glob("month=*/*.parquet")):
        raise ValueError("Canonical monthly partitions are missing")
    path = str(folder.resolve() / "month=*/*.parquet").replace("'", "''")
    con.execute(
        f"CREATE TEMP VIEW {name} AS SELECT customer_id,article_id,t_dat,price,"
        f"sales_channel_id FROM read_parquet('{path}', hive_partitioning=false)"
    )


def canonical(run: Path, source: Path, profile: dict) -> None:
    """Project the complete catalog-matched history to disk, excluding final Test.

    No transaction rows become Python objects here. Canonical row groups are
    sorted by customer/date for selective query history access. Duplicate lines
    remain distinct. The cohort controls gradients only, never global statistics.
    """
    raw = source / "original data"
    require_space(
        run, (raw / "transactions_train.csv").stat().st_size * 5 + 10 * 1024**3
    )
    folder = run / "canonical"
    folder.mkdir()
    con = connect(folder / "history.duckdb", run, readonly=False)
    try:
        con.execute(
            "CREATE TABLE articles AS SELECT * FROM read_csv(?, all_varchar=true)",
            [str(raw / "articles.csv")],
        )
        con.execute(
            "CREATE TABLE customers AS SELECT * FROM read_csv(?, all_varchar=true)",
            [str(raw / "customers.csv")],
        )
        for table, key in (("articles", "article_id"), ("customers", "customer_id")):
            if con.execute(
                f"SELECT count(*)-count(DISTINCT {key}) FROM {table}"
            ).fetchall()[0][0]:
                raise ValueError(f"Duplicate {table} identity")
        if con.execute(
            "SELECT count(*) FROM articles WHERE NOT "
            "regexp_full_match(article_id, '[0-9]{10}')"
        ).fetchall()[0][0]:
            raise ValueError("Noncanonical article ID")
        con.execute(
            """CREATE TABLE transactions AS
            SELECT t.customer_id, t.article_id, cast(t.t_dat AS DATE) AS t_dat,
                   cast(t.price AS DOUBLE) AS price,
                   cast(t.sales_channel_id AS INTEGER) AS sales_channel_id
           
            FROM read_csv(?, all_varchar=true) t
           
            WHERE cast(t.t_dat AS DATE) < DATE '2020-09-16'
           
            ORDER BY customer_id, t_dat, article_id, price, sales_channel_id""",
            [str(raw / "transactions_train.csv")],
        )
        excluded = con.execute("""SELECT count(*) FROM transactions t
            ANTI JOIN articles a USING(article_id)""").fetchall()[0][0]
        con.execute("""DELETE FROM transactions WHERE article_id IS NULL
            OR article_id NOT IN (SELECT article_id FROM articles)""")
        if con.execute("""SELECT count(*)
            FROM transactions t LEFT JOIN customers c
                          USING(customer_id)
            WHERE c.customer_id IS NULL OR
            price IS NULL OR NOT isfinite(price) OR t_dat IS NULL
            OR sales_channel_id IS NULL""").fetchall()[0][0]:
            raise ValueError("Invalid transaction/customer join or price")
        con.execute(
            """CREATE TABLE cohort AS SELECT customer_id
            FROM transactions
                      
            WHERE t_dat < ?
            GROUP BY customer_id
                      
            ORDER BY sha256('42:' || customer_id), customer_id LIMIT ?""",
            [TT_END, profile["training_subset"]["base_users"]],
        )
        counts = {
            name: con.execute(f"SELECT count(*) FROM {name}").fetchall()[0][0]
            for name in ("transactions", "articles", "customers", "cohort")
        }
        write_json(
            run / "population_manifest.json",
            {
                "profile": "pilot",
                "global_history_rows": counts["transactions"],
                "catalog_unmatched_history_rows": excluded,
                "catalog_items": counts["articles"],
                "global_customers": counts["customers"],
                "base_users": counts["cohort"],
                "base_selection": profile["training_subset"],
                "statistics_scope": "all catalog-matched past transactions, not cohort",
                "history_end_exclusive": "2020-09-16",
                "final_test_read": False,
                "static_metadata_snapshot": True,
            },
        )
        write_json(
            run / "cohort_manifest.json",
            {
                "users": [
                    r[0]
                    for r in con.execute(
                        "SELECT customer_id FROM cohort ORDER BY customer_id"
                    ).fetchall()
                ],
                "cutoff_exclusive": str(TT_END),
            },
        )
        partition_root = str((folder / "transactions").resolve()).replace("'", "''")
        con.execute(
            "COPY (SELECT *, strftime(t_dat, '%Y-%m') AS month FROM transactions) "
            f"TO '{partition_root}' (FORMAT PARQUET, PARTITION_BY (month), "
            "COMPRESSION ZSTD, ROW_GROUP_SIZE 122880)"
        )
        con.execute("DROP TABLE transactions")
        con.execute("CHECKPOINT")
    finally:
        con.close()


class History:
    """Query bounded histories and metadata from immutable canonical storage."""

    def __init__(self, run: Path) -> None:
        self.run = run
        self.con = connect(run / "canonical/history.duckdb", run)
        transaction_view(self.con, run)
        articles = (
            self.con.execute("SELECT * FROM articles ORDER BY article_id")
            .to_arrow_table()
            .to_pylist()
        )
        self.articles = {row["article_id"]: row for row in articles}
        self.catalog = list(self.articles)

    def close(self) -> None:
        """Release the read-only database before another isolated stage starts."""
        self.con.close()

    def customers(self, users: list[str]) -> dict:
        self.con.register("requested_users", pa.table({"customer_id": users}))
        rows = (
            self.con.execute(
                "SELECT c.* FROM customers c JOIN requested_users USING(customer_id)"
            )
            .to_arrow_table()
            .to_pylist()
        )
        return {
            r["customer_id"]: {**r, "age": float(r["age"]) if r.get("age") else None}
            for r in rows
        }

    def sample(self, users: list[str]) -> Sample:
        """Return static metadata only; this never contains global transactions."""
        return Sample([], self.articles, self.customers(users), self.catalog, {})

    def histories(self, users: list[str], before: date) -> dict[str, list[Transaction]]:
        """Fetch one feature batch; reject large batches before Python allocation."""
        if len(users) > 32:
            raise ValueError("History query batch exceeds 32 users")
        self.con.register("history_users", pa.table({"customer_id": users}))
        sql = """FROM transactions t JOIN history_users USING(customer_id)
                 WHERE t_dat < ?"""
        count = self.con.execute("SELECT count(*) " + sql, [before]).fetchall()[0][0]
        if count > 100000:
            raise RuntimeError(
                "blocked_resource: feature batch exceeds 100000 history lines"
            )
        rows = self.con.execute(
            "SELECT t.* "
            + sql
            + " ORDER BY customer_id, t_dat, article_id, price, sales_channel_id",
            [before],
        ).fetchall()
        result: dict[str, list[Transaction]] = defaultdict(list)
        for u, i, d, p, c in rows:
            result[u].append(Transaction(u, i, d, p, c))
        return result

    def event_histories(
        self, queries: list[tuple[str, date]]
    ) -> list[list[Transaction]]:
        """Latest 20 raw lines per event, excluding all same-day observations."""
        if len(queries) > 512:
            raise ValueError("Tower query batch exceeds 512")
        self.con.register(
            "event_queries",
            pa.table(
                {
                    "q": list(range(len(queries))),
                    "customer_id": [u for u, _ in queries],
                    "before": [d for _, d in queries],
                }
            ),
        )
        rows = self.con.execute("""SELECT q, t.*
            FROM transactions t JOIN event_queries q
          ON t.customer_id=q.customer_id AND t.t_dat<q.before
         
            WHERE t.t_dat>=DATE '2020-05-19'
          QUALIFY row_number() OVER(PARTITION BY q
            ORDER BY t_dat DESC,
            article_id DESC,
            price DESC, sales_channel_id DESC)<=20
         
            ORDER BY q, t_dat DESC, article_id DESC,
            price DESC, sales_channel_id DESC""").fetchall()
        result: list[list[Transaction]] = [[] for _ in queries]
        for q, u, i, d, p, c in rows:
            result[q].append(Transaction(u, i, d, p, c))
        return result

    def targets(self, limit: int) -> list[tuple[str, date, str]]:
        """Stable unique cohort events and their original LogQ sampling population."""
        return self.con.execute(
            """SELECT DISTINCT customer_id, t_dat, article_id
         
            FROM transactions JOIN cohort USING(customer_id)
            WHERE t_dat>=? AND t_dat<?
         
            ORDER BY sha256('42:' || customer_id || ':' || cast(t_dat AS
            VARCHAR) || ':' || article_id), customer_id, t_dat, article_id
            LIMIT ?""",
            [TT_START, TT_END, limit],
        ).fetchall()

    def positives(self, queries: list[tuple[str, date]]) -> dict:
        """Complete same-day positives, including events outside the target cap."""
        self.con.register(
            "positive_queries",
            pa.table(
                {"customer_id": [u for u, _ in queries], "day": [d for _, d in queries]}
            ),
        )
        rows = self.con.execute("""SELECT DISTINCT t.customer_id, t.t_dat, article_id
           
            FROM transactions t JOIN positive_queries q
            ON t.customer_id=q.customer_id AND t.t_dat=q.day""").fetchall()
        result: dict[tuple[str, date], set[str]] = defaultdict(set)
        for u, d, i in rows:
            result[u, d].add(i)
        return result

    def active(self, before: date) -> set[str]:
        """Use complete global legal history, never the gradient-training cohort."""
        return {
            r[0]
            for r in self.con.execute(
                """SELECT article_id
            FROM transactions
         
            WHERE t_dat>=? AND t_dat<?
            GROUP BY article_id HAVING count(*)>=3""",
                [before - timedelta(days=28), before],
            ).fetchall()
        }


def rosters(run: Path, profile: dict) -> None:
    """Select query rosters independently of recall and preserve complete truth."""
    history = History(run)
    try:
        for part in split_manifest():
            if part["partition"] not in profile["folds"]:
                continue
            start, end = (
                date.fromisoformat(part["start"]),
                date.fromisoformat(part["end"]),
            )
            users = [
                r[0]
                for r in history.con.execute(
                    """SELECT customer_id
            FROM transactions
               
            WHERE t_dat>=? AND t_dat<?
            GROUP BY customer_id
               
            ORDER BY sha256('42:' || ? || ':' || customer_id), customer_id LIMIT ?""",
                    [
                        start,
                        end,
                        str(start),
                        profile["compute"]["max_queries_per_fold"],
                    ],
                ).fetchall()
            ]
            if not users:
                raise ValueError("Empty pilot query partition")
            history.con.register("label_users", pa.table({"customer_id": users}))
            labels: dict[str, list[str]] = {u: [] for u in users}
            for u, i in history.con.execute(
                """SELECT DISTINCT customer_id,article_id
               
            FROM transactions JOIN label_users USING(customer_id) WHERE
            t_dat>=? AND t_dat<?
            ORDER BY customer_id,article_id""",
                [start, end],
            ).fetchall():
                labels[u].append(i)
            write_json(
                run / "queries" / f"{part['partition']}.json",
                {
                    **part,
                    "users": sorted(users),
                    "selection": (
                        "purchase-active then SHA256(42:start:user), no hit filtering"
                    ),
                },
            )
            write_json(run / "labels" / f"{part['partition']}.json", labels)
    finally:
        history.close()


def adapt_content(run: Path, source: Path) -> None:
    """Normalize derived disk arrays in batches; supplied embeddings stay frozen."""
    history = History(run)
    ids = history.catalog
    history.close()
    folder = run / "content"
    folder.mkdir()
    write_json(folder / "mapping.json", {"article_ids": ids})
    reports = {}
    for modality in ("text", "image"):
        src = ContentSource(source, modality, set(ids))
        shape = (len(ids), src.array.shape[1])
        require_space(run, np.prod(shape).item() * 4 * 3 + 10 * 1024**3)
        vectors = np.lib.format.open_memmap(
            folder / f"{modality}.npy", mode="w+", dtype=np.float32, shape=shape
        )
        masks = np.zeros(len(ids), dtype=bool)
        for start in range(0, len(ids), 256):
            vectors[start : start + 256], masks[start : start + 256] = src.fetch(
                ids[start : start + 256]
            )
        vectors.flush()
        del vectors
        np.save(folder / f"{modality}_mask.npy", masks, allow_pickle=False)
        reports[modality] = src.report
    write_json(folder / "acceptance.json", reports)


class MappedContent(ContentBank):
    """Read-only disk adapters with the same FeatureBuilder/TowerEncoder API."""

    def __init__(self, run: Path) -> None:
        from hm_recsys.contracts import read_json

        folder = run / "content"
        self.ids = read_json(folder / "mapping.json")["article_ids"]
        self.position = {item: j for j, item in enumerate(self.ids)}
        self.vectors = {
            m: np.load(folder / f"{m}.npy", mmap_mode="r", allow_pickle=False)
            for m in ("text", "image")
        }
        self.masks = {
            m: np.load(folder / f"{m}_mask.npy", allow_pickle=False)
            for m in ("text", "image")
        }
        self.reports = read_json(folder / "acceptance.json")
