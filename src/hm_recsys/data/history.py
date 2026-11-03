"""Build canonical purchase storage and bounded, cutoff-safe query histories.

CSV inputs and content artifacts are immutable. Full mode retains complete legal
history; smoke chooses a recorded deterministic cohort and candidate catalog.
Derived monthly Parquet, independent rosters and full unique labels live under
one run. History converts only bounded query batches to Python purchase records.
Final Test labels require a separate explicit authorization call.
"""

from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

import duckdb
import psutil
import pyarrow as pa

from hm_recsys.core.contracts import (
    BASE_END,
    TT_END,
    TT_START,
    split_manifest,
    write_json,
)
from hm_recsys.core.progress import emit
from hm_recsys.data.sample import Sample, Transaction
from hm_recsys.runtime.files import require_space


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
    con.execute("SET threads=12")
    con.execute("SET memory_limit='64GB'")
    con.execute("SET max_temp_directory_size='64GiB'")
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
    """Write canonical SQL/Parquet data using explicit profile population limits.

    Full keeps complete pre-Test history. Smoke selects the original deterministic
    historical cohort and candidate catalog before materializing purchase lines.
    Raw repeats and all catalog-matched future positives of selected users remain;
    candidate eligibility is stored separately from the metadata/label universe.
    """
    from hm_recsys.data.streaming import export_query

    raw = source / "original data"
    compute = profile["compute"]
    smoke = profile["profile"] == "smoke"
    disk_estimate = (
        compute["max_transactions"] * 1024
        if smoke
        else (raw / "transactions_train.csv").stat().st_size * 5
    )
    require_space(run, disk_estimate + 10 * 1024**3)
    folder = run / "canonical"
    folder.mkdir()
    con = connect(folder / "history.duckdb", run, readonly=False)
    emit(
        "canonical",
        0,
        None,
        "retained_rows",
        status="started",
        message="Selecting canonical purchases; scanned-row count unknown",
        force=True,
    )
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
        con.read_csv(str(raw / "transactions_train.csv"), all_varchar=True).create_view(
            "raw_transactions"
        )
        con.execute(
            "CREATE TEMP VIEW source_transactions AS SELECT customer_id,"
            "\n            article_id, cast(t_dat AS DATE) AS t_dat, "
            "cast(price AS DOUBLE) AS price,\n            "
            "cast(sales_channel_id AS INTEGER) AS sales_channel_id"
            "\n            FROM raw_transactions WHERE cast(t_dat AS "
            "DATE)<DATE '2020-09-16'"
        )
        selection = "SELECT * FROM source_transactions"
        if smoke:
            con.execute(
                (
                    "CREATE TABLE smoke_users AS SELECT customer_id\n                "
                    "FROM source_transactions WHERE t_dat>=DATE '2020-02-01'"
                    "\n                AND t_dat<DATE '2020-08-05' GROUP BY "
                    "customer_id\n                HAVING count(DISTINCT t_dat)>=20"
                    "\n                ORDER BY sha256('42:' || "
                    "customer_id),customer_id LIMIT ?"
                ),
                [compute["max_users"]],
            )
            selection += (
                " SEMI JOIN smoke_users USING(customer_id) WHERE t_dat>=DATE "
                "'2020-02-01'"
            )
            count = con.execute("SELECT count(*) FROM (" + selection + ")").fetchall()[
                0
            ][0]
            if not 0 < count <= compute["max_transactions"]:
                raise ValueError(
                    "Complete selected cohort exceeds transaction cap or is empty"
                )
        con.execute(
            "CREATE TABLE transactions AS "
            + selection
            + " ORDER BY customer_id,t_dat,article_id,price,sales_channel_id"
        )
        excluded = con.execute(
            "SELECT count(*) FROM transactions ANTI JOIN articles USING(article_id)"
        ).fetchall()[0][0]
        if smoke and excluded:
            raise ValueError("Catalog join failures in selected transactions")
        con.execute(
            "DELETE FROM transactions WHERE article_id IS NULL OR article_id "
            "NOT IN (SELECT article_id FROM articles)"
        )
        if con.execute(
            "SELECT count(*) FROM transactions t\n            LEFT JOIN "
            "customers c USING(customer_id) WHERE c.customer_id IS NULL"
            "\n            OR price IS NULL OR NOT isfinite(price) OR t_dat IS"
            " NULL\n            OR sales_channel_id IS NULL"
        ).fetchall()[0][0]:
            raise ValueError("Invalid transaction/customer join or price")
        if smoke:
            con.execute(
                (
                    "CREATE TABLE candidate_catalog AS SELECT article_id"
                    "\n                FROM transactions WHERE t_dat<DATE '2020-08-05'"
                    "\n                GROUP BY article_id ORDER BY count(*) "
                    "DESC,article_id LIMIT ?"
                ),
                [compute["max_catalog"]],
            )
            remaining = (
                compute["max_catalog"]
                - con.execute("SELECT count(*) FROM candidate_catalog").fetchall()[0][0]
            )
            if remaining:
                con.execute(
                    (
                        "INSERT INTO candidate_catalog SELECT DISTINCT a.article_id"
                        "\n                    FROM articles a JOIN (SELECT DISTINCT "
                        "product_code FROM articles\n                    SEMI JOIN "
                        "candidate_catalog USING(article_id)) p USING(product_code)"
                        "\n                    ANTI JOIN candidate_catalog c ON "
                        "a.article_id=c.article_id\n                    ORDER BY "
                        "a.article_id LIMIT ?"
                    ),
                    [remaining],
                )
            con.execute(
                "DELETE FROM articles WHERE article_id NOT IN\n                "
                "(SELECT article_id FROM transactions UNION SELECT article_id "
                "FROM candidate_catalog)"
            )
            con.execute(
                "DELETE FROM customers WHERE customer_id NOT IN (SELECT "
                "customer_id FROM smoke_users)"
            )
            con.execute(
                "CREATE TABLE cohort AS SELECT customer_id FROM smoke_users ORDER"
                " BY customer_id"
            )
            subset = {
                "base_users": compute["max_users"],
                "selection": "pre-Aug-5 >=20 dates; SHA256(42:ID)",
            }
            boundary = TT_END
        else:
            con.execute(
                "CREATE TABLE candidate_catalog AS SELECT article_id FROM "
                "articles ORDER BY article_id"
            )
            subset = profile.get("training_subset")
            boundary = TT_END if subset is not None else BASE_END
            sql = (
                "CREATE TABLE cohort AS SELECT customer_id FROM transactions "
                "WHERE t_dat<? GROUP BY customer_id ORDER BY sha256('42:' || "
                "customer_id),customer_id"
            )
            parameters = [boundary]
            if subset is not None:
                sql += " LIMIT ?"
                parameters.append(subset["base_users"])
            con.execute(sql, parameters)
        counts = {
            name: con.execute(f"SELECT count(*) FROM {name}").fetchall()[0][0]
            for name in (
                "transactions",
                "articles",
                "customers",
                "cohort",
                "candidate_catalog",
            )
        }
        scope = (
            "complete selected-customer purchase lines"
            if smoke
            else "all catalog-matched past transactions, not cohort"
        )
        write_json(
            run / "population_manifest.json",
            {
                "profile": profile["profile"],
                "global_history_rows": counts["transactions"],
                "catalog_unmatched_history_rows": excluded,
                "catalog_items": counts["candidate_catalog"],
                "metadata_items": counts["articles"],
                "global_customers": counts["customers"],
                "base_users": counts["cohort"],
                "base_selection": subset,
                "statistics_scope": scope,
                "history_end_exclusive": "2020-09-16",
                "final_test_read": False,
                "static_metadata_snapshot": True,
            },
        )
        export_query(
            con,
            "SELECT customer_id FROM cohort ORDER BY customer_id",
            [],
            folder / "cohort.parquet",
        )
        write_json(
            run / "cohort_manifest.json",
            {
                "file": "canonical/cohort.parquet",
                "count": counts["cohort"],
                "cutoff_exclusive": str(boundary),
                "subset": subset,
            },
        )
        target = str((folder / "transactions").resolve()).replace("'", "''")
        emit(
            "canonical_partitions", 0, None, "partitions", status="started", force=True
        )
        con.execute(
            f"COPY (SELECT *,strftime(t_dat,'%Y-%m') AS month FROM transactions) "
            f"TO '{target}' (FORMAT PARQUET,PARTITION_BY(month),"
            "COMPRESSION ZSTD,ROW_GROUP_SIZE 122880)"
        )
        partitions = len(list((folder / "transactions").glob("month=*")))
        con.execute("DROP TABLE transactions")
        con.execute("CHECKPOINT")
        emit(
            "canonical",
            counts["transactions"],
            None,
            "retained_rows",
            status="completed",
            metrics={
                "catalog_items": counts["candidate_catalog"],
                "partitions_written": partitions,
            },
            force=True,
        )
    finally:
        con.close()


def _to_arrow_table(rel) -> pa.Table:
    """Convert a DuckDB relation/connection result to PyArrow across DuckDB versions."""
    fn = getattr(rel, "to_arrow_table", None) or getattr(rel, "fetch_arrow_table")
    return fn()


class History:
    """Query bounded histories and metadata from immutable canonical storage."""

    def __init__(self, run: Path) -> None:
        self.run = run
        self.con = connect(run / "canonical/history.duckdb", run)
        transaction_view(self.con, run)
        articles = _to_arrow_table(
            self.con.execute("SELECT * FROM articles ORDER BY article_id")
        ).to_pylist()
        self.articles = {row["article_id"]: row for row in articles}
        self.candidate_catalog = [
            r[0]
            for r in self.con.execute(
                "SELECT article_id FROM candidate_catalog ORDER BY article_id"
            ).fetchall()
        ]
        # A candidate cap must not make legal training targets/history unindexable.
        # Model/input rows cover this selected metadata scope; active candidates
        # are independently restricted by candidate_catalog at each cutoff.
        self.catalog = sorted(self.articles)

    def close(self) -> None:
        """Release the read-only database before another isolated stage starts."""
        self.con.close()

    def customers(self, users: list[str]) -> dict:
        self.con.register("requested_users", pa.table({"customer_id": users}))
        rows = _to_arrow_table(
            self.con.execute(
                "SELECT c.* FROM customers c JOIN requested_users USING(customer_id)"
            )
        ).to_pylist()
        return {
            r["customer_id"]: {**r, "age": float(r["age"]) if r.get("age") else None}
            for r in rows
        }

    def sample(self, users: list[str]) -> Sample:
        """Return static metadata only; this never contains global transactions."""
        return Sample([], self.articles, self.customers(users), self.catalog, {})

    def histories(self, users: list[str], before: date) -> dict[str, list[Transaction]]:
        """Fetch one feature batch; reject large batches before Python allocation."""
        if len(users) > 8192:
            raise ValueError("History query batch exceeds 8192 users")
        self.con.register("history_users", pa.table({"customer_id": users}))
        sql = (
            "FROM transactions t JOIN history_users USING(customer_id)"
            "\n                 WHERE t_dat < ? AND t.customer_id>=? AND "
            "t.customer_id<=?"
        )
        count = self.con.execute(
            "SELECT count(*) " + sql,
            [before, min(users, default=""), max(users, default="")],
        ).fetchall()[0][0]
        if count > 10000000:
            raise RuntimeError(
                "blocked_resource: feature batch exceeds 10000000 history lines"
            )
        rows = self.con.execute(
            "SELECT t.* "
            + sql
            + " ORDER BY customer_id, t_dat, article_id, price, sales_channel_id",
            [before, min(users, default=""), max(users, default="")],
        ).fetchall()
        result: dict[str, list[Transaction]] = defaultdict(list)
        for u, i, d, p, c in rows:
            result[u].append(Transaction(u, i, d, p, c))
        return result

    def event_histories(
        self, queries: list[tuple[str, date]]
    ) -> list[list[Transaction]]:
        """Latest 20 raw lines per event, excluding all same-day observations."""
        if len(queries) > 8192:
            raise ValueError("Tower query batch exceeds 8192")
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
        rows = self.con.execute(
            "SELECT q, t.*\n            FROM transactions t JOIN event_queries"
            " q\n          ON t.customer_id=q.customer_id AND t.t_dat<q.before"
            "\n         \n            WHERE t.t_dat>=DATE '2020-05-19'"
            "\n          QUALIFY row_number() OVER(PARTITION BY q\n            "
            "ORDER BY t_dat DESC,\n            article_id DESC,\n            "
            "price DESC, sales_channel_id DESC)<=20\n         \n            "
            "ORDER BY q, t_dat DESC, article_id DESC,\n            price DESC,"
            " sales_channel_id DESC"
        ).fetchall()
        result: list[list[Transaction]] = [[] for _ in queries]
        for q, u, i, d, p, c in rows:
            result[q].append(Transaction(u, i, d, p, c))
        return result

    def targets(self, limit: int) -> list[tuple[str, date, str]]:
        """Stable unique cohort events and their original LogQ sampling population."""
        if limit is None or not 0 < limit <= 20000:
            raise ValueError("Use TargetStream for uncapped or larger target reads")
        return self.con.execute(
            (
                "SELECT DISTINCT customer_id, t_dat, article_id"
                "\n         \n            FROM transactions JOIN cohort "
                "USING(customer_id)\n            WHERE t_dat>=? AND t_dat<?"
                "\n         \n            ORDER BY sha256('42:' || customer_id || "
                "':' || cast(t_dat AS\n            VARCHAR) || ':' || article_id),"
                " customer_id, t_dat, article_id\n            LIMIT ?"
            ),
            [TT_START, TT_END, limit],
        ).fetchall()

    def positives(self, queries: list[tuple[str, date]]) -> dict:
        """Complete same-day positives, including events outside the target cap."""
        if len(queries) > 8192:
            raise ValueError("Positive-set query batch exceeds 8192 events")
        self.con.register(
            "positive_queries",
            pa.table(
                {"customer_id": [u for u, _ in queries], "day": [d for _, d in queries]}
            ),
        )
        rows = self.con.execute(
            "SELECT DISTINCT t.customer_id, t.t_dat, article_id"
            "\n           \n            FROM transactions t JOIN "
            "positive_queries q\n            ON t.customer_id=q.customer_id "
            "AND t.t_dat=q.day"
        ).fetchall()
        result: dict[tuple[str, date], set[str]] = defaultdict(set)
        for u, d, i in rows:
            result[u, d].add(i)
        return result

    def active(self, before: date) -> set[str]:
        """Use complete global legal history, never the gradient-training cohort."""
        return {
            r[0]
            for r in self.con.execute(
                (
                    "SELECT article_id FROM transactions SEMI JOIN candidate_catalog "
                    "USING(article_id) WHERE t_dat>=? AND t_dat<? GROUP BY article_id"
                    " HAVING count(*)>=3"
                ),
                [before - timedelta(days=28), before],
            ).fetchall()
        }


def rosters(run: Path, profile: dict) -> None:
    """Export complete truth for an optionally capped independent query roster.

    Caps choose queries by the existing seeded hash, never by retrieval success.
    Full null limits select all eligible queries. Both modes use the same SQL and
    bounded Parquet interfaces. Test is prepared only by its separately gated API.
    """
    from hm_recsys.data.streaming import export_query

    history = History(run)
    try:
        cap = profile["compute"]["max_queries_per_fold"]
        if profile["profile"] == "full" and cap is not None:
            raise ValueError("Full rosters cannot be sampled")
        for part in split_manifest():
            name = part["partition"]
            if name == "Test" or name not in profile["folds"]:
                continue
            emit(f"roster_{name}", 0, None, "queries", status="started", force=True)
            history.con.execute(
                (
                    "CREATE OR REPLACE TEMP TABLE fold_labels AS\n                "
                    "SELECT DISTINCT customer_id,article_id FROM transactions"
                    "\n                WHERE t_dat>=? AND t_dat<?"
                ),
                [part["start"], part["end"]],
            )
            if cap is not None:
                history.con.execute(
                    (
                        "CREATE OR REPLACE TEMP TABLE fold_selected AS"
                        "\n                    SELECT DISTINCT customer_id FROM "
                        "fold_labels\n                    ORDER BY "
                        "sha256('42:' || ? || "
                        "':' || customer_id),customer_id LIMIT ?"
                    ),
                    [part["start"], cap],
                )
                history.con.execute(
                    "DELETE FROM fold_labels WHERE customer_id NOT IN (SELECT "
                    "customer_id FROM fold_selected)"
                )
            count = history.con.execute(
                "SELECT count(DISTINCT customer_id) FROM fold_labels"
            ).fetchall()[0][0]
            if not count:
                raise ValueError("Empty eligible query partition")
            export_query(
                history.con,
                "SELECT DISTINCT customer_id FROM fold_labels ORDER BY customer_id",
                [],
                run / "queries" / f"{name}.parquet",
            )
            export_query(
                history.con,
                "SELECT * FROM fold_labels ORDER BY customer_id,article_id",
                [],
                run / "labels" / f"{name}.parquet",
            )
            pairs = history.con.execute("SELECT count(*) FROM fold_labels").fetchall()[
                0
            ][0]
            write_json(
                run / "queries" / f"{name}.json",
                {
                    **part,
                    "file": f"queries/{name}.parquet",
                    "count": count,
                    "ground_truth_pairs": pairs,
                    "query_cap": cap,
                    "selection": "all catalog-matched purchase-active queries; no cap"
                    if cap is None
                    else "purchase-active then SHA256(42:start:user); no hit filtering",
                },
            )
            emit(
                f"roster_{name}",
                count,
                count,
                "queries",
                status="completed",
                metrics={"ground_truth_pairs": pairs},
                force=True,
            )
    finally:
        history.close()


def full_rosters(run: Path, profile: dict) -> None:
    """Validate the full-mode contract and use the shared roster implementation."""
    if profile["profile"] != "full":
        raise ValueError("Full rosters require the full profile")
    rosters(run, profile)


def prepare_test_roster(run: Path, source: Path, *, authorized: bool) -> None:
    """Read final Test truth only after the runtime's separate authorization gates.

    The caller must additionally lock model/config identities and enforce one-time
    evaluation. This function refuses implicit access and existing Test outputs.
    It never refits models or modifies the pre-Test canonical history.
    """
    if not authorized:
        raise PermissionError("Final Test label access requires explicit authorization")
    from hm_recsys.data.streaming import export_query

    if any(
        (run / folder / "Test.parquet").exists() for folder in ("queries", "labels")
    ):
        raise FileExistsError("Final Test roster already exists; refusing replacement")
    history = History(run)
    try:
        part = next(p for p in split_manifest() if p["partition"] == "Test")
        history.con.execute(
            (
                "CREATE TEMP TABLE test_labels AS\n            SELECT DISTINCT "
                "t.customer_id,t.article_id FROM read_csv(?,all_varchar=true) t"
                "\n            JOIN articles a USING(article_id) WHERE "
                "cast(t.t_dat AS DATE)>=?\n            AND cast(t.t_dat AS DATE)<?"
            ),
            [
                str(source / "original data/transactions_train.csv"),
                part["start"],
                part["end"],
            ],
        )
        count = history.con.execute(
            "SELECT count(DISTINCT customer_id) FROM test_labels"
        ).fetchall()[0][0]
        if not count:
            raise ValueError("No valid final Test queries")
        export_query(
            history.con,
            "SELECT DISTINCT customer_id FROM test_labels ORDER BY customer_id",
            [],
            run / "queries/Test.parquet",
        )
        export_query(
            history.con,
            "SELECT * FROM test_labels ORDER BY customer_id,article_id",
            [],
            run / "labels/Test.parquet",
        )
        write_json(
            run / "queries/Test.json",
            {
                **part,
                "file": "queries/Test.parquet",
                "count": count,
                "selection": (
                    "all catalog-matched purchase-active final Test queries; no cap"
                ),
            },
        )
    finally:
        history.close()
