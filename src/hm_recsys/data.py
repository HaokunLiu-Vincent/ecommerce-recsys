"""Projected CSV extraction and bounded, explicitly validated sample tables."""

import csv
import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from hm_recsys.contracts import (
    EXTRACT_END,
    START,
    Config,
    article_id,
    digest,
    read_json,
    source_identity,
    stable_key,
    write_json,
)

ARTICLE_FIELDS = (
    "product_code",
    "product_type_no",
    "graphical_appearance_no",
    "colour_group_code",
    "section_no",
    "garment_group_no",
    "index_group_no",
)
CUSTOMER_FIELDS = ("age", "club_member_status", "fashion_news_frequency")


@dataclass(frozen=True)
class Transaction:
    """A purchase line; same-day repeated lines remain separate observations."""

    customer: str
    article: str
    day: date
    price: float
    channel: int


@dataclass
class Sample:
    """Bounded transactions, static metadata, candidate catalog, and lineage."""

    transactions: list[Transaction]
    articles: dict[str, dict]
    customers: dict[str, dict]
    catalog: list[str]
    manifest: dict

    def validate_training(self, config: Config) -> None:
        """Reject oversized or misdeclared in-memory inputs before fitting."""
        config.check_sample(self.manifest)
        actual = {
            "users": len(self.customers),
            "catalog": len(self.catalog),
            "transactions": len(self.transactions),
        }
        if any(self.manifest[key] != count for key, count in actual.items()):
            raise ValueError("Training input counts do not match sample manifest")
        if len(set(self.catalog)) != len(self.catalog):
            raise ValueError("Duplicate candidate catalog identity")

    @classmethod
    def load(cls, run: Path, config: Config) -> "Sample":
        """Validate counts and checksums before loading the bounded extraction."""
        manifest = read_json(run / "sample_manifest.json")
        config.check_sample(manifest)
        for name, sha in manifest["file_sha256"].items():
            if digest(run / "sample" / name) != sha:
                raise ValueError(f"Sample integrity mismatch: {name}")
        tx_path = run / "sample/transactions.parquet"
        if pq.ParquetFile(tx_path).metadata.num_rows != manifest["transactions"]:
            raise ValueError("Sample transaction count mismatch")
        rows = pq.read_table(tx_path).to_pylist()
        transactions = [
            Transaction(
                r["customer_id"],
                article_id(r["article_id"]),
                r["t_dat"],
                float(r["price"]),
                int(r["sales_channel_id"]),
            )
            for r in rows
        ]
        if any(not START <= row.day < EXTRACT_END for row in transactions):
            raise ValueError("Sample contains out-of-boundary dates")
        articles = {
            article_id(row["article_id"]): row
            for row in pq.read_table(run / "sample/articles.parquet").to_pylist()
        }
        customers = {
            row["customer_id"]: row
            for row in pq.read_table(run / "sample/customers.parquet").to_pylist()
        }
        catalog = read_json(run / "sample/catalog.json")["article_ids"]
        if len(catalog) != manifest["catalog"] or len(set(catalog)) != len(catalog):
            raise ValueError("Sample catalog size or uniqueness mismatch")
        if len(customers) != manifest["users"]:
            raise ValueError("Sample user count mismatch")
        if any(
            t.article not in articles or t.customer not in customers
            for t in transactions
        ):
            raise ValueError("Transaction join failure")
        if not set(catalog) <= articles.keys():
            raise ValueError("Candidate catalog join failure")
        return cls(transactions, articles, customers, catalog, manifest)


def connection(run: Path, config: Config) -> duckdb.DuckDBPyConnection:
    """Create a sequential spill-enabled connection scoped to this smoke run."""
    temp = run / "tmp/duckdb"
    temp.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"SET threads={config.limit('threads')}")
    con.execute("SET memory_limit=?", [config.values["duckdb_memory"]])
    con.execute("SET max_temp_directory_size=?", [config.values["duckdb_temp"]])
    con.execute("SET temp_directory=?", [str(temp)])
    con.execute("SET preserve_insertion_order=false")
    return con


def inspect_csv(path: Path, required: set[str]) -> dict:
    """Validate a header without exposing customer histories or reading rows."""
    with path.open(newline="") as handle:
        header = next(csv.reader(handle))
    if not required <= set(header):
        raise ValueError(f"Missing required columns in {path.name}")
    return {**source_identity(path), "columns": header}


def _csv_view(con: duckdb.DuckDBPyConnection, source: Path) -> None:
    # all_varchar prevents numeric inference from stripping leading zeros.
    con.read_csv(str(source / "transactions_train.csv"), all_varchar=True).create_view(
        "raw_tx"
    )
    con.execute("""
        CREATE VIEW tx AS SELECT customer_id, article_id,
            CAST(t_dat AS DATE) AS t_dat, CAST(price AS DOUBLE) AS price,
            CAST(sales_channel_id AS INTEGER) AS sales_channel_id
        FROM raw_tx
        WHERE t_dat >= '2020-02-01' AND t_dat < '2020-09-16'
    """)


def extract_sample(run: Path, source_root: Path, config: Config) -> None:
    """Select users before Aug 5, then extract complete bounded user histories.

    Customer selection uses SHA256(seed:ID), with at least 20 pre-Aug-5 dates.
    Catalog priority is sample pre-Aug-5 purchase count, then article ID;
    static siblings fill remaining capacity. Future positives never select items.
    """
    folder = run / "sample"
    folder.mkdir(exist_ok=False)
    source = source_root / "original data"
    con = connection(run, config)
    try:
        _csv_view(con, source)
        users = con.execute(
            """
            SELECT customer_id FROM tx WHERE t_dat < DATE '2020-08-05'
            GROUP BY customer_id HAVING count(DISTINCT t_dat) >= 20
            ORDER BY sha256('42:' || customer_id), customer_id LIMIT ?
        """,
            [config.limit("max_users")],
        ).fetchall()
        if not users:
            raise ValueError("No historically eligible users")
        user_ids = [r[0] for r in users]
        con.register("selected_users", pa.table({"customer_id": user_ids}))
        count = con.execute("""
            SELECT count(*) FROM tx SEMI JOIN selected_users USING(customer_id)
        """).fetchone()
        if count is None or not 0 < count[0] <= config.limit("max_transactions"):
            raise ValueError("Complete selected cohort exceeds transaction cap")
        # Count first; no arbitrary prefix truncation or late-date removal.
        con.execute(
            """
            COPY (SELECT * FROM tx SEMI JOIN selected_users USING(customer_id)
                  ORDER BY customer_id, t_dat, article_id, price, sales_channel_id)
            TO ? (FORMAT PARQUET)
        """,
            [str(folder / "transactions.parquet")],
        )
        con.read_parquet(str(folder / "transactions.parquet")).create_view("sample_tx")
        con.read_csv(str(source / "articles.csv"), all_varchar=True).create_view(
            "raw_articles"
        )
        unmatched = con.execute("""
            SELECT count(*) FROM sample_tx ANTI JOIN raw_articles USING(article_id)
        """).fetchone()
        if unmatched and unmatched[0]:
            raise ValueError("Catalog join failures in selected transactions")
        primary = con.execute(
            """
            SELECT article_id FROM sample_tx WHERE t_dat < DATE '2020-08-05'
            GROUP BY article_id ORDER BY count(*) DESC, article_id LIMIT ?
        """,
            [config.limit("max_catalog")],
        ).fetchall()
        catalog = [r[0] for r in primary]
        con.register("primary_catalog", pa.table({"article_id": catalog}))
        remaining = config.limit("max_catalog") - len(catalog)
        if remaining:
            siblings = con.execute(
                """
                SELECT DISTINCT a.article_id FROM raw_articles a
                JOIN (SELECT DISTINCT product_code FROM raw_articles
                      SEMI JOIN primary_catalog USING(article_id)) p
                  USING(product_code)
                ANTI JOIN primary_catalog c ON a.article_id=c.article_id
                ORDER BY a.article_id LIMIT ?
            """,
                [remaining],
            ).fetchall()
            catalog.extend(r[0] for r in siblings)
        catalog = sorted(catalog)
        write_json(folder / "catalog.json", {"article_ids": catalog})
        con.register("candidate_catalog", pa.table({"article_id": catalog}))
        fields = ", ".join(ARTICLE_FIELDS)
        con.execute(
            f"""
            COPY (SELECT article_id, {fields} FROM raw_articles
                  WHERE article_id IN (SELECT article_id FROM sample_tx)
                     OR article_id IN (SELECT article_id FROM candidate_catalog)
                  ORDER BY article_id) TO ? (FORMAT PARQUET)
        """,
            [str(folder / "articles.parquet")],
        )
        con.read_csv(str(source / "customers.csv"), all_varchar=True).create_view(
            "raw_customers"
        )
        con.execute(
            """
            COPY (SELECT customer_id, TRY_CAST(age AS DOUBLE) AS age,
                         club_member_status, fashion_news_frequency
                  FROM raw_customers SEMI JOIN selected_users USING(customer_id)
                  ORDER BY customer_id) TO ? (FORMAT PARQUET)
        """,
            [str(folder / "customers.parquet")],
        )
        dates = con.execute("SELECT min(t_dat), max(t_dat) FROM sample_tx").fetchone()
        if dates is None:
            raise ValueError("Empty extraction")
        manifest = {
            "sampling_scope": "smoke",
            "validated": True,
            "users": len(user_ids),
            "catalog": len(catalog),
            "transactions": count[0],
            "seed": 42,
            "history_start_inclusive": START.isoformat(),
            "history_end_exclusive": EXTRACT_END.isoformat(),
            "observed_min_date": str(dates[0]),
            "observed_max_date": str(dates[1]),
            "customer_rule": "pre-Aug-5 >=20 distinct dates; SHA256(42:ID)",
            "catalog_rule": "pre-Aug-5 sample count DESC, ID ASC; static siblings fill",
            "statistics_universe": "complete selected-customer purchase lines",
            "selected_customer_ids": user_ids,
            "catalog_join_failures": 0,
            "source": {p.name: source_identity(p) for p in source.glob("*.csv")},
            "file_sha256": {p.name: digest(p) for p in folder.iterdir()},
        }
        config.check_sample(manifest)
        write_json(run / "sample_manifest.json", manifest)
    finally:
        con.close()


def query_users(
    sample: Sample, *, start: date, cap: int
) -> tuple[list[str], dict[str, set[str]]]:
    """Select purchase-active queries by stable hash, independent of recall hits."""
    from datetime import timedelta

    labels: dict[str, set[str]] = {}
    for row in sample.transactions:
        if start <= row.day < start + timedelta(days=7):
            labels.setdefault(row.customer, set()).add(row.article)
    users = sorted(labels, key=lambda u: (stable_key(f"{start}:{u}"), u))[:cap]
    return users, {user: labels[user] for user in users}


def age_bucket(value: object) -> str:
    """Apply the non-overlapping documented age buckets with explicit UNKNOWN."""
    if value is None:
        return "UNKNOWN"
    age = float(str(value))
    if not math.isfinite(age) or age <= 0 or age > 120:
        return "UNKNOWN"
    if age < 25:
        return "UNDER25"
    if age < 35:
        return "25TO34"
    if age <= 50:
        return "35TO50"
    return "OVER50"
