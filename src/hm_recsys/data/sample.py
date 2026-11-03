"""Define immutable purchase-line and bounded metadata contracts.

Transaction preserves raw repeats and source calendar dates. Sample is a bounded
fixture/static-metadata adapter used by model encoders and cutoff snapshots; it
is not an alternative production pipeline. CSV header validation and age buckets
are shared by canonical data acceptance and content/ranking preparation.
"""

import csv
import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pyarrow.parquet as pq

from hm_recsys.core.contracts import (
    EXTRACT_END,
    START,
    Config,
    article_id,
    digest,
    read_json,
    source_identity,
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
        if any((self.manifest[key] != count for key, count in actual.items())):
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


def inspect_csv(path: Path, required: set[str]) -> dict:
    """Validate a header without exposing customer histories or reading rows."""
    with path.open(newline="") as handle:
        header = next(csv.reader(handle))
    if not required <= set(header):
        raise ValueError(f"Missing required columns in {path.name}")
    return {**source_identity(path), "columns": header}


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
