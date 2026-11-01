"""Immutable smoke configuration, date contracts, and artifact helpers."""

import hashlib
import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
START = date(2020, 2, 1)
TT_START = date(2020, 5, 19)
TT_END = date(2020, 8, 5)
BASE_END = date(2020, 8, 12)
EXTRACT_END = date(2020, 9, 16)
VOCAB_START = date(2020, 6, 16)
CHANNELS = ("trend", "repurchase", "sibling", "itemcf", "wals", "twotower")
QUOTAS = dict(zip(CHANNELS, (60, 30, 30, 50, 50, 50), strict=True))
HARD_CAPS = {
    "max_users": 500,
    "max_catalog": 5000,
    "max_transactions": 200000,
    "max_queries_per_fold": 50,
    "max_feature_rows": 100000,
    "threads": 2,
    "wals_factors": 16,
    "wals_iterations": 2,
    "tt_batch_size": 32,
    "tt_max_targets": 5000,
    "tt_epochs": 1,
    "tt_max_steps": 50,
    "cluster_max_rows": 1000,
    "cluster_max_iter": 10,
    "ranker_rounds": 30,
    "rss_target_gib": 4,
    "rss_abort_gib": 5,
    "monitor_seconds": 0.2,
    "max_itemcf_pairs": 10000000,
}
FIXED_SETTINGS = {
    "profile": "smoke",
    "device": "cpu",
    "duckdb_memory": "1GB",
    "duckdb_temp": "2GiB",
    "interop_threads": 1,
    "loader_workers": 0,
    "seed": 42,
    "tt_dimension": 128,
    "cluster_count": 8,
    "cluster_n_init": 1,
    "ranker_num_leaves": 7,
    "ranker_max_depth": 3,
    "ranker_min_child_samples": 5,
}


def write_json(path: Path, value: object) -> None:
    """Write a readable local artifact, rejecting nonstandard JSON numbers."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def read_json(path: Path) -> dict:
    """Read an object manifest; reject a non-object top level."""
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected an object manifest: {path.name}")
    return value


def digest(path: Path) -> str:
    """Hash a file incrementally without retaining its contents."""
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def stable_key(value: str, seed: int = 42) -> str:
    """Return a cross-process deterministic sampling key."""
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def article_id(value: object) -> str:
    """Accept canonical strings only; never silently truncate or lose zeros."""
    if not isinstance(value, str) or len(value) != 10 or not value.isdigit():
        raise ValueError("Article IDs must be ten-digit strings")
    return value


def split_manifest() -> list[dict[str, str]]:
    """Return the exact six disjoint weeks, including disabled Test dates."""
    return [
        {
            "partition": name,
            "start": (BASE_END + timedelta(days=7 * i)).isoformat(),
            "end": (BASE_END + timedelta(days=7 * (i + 1))).isoformat(),
            "cutoff": (BASE_END + timedelta(days=7 * i - 1)).isoformat(),
        }
        for i, name in enumerate(("F1", "F2", "F3", "F4", "Validation", "Test"))
    ]


def validate_splits(splits: list[dict[str, str]]) -> None:
    """Fail closed on altered dates, legacy anchors, or overlap."""
    if splits != split_manifest():
        raise ValueError("Split manifest violates protocol v2.0 exact dates")


@dataclass(frozen=True)
class Config:
    """Smoke-only validated limits; no implicit full-training fallback."""

    values: dict

    @classmethod
    def load(cls, path: Path | None = None) -> "Config":
        """Load explicit settings and enforce the task's maximum budgets."""
        baseline = read_json(ROOT / "configs/smoke.json")
        values = read_json(path) if path else baseline.copy()
        for name, cap in HARD_CAPS.items():
            value = values.get(name)
            if not isinstance(value, (int, float)) or not 0 < value <= cap:
                raise ValueError(f"Smoke hard budget exceeded: {name}")
        for name, expected in FIXED_SETTINGS.items():
            if values.get(name) != expected:
                raise ValueError(f"Unsupported fixed smoke setting: {name}")
        if set(values) != set(baseline):
            raise ValueError("Missing or unexpected smoke configuration keys")
        fixed = (
            "profile",
            "device",
            "duckdb_memory",
            "duckdb_temp",
            "interop_threads",
            "loader_workers",
            "seed",
            "tt_dimension",
            "cluster_count",
            "cluster_n_init",
            "ranker_num_leaves",
            "ranker_max_depth",
            "ranker_min_child_samples",
        )
        for name in fixed:
            if values[name] != baseline[name]:
                raise ValueError(f"Unsupported smoke setting: {name}")
        for name, limit in baseline.items():
            if name in fixed:
                continue
            value = values[name]
            if not isinstance(value, (float, int)) or value <= 0:
                raise ValueError(f"Invalid positive limit: {name}")
            if name == "available_floor_gib":
                if value < 2:
                    raise ValueError("Cannot lower the available-memory floor")
            elif value > limit:
                raise ValueError(f"Smoke budget exceeded: {name}")
        return cls(values)

    def limit(self, key: str) -> int:
        """Read an integer count limit from the validated profile."""
        return int(self.values[key])

    def check_sample(self, manifest: dict) -> None:
        """Require a bounded, verified sample before any training allocation."""
        if manifest.get("sampling_scope") != "smoke":
            raise ValueError("Training requires a smoke sample manifest")
        if manifest.get("validated") is not True:
            raise ValueError("Unvalidated sample cannot enter training")
        for key, cap in (
            ("users", "max_users"),
            ("catalog", "max_catalog"),
            ("transactions", "max_transactions"),
        ):
            count = manifest.get(key, -1)
            if not isinstance(count, int) or not 0 < count <= self.limit(cap):
                raise ValueError(f"Sample {key} exceeds smoke limit or is empty")
        if manifest.get("history_end_exclusive") != EXTRACT_END.isoformat():
            raise ValueError("Unexpected extraction boundary")


def source_identity(path: Path) -> dict:
    """Record cheap identity without claiming a verified file checksum."""
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "checksum_status": "not_verified",
    }
