"""Immutable smoke configuration, date contracts, and artifact helpers."""

import hashlib
import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

# Fixed historical split and model boundary dates.
START = date(2020, 2, 1)
TT_START = date(2020, 5, 19)
TT_END = date(2020, 8, 5)
BASE_END = date(2020, 8, 12)
EXTRACT_END = date(2020, 9, 16)
VOCAB_START = date(2020, 6, 16)

# Canonical 6 recall channels and their evaluation quotas.
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


# Return a cross-process deterministic sampling key.
def stable_key(value: str, seed: int = 42) -> str:
    """Return a cross-process deterministic sampling key."""
    # Return the computed result for return a cross-process deterministic sampling
    # key with the stated dtype/order and empty-input behavior.
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


# Accept canonical strings only; never silently truncate or lose zeros.
def article_id(value: object) -> str:
    """Accept canonical strings only; never silently truncate or lose zeros."""
    # Check this validity/resource condition before proceeding; its failure must stop
    # the operation explicitly.
    if not isinstance(value, str) or len(value) != 10 or not value.isdigit():
        # Raise the explicit ValueError for this violated contract; the caller must
        # not publish incomplete success.
        raise ValueError("Article IDs must be ten-digit strings")
    # Return current typed value being validated or accumulated to the caller without
    # changing its row/key associations.
    return value


# Return the exact six disjoint weeks, including disabled Test dates.
def split_manifest() -> list[dict[str, str]]:
    """Return the exact six disjoint weeks, including disabled Test dates."""
    # Return the computed result for return the exact six disjoint weeks, including
    # disabled test dates with the stated dtype/order and empty-input behavior.
    return [
        {
            "partition": name,
            "start": (BASE_END + timedelta(days=7 * i)).isoformat(),
            "end": (BASE_END + timedelta(days=7 * (i + 1))).isoformat(),
            "cutoff": (BASE_END + timedelta(days=7 * i - 1)).isoformat(),
        }
        for i, name in enumerate(("F1", "F2", "F3", "F4", "Validation", "Test"))
    ]


# Fail closed on altered dates, legacy anchors, or overlap.
def validate_splits(splits: list[dict[str, str]]) -> None:
    """Fail closed on altered dates, legacy anchors, or overlap."""
    # Check this validity/resource condition before proceeding; its failure must stop
    # the operation explicitly.
    if splits != split_manifest():
        # Raise the explicit ValueError for this violated contract; the caller must
        # not publish incomplete success.
        raise ValueError("Split manifest violates protocol v2.0 exact dates")


# Smoke-only validated limits; no implicit full-training fallback.
@dataclass(frozen=True)
class Config:
    """Smoke-only validated limits; no implicit full-training fallback."""

    # Declare aligned numeric/category values consumed by this step in the typed
    # record/interface contract.
    values: dict

    # Load explicit settings and enforce the task's maximum budgets.
    @classmethod
    def load(cls, path: Path | None = None) -> "Config":
        """Load explicit settings and enforce the task's maximum budgets."""
        # For baseline: load object metadata before trusting its identity,
        # dimensions, counts or completion claim.
        baseline = read_json(ROOT / "configs/smoke.json")
        # For aligned numeric/category values consumed by this step: load object
        # metadata before trusting its identity, dimensions, counts or completion
        # claim.
        values = read_json(path) if path else baseline.copy()
        # Process each (name, cap) from the declared items() sequence for load
        # explicit settings and enforce the task's maximum budgets.
        for name, cap in HARD_CAPS.items():
            # Resolve current typed value being validated or accumulated for load
            # explicit settings and enforce the task's maximum budgets before the
            # downstream consumer uses it.
            value = values.get(name)
            # Check this validity/resource condition before proceeding; its failure
            # must stop the operation explicitly.
            if not isinstance(value, (int, float)) or not 0 < value <= cap:
                # Raise the explicit ValueError for this violated contract; the
                # caller must not publish incomplete success.
                raise ValueError(f"Smoke hard budget exceeded: {name}")
        # Process each (name, expected) from the declared items() sequence for load
        # explicit settings and enforce the task's maximum budgets.
        for name, expected in FIXED_SETTINGS.items():
            # Check this validity/resource condition before proceeding; its failure
            # must stop the operation explicitly.
            if values.get(name) != expected:
                # Raise the explicit ValueError for this violated contract; the
                # caller must not publish incomplete success.
                raise ValueError(f"Unsupported fixed smoke setting: {name}")
        # Check this validity/resource condition before proceeding; its failure must
        # stop the operation explicitly.
        if set(values) != set(baseline):
            # Raise the explicit ValueError for this violated contract; the caller
            # must not publish incomplete success.
            raise ValueError("Missing or unexpected smoke configuration keys")
        # Initialize fixed with the explicit structured fields/empty state used by
        # this operation.
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
        # Process each artifact, feature or stage identifier used in this operation
        # from the declared fixed sequence for load explicit settings and enforce the
        # task's maximum budgets.
        for name in fixed:
            # Check this validity/resource condition before proceeding; its failure
            # must stop the operation explicitly.
            if values[name] != baseline[name]:
                # Raise the explicit ValueError for this violated contract; the
                # caller must not publish incomplete success.
                raise ValueError(f"Unsupported smoke setting: {name}")
        # Process each (name, limit) from the declared items() sequence for load
        # explicit settings and enforce the task's maximum budgets.
        for name, limit in baseline.items():
            # Choose the name in fixed case required by load explicit settings and
            # enforce the task's maximum budgets.
            if name in fixed:
                # Skip this explicitly inapplicable/empty record while retaining the
                # order of all remaining valid inputs.
                continue
            # Read current typed value being validated or accumulated from aligned
            # numeric/category values consumed by this step at the explicit row/key
            # coordinates.
            value = values[name]
            # Check this validity/resource condition before proceeding; its failure
            # must stop the operation explicitly.
            if not isinstance(value, (float, int)) or value <= 0:
                # Raise the explicit ValueError for this violated contract; the
                # caller must not publish incomplete success.
                raise ValueError(f"Invalid positive limit: {name}")
            # Choose the name == 'available_floor_gib' case required by load explicit
            # settings and enforce the task's maximum budgets.
            if name == "available_floor_gib":
                # Check this validity/resource condition before proceeding; its
                # failure must stop the operation explicitly.
                if value < 2:
                    # Raise the explicit ValueError for this violated contract; the
                    # caller must not publish incomplete success.
                    raise ValueError("Cannot lower the available-memory floor")
            # Check this validity/resource condition before proceeding; its failure
            # must stop the operation explicitly.
            elif value > limit:
                # Raise the explicit ValueError for this violated contract; the
                # caller must not publish incomplete success.
                raise ValueError(f"Smoke budget exceeded: {name}")
        # Return the computed result for load explicit settings and enforce the
        # task's maximum budgets with the stated dtype/order and empty-input
        # behavior.
        return cls(values)

    # Read an integer count limit from the validated profile.
    def limit(self, key: str) -> int:
        """Read an integer count limit from the validated profile."""
        # Return the computed result for read an integer count limit from the
        # validated profile with the stated dtype/order and empty-input behavior.
        return int(self.values[key])

    # Require a bounded, verified sample before any training allocation.
    def check_sample(self, manifest: dict) -> None:
        """Require a bounded, verified sample before any training allocation."""
        # Check this validity/resource condition before proceeding; its failure must
        # stop the operation explicitly.
        if manifest.get("sampling_scope") != "smoke":
            # Raise the explicit ValueError for this violated contract; the caller
            # must not publish incomplete success.
            raise ValueError("Training requires a smoke sample manifest")
        # Check this validity/resource condition before proceeding; its failure must
        # stop the operation explicitly.
        if manifest.get("validated") is not True:
            # Raise the explicit ValueError for this violated contract; the caller
            # must not publish incomplete success.
            raise ValueError("Unvalidated sample cannot enter training")
        # Process each (key, cap) from the declared (('users', 'max users'),
        # ('catalog', 'max catalog'), ('transactions', 'max transactions')) sequence
        # for require a bounded, verified sample before any training allocation.
        for key, cap in (
            ("users", "max_users"),
            ("catalog", "max_catalog"),
            ("transactions", "max_transactions"),
        ):
            # Resolve observed count used for alignment, reporting or a resource
            # guard for require a bounded, verified sample before any training
            # allocation before the downstream consumer uses it.
            count = manifest.get(key, -1)
            # Check this validity/resource condition before proceeding; its failure
            # must stop the operation explicitly.
            if not isinstance(count, int) or not 0 < count <= self.limit(cap):
                # Raise the explicit ValueError for this violated contract; the
                # caller must not publish incomplete success.
                raise ValueError(f"Sample {key} exceeds smoke limit or is empty")
        # Check this validity/resource condition before proceeding; its failure must
        # stop the operation explicitly.
        if manifest.get("history_end_exclusive") != EXTRACT_END.isoformat():
            # Raise the explicit ValueError for this violated contract; the caller
            # must not publish incomplete success.
            raise ValueError("Unexpected extraction boundary")


# Record cheap identity without claiming a verified file checksum.
def source_identity(path: Path) -> dict:
    """Record cheap identity without claiming a verified file checksum."""
    # Resolve stat for record cheap identity without claiming a verified file
    # checksum before the downstream consumer uses it.
    stat = path.stat()
    # Return the explicit contract/report fields rather than inferring success from
    # file or directory presence.
    return {
        "path": str(path.resolve()),
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "checksum_status": "not_verified",
    }
