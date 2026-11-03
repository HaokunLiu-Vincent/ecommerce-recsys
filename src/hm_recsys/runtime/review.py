"""Aggregate-only diagnostic exports for verified completed runs.

Consumes a validated artifact view, source identity and stage status; emits a
checksummed review bundle. Depends on atomic file transfer helpers. Customer rows,
raw histories, source vectors and model weights are excluded from the diagnostic
export.
"""

import json
import zipfile
from pathlib import Path

from hm_recsys.core.contracts import digest, read_json
from hm_recsys.runtime.files import atomic_json, canonical_hash, verified_copy


# Export explicit summary fields only; no arbitrary log or model inclusion.
def publish_review(
    local: Path, persistent: Path, identity: dict, status: dict, view: dict[str, Path]
) -> dict:
    """Export explicit summary fields only; no arbitrary log or model inclusion."""

    # optional.
    def optional(name: str) -> dict:
        return read_json(view[name]) if name in view else {}

    population = optional("population_manifest.json")
    ranker = optional(f"{identity['profile']['profile']}_results.json")
    wals = optional("models/wals.json")
    tower = optional("models/twotower.json")
    schema = optional("feature_schema.json")
    counts = {}
    for name in identity["profile"]["folds"]:
        roster = optional(f"queries/{name}.json")
        counts[name] = {
            "queries": len(roster["users"])
            if "users" in roster
            else roster.get("count"),
            "start": roster.get("start"),
            "end": roster.get("end"),
            "cutoff": roster.get("cutoff"),
        }
    failures = []
    for path in sorted((persistent / "failures").glob("*.json")):
        failure = read_json(path)
        failures.append(
            {
                "stage": failure.get("stage"),
                "error_type": failure.get("error_type"),
                "status": failure.get("status"),
                "resource_measurements": failure.get("resource_measurements", []),
                "resource_estimates": failure.get("resource_estimates", {}),
            }
        )
    payload = {
        "format_version": 1,
        "scope": "aggregate diagnostics; no customer-level data",
        "capacity": {key: optional(key) for key in view if key.startswith("capacity/")},
        "run_signature": canonical_hash(identity),
        "profile": identity["profile"],
        "code_identity": identity["code"],
        "input_identity": identity["inputs"],
        "runtime": identity["runtime"],
        "splits": identity["splits"],
        "status": status,
        "population": population,
        "queries": counts,
        "coverage": {
            "catalog": population.get("catalog_items"),
            "wals_global_vocabulary": len(wals.get("global_vocabulary", [])),
            "wals_represented_items": len(wals.get("items", [])),
            "tower_items": population.get("catalog_items") if tower else None,
        },
        "schema": {
            key: schema.get(key)
            for key in ("version", "names", "dtype", "categorical", "cluster_count")
        },
        "tower": {
            key: tower.get(key)
            for key in (
                "device",
                "requested_device",
                "steps",
                "epochs",
                "selected_epoch",
                "internal_daily_recall50",
                "internal_validation_queries",
                "peak_cuda_allocated",
                "peak_cuda_reserved",
            )
        },
        "wals": {key: wals.get(key) for key in ("iterations", "factors", "nnz")},
        "ranker": ranker.get("ranker", {}),
        "validation": ranker.get("validation", {}),
        "retrieval": optional("candidate_results.json"),
        "ranker_memory": optional("ranker_memory_estimate.json"),
        "reload": optional("reload_results.json"),
        "resources": optional("resource_usage.json"),
        "failures": failures,
        "final_test": "not_run",
    }
    name = f"{identity['profile']['profile']}_review_{canonical_hash(payload)[:16]}.zip"
    path = local / name
    content = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if not path.exists():
        info = zipfile.ZipInfo(
            f"{identity['profile']['profile']}_review.json",
            date_time=(2020, 1, 1, 0, 0, 0),
        )
        info.compress_type = zipfile.ZIP_DEFLATED
        with zipfile.ZipFile(path, "x") as archive:
            archive.writestr(info, content)
    # Transfer through a temporary file and verify both ends before publishing
    # completion.
    verified_copy(path, persistent / name, expected_sha256=digest(path))
    record = {
        "archive": name,
        "sha256": digest(path),
        "bytes": path.stat().st_size,
        "contains_raw_customer_data": False,
    }
    atomic_json(local / "review_manifest.json", record)
    atomic_json(persistent / "review_manifest.json", record)
    return record
