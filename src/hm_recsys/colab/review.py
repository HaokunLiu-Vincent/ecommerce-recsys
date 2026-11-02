"""Small aggregate-only pilot review bundles, excluding customer-level artifacts."""

import json
import zipfile
from pathlib import Path

from hm_recsys.colab.files import atomic_json, canonical_hash, verified_copy
from hm_recsys.contracts import digest, read_json


def publish_review(
    local: Path, persistent: Path, identity: dict, status: dict, view: dict[str, Path]
) -> dict:
    """Export explicit summary fields only; no arbitrary log or model inclusion."""

    def optional(name: str) -> dict:
        return read_json(view[name]) if name in view else {}

    population = optional("population_manifest.json")
    ranker = optional("pilot_results.json")
    wals = optional("models/wals.json")
    tower = optional("models/twotower.json")
    schema = optional("feature_schema.json")
    counts = {}
    for name in identity["profile"]["folds"]:
        roster = optional(f"queries/{name}.json")
        counts[name] = {
            "queries": len(roster["users"]) if "users" in roster else None,
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
            }
        )
    payload = {
        "format_version": 1,
        "scope": "pilot aggregate diagnostics; no customer-level data",
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
    name = f"pilot_review_{canonical_hash(payload)[:16]}.zip"
    path = local / name
    content = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if not path.exists():
        info = zipfile.ZipInfo("pilot_review.json", date_time=(2020, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        with zipfile.ZipFile(path, "x") as archive:
            archive.writestr(info, content)
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
