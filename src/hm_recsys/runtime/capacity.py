"""Resource estimates for complete feature, training and Validation workloads.

Consumes canonical counts and resolved budgets; emits capacity JSON and progress
events. Uses data allocation guards and the frozen feature schema. Estimates never
authorize sampling or treat LightGBM memmaps as unbounded streaming.
"""

from pathlib import Path

from hm_recsys.core.contracts import read_json, write_json

# Publish aggregate work counters without exposing source rows or changing RNG.
from hm_recsys.core.progress import emit
from hm_recsys.data.history import allocation
from hm_recsys.features.builder import FEATURES
from hm_recsys.runtime.files import require_space


# Persist estimates before checking available RAM and local disk reserve.
def check(run: Path, name: str, *, memory: int, disk: int, details: dict) -> None:
    """Persist estimates before checking available RAM and local disk reserve."""
    # Write measured state/lineage to the owned run path without mutating source
    # manifests.
    write_json(
        run / "capacity" / f"{name}.json",
        {
            **details,
            "estimated_memory_bytes": int(memory),
            "estimated_additional_disk_bytes": int(disk),
            "assumption": "working-set estimate; sampled RSS guard remains",
        },
    )
    # Reserve headroom against actual available RAM before materializing this working
    # set.
    # Estimates describe required resources, not rows already processed.
    emit(
        f"capacity_{name}",
        0,
        1,
        unit="gates",
        status="started",
        message="Checking estimated working set and disk reserve",
        metrics={
            "estimated_memory_bytes": int(memory),
            "estimated_additional_disk_bytes": int(disk),
            **details,
        },
        force=True,
    )
    # Resource rejection must propagate without reducing a full population.
    try:
        # Apply the existing RAM estimate unchanged.
        allocation(int(memory), name)
        # Apply the existing disk headroom unchanged.
        require_space(run, int(disk) + 10 * 1024**3)
    # Never present an insufficient-resource gate as passed.
    except BaseException as error:
        # The error class is safe aggregate evidence; the original error is re-raised.
        emit(
            f"capacity_{name}",
            0,
            1,
            unit="gates",
            status="failed",
            message="Resource gate rejected; no population reduction applied",
            metrics={"error_type": type(error).__name__},
            force=True,
        )
        # Keep the original failure boundary and worker exit behavior.
        raise
    # Both existing checks have passed before the one-gate counter advances.
    emit(
        f"capacity_{name}",
        1,
        1,
        unit="gates",
        status="completed",
        message="Resource estimate accepted; runtime RSS monitoring still applies",
        metrics={
            "estimated_memory_bytes": int(memory),
            "estimated_additional_disk_bytes": int(disk),
            **details,
        },
        force=True,
    )


# Reserve four-fold and complete Validation outputs before model fitting.
def feature_capacity(run: Path, profile: dict) -> None:
    """Reserve four-fold and complete Validation outputs before model fitting.

    Disk uses the native quota upper bound, two local copies and uncompressed
    features/provenance. Compression is deliberately not assumed. This is not a
    substitute for the ranker's later actual-sampled-row memory estimate.
    """
    partitions = {}
    for name in profile["folds"]:
        meta = read_json(run / "queries" / f"{name}.json")
        count = meta["count"] if "count" in meta else len(meta["users"])
        partitions[name] = count
    # Numeric features, IDs and up to six provenance records per native row.
    row_bytes = len(FEATURES) * 4 + 160 + 6 * 112
    for label, names in (
        ("training_features", profile["folds"][:-1]),
        ("complete_validation", ["Validation"]),
    ):
        count = sum(partitions[n] for n in names)
        # Apply the independent resource estimate before expensive allocation;
        # rejection never shrinks the population.
        check(
            run,
            label,
            memory=512 * 1024**2,
            disk=count * 270 * row_bytes * 2,
            details={
                "queries": count,
                "partitions": names,
                "candidate_row_upper_bound": count * 270,
            },
        )
    total = sum(partitions.values()) * 270 * row_bytes * 2
    # Check scratch/output disk capacity without deleting any existing run or input.
    require_space(run, total + 10 * 1024**3)
    # Write measured state/lineage to the owned run path without mutating source
    # manifests.
    write_json(
        run / "capacity/feature_total.json",
        {
            "partitions": partitions,
            "additional_disk_upper_bound": total,
            "sampling": False,
            "candidate_budget": 270,
        },
    )
