"""Validated smoke/full budgets, host gates and measured resource planning.

Consumes existing versioned JSON configuration and runtime evidence; emits resolved
stage plans and guarded resource limits. Depends on core temporal contracts and
system metadata. Full retains null/no-limit semantics and is prohibited on the
development host.
"""

import importlib.metadata
import os
import platform
import subprocess
import sys
from pathlib import Path

import psutil

from hm_recsys.core.contracts import Config, digest, read_json, split_manifest
from hm_recsys.runtime.files import canonical_hash


def stages_for(profile: dict) -> tuple[str, ...]:
    """Return the same functional stage graph for smoke and full."""
    from hm_recsys.runtime.modular import ORDER

    if profile["profile"] not in ("smoke", "full"):
        raise ValueError("Only smoke and full are active profiles")
    return ORDER


# Validate semantic settings independently of notebook switches.
def load_profile(project: Path, name: str) -> dict:
    """Validate semantic settings independently of notebook switches."""
    if name not in ("smoke", "full"):
        raise ValueError("Unknown training profile")
    profile = read_json(project / "configs" / f"colab_{name}.json")
    if profile["profile"] != name or profile["run_final_test"] is not False:
        raise ValueError("Profile identity or final-Test contract violation")
    if profile["quotas"] != [60, 30, 30, 50, 50, 50]:
        raise ValueError("Retrieval architecture cannot change during migration")
    if name == "smoke":
        if profile["compute"] != Config.load(project / "configs/smoke.json").values:
            raise ValueError("Colab CPU smoke must retain the local smoke limits")
    if name == "full":
        if profile["training_subset"] is not None:
            raise ValueError("Full cannot restrict the base training cohort")
        for key in (
            "max_users",
            "max_catalog",
            "max_transactions",
            "max_queries_per_fold",
            "tt_max_targets",
            "tt_max_steps",
            "cluster_max_rows",
            "historical_activity_min_dates",
        ):
            if profile["compute"].get(key) is not None:
                raise ValueError(f"Full cannot inherit smoke/pilot sampling: {key}")
        for key, expected in {
            "wals_factors": 64,
            "wals_iterations": 15,
            "tt_epochs": 10,
            "cluster_count": 128,
            "ranker_rounds": 1500,
            "ranker_num_leaves": 63,
            "ranker_max_depth": 6,
            "ranker_min_child_samples": 100,
            "ranker_early_stopping": 50,
        }.items():
            if profile["compute"].get(key) != expected:
                raise ValueError(f"Full architecture setting mismatch: {key}")
        if profile["folds"] != ["F1", "F2", "F3", "F4", "Validation"]:
            raise ValueError("Full requires all four folds and complete Validation")
    if profile["negative_sampling"] != {
        "train_only": True,
        "negatives_per_positive": 20,
        "hard_fraction": 0.5,
    }:
        raise ValueError("Train-only sampling contract mismatch")
    return profile


# Detect the Linux Colab environment; never infer it from a GPU alone.
def is_colab_runtime() -> bool:
    """Detect the Linux Colab environment; never infer it from a GPU alone."""
    return sys.platform == "linux" and bool(
        os.environ.get("COLAB_RELEASE_TAG") or os.environ.get("COLAB_BACKEND_VERSION")
    )


# Reject local scale-up, final Test, missing opt-in, and unsupported paths.
def execution_gate(profile: dict, *, allow_full: bool, final_test: bool) -> None:
    """Reject local scale-up, final Test, missing opt-in, and unsupported paths."""
    if final_test or profile["run_final_test"]:
        raise ValueError("Final Test is disabled for this workflow")
    if profile["profile"] == "full" and not allow_full:
        raise ValueError("Full requires explicit --allow-full")
    if profile["profile"] not in ("smoke", "full"):
        raise ValueError("Only smoke and full are active profiles")
    if profile["profile"] != "smoke":
        if not is_colab_runtime():
            raise ValueError("Full training is prohibited on the development host")


def runtime_identity(project: Path) -> dict:
    """Record execution environment without initializing numerical runtimes."""
    accelerator = []
    if is_colab_runtime():
        # Protect this operation with explicit failure/cleanup handling so partial
        # work is never reported as complete.
        try:
            probe = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,driver_version,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            if probe.returncode == 0:
                accelerator = sorted(probe.stdout.strip().splitlines())
        # Handle the declared exception type here and preserve the surrounding
        # failure/recovery contract.
        except (FileNotFoundError, subprocess.TimeoutExpired):
            # Preserve this intentionally empty handler; the surrounding branch
            # defines which failure is optional.
            pass
    return {
        "python": platform.python_version(),
        "platform": sys.platform,
        "machine": platform.machine(),
        "colab": is_colab_runtime(),
        "accelerator_inventory": accelerator,
        "lock_sha256": digest(project / "uv.lock"),
        "versions": {
            name: importlib.metadata.version(name)
            for name in (
                "numpy",
                "pyarrow",
                "duckdb",
                "scipy",
                "scikit-learn",
                "implicit",
                "torch",
                "faiss-cpu",
                "lightgbm",
                "psutil",
            )
        },
    }


def resource_plan(profile: dict, *, available: int | None = None) -> dict:
    """Resolve reserve-based process limits; settings are not memory guarantees."""
    memory = psutil.virtual_memory()
    usable = memory.available if available is None else available
    gib = 1024**3
    if usable < 3 * gib:
        raise RuntimeError("blocked_resource: less than 3 GiB available memory")
    if not is_colab_runtime():
        abort = min(5 * gib, int(usable * 0.70))
        reserve = 2 * gib
    else:
        reserve = max(2 * gib, int(usable * 0.20))
        abort = int((usable - reserve) * 0.85)
    return {
        "observed_available_bytes": usable,
        "rss_abort_gib": abort / gib,
        "rss_target_gib": abort * 0.80 / gib,
        "available_floor_gib": reserve / gib,
        "monitor_seconds": 0.2,
        "threads": profile["compute"]["threads"],
        "disk_reserve_bytes": 10 * gib,
        "environment_disk_estimate_bytes": 15 * gib,
        "os_hard_cap": False,
    }


def plan(
    project: Path,
    name: str,
    data_root: Path,
    run: Path,
    persistent: Path,
    *,
    device: str | None = None,
    performance: str = "reference",
) -> dict:
    """Return a non-training plan with explicit implementation and runtime gates."""
    from hm_recsys.runtime.settings import apply

    profile = apply(load_profile(project, name), project, performance)
    if device is not None:
        if name not in ("full",) or device not in ("cpu", "cuda"):
            raise ValueError("Device override requires full")
        profile["device"] = device
    return {
        "profile": profile,
        "semantic_hash": canonical_hash(profile),
        "splits": split_manifest(),
        "final_test_enabled": False,
        "local_data_root": str(data_root),
        "local_run": str(run),
        "persistent_run": str(persistent),
        "resources": resource_plan(profile),
        "runtime": runtime_identity(project),
        "implementation_status": "supported_bounded_cpu"
        if name == "smoke"
        else "implemented_not_cloud_verified",
        "blockers": [],
        "stages": list(stages_for(profile)),
        "required_operator_evidence": []
        if name == "smoke"
        else [
            "passing Colab CPU smoke and persistence/reload",
            "CUDA probe and tiny tower update/reload for CUDA",
            "compatible pilot resource/persistence evidence before full",
            "full requires explicit opt-in and per-stage resource acceptance",
        ],
        "checkpoint_policy": (
            "complete deterministic batches; same-runtime continuation "
            "requires explicit parity verification"
        ),
    }
