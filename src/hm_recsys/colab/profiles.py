"""Explicit migration profiles, platform gates, and honest scale-readiness plans."""

import importlib.metadata
import os
import platform
import subprocess
import sys
from pathlib import Path

import psutil

from hm_recsys.colab.files import canonical_hash
from hm_recsys.contracts import Config, digest, read_json, split_manifest

# The current Python snapshot and fit adapters are deliberately not relabelled
# as full-data backends. These are actionable implementation gates, not claimed
# resource measurements. A profile cannot disable them through configuration.
SCALE_BLOCKERS = (
    (
        "Full population training and all-fold resource acceptance "
        "remain outside the pilot milestone"
    ),
    "Full/CUDA/cloud acceptance evidence has not been established for this release",
)


def stages_for(profile: dict) -> tuple[str, ...]:
    """Return only the stages actually required by the selected profile."""
    if profile["profile"] == "pilot":
        from hm_recsys.pilot.worker import STAGES
    else:
        from hm_recsys.colab.state import STAGES
    return STAGES


def load_profile(project: Path, name: str) -> dict:
    """Validate semantic settings independently of notebook switches."""
    if name not in ("smoke", "pilot", "full"):
        raise ValueError("Unknown training profile")
    profile = read_json(project / "configs" / f"colab_{name}.json")
    if profile["profile"] != name or profile["run_final_test"] is not False:
        raise ValueError("Profile identity or final-Test contract violation")
    if profile["quotas"] != [60, 30, 30, 50, 50, 50]:
        raise ValueError("Retrieval architecture cannot change during migration")
    if name == "smoke":
        if profile["compute"] != Config.load(project / "configs/smoke.json").values:
            raise ValueError("Colab CPU smoke must retain the local smoke limits")
    if name == "pilot":
        if profile["folds"] != ["F1", "Validation"] or profile["device"] not in (
            "cpu",
            "cuda",
        ):
            raise ValueError("Pilot fold/device contract mismatch")
        bounds = {
            "max_queries_per_fold": 1000,
            "wals_factors": 64,
            "wals_iterations": 3,
            "tt_batch_size": 512,
            "tt_max_targets": 20000,
            "tt_epochs": 1,
            "tt_max_steps": 100,
            "ranker_rounds": 100,
        }
        if any(
            not 0 < profile["compute"][key] <= limit for key, limit in bounds.items()
        ):
            raise ValueError("Pilot compute budget exceeds its declared cap")
        if (
            profile["compute"]["cluster_count"] != 128
            or profile["compute"]["cluster_max_rows"] is not None
        ):
            raise ValueError("Pilot must fit all legal K=128 visual preprocessing rows")
        if (
            not 0 < profile["training_subset"]["base_users"] <= 5000
            or profile["feature_query_batch"] != 32
        ):
            raise ValueError("Pilot population/batch contract mismatch")
        if any(
            profile["compute"].get(key) is not None
            for key in (
                "max_users",
                "max_catalog",
                "max_transactions",
                "historical_activity_min_dates",
            )
        ):
            raise ValueError("Pilot global data must not inherit smoke population caps")
    if name == "full":
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


def is_colab_runtime() -> bool:
    """Detect the Linux Colab environment; never infer it from a GPU alone."""
    return sys.platform == "linux" and bool(
        os.environ.get("COLAB_RELEASE_TAG") or os.environ.get("COLAB_BACKEND_VERSION")
    )


def execution_gate(
    profile: dict, *, allow_full: bool, final_test: bool, allow_pilot: bool = False
) -> None:
    """Reject local scale-up, final Test, missing opt-in, and unsupported paths."""
    if final_test or profile["run_final_test"]:
        raise ValueError("Final Test is disabled for this workflow")
    if profile["profile"] == "full" and not allow_full:
        raise ValueError("Full requires explicit --allow-full")
    if profile["profile"] == "pilot" and not allow_pilot:
        raise ValueError("Pilot requires explicit --allow-pilot")
    if profile["profile"] != "smoke":
        if not is_colab_runtime():
            raise ValueError(
                "Pilot/full training is prohibited on the development host"
            )
        if profile["profile"] == "full":
            raise RuntimeError("blocked_implementation: " + "; ".join(SCALE_BLOCKERS))


def runtime_identity(project: Path) -> dict:
    """Record execution environment without initializing numerical runtimes."""
    accelerator = []
    if is_colab_runtime():
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
        except (FileNotFoundError, subprocess.TimeoutExpired):
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
) -> dict:
    """Return a non-training plan with explicit implementation and runtime gates."""
    profile = load_profile(project, name)
    if device is not None:
        if name != "pilot" or device not in ("cpu", "cuda"):
            raise ValueError("Device override is pilot-only")
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
        else "implemented_not_cloud_verified"
        if name == "pilot"
        else "blocked_implementation",
        "blockers": list(SCALE_BLOCKERS) if name == "full" else [],
        "stages": list(stages_for(profile)),
        "required_operator_evidence": []
        if name == "smoke"
        else [
            "passing Colab CPU smoke and persistence/reload",
            "CUDA probe and tiny tower update/reload for CUDA",
            "compatible pilot resource/persistence evidence before full",
            "no full execution in the pilot milestone",
        ],
        "checkpoint_policy": (
            "complete deterministic batches; same-runtime continuation "
            "requires explicit parity verification"
        ),
    }
