"""Portable identity gates and resumable execution of existing smoke stages."""

from pathlib import Path

from hm_recsys.colab.files import (
    atomic_json,
    canonical_hash,
    require_space,
    verify_files,
)
from hm_recsys.colab.profiles import (
    execution_gate,
    load_profile,
    resource_plan,
    runtime_identity,
    stages_for,
)
from hm_recsys.colab.state import StageStore, writer_lease
from hm_recsys.contracts import Config, digest, read_json, split_manifest
from hm_recsys.resources import run_stage

REQUIRED = {
    "preflight": ["input_acceptance.json", "split_manifest.json"],
    "sample": ["sample_manifest.json", "sample/transactions.parquet"],
    "wals": ["models/wals.json", "indexes/wals.faiss"],
    "towers": ["models/twotower.pt", "models/twotower.json", "checkpoints/latest.json"],
    "index_towers": ["indexes/twotower.faiss"],
    "encode": ["queries/query_vectors.npy", "queries/query_vectors.json"],
    "preprocessing": ["feature_schema.json", "models/visual_centers.npy"],
    "ranker": [
        "models/lambdamart.txt",
        "smoke_results.json",
        "predictions/reference_top12.json",
    ],
    "reload_encode": ["predictions/reloaded_query_vectors.npy"],
    "reload_features": ["predictions/reloaded_features.parquet"],
    "reload": ["reload_results.json", "predictions/reloaded_top12.json"],
}


def code_identity(project: Path) -> dict:
    """Hash the actual checkout, never relying on a commit or build marker alone."""
    paths = [project / "pyproject.toml", project / "uv.lock"]
    for folder, suffix in (("src", ".py"), ("scripts", ".py"), ("configs", ".json")):
        paths.extend(sorted((project / folder).rglob(f"*{suffix}")))
    return {path.relative_to(project).as_posix(): digest(path) for path in paths}


def execution_identity(project: Path, profile: dict, data_root: Path) -> dict:
    """Bind source hashes, code, lock, schema definition, cutoffs and runtime."""
    staging = read_json(data_root / "staging_manifest.json")
    if staging.get("complete") is not True:
        raise ValueError("Input staging is incomplete")
    verify_files(data_root, staging["files"])
    files = {
        name: {key: entry[key] for key in ("bytes", "sha256")}
        for name, entry in staging["files"].items()
    }
    if canonical_hash(files) != staging["input_digest"]:
        raise ValueError("Input staging identity mismatch")
    return {
        "version": 1,
        "profile": profile,
        "code": code_identity(project),
        "inputs": files,
        "splits": split_manifest(),
        "runtime": runtime_identity(project),
        "schema_definition": digest(project / "src/hm_recsys/features.py"),
        "run_final_test": False,
    }


def validate_stage(stage: str, work: Path) -> None:
    """Validate expected outputs and lightweight semantic completion conditions."""
    if stage.startswith("features_"):
        partition = stage.removeprefix("features_")
        required = [
            f"features/{partition}.parquet",
            f"queries/{partition}.json",
            f"labels/{partition}.json",
            "candidate_results.json",
        ]
    elif stage.startswith("probe_"):
        required = ["probe/device_probe.json", "probe/probe_weights.pt"]
    else:
        required = REQUIRED[stage]
    for name in required:
        if not (work / name).is_file():
            raise ValueError(f"Stage missing required output: {name}")
    if stage == "sample":
        Config.load(work / "config.resolved.json").check_sample(
            read_json(work / "sample_manifest.json")
        )
    if stage == "towers":
        result = read_json(work / "models/twotower.json")
        if result["steps"] <= 0 or result["projection_max_parameter_change"] <= 0:
            raise ValueError("Tower stage did not actually train")
    if (
        stage == "reload"
        and read_json(work / "reload_results.json")["status"] != "passed"
    ):
        raise ValueError("Reload parity did not pass")
    if stage.startswith("features_"):
        import pyarrow.parquet as pq

        partition = stage.removeprefix("features_")
        table = pq.ParquetFile(work / "features" / f"{partition}.parquet")
        schema = read_json(work / "feature_schema.json")
        if table.schema_arrow.names != ["customer_id", "article_id", *schema["names"]]:
            raise ValueError("Feature shard schema mismatch")
        if (
            table.metadata.num_rows
            != read_json(work / "candidate_results.json")["partitions"][partition][
                "feature_rows"
            ]
        ):
            raise ValueError("Feature shard row count mismatch")


def execute(
    project: Path,
    data_root: Path,
    local: Path,
    persistent: Path,
    *,
    profile_name: str = "smoke",
    until: str = "reload",
    allow_full: bool = False,
    allow_pilot: bool = False,
    device: str | None = None,
    final_test: bool = False,
    recover_lock: bool = False,
    mount_root: Path | None = None,
) -> dict:
    """Execute or restore complete boundaries; never treat partial files as done."""
    profile = load_profile(project, profile_name)
    if device is not None:
        if profile_name != "pilot" or device not in ("cpu", "cuda"):
            raise ValueError("Device override is only supported for pilot")
        profile["device"] = device
    execution_gate(
        profile, allow_full=allow_full, final_test=final_test, allow_pilot=allow_pilot
    )
    return _execute_profile(
        project,
        data_root,
        local,
        persistent,
        profile=profile,
        until=until,
        recover_lock=recover_lock,
        mount_root=mount_root,
    )


def _execute_profile(
    project: Path,
    data_root: Path,
    local: Path,
    persistent: Path,
    *,
    profile: dict,
    until: str = "reload",
    recover_lock: bool = False,
    mount_root: Path | None = None,
    synthetic_fixture: bool = False,
) -> dict:
    """Internal shared orchestration; public entry points enforce host/input gates."""
    stages = stages_for(profile)
    if until not in stages:
        raise ValueError("Unknown or disabled stage")
    if data_root.resolve() == persistent.resolve() or local.resolve().is_relative_to(
        persistent.resolve()
    ):
        raise ValueError("Training and persistence roots must be distinct")
    identity = execution_identity(project, profile, data_root)
    if not persistent.parent.is_dir():
        raise FileNotFoundError("Persistent run parent is unavailable")
    if mount_root is not None:
        import os

        if not os.path.ismount(mount_root):
            raise OSError("Persistence mount is unavailable")
    with (
        writer_lease(persistent, recover=recover_lock),
        writer_lease(local, recover=recover_lock),
    ):
        store = StageStore(
            local, persistent, identity, mount_root=mount_root, stages=stages
        )
        resources_path = local / "resources.resolved.json"
        persistent_resources = persistent / "resources.resolved.json"
        if persistent_resources.exists():
            resources = read_json(persistent_resources)
        else:
            resources = resource_plan(profile)
            atomic_json(persistent_resources, resources)
        atomic_json(resources_path, resources)
        current_resources = resource_plan(profile)
        resources["rss_abort_gib"] = min(
            resources["rss_abort_gib"], current_resources["rss_abort_gib"]
        )
        resources["rss_target_gib"] = resources["rss_abort_gib"] * 0.8
        resources["available_floor_gib"] = max(
            resources["available_floor_gib"], current_resources["available_floor_gib"]
        )
        atomic_json(local / "resources.current.json", resources)
        policy = Config({**profile["compute"], **resources})
        require_space(local, resources["disk_reserve_bytes"])
        (persistent / "checkpoints").mkdir(exist_ok=True)
        records = []
        for stage in stages:
            dependencies = {
                record["stage"]: record["manifest_sha256"] for record in records
            }
            record = store.restore(stage, dependencies)
            if record is None:
                print(f"Running isolated stage: {stage}", flush=True)
                view = store.view(records)

                def worker(work: Path, current: str = stage) -> None:
                    require_space(local, resources["disk_reserve_bytes"])
                    # These are ephemeral runtime bindings, not semantic inputs.
                    manifest = (
                        read_json(work / "run_manifest.json")
                        if (work / "run_manifest.json").exists()
                        else {}
                    )
                    manifest.update(
                        source_root=str(data_root.resolve()),
                        is_smoke=profile["profile"] == "smoke",
                        profile=profile["profile"],
                        evidence="synthetic_fixture"
                        if synthetic_fixture
                        else "operator_execution",
                        is_benchmark=False,
                        portable_inputs=True,
                        migration_signature=store.signature,
                        code_sha256=identity["code"],
                    )
                    atomic_json(work / "run_manifest.json", manifest)
                    atomic_json(work / "config.resolved.json", profile["compute"])
                    atomic_json(work / "profile.resolved.json", profile)
                    atomic_json(
                        work / "colab_stage.json",
                        {
                            "data_root": str(data_root.resolve()),
                            "synthetic_fixture": synthetic_fixture,
                            "persistent_checkpoint": str(
                                (persistent / "checkpoints/towers").resolve()
                            ),
                            "checkpoint_every_steps": profile["checkpoint_every_steps"],
                            "mount_root": str(mount_root) if mount_root else None,
                        },
                    )
                    for folder in ("models", "indexes", "predictions", "tmp"):
                        (work / folder).mkdir(exist_ok=True)
                    run_stage(
                        current,
                        work,
                        policy,
                        module="hm_recsys.colab.worker",
                        cwd=project,
                    )

                record = store.execute(
                    stage,
                    dependencies,
                    view,
                    worker,
                    lambda work, current=stage: _validate_profile_stage(
                        current, work, profile
                    ),
                )
            else:
                print(f"Verified/restored stage: {stage}", flush=True)
            records.append(record)
            if stage == until:
                break
        status = store.status()
        if status["complete"] and len(records) < len(stages):
            return status
        locations = {
            name: path.relative_to(local).as_posix()
            for name, path in store.view(records).items()
        }
        atomic_json(local / "artifacts.json", locations)
        atomic_json(local / "status.json", status)
        if status["complete"]:
            atomic_json(persistent / "status.json", status)
            atomic_json(persistent / "artifacts.json", locations)
        if status["complete"]:
            view = store.view(records)
            summary = read_json(
                view[
                    "smoke_results.json"
                    if profile["profile"] == "smoke"
                    else "pilot_results.json"
                ]
            )
            summary.update(
                status="passed",
                run_signature=store.signature,
                sample=read_json(
                    view[
                        "sample_manifest.json"
                        if profile["profile"] == "smoke"
                        else "population_manifest.json"
                    ]
                ),
                wals=read_json(view["models/wals.json"]),
                twotower={
                    key: read_json(view["models/twotower.json"])[key]
                    for key in ("steps", "epochs", "processed_targets")
                },
                reload=read_json(view["reload_results.json"]),
            )
            atomic_json(local / "run_summary.json", summary)
            store.check_mount()
            atomic_json(persistent / "run_summary.json", summary)
            if profile["profile"] == "pilot":
                from hm_recsys.colab.review import publish_review

                publish_review(local, persistent, identity, status, view)
        return status


def _validate_profile_stage(stage: str, work: Path, profile: dict) -> None:
    if profile["profile"] == "pilot":
        from hm_recsys.pilot.worker import validate

        validate(stage, work, profile)
    else:
        validate_stage(stage, work)
