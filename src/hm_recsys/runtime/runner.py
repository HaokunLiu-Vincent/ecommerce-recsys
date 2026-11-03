"""Bind modular runs to immutable source, data, schema and runtime identities.

The stage client delegates execution to runtime.modular; this module owns only
identity validation and the whole-run convenience interface. Changed code layout
requires a new run ID; old ZIP and Git run identities are never relabelled.
"""

from pathlib import Path

from hm_recsys.core.contracts import digest, read_json, split_manifest
from hm_recsys.runtime.files import (
    canonical_hash,
    verify_files,
)
from hm_recsys.runtime.profiles import (
    execution_gate,
    runtime_identity,
)


def code_identity(project: Path) -> dict:
    """Hash the actual checkout, never relying on a commit or build marker alone."""
    paths = [project / "pyproject.toml", project / "uv.lock"]
    for folder, suffix in (("src", ".py"), ("scripts", ".py"), ("configs", ".json")):
        paths.extend(sorted((project / folder).rglob(f"*{suffix}")))
    return {path.relative_to(project).as_posix(): digest(path) for path in paths}


def execution_identity(
    project: Path,
    profile: dict,
    data_root: Path,
    *,
    git_source: dict | None = None,
) -> dict:
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
    identity = {
        "version": 3,
        "source_layout": "single-source-modular-v1",
        "profile": profile,
        "code": code_identity(project),
        "inputs": files,
        "splits": split_manifest(),
        "runtime": runtime_identity(project),
        "schema_definition": digest(project / "src/hm_recsys/features/builder.py"),
        "run_final_test": False,
    }
    # Git runs deliberately cannot impersonate an archive/local-development run.
    if git_source is not None:
        # Recheck the current checkout at execution/status time, not just bootstrap.
        from hm_recsys.runtime.git_checkout import inspect_checkout

        # Bind the actual clean tree to the requested detached commit and origin.
        identity["source_control"] = inspect_checkout(
            project,
            repo_url=git_source["remote_url"],
            ref=git_source["requested_ref"],
            commit=git_source["resolved_commit"],
        )
        # Protocol text is part of the reproducible Git path's artifact contract.
        identity["protocol_sha256"] = digest(
            project / "docs/archi/evaluation_protocol.md"
        )
        # Retain an explicit lock identity alongside the per-file source hashes.
        identity["lock_sha256"] = digest(project / "uv.lock")
    # A changed source layout always has a distinct immutable run signature.
    return identity


def execute(
    project: Path,
    data_root: Path,
    local: Path,
    persistent: Path,
    *,
    profile_name: str = "smoke",
    until: str = "reload",
    allow_full: bool = False,
    device: str | None = None,
    performance: str = "optimized",
    final_test: bool = False,
    recover_lock: bool = False,
    mount_root: Path | None = None,
    git_source: dict | None = None,
) -> dict:
    """Run the canonical graph with host/resource gates and no implicit Test."""
    from hm_recsys.runtime.modular import resolved_profile, run_all

    if final_test:
        raise ValueError("Use the separately approved final_test stage")
    profile = resolved_profile(
        project, profile_name, device=device, performance=performance
    )
    execution_gate(profile, allow_full=allow_full, final_test=False)
    return run_all(
        project,
        data_root,
        local,
        persistent,
        profile=profile,
        until=until,
        mount_root=mount_root,
        recover_lock=recover_lock,
        git_source=git_source,
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
    """Use the canonical graph for byte-authenticated bounded fixture tests."""
    from hm_recsys.runtime.modular import run_all

    return run_all(
        project,
        data_root,
        local,
        persistent,
        profile=profile,
        until=until,
        recover_lock=recover_lock,
        mount_root=mount_root,
        synthetic_fixture=synthetic_fixture,
    )
