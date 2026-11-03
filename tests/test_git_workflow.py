"""Verify pinned Git identities around the real canonical smoke stage graph.

Temporary source checkouts and generated tiny data exercise failed-stage
recovery, incompatible contracts, actual Notebook stage-cell binding and
candidate/feature/Top-12 agreement with the same nongit production path.
Owned offline Git commits and model artifacts stay in pytest directories;
no remote is contacted and the user's repository/history is never mutated.
"""

import copy
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from hm_recsys.core.contracts import ROOT, read_json
from hm_recsys.core.synthetic import create_fixture
from hm_recsys.runtime.files import stage_inputs
from hm_recsys.runtime.profiles import load_profile
from hm_recsys.runtime.runner import execute, execution_identity
from hm_recsys.runtime.state import StageStore


def _git(project: Path, *arguments: str) -> str:
    """Execute a temporary fixture's Git operation and return trimmed stdout."""
    result = subprocess.run(
        [
            "git",
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "-C",
            str(project),
            *arguments,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _project(tmp_path: Path) -> tuple[Path, dict]:
    """Create an offline code checkout whose origin is metadata, never contacted."""
    project = tmp_path / "checkout"
    project.mkdir()
    for name in ("src", "scripts", "configs"):
        shutil.copytree(
            ROOT / name, project / name, ignore=shutil.ignore_patterns("__pycache__")
        )
    for name in ("pyproject.toml", "uv.lock"):
        shutil.copy2(ROOT / name, project / name)
    (project / "docs/archi").mkdir(parents=True)
    shutil.copy2(ROOT / "docs/archi/evaluation_protocol.md", project / "docs/archi")
    (project / ".gitignore").write_text("data/\n__pycache__/\n", encoding="utf-8")
    _git(project, "init", "-b", "main")
    _git(project, "add", ".")
    _git(project, "commit", "-m", "offline fixture")
    remote = "https://github.com/example/hm-offline-fixture.git"
    _git(project, "remote", "add", "origin", remote)
    commit = _git(project, "rev-parse", "HEAD")
    _git(project, "checkout", "--detach", commit)
    return (
        project,
        {"remote_url": remote, "requested_ref": "main", "resolved_commit": commit},
    )


def test_git_identity_failed_stage_resume_and_incompatibility(tmp_path: Path) -> None:
    """Reject changed contracts and archive identity while restoring valid stages."""
    project, source_control = _project(tmp_path)
    source, data = (tmp_path / "source", project / "data")
    create_fixture(source)
    for modality in ("text", "image"):
        (source / f"{modality}_features/{modality}_faiss.index").write_bytes(b"unused")
    stage_inputs(source, data)
    profile = load_profile(project, "smoke")
    identity = execution_identity(project, profile, data, git_source=source_control)
    assert identity["version"] == 3 and identity["source_control"]["clean"] is True
    persistent = tmp_path / "drive/run"
    persistent.parent.mkdir()
    store = StageStore(data / "run", persistent, identity, stages=("prepare",))

    def fail(work: Path) -> None:
        (work / "partial.txt").write_text("partial", encoding="utf-8")
        raise RuntimeError("fixture stage interruption")

    with pytest.raises(RuntimeError, match="interruption"):
        store.execute("prepare", {}, {}, fail, lambda _: None)
    assert store.record("prepare", persistent=True) is None

    def succeed(work: Path) -> None:
        (work / "ok.txt").write_text("verified", encoding="utf-8")

    record = store.execute("prepare", {}, {}, succeed, lambda _: None)
    fresh = StageStore(
        tmp_path / "replacement", persistent, identity, stages=("prepare",)
    )
    assert fresh.restore("prepare", {}) == record
    for field in (
        "source_control",
        "profile",
        "inputs",
        "runtime",
        "schema_definition",
        "protocol_sha256",
        "lock_sha256",
    ):
        changed = copy.deepcopy(identity)
        changed[field] = {"incompatible": True}
        with pytest.raises(ValueError, match="Incompatible"):
            StageStore(tmp_path / "bad", persistent, changed)
    legacy = execution_identity(project, profile, data)
    with pytest.raises(ValueError, match="Incompatible"):
        StageStore(tmp_path / "legacy", persistent, legacy)
    _git(project, "commit", "--allow-empty", "-m", "new commit same code")
    changed_source = {
        **source_control,
        "resolved_commit": _git(project, "rev-parse", "HEAD"),
    }
    new_commit = execution_identity(project, profile, data, git_source=changed_source)
    assert new_commit["code"] == identity["code"]
    with pytest.raises(ValueError, match="Incompatible"):
        StageStore(tmp_path / "new_commit", persistent, new_commit)


def test_git_smoke_pipeline_matches_local_reference(tmp_path: Path) -> None:
    """Git identity changes provenance while canonical numerical results agree."""
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    from test_modular_notebook import _cell, _settings, _stage_namespace

    from hm_recsys.runtime.modular import ORDER, resolved_profile, run_one

    project, source_control = _project(tmp_path)
    source, data = (tmp_path / "source", project / "data")
    create_fixture(source)
    for modality in ("text", "image"):
        (source / f"{modality}_features/{modality}_faiss.index").write_bytes(b"unused")
    stage_inputs(source, data)
    persistent = tmp_path / "drive/git"
    persistent.parent.mkdir()
    local = data / "runs/colab/git"
    profile = resolved_profile(project, "smoke")
    namespace, calls = _stage_namespace(tmp_path / "notebook_logs")
    namespace["CLI"] = [sys.executable, str(project / "scripts/run_pipeline.py")]
    namespace["OPTIONS"] = [
        "--profile",
        "smoke",
        "--run-id",
        "git",
        "--data-root",
        str(data),
        "--persistent-root",
        str(persistent.parent),
        "--git-remote",
        source_control["remote_url"],
        "--git-ref",
        source_control["requested_ref"],
        "--git-commit",
        source_control["resolved_commit"],
    ]
    namespace["LOG_ROOT"] = local / "notebook_logs"
    namespace["REPORT_ROOT"] = local / "reports"

    def dispatch(argv: list[str], log: Path) -> None:
        """Replace only the Colab command endpoint; run genuine stage workers."""
        calls.append((argv, log))
        run_one(
            project,
            data,
            local,
            persistent,
            profile=profile,
            stage=argv[argv.index("--stage") + 1],
            action=argv[argv.index("--stage-action") + 1],
            git_source=source_control,
        )

    namespace["run_command"] = dispatch
    for stage in ORDER:
        exec(
            compile(_cell("stage:" + stage), "actual Notebook stage", "exec"), namespace
        )
    defaults = _settings()
    for name in (
        "RUN_PROFILE",
        "RUN_FINAL_TEST",
        "ALLOW_FINAL_TEST",
        "LOCKED_RUN_SIGNATURE",
    ):
        namespace[name] = defaults[name]
    exec(
        compile(_cell("stage:final_test"), "actual gated Test cell", "exec"), namespace
    )
    assert [argv[argv.index("--stage") + 1] for argv, _ in calls] == list(ORDER)
    result = read_json(persistent / "status.json")
    assert result["complete"]
    paths = read_json(persistent / "artifacts.json")
    assert read_json(persistent / paths["reload_results.json"])["status"] == "passed"
    assert (
        read_json(persistent / "identity.json")["source_control"]["resolved_commit"]
        == source_control["resolved_commit"]
    )
    summary = read_json(persistent / "run_summary.json")
    reference = tmp_path / "drive/local_reference"
    assert execute(project, data, data / "runs/colab/local_reference", reference)[
        "complete"
    ]
    expected_paths = read_json(reference / "artifacts.json")
    compared = 0
    for name, location in paths.items():
        if not name.endswith(".parquet") or not name.startswith(
            (
                "channels/",
                "fusion/",
                "features/",
                "queries/",
                "labels/",
                "predictions/reference_top12",
            )
        ):
            continue
        left, right = (
            pq.read_table(persistent / location),
            pq.read_table(reference / expected_paths[name]),
        )
        assert left.num_rows < 100000 and left.schema == right.schema
        for column in left.column_names:
            if pa.types.is_floating(left[column].type):
                np.testing.assert_allclose(
                    left[column].to_numpy(),
                    right[column].to_numpy(),
                    rtol=1e-05,
                    atol=1e-06,
                    equal_nan=True,
                )
            else:
                assert left[column].equals(right[column]), (name, column)
        compared += 1
    assert compared >= 50
    assert (
        read_json(reference / "run_summary.json")["validation"] == summary["validation"]
    )
    assert execute(project, data, local, persistent, git_source=source_control)[
        "complete"
    ]
    replacement = tmp_path / "new_runtime/data"
    stage_inputs(source, replacement)
    assert execute(
        project,
        replacement,
        replacement / "runs/colab/git",
        persistent,
        git_source=source_control,
    )["complete"]
