"""Verify input staging, manifest recovery and bounded canonical execution.

Inputs are generated source artifacts and owned temporary local/persistent
roots. Tests cover interruption, corruption, mount/identity/resource rejection,
safe Notebook defaults and actual small CPU model reload. Files are written
only beneath pytest roots; no real Drive, Colab, CUDA or customer input is used.
"""

import ast
import copy
import shutil
from pathlib import Path

import nbformat
import pytest

from hm_recsys.core.contracts import ROOT, digest, read_json
from hm_recsys.core.synthetic import create_fixture
from hm_recsys.runtime.files import atomic_json, stage_inputs, verified_copy
from hm_recsys.runtime.profiles import execution_gate, load_profile, resource_plan
from hm_recsys.runtime.runner import execute
from hm_recsys.runtime.state import StageStore, writer_lease


def test_input_whitelist_reuse_corruption_and_source_integrity(tmp_path: Path) -> None:
    """Copy with spaces; never copy old runs or mutate prepared sources."""
    source, local = (tmp_path / "source data", tmp_path / "local data")
    create_fixture(source)
    for modality in ("text", "image"):
        (source / f"{modality}_features/{modality}_faiss.index").write_bytes(
            b"opaque fixture index"
        )
    (source / "runs/old").mkdir(parents=True)
    (source / "runs/old/private.txt").write_text("must not copy")
    before = {
        str(p.relative_to(source)): digest(p) for p in source.rglob("*") if p.is_file()
    }
    report = stage_inputs(source, local)
    assert report["complete"] and len(report["files"]) == 12
    assert not (local / "runs").exists()
    assert stage_inputs(source, local) == report
    target = local / "text_features/text_embeddings.npy"
    target.write_bytes(b"corrupted")
    assert stage_inputs(source, local)["complete"]
    assert digest(target) == digest(source / "text_features/text_embeddings.npy")
    assert list(target.parent.glob("*.rejected-*"))
    assert before == {
        str(p.relative_to(source)): digest(p) for p in source.rglob("*") if p.is_file()
    }
    assert (
        stage_inputs(source, local, reverify=True)["input_digest"]
        == report["input_digest"]
    )


def test_copy_interruption_checksum_missing_and_space(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Incomplete copies and bad digests never become successful artifacts."""
    source = tmp_path / "input"
    source.write_bytes(b"a" * 2048)
    target = tmp_path / "output"

    def interrupt(_: int) -> None:
        raise OSError("simulated interrupted copy")

    with pytest.raises(OSError, match="interrupted"):
        verified_copy(source, target, after_chunk=interrupt)
    assert not target.exists()
    with pytest.raises(ValueError, match="checksum"):
        verified_copy(source, target, expected_sha256="0" * 64)
    assert not target.exists()
    with pytest.raises(ValueError, match="regular"):
        verified_copy(tmp_path / "missing", target)
    monkeypatch.setattr(
        "hm_recsys.runtime.files.shutil.disk_usage",
        lambda _: shutil._ntuple_diskusage(10, 10, 0),
    )
    with pytest.raises(RuntimeError, match="blocked_resource"):
        verified_copy(source, target)


def test_stage_failure_resume_fresh_runtime_and_incompatible_identity(
    tmp_path: Path,
) -> None:
    """Simulated persistence restores verified deltas, not markers or directories."""
    local, persistent = (tmp_path / "runtime1/run", tmp_path / "persistent/run")
    persistent.parent.mkdir()
    identity = {
        "config": 1,
        "build": "a",
        "schema": "v1",
        "cutoff": "2020-08-11",
        "input": "hash",
        "device": "cpu",
        "runtime": "python/numerical runtime identity",
    }
    store = StageStore(local, persistent, identity)

    def fail(work: Path) -> None:
        (work / "partial.txt").write_text("partial")
        raise RuntimeError("child exit 7")

    with pytest.raises(RuntimeError, match="exit 7"):
        store.execute("prepare", {}, {}, fail, lambda _: None)
    assert store.record("prepare") is None
    record = store.execute(
        "prepare",
        {},
        {},
        lambda work: write_text(work / "complete.txt", "complete"),
        lambda work: check_text(work / "complete.txt", "complete"),
    )
    assert store.record("prepare", persistent=True) == record
    shutil.rmtree(local)
    restored = StageStore(tmp_path / "runtime2/run", persistent, identity)
    assert restored.restore("prepare", {}) == record
    assert restored.view([record])["complete.txt"].read_text() == "complete"
    with pytest.raises(ValueError, match="Upstream"):
        restored.restore("prepare", {"changed": "dependency"})
    for key in identity:
        with pytest.raises(ValueError, match="Incompatible"):
            StageStore(
                tmp_path / f"mismatch-{key}", persistent, {**identity, key: "changed"}
            )
    file = persistent / "stages/prepare" / record["attempt"] / "files/complete.txt"
    file.write_text("corrupt")
    with pytest.raises(ValueError, match="artifact|Artifact"):
        restored.record("prepare", persistent=True)


def test_local_completion_survives_failed_persistence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A remote copy failure can be resumed without rerunning successful work."""
    persistent = tmp_path / "drive/run"
    persistent.parent.mkdir()
    store = StageStore(tmp_path / "local/run", persistent, {"id": "1"})
    original = store.persist
    monkeypatch.setattr(
        store, "persist", lambda _: (_ for _ in ()).throw(OSError("mount unavailable"))
    )
    with pytest.raises(OSError, match="unavailable"):
        store.execute(
            "prepare",
            {},
            {},
            lambda p: write_text(p / "value.json", "{}"),
            lambda _: None,
        )
    assert store.record("prepare") is not None
    assert store.record("prepare", persistent=True) is None
    monkeypatch.setattr(store, "persist", original)
    record = store.restore("prepare", {})
    assert record == store.record("prepare", persistent=True)
    with writer_lease(persistent), pytest.raises(RuntimeError, match="writer"):
        with writer_lease(persistent):
            pass


def test_profiles_full_opt_in_local_gate_and_no_smoke_caps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Full is neither an accidental Mac run nor relabelled smoke execution."""
    full = load_profile(ROOT, "full")
    assert full["compute"]["max_users"] is None
    assert full["compute"]["max_catalog"] is None
    assert full["compute"]["tt_max_steps"] is None
    assert full["compute"]["wals_factors"] == 64
    with pytest.raises(ValueError, match="explicit"):
        execution_gate(full, allow_full=False, final_test=False)
    monkeypatch.setattr("hm_recsys.runtime.profiles.is_colab_runtime", lambda: False)
    with pytest.raises(ValueError, match="development"):
        execution_gate(full, allow_full=True, final_test=False)
    monkeypatch.setattr("hm_recsys.runtime.profiles.is_colab_runtime", lambda: True)
    execution_gate(full, allow_full=True, final_test=False)
    with pytest.raises(ValueError, match="Test"):
        execution_gate(load_profile(ROOT, "smoke"), allow_full=False, final_test=True)
    (tmp_path / "configs").mkdir()
    bad = copy.deepcopy(full)
    bad["compute"]["max_users"] = 500
    atomic_json(tmp_path / "configs/colab_full.json", bad)
    with pytest.raises(ValueError, match="sampling"):
        load_profile(tmp_path, "full")
    with pytest.raises(RuntimeError, match="resource"):
        resource_plan(full, available=2 * 1024**3)


def test_notebook_structure_syntax_and_safe_defaults() -> None:
    """Validate notebook source without mounting, installing, or executing cells."""
    path = ROOT / "colab_training_val.ipynb"
    book = nbformat.read(path, as_version=4)
    nbformat.validate(book)
    source = []
    for cell in book.cells:
        if cell.cell_type == "code":
            compile(cell.source, str(path), "exec")
            assert cell.execution_count is None and (not cell.outputs)
            source.append(cell.source)
    text = "\n".join(source)
    assert 'RUN_PROFILE = "smoke"' in text
    assert "ALLOW_FULL_RUN = False" in text and "RUN_FINAL_TEST = False" in text
    assert "fit_towers" not in text and "LGBMRanker" not in text
    assert all(name not in text for name in ("RELEASE_NAME", "zipfile", "unpack("))
    assert "GITHUB_REPO_URL" in text and "GIT_COMMIT" in text
    choices: set[tuple[str, ...]] = set()
    for node in ast.walk(ast.parse(text)):
        if not isinstance(node, ast.Tuple):
            continue
        values = [
            value.value
            for value in node.elts
            if isinstance(value, ast.Constant) and isinstance(value.value, str)
        ]
        if len(values) == len(node.elts):
            choices.add(tuple(values))
    assert ("run", "resume", "status") in choices
    assert ("smoke", "full") in choices


def test_resumable_existing_pipeline_on_synthetic_data(tmp_path: Path) -> None:
    """Execute real shared models, restore stage deltas and verify cached reload."""
    source, data = (tmp_path / "source", tmp_path / "runtime/data")
    create_fixture(source)
    for modality in ("text", "image"):
        (source / f"{modality}_features/{modality}_faiss.index").write_bytes(
            b"not used"
        )
    stage_inputs(source, data)
    persistent = tmp_path / "simulated_drive/runs/test"
    persistent.parent.mkdir(parents=True)
    local = data / "runs/colab/test"
    result = execute(ROOT, data, local, persistent, until="twotower")
    assert result["stages"]["twotower"]["persisted_complete"]
    assert not result["complete"]
    replacement = tmp_path / "new_runtime/data"
    stage_inputs(source, replacement)
    shutil.rmtree(data)
    restored = replacement / "runs/colab/test"
    final = execute(ROOT, replacement, restored, persistent)
    assert final["complete"]
    locations = read_json(persistent / "artifacts.json")
    assert (
        read_json(persistent / locations["reload_results.json"])["status"] == "passed"
    )
    assert (
        read_json(persistent / "validation_lock.json")["signature"]
        == final["signature"]
    )
    for partition in ("F1", "F2", "F3", "F4", "Validation"):
        assert f"fusion/{partition}/candidates.parquet" in locations
        assert f"features/{partition}/manifest.json" in locations
        for channel in ("trend", "repurchase", "sibling", "itemcf", "wals", "twotower"):
            assert f"channels/{partition}/{channel}.parquet" in locations
    assert not any(name.startswith("labels/Test") for name in locations)
    before = {p: digest(p) for p in persistent.rglob("complete.json")}
    assert execute(ROOT, replacement, restored, persistent)["complete"]
    assert before == {p: digest(p) for p in persistent.rglob("complete.json")}
    assert execute(ROOT, replacement, restored, persistent, until="twotower")[
        "complete"
    ]


def test_disconnected_mount_rejects_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An existing directory cannot substitute for an actual mounted destination."""
    persistent = tmp_path / "drive/run"
    persistent.parent.mkdir()
    monkeypatch.setattr("hm_recsys.runtime.state.os.path.ismount", lambda _: True)
    store = StageStore(
        tmp_path / "local", persistent, {"id": "test"}, mount_root=tmp_path / "drive"
    )
    monkeypatch.setattr("hm_recsys.runtime.state.os.path.ismount", lambda _: False)
    with pytest.raises(OSError, match="mount"):
        store.record("prepare", persistent=True)
    with pytest.raises(OSError, match="mount"):
        store.persist({})


def write_text(path: Path, value: str) -> None:
    """Write a tiny stage fixture."""
    path.write_text(value, encoding="utf-8")


def check_text(path: Path, expected: str) -> None:
    """Validate the stage fixture's content."""
    assert path.read_text(encoding="utf-8") == expected
