"""Bounded filesystem/CPU migration tests; no real Drive, Colab or CUDA access."""

import copy
import importlib.util
import shutil
import zipfile
from pathlib import Path

import nbformat
import pytest

from hm_recsys.colab.bootstrap import unpack
from hm_recsys.colab.files import (
    atomic_json,
    canonical_hash,
    stage_inputs,
    verified_copy,
)
from hm_recsys.colab.profiles import execution_gate, load_profile, resource_plan
from hm_recsys.colab.release import allowed, build_release, extract_release
from hm_recsys.colab.runner import execute
from hm_recsys.colab.state import StageStore, writer_lease
from hm_recsys.contracts import ROOT, digest, read_json
from hm_recsys.synthetic import create_fixture


def test_input_whitelist_reuse_corruption_and_source_integrity(tmp_path: Path) -> None:
    """Copy with spaces; never copy old runs or mutate prepared sources."""
    source, local = tmp_path / "source data", tmp_path / "local data"
    create_fixture(source)
    # Source indexes are copied intact but never loaded in the migration test.
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
        "hm_recsys.colab.files.shutil.disk_usage",
        lambda _: shutil._ntuple_diskusage(10, 10, 0),
    )
    with pytest.raises(RuntimeError, match="blocked_resource"):
        verified_copy(source, target)


def test_stage_failure_resume_fresh_runtime_and_incompatible_identity(
    tmp_path: Path,
) -> None:
    """Simulated persistence restores verified deltas, not markers or directories."""
    local, persistent = tmp_path / "runtime1/run", tmp_path / "persistent/run"
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
        store.execute("preflight", {}, {}, fail, lambda _: None)
    assert store.record("preflight") is None
    record = store.execute(
        "preflight",
        {},
        {},
        lambda work: write_text(work / "complete.txt", "complete"),
        lambda work: check_text(work / "complete.txt", "complete"),
    )
    assert store.record("preflight", persistent=True) == record
    shutil.rmtree(local)
    restored = StageStore(tmp_path / "runtime2/run", persistent, identity)
    assert restored.restore("preflight", {}) == record
    assert restored.view([record])["complete.txt"].read_text() == "complete"
    with pytest.raises(ValueError, match="Upstream"):
        restored.restore("preflight", {"changed": "dependency"})
    for key in identity:
        with pytest.raises(ValueError, match="Incompatible"):
            StageStore(
                tmp_path / f"mismatch-{key}", persistent, {**identity, key: "changed"}
            )
    file = persistent / "stages/preflight" / record["attempt"] / "files/complete.txt"
    file.write_text("corrupt")
    with pytest.raises(ValueError, match="artifact|Artifact"):
        restored.record("preflight", persistent=True)


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
            "preflight",
            {},
            {},
            lambda p: write_text(p / "value.json", "{}"),
            lambda _: None,
        )
    assert store.record("preflight") is not None
    assert store.record("preflight", persistent=True) is None
    monkeypatch.setattr(store, "persist", original)
    record = store.restore("preflight", {})
    assert record == store.record("preflight", persistent=True)
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
    monkeypatch.setattr("hm_recsys.colab.profiles.is_colab_runtime", lambda: False)
    with pytest.raises(ValueError, match="development"):
        execution_gate(full, allow_full=True, final_test=False)
    monkeypatch.setattr("hm_recsys.colab.profiles.is_colab_runtime", lambda: True)
    with pytest.raises(RuntimeError, match="blocked_implementation"):
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
    path = ROOT / "notebooks/01_Colab_Training.ipynb"
    book = nbformat.read(path, as_version=4)
    nbformat.validate(book)
    source = []
    for cell in book.cells:
        if cell.cell_type == "code":
            compile(cell.source, str(path), "exec")
            assert cell.execution_count is None and not cell.outputs
            source.append(cell.source)
    text = "\n".join(source)
    assert 'RUN_PROFILE = "smoke"' in text
    assert "ALLOW_FULL_RUN = False" in text and "RUN_FINAL_TEST = False" in text
    assert "fit_towers" not in text and "LGBMRanker" not in text


def test_release_clean_extract_and_hostile_members(tmp_path: Path) -> None:
    """Exercise deterministic packaging and bootstrap without project imports."""
    metadata = build_release(ROOT, tmp_path / "release")
    again = build_release(ROOT, tmp_path / "release")
    assert metadata == again
    assert not any(
        n.startswith(("data/", ".venv/", "docs/achive/")) for n in metadata["files"]
    )
    for name in (
        ".env",
        "data/secrets.json",
        "docs/achive/old.py",
        "src/hm_recsys/credentials.py",
    ):
        assert not allowed(name)
    archive = tmp_path / "release" / metadata["archive"]
    manifest = archive.with_suffix(".manifest.json")
    destination = tmp_path / "clean project"
    extract_release(archive, manifest, destination)
    assert (destination / "scripts/run_pipeline.py").is_file()
    assert (destination / "scripts/local_smoke.py").is_file()
    assert extract_release(archive, manifest, destination) == metadata
    (destination / "scripts/run_pipeline.py").write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        extract_release(archive, manifest, destination)
    # Loading this helper requires only stdlib, just like the notebook bootstrap.
    spec = importlib.util.spec_from_file_location(
        "standalone_bootstrap", ROOT / "src/hm_recsys/colab/bootstrap.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in (
        "../escape",
        "/absolute",
        "data/overwrite",
        "scripts/../../escape",
        "docs/achive/revive.py",
    ):
        with pytest.raises(ValueError):
            module.safe_member(name)
    with zipfile.ZipFile(tmp_path / "evil.zip", "w") as z:
        z.writestr("../escape", "bad")
    evil = {
        "files": {"../escape": {"bytes": 3, "sha256": "bad"}},
        "archive_sha256": digest(tmp_path / "evil.zip"),
    }
    evil["build_id"] = canonical_hash(evil["files"])[:16]
    with pytest.raises(ValueError):
        unpack(tmp_path / "evil.zip", evil, tmp_path / "evil")


def test_resumable_existing_pipeline_on_synthetic_data(tmp_path: Path) -> None:
    """Exercise all real shared CPU models, five folds, recovery and fresh reload."""
    source, data = tmp_path / "source", tmp_path / "runtime/data"
    create_fixture(source)
    for modality in ("text", "image"):
        (source / f"{modality}_features/{modality}_faiss.index").write_bytes(
            b"not used"
        )
    stage_inputs(source, data)
    persistent = tmp_path / "simulated_drive/runs/test"
    persistent.parent.mkdir(parents=True)
    local = data / "runs/colab/test"
    result = execute(ROOT, data, local, persistent, until="index_towers")
    assert result["stages"]["index_towers"]["persisted_complete"]
    assert not result["complete"]
    # A replacement runtime has no previous notebook variables or local models.
    replacement = tmp_path / "new_runtime/data"
    stage_inputs(source, replacement)
    shutil.rmtree(data)
    final = execute(ROOT, replacement, replacement / "runs/colab/test", persistent)
    assert final["complete"]
    locations = read_json(persistent / "artifacts.json")
    assert (
        read_json(persistent / locations["reload_results.json"])["status"] == "passed"
    )
    before = {p: digest(p) for p in persistent.rglob("complete.json")}
    assert execute(ROOT, replacement, replacement / "runs/colab/test", persistent)[
        "complete"
    ]
    assert before == {p: digest(p) for p in persistent.rglob("complete.json")}
    # A completed run may be revisited through an earlier notebook cell.
    assert execute(
        ROOT,
        replacement,
        replacement / "runs/colab/test",
        persistent,
        until="index_towers",
    )["complete"]
    from hm_recsys.cli import execute as legacy_execute
    from hm_recsys.contracts import Config

    legacy = tmp_path / "legacy"
    legacy.mkdir()
    legacy_execute("synthetic", legacy, Config.load(), legacy / "inputs")
    for partition in ("F1", "F2", "F3", "F4", "Validation"):
        for folder, suffix in (
            ("features", ".parquet"),
            ("queries", ".json"),
            ("labels", ".json"),
            ("candidates", ".parquet"),
        ):
            name = f"{folder}/{partition}{suffix}"
            assert digest(legacy / name) == digest(persistent / locations[name])
    for name in ("ranker_sampling.json", "predictions/reference_top12.json"):
        assert read_json(legacy / name) == read_json(persistent / locations[name])
    old_metrics = read_json(legacy / "smoke_results.json")["validation"]
    assert old_metrics == read_json(persistent / "run_summary.json")["validation"]


def test_disconnected_mount_rejects_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An existing directory cannot substitute for an actual mounted destination."""
    persistent = tmp_path / "drive/run"
    persistent.parent.mkdir()
    monkeypatch.setattr("hm_recsys.colab.state.os.path.ismount", lambda _: True)
    store = StageStore(
        tmp_path / "local", persistent, {"id": "test"}, mount_root=tmp_path / "drive"
    )
    monkeypatch.setattr("hm_recsys.colab.state.os.path.ismount", lambda _: False)
    with pytest.raises(OSError, match="mount"):
        store.record("preflight", persistent=True)
    with pytest.raises(OSError, match="mount"):
        store.persist({})


def write_text(path: Path, value: str) -> None:
    """Write a tiny stage fixture."""
    path.write_text(value, encoding="utf-8")


def check_text(path: Path, expected: str) -> None:
    """Validate the stage fixture's content."""
    assert path.read_text(encoding="utf-8") == expected
