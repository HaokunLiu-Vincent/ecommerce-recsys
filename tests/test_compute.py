"""Verify bounded canonical inputs, gradients, cache invalidation and publication.

Generated fixtures exercise raw-input preparation, unchanged live tower updates,
ordered prefetch, complete-state remote failure recovery and real CLI argument
contracts. Full/Colab execution is never inferred from CPU or plan-only checks.
The separate root notebook intentionally omits diagnostic benchmark controls.
"""

import copy
from pathlib import Path

import numpy as np
import pytest

from hm_recsys.core.contracts import ROOT, read_json, write_json
from hm_recsys.runtime.files import stage_inputs
from hm_recsys.runtime.profiles import load_profile, stages_for
from hm_recsys.runtime.settings import Cadence, apply
from hm_recsys.runtime.testing import create_full_source


def test_cadence_failure_clock_and_overlay() -> None:
    """Local and remote clocks are independent; failed publish cannot advance time."""
    profile = apply(load_profile(ROOT, "full"), ROOT, "optimized")
    original = load_profile(ROOT, "full")
    assert profile["compute"] == original["compute"]
    assert "prepare" in stages_for(profile)
    assert stages_for(profile).index("prepare") < stages_for(profile).index("twotower")
    c = Cadence(profile["performance"], 0.0)
    assert c.due(1, 1) == (True, True)
    c.completed(1, 1, remote=True)
    assert c.due(2, 59) == (False, False)
    assert c.due(2, 61) == (True, False)
    c.completed(2, 61, remote=False)
    assert c.due(102, 62) == (True, False)
    assert c.due(102, 302) == (True, True)
    assert c.due(103, 303) == (True, True)
    assert c.remote_time == 1
    assert c.due(103, 63, force=True) == (True, True)
    assert Cadence(profile["performance"], 303).due(104, 304) == (True, True)
    smoke = load_profile(ROOT, "smoke")
    assert apply(smoke, ROOT, "optimized")["compute"] == smoke["compute"]


def test_prepared_inputs_updates_invalidation_and_prefetch(tmp_path: Path) -> None:
    """Exact input/mask/Q and live gradients across authentic partial FP32 batches."""
    import torch

    from hm_recsys.data.history import History, canonical
    from hm_recsys.data.streaming import TargetStream
    from hm_recsys.features.content import MappedContent, adapt_content
    from hm_recsys.retrieval.two_tower.execution import prefetch
    from hm_recsys.retrieval.two_tower.inputs import SqlEncoder
    from hm_recsys.retrieval.two_tower.prepared import Prepared, prepare, verify
    from hm_recsys.runtime.benchmark import parity

    torch.set_num_threads(2)
    source = tmp_path / "source"
    create_full_source(source)
    run = tmp_path / "run"
    run.mkdir()
    profile = load_profile(ROOT, "full")
    canonical(run, source, profile)
    adapt_content(run, source)
    write_json(
        run / "run_manifest.json", {"migration_signature": "fixture-current-code"}
    )
    history = History(run)
    encoder = SqlEncoder(history, MappedContent(run))
    targets = TargetStream(history, None)
    meta = prepare(run, encoder, targets)
    assert meta["transaction_rows_exported"] > 0
    assert meta["history_query_rows_scanned"] > 0
    assert read_json(run / "prepared/tower/history_query_profile.json")["children"]
    prepared = Prepared(run / "prepared/tower")
    with pytest.raises(ValueError, match="batch"):
        prepared.users(np.zeros(513, dtype=np.int64))
    with pytest.raises(ValueError, match="batch"):
        next(prepared.batches(0, 513))
    report = parity(encoder, targets, prepared, torch.device("cpu"), run)
    assert report["parameter_max_abs_difference"] == 0
    assert report["gradient_max_abs_difference"] == 0
    assert report["input_examples"] == 1584
    from hm_recsys.retrieval.two_tower.execution import ItemBank

    bank = ItemBank(encoder, torch.device("cpu"))
    for start in range(0, len(history.catalog), 32):
        ids = history.catalog[start : start + 32]
        positions = np.arange(start, start + len(ids), dtype=np.int64)
        for left, right in zip(encoder.items(ids), bank.items(positions), strict=True):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
    import csv

    path = source / "original data/transactions_train.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        first = next(csv.DictReader(handle))
    with path.open("a", encoding="utf-8", newline="") as handle:
        csv.DictWriter(handle, fieldnames=list(first)).writerow(
            {**first, "t_dat": "2020-09-14"}
        )
    future = tmp_path / "future"
    future.mkdir()
    canonical(future, source, profile)
    adapt_content(future, source)
    write_json(
        future / "run_manifest.json", {"migration_signature": "different-future-source"}
    )
    later = History(future)
    later_encoder = SqlEncoder(later, MappedContent(future))
    later_targets = TargetStream(later, None)
    prepare(future, later_encoder, later_targets)
    future_inputs = Prepared(future / "prepared/tower")
    for name, array in prepared.arrays.items():
        np.testing.assert_array_equal(array, future_inputs.arrays[name])
    with pytest.raises(ValueError, match="identity"):
        verify(prepared.folder, future_inputs.meta["identity"])
    later.close()
    contexts = prepared.arrays["target_context"][:1]
    candidates = prepared.arrays["target_item"][:1]
    single = prepared.mask(contexts, candidates)
    assert single.all() and (not (~single).any())
    with pytest.raises(ValueError, match="Prefetch"):
        next(prefetch(iter([]), 3))

    def forbidden(*args, **kwargs):
        raise AssertionError("Unexpected steady-state SQL")

    history.event_histories = forbidden
    history.positives = forbidden
    history.customers = forbidden
    source_iter = prefetch(iter(prepared.batches(0, 512)), 2)
    first = next(source_iter)
    source_iter.close()
    resumed = list(prefetch(iter(prepared.batches(len(first[0]), 512)), 2))
    np.testing.assert_array_equal(
        np.concatenate([b[0] for b in resumed]), prepared.arrays["target_context"][512:]
    )
    changed = copy.deepcopy(meta["identity"])
    changed["history"][2] = 19
    with pytest.raises(ValueError, match="identity"):
        verify(prepared.folder, changed)
    path = prepared.folder / "weights.npy"
    with path.open("r+b") as handle:
        handle.seek(-1, 2)
        handle.write(b"X")
    with pytest.raises(ValueError, match="integrity"):
        Prepared(prepared.folder)
    history.close()


@pytest.mark.parametrize("failure", ["copy", "pointer"])
def test_remote_publication_failure_preserves_last_valid(
    tmp_path: Path, monkeypatch, failure: str
) -> None:
    """A failed upload leaves the old verified remote pointer and snapshot usable."""
    import torch

    from hm_recsys.runtime import checkpoints

    identity = {"fixture": "checkpoint-failure"}
    local, remote = (tmp_path / "local", tmp_path / "remote")
    payload = {
        "global_step": 1,
        "next_target_offset": 512,
        "model": {"weight": torch.tensor([1.0])},
        "optimizer": {},
        "scheduler": {},
        "epoch": 0,
    }
    checkpoints.save_checkpoint(
        local, payload, identity, persistent=remote, retain_latest_only=True
    )
    before = read_json(remote / "latest.json")

    def fail(*args, **kwargs):
        raise OSError("injected upload failure")

    if failure == "copy":
        monkeypatch.setattr(checkpoints, "verified_copy", fail)
    else:
        original = checkpoints.atomic_json

        def pointer(path, value):
            if path == remote / "latest.json":
                raise OSError("injected upload pointer failure")
            return original(path, value)

        monkeypatch.setattr(checkpoints, "atomic_json", pointer)
    with pytest.raises(OSError, match="upload"):
        checkpoints.save_checkpoint(
            local,
            {**payload, "global_step": 2, "next_target_offset": 1024},
            identity,
            persistent=remote,
            retain_latest_only=True,
        )
    assert read_json(remote / "latest.json") == before
    assert checkpoints.load_checkpoint(remote, identity)["global_step"] == 1
    assert checkpoints.load_checkpoint(local, identity)["global_step"] == 2
    record = read_json(local / "latest.json")
    write_json(local / "latest.json", {**record, "global_step": 999})
    with pytest.raises(ValueError, match="progress"):
        checkpoints.load_checkpoint(local, identity)


def test_cli_contracts_and_guards(tmp_path: Path) -> None:
    """Parse actual CLI arguments; local real inputs and excessive caps are rejected."""
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_pipeline.py",
            "plan",
            "--profile",
            "full",
            "--performance",
            "optimized",
            "--persistent-root",
            str(tmp_path),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    import json

    plan = json.loads(result.stdout)
    assert (
        "prepare" in plan["stages"]
        and plan["profile"]["compute"]["tt_max_steps"] is None
    )
    for args in (["--input-run", str(tmp_path)], ["--synthetic", "--batches", "101"]):
        result = subprocess.run(
            [
                sys.executable,
                "scripts/benchmark_compute.py",
                "--run-id",
                "guard",
                *args,
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0


@pytest.mark.skipif(
    __import__("sys").platform != "linux", reason="CUDA compute parity not run on Mac"
)
def test_cuda_inputs_are_not_checkpoint_parameters(tmp_path: Path) -> None:
    """CUDA requires explicit invocation and hardware; never infer success on CPU."""
    import os

    if os.environ.get("HM_TEST_CUDA") != "1":
        pytest.skip("CUDA not requested")
    import torch

    if not torch.cuda.is_available():
        pytest.fail("Explicit CUDA requested but unavailable")
    from hm_recsys.runtime.benchmark import run_benchmark

    report = run_benchmark(
        tmp_path,
        {
            "device": "cuda",
            "synthetic": True,
            "input_run": None,
            "batches": 4,
            "warmup": 1,
            "mode": "all",
            "validation": "complete",
            "trace": False,
        },
    )
    assert report["parity"]["status"] == "passed"


def test_standalone_benchmark_contract_after_notebook_simplification(
    tmp_path: Path,
) -> None:
    """Keep diagnostics callable while excluding benchmark switches from the client."""
    import subprocess
    import sys

    import nbformat

    book = nbformat.read(ROOT / "colab_training_val.ipynb", as_version=4)
    text = "\n".join(c.source for c in book.cells if c.cell_type == "code")
    assert "RUN_BENCHMARK" not in text and "BENCHMARK_ONLY" not in text
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/benchmark_compute.py"),
            "--run-id",
            "contract_only",
            "--synthetic",
            "--batches",
            "4",
            "--warmup",
            "1",
            "--plan",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert '"batches": 4' in result.stdout


def test_optimized_full_stage_restore(tmp_path: Path) -> None:
    """Restore completed stages into a fresh runtime and verify real outputs."""
    import json
    import shutil
    import subprocess
    import sys

    from hm_recsys.runtime.testing import execute_full_fixture

    source = tmp_path / "source"
    create_full_source(source)
    data = tmp_path / "runtime/data"
    stage_inputs(source, data)
    persistent = tmp_path / "drive/full"
    persistent.parent.mkdir()
    local = data / "runs/full"
    execute_full_fixture(
        ROOT,
        data,
        local,
        persistent,
        until="twotower",
        allow_full=True,
        performance="optimized",
    )
    before = read_json(persistent / "stages/twotower/complete.json")
    shutil.rmtree(data)
    replacement = tmp_path / "replacement/data"
    stage_inputs(source, replacement)
    run = replacement / "runs/full"
    result = execute_full_fixture(
        ROOT, replacement, run, persistent, allow_full=True, performance="optimized"
    )
    assert result["complete"]
    assert read_json(persistent / "stages/twotower/complete.json") == before
    summary = read_json(persistent / "run_summary.json")
    assert summary["twotower"] == {
        "steps": 40,
        "epochs": 10,
        "processed_targets": 15840,
    }
    assert summary["reload"]["max_score_absolute_difference"] == 0
    assert summary["validation"]["eligible_queries"] == 24
    assert summary["validation"]["zero_hit_queries"] >= 1
    assert set(summary["partitions"]) == {"F1", "F2", "F3", "F4", "Validation"}
    paths = read_json(run / "artifacts.json")
    for phase in (
        "wals_fit",
        "towers_fit",
        "features_F1",
        "ranker_fit",
        "reload_score",
    ):
        events = [
            json.loads(line)
            for line in (run / paths[f"progress/{phase}.jsonl"])
            .read_text()
            .splitlines()
        ]
        assert events and all(event["stage"] == phase for event in events)
        assert all(
            event["total"] is None or event["completed"] <= event["total"]
            for event in events
        )
    timing = read_json(run / paths["tower_performance.json"])
    assert timing["calls"].get("history_sql", 0) == 0
    assert "models/visual_assignments.npy" in paths
    view = tmp_path / "feature_view"
    for name, relative in paths.items():
        destination = view / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(run / relative, destination)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "hm_recsys.features.benchmark",
            str(view),
            str(tmp_path / "feature_parity.json"),
        ],
        check=True,
        timeout=60,
    )
    assert read_json(tmp_path / "feature_parity.json")["top12_exact"]


def test_benchmark_reads_canonical_prepare_not_retired_stages(tmp_path: Path) -> None:
    """Verify source-bound diagnostic copying without importing old stage names."""
    from hm_recsys.runtime.benchmark import input_view
    from hm_recsys.runtime.modular import ALL_STAGES, GRAPH, resolved_profile
    from hm_recsys.runtime.runner import code_identity
    from hm_recsys.runtime.state import StageStore

    source = tmp_path / "run"
    store = StageStore(
        source,
        source,
        {"code": code_identity(ROOT), "profile": resolved_profile(ROOT, "smoke")},
        stages=ALL_STAGES,
        dependency_graph=GRAPH,
        optional_stages=("final_test",),
    )

    def produce(folder: Path) -> None:
        for name in (
            "canonical/input.txt",
            "content/mapping.json",
            "models/excluded.txt",
        ):
            target = folder / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("synthetic transfer fixture", encoding="utf-8")

    record = store.execute("prepare", {}, {}, produce, lambda p: None)
    destination = tmp_path / "diagnostic"
    input_view(source, destination)
    assert (destination / "canonical/input.txt").is_file()
    assert not (destination / "models").exists()
    dependency = read_json(destination / "benchmark_dependency.json")
    assert dependency["stages"] == {"prepare": record["manifest_sha256"]}
    (destination / "canonical/input.txt").write_text("changed copy", encoding="utf-8")
    original = store.view([record])["canonical/input.txt"]
    assert original.read_text(encoding="utf-8") == "synthetic transfer fixture"
