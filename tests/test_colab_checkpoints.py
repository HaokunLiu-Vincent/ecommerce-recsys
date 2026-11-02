"""Exact CPU continuation of the shared tower in independent processes."""

import os
from pathlib import Path

import pytest

from hm_recsys.contracts import Config, read_json, write_json
from hm_recsys.resources import run_stage

WORKER = '''"""Bounded test worker; never handles real inputs."""
import sys
from pathlib import Path
import numpy as np
import torch
from hm_recsys.colab import checkpoints
from hm_recsys.contracts import Config, read_json, write_json
from hm_recsys.data import Sample, extract_sample
from hm_recsys.pipeline import preflight, load_content
from hm_recsys.synthetic import create_fixture
from hm_recsys.towers import fit_towers

mode, location = sys.argv[1:]
run = Path(location)
source = run / "inputs"
config = Config.load(run / "config.resolved.json")
if mode == "prepare":
    create_fixture(source)
    preflight(run, source, config)
    extract_sample(run, source, config)
elif mode == "compare":
    baseline = torch.load(run / "baseline/models/twotower.pt", weights_only=True)
    resumed = torch.load(run / "resumed/models/twotower.pt", weights_only=True)
    final = torch.load(run / "final_resumed/models/twotower.pt", weights_only=True)
    for key in baseline:
        torch.testing.assert_close(baseline[key], final[key], atol=0, rtol=0)
        torch.testing.assert_close(baseline[key], resumed[key], atol=0, rtol=0)
    for name in ("losses", "processed_events", "steps", "internal_daily_recall50"):
        assert read_json(run / "baseline/models/twotower.json")[name] == read_json(
            run / "resumed/models/twotower.json")[name]
    np.testing.assert_array_equal(np.load(run / "baseline/models/twotower_items.npy"),
                                  np.load(run / "resumed/models/twotower_items.npy"))
    record = read_json(run / "checkpoints/latest.json")
    assert record["global_step"] == 2
    try:
        checkpoints.load_checkpoint(run / "checkpoints", {"wrong": "identity"})
    except ValueError:
        pass
    else:
        raise AssertionError("Changed identity accepted")
    write_json(run / "checkpoint_parity.json", {"status": "passed", "atol": 0,
               "rtol": 0, "boundary_step": 2, "fresh_process": True})
else:
    sample = Sample.load(run, config)
    content = load_content(run, sample, source)
    destination = run / ("baseline" if mode == "baseline" else
                         "final_resumed" if mode == "resume_final" else "resumed")
    (destination / "models").mkdir(parents=True, exist_ok=True)
    write_json(destination / "run_manifest.json", {"migration_signature": "fixture"})
    kwargs = {}
    if mode == "interrupted":
        original = checkpoints.save_checkpoint
        def interrupt(folder, payload, identity, **options):
            value = original(folder, payload, identity, **options)
            if payload["global_step"] == 2:
                raise InterruptedError("intentional complete-batch interruption")
            return value
        checkpoints.save_checkpoint = interrupt
        kwargs = {"checkpoint_folder": run / "checkpoints", "checkpoint_every_steps": 1}
    elif mode == "resume":
        kwargs = {"checkpoint_folder": run / "continued_checkpoints",
                  "resume_folder": run / "checkpoints", "checkpoint_every_steps": 1}
    elif mode == "resume_final":
        kwargs = {"checkpoint_folder": run / "final_checkpoints",
                  "resume_folder": run / "continued_checkpoints"}
    report = fit_towers(sample, content, config, destination, **kwargs)
    if mode == "resume_final":
        assert (run / "final_checkpoints/latest.json").exists()
    assert report["steps"] > 2
'''


def test_real_tower_complete_batch_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Interrupt after two optimizer steps, then compare exact fresh-process output."""
    (tmp_path / "checkpoint_worker.py").write_text(WORKER, encoding="utf-8")
    monkeypatch.setenv(
        "PYTHONPATH", str(tmp_path) + os.pathsep + os.environ["PYTHONPATH"]
    )
    run = tmp_path / "run"
    run.mkdir()
    config = Config.load()
    write_json(run / "config.resolved.json", config.values)
    write_json(run / "run_manifest.json", {"source_root": str(run / "inputs")})
    for stage in ("prepare", "baseline"):
        run_stage(stage, run, config, module="checkpoint_worker")
    with pytest.raises(RuntimeError, match="interrupted"):
        run_stage("interrupted", run, config, module="checkpoint_worker")
    assert read_json(run / "checkpoints/latest.json")["global_step"] == 2
    for stage in ("resume", "resume_final", "compare"):
        run_stage(stage, run, config, module="checkpoint_worker")
    assert read_json(run / "checkpoint_parity.json")["status"] == "passed"
