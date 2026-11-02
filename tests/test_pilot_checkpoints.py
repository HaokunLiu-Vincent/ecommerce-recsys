"""Fresh-process mid-epoch continuation of the SQL-backed shared tower."""

import os
import shutil
from pathlib import Path

import pytest

from hm_recsys.colab.files import stage_inputs
from hm_recsys.contracts import ROOT, Config, read_json
from hm_recsys.pilot.testing import create_source, execute_fixture
from hm_recsys.resources import run_stage

WORKER = '''"""Test-only bounded worker with an intentional checkpoint interruption."""
import sys
from pathlib import Path
import numpy as np
import torch
from hm_recsys.contracts import read_json, write_json
from hm_recsys.pilot.testing import assert_fixture
from hm_recsys.pilot import towers

mode, location = sys.argv[1:]
run = Path(location)
options = read_json(run / 'colab_stage.json')
assert_fixture(Path(options['data_root']))
profile = read_json(run / 'profile.resolved.json')
if mode == 'compare':
    baseline = Path(options['baseline'])
    a = torch.load(baseline / 'models/twotower.pt', weights_only=True)
    b = torch.load(run / 'models/twotower.pt', weights_only=True)
    for key in a:
        torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
    for key in ('steps', 'epochs', 'losses', 'epoch_recalls', 'selected_epoch'):
        assert read_json(baseline / 'models/twotower.json')[key] == read_json(
            run / 'models/twotower.json')[key]
    np.testing.assert_array_equal(
        np.load(baseline / 'models/twotower_items.npy'),
        np.load(run / 'models/twotower_items.npy'))
    write_json(run / 'pilot_checkpoint_parity.json', {'status': 'passed'})
else:
    if mode == 'interrupted':
        original = towers.save_checkpoint
        def interrupt(folder, payload, identity, **kwargs):
            path = original(folder, payload, identity, **kwargs)
            if payload['global_step'] == 2:
                raise InterruptedError('intentional pilot mid-epoch interruption')
            return path
        towers.save_checkpoint = interrupt
    towers.fit(run, profile, options)
'''


def test_pilot_epoch_checkpoint_continuation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Train two real epochs uninterrupted and after an identical step-2 restart."""
    from hm_recsys.colab.files import atomic_json

    source = tmp_path / "source"
    create_source(source)
    data = tmp_path / "data"
    stage_inputs(source, data)
    persistent = tmp_path / "drive/run"
    persistent.parent.mkdir()
    local = data / "runs/pilot"
    execute_fixture(ROOT, data, local, persistent, until="rosters")
    locations = read_json(local / "artifacts.json")
    base, resumed = tmp_path / "baseline", tmp_path / "resumed"
    for target in (base, resumed):
        for name, relative in locations.items():
            path = target / name
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(local / relative, path)
        for folder in ("models", "tmp"):
            (target / folder).mkdir(exist_ok=True)
        options = read_json(target / "colab_stage.json")
        options.update(
            persistent_checkpoint=str(tmp_path / (target.name + "_checkpoint")),
            baseline=str(base),
        )
        atomic_json(target / "colab_stage.json", options)
    (tmp_path / "pilot_checkpoint_worker.py").write_text(WORKER, encoding="utf-8")
    monkeypatch.setenv(
        "PYTHONPATH", str(tmp_path) + os.pathsep + os.environ["PYTHONPATH"]
    )
    policy = Config.load()
    run_stage("baseline", base, policy, module="pilot_checkpoint_worker")
    with pytest.raises(RuntimeError, match="interrupted"):
        run_stage("interrupted", resumed, policy, module="pilot_checkpoint_worker")
    for stage in ("resume", "compare"):
        run_stage(stage, resumed, policy, module="pilot_checkpoint_worker")
    assert read_json(resumed / "pilot_checkpoint_parity.json")["status"] == "passed"
