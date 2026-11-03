"""Verify complete-batch continuation in the canonical production tower trainer.

Only generated 24-user/170-item inputs are used. Full compute settings remain
unchanged; each trainer invocation runs in a fresh native-library process.
Interrupted and uninterrupted runs must preserve weights, losses, selection,
learned vectors and complete final-checkpoint reload behavior exactly on CPU.
"""

import os
import shutil
from pathlib import Path

import pytest

from hm_recsys.core.contracts import ROOT, Config, read_json
from hm_recsys.runtime.files import atomic_json, stage_inputs
from hm_recsys.runtime.resources import run_stage
from hm_recsys.runtime.testing import create_full_source, execute_full_fixture

WORKER = '''"""Train generated canonical fixtures in isolated CPU processes."""
import sys
from pathlib import Path
import numpy as np
import torch
from hm_recsys.core.contracts import read_json, write_json
from hm_recsys.data.history import History
from hm_recsys.data.streaming import TargetStream
from hm_recsys.features.content import MappedContent
from hm_recsys.retrieval.two_tower.inputs import SqlEncoder
from hm_recsys.retrieval.two_tower.prepared import prepare
from hm_recsys.retrieval.two_tower import training

mode, location = sys.argv[1:]
run = Path(location)
options = read_json(run / 'colab_stage.json')
profile = read_json(run / 'profile.resolved.json')
assert profile['compute']['tt_batch_size'] == 512
assert profile['compute']['tt_epochs'] == 10
assert profile['compute']['tt_max_targets'] is None
assert profile['compute']['tt_max_steps'] is None
if mode in ('compare', 'compare_final'):
    baseline = Path(options['baseline'])
    left = torch.load(baseline / 'models/twotower.pt', weights_only=True)
    right = torch.load(run / 'models/twotower.pt', weights_only=True)
    for key in left:
        torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)
    for key in ('steps', 'epochs', 'losses', 'epoch_recalls', 'selected_epoch'):
        assert read_json(baseline / 'models/twotower.json')[key] == read_json(
            run / 'models/twotower.json')[key]
    np.testing.assert_array_equal(np.load(baseline / 'models/twotower_items.npy'),
                                  np.load(run / 'models/twotower_items.npy'))
    write_json(run / 'checkpoint_parity.json', {'status': 'passed'})
else:
    if profile['performance']['backend'] == 'optimized' and not (
        run / 'prepared/tower/manifest.json').exists():
        history = History(run)
        try:
            encoder = SqlEncoder(history, MappedContent(run))
            prepare(run, encoder, TargetStream(history, None))
        finally:
            history.close()
    if mode == 'interrupted':
        original = training.save_checkpoint
        def interrupt(folder, payload, identity, **kwargs):
            result = original(folder, payload, identity, **kwargs)
            if payload['global_step'] == options['interrupt_step']:
                raise InterruptedError('intentional complete-batch interruption')
            return result
        training.save_checkpoint = interrupt
    training.fit(run, profile, options)
'''


@pytest.mark.parametrize("backend", ["reference", "optimized"])
def test_full_checkpoint_exact_continuation(
    tmp_path: Path, monkeypatch, backend: str
) -> None:
    """Restore a real full-profile tiny-input update and the final selected state."""
    source, data = tmp_path / "source", tmp_path / "data"
    create_full_source(source)
    stage_inputs(source, data)
    persistent = tmp_path / "drive/full"
    persistent.parent.mkdir()
    local = data / "runs/full"
    execute_full_fixture(
        ROOT,
        data,
        local,
        persistent,
        until="prepare",
        allow_full=True,
        performance=backend,
    )
    artifacts = read_json(local / "artifacts.json")
    baseline, resumed = tmp_path / "baseline", tmp_path / "resumed"
    for target in (baseline, resumed):
        for name, relative in artifacts.items():
            destination = target / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(local / relative, destination)
        for folder in ("models", "tmp"):
            (target / folder).mkdir(exist_ok=True)
        profile = read_json(target / "profile.resolved.json")
        profile.setdefault("performance", {"backend": backend})
        profile["performance"]["local_checkpoint_steps"] = 1
        profile["checkpoint_every_steps"] = 1
        atomic_json(target / "profile.resolved.json", profile)
        options = read_json(target / "colab_stage.json")
        options.update(
            persistent_checkpoint=str(tmp_path / (target.name + "_remote")),
            baseline=str(baseline),
            local_checkpoint=str(target / "checkpoints"),
            interrupt_step=2,
        )
        atomic_json(target / "colab_stage.json", options)
    (tmp_path / "canonical_checkpoint_worker.py").write_text(WORKER, encoding="utf-8")
    monkeypatch.setenv(
        "PYTHONPATH",
        str(tmp_path) + os.pathsep + os.environ.get("PYTHONPATH", str(ROOT / "src")),
    )
    policy = Config.load()
    run_stage("baseline", baseline, policy, module="canonical_checkpoint_worker")
    with pytest.raises(RuntimeError, match="interrupted"):
        run_stage("interrupted", resumed, policy, module="canonical_checkpoint_worker")
    boundary = read_json(resumed / "checkpoints/latest.json")
    assert boundary["global_step"] == 2
    assert boundary["boundary"] == "complete_batch"
    for mode in ("resume", "compare"):
        run_stage(mode, resumed, policy, module="canonical_checkpoint_worker")
    assert read_json(resumed / "checkpoint_parity.json")["status"] == "passed"
    before = read_json(resumed / "models/twotower.json")
    run_stage("resume_final", resumed, policy, module="canonical_checkpoint_worker")
    run_stage("compare_final", resumed, policy, module="canonical_checkpoint_worker")
    assert read_json(resumed / "models/twotower.json")["steps"] == before["steps"]
    assert len(list((tmp_path / "resumed_remote").glob("*.pt"))) == 1
