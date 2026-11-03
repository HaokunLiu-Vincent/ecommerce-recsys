"""Validate canonical complete-state checkpoint contents using an actual tower update.

A tiny real Two-Tower update initializes optimizer and scheduler state. The
checkpoint API must restore those tensors and Python/NumPy/Torch RNG streams,
reject a different identity, and preserve identical frozen inference outputs.
This unit test does not claim Colab, Drive or full-data execution.
"""

import random
from pathlib import Path

import numpy as np
import pytest
import torch

from hm_recsys.retrieval.two_tower.model import (
    ITEM_KEYS,
    PROFILE_KEYS,
    TwoTower,
    logq_loss,
)
from hm_recsys.runtime.checkpoints import load_checkpoint, save_checkpoint


def trained_state() -> tuple[TwoTower, dict, tuple]:
    """Return complete state after one genuine multimodal CPU optimization step."""
    torch.manual_seed(42)
    torch.set_num_threads(2)
    maps = {key: {"first": 1, "second": 2} for key in (*ITEM_KEYS, *PROFILE_KEYS)}
    model = TwoTower(maps, 4, 3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)
    profiles = torch.ones((2, 3), dtype=torch.long)
    histories = torch.ones((2, 20, 3), dtype=torch.long)
    weights = torch.ones((2, 20))
    codes = torch.tensor([[1] * 6, [2] * 6])
    text = torch.arange(8, dtype=torch.float32).reshape(2, 4) / 8
    image = torch.arange(6, dtype=torch.float32).reshape(2, 3) / 6
    masks = torch.ones((2, 2))
    loss = logq_loss(
        model.user_vectors(profiles, histories, weights),
        model.item_vectors(codes, text, image, masks),
        torch.tensor([0.5, 0.5]),
        torch.eye(2, dtype=torch.bool),
    )
    loss.backward()
    optimizer.step()
    scheduler.step()
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": 0,
        "global_step": 1,
        "next_target_offset": 2,
        "losses": [float(loss.detach())],
    }
    return (
        model,
        payload,
        (maps, profiles, histories, weights, codes, text, image, masks),
    )


def test_complete_model_optimizer_scheduler_and_rng_restore(tmp_path: Path) -> None:
    """Restore one genuine update, all random streams and exact eval representations."""
    model, payload, inputs = trained_state()
    maps, profiles, histories, weights, codes, text, image, masks = inputs
    identity = {"fixture": "canonical-model-update", "batch_size": 2}
    save_checkpoint(tmp_path, payload, identity)
    expected = (random.random(), float(np.random.random()), torch.rand(3))
    state = load_checkpoint(tmp_path, identity)
    assert random.random() == expected[0]
    assert float(np.random.random()) == expected[1]
    torch.testing.assert_close(torch.rand(3), expected[2], rtol=0, atol=0)
    restored = TwoTower(maps, 4, 3)
    restored.load_state_dict(state["model"])
    optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    optimizer.load_state_dict(state["optimizer"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)
    scheduler.load_state_dict(state["scheduler"])
    assert optimizer.state and scheduler.last_epoch == 1
    model.eval()
    restored.eval()
    with torch.no_grad():
        torch.testing.assert_close(
            model.user_vectors(profiles, histories, weights),
            restored.user_vectors(profiles, histories, weights),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            model.item_vectors(codes, text, image, masks),
            restored.item_vectors(codes, text, image, masks),
            rtol=0,
            atol=0,
        )
    with pytest.raises(ValueError, match="identity"):
        load_checkpoint(tmp_path, {"fixture": "different"})
