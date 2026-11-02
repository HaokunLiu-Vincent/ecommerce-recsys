"""Isolated actual CUDA forward/backward/update/reload probe; no CPU fallback."""

import platform
from pathlib import Path

from hm_recsys.colab.files import atomic_json


def probe(output: Path, device: str) -> dict:
    """Exercise the shared TwoTower architecture on bounded synthetic tensors."""
    import torch

    from hm_recsys.towers import ITEM_KEYS, PROFILE_KEYS, TwoTower, logq_loss

    if device not in ("cpu", "cuda"):
        raise ValueError("Only explicit cpu/cuda devices are supported")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable in the project interpreter")
    torch.set_num_threads(2)
    torch.manual_seed(42)
    if device == "cuda":
        free, total = torch.cuda.mem_get_info()
        if free < 2 * 1024**3:
            raise RuntimeError("blocked_resource: CUDA probe needs 2 GiB free")
        torch.cuda.reset_peak_memory_stats()
    else:
        free = total = 0
    maps = {
        key: {str(i): i + 1 for i in range(4)} for key in (*ITEM_KEYS, *PROFILE_KEYS)
    }
    model = TwoTower(maps, 12, 16).to(device)
    original = model.text.weight.detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    codes = torch.tensor([[i] * 6 for i in range(1, 5)], device=device)
    profiles = torch.ones((4, 3), dtype=torch.long, device=device)
    histories = torch.ones((4, 20, 3), dtype=torch.long, device=device)
    weights = torch.ones((4, 20), device=device)
    text = torch.randn(4, 12, device=device)
    image = torch.randn(4, 16, device=device)
    masks = torch.ones(4, 2, device=device)
    loss = logq_loss(
        model.user_vectors(profiles, histories, weights),
        model.item_vectors(codes, text, image, masks),
        torch.full((4,), 0.25, device=device),
        torch.eye(4, dtype=torch.bool, device=device),
    )
    if not torch.isfinite(loss):
        raise ValueError("Nonfinite device probe loss")
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    changed = float((model.text.weight.detach() - original).abs().max())
    if changed <= 0:
        raise ValueError("Device probe did not update shared tower parameters")
    output.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output / "probe_weights.pt")
    model.eval()
    with torch.no_grad():
        expected = model.item_vectors(codes, text, image, masks)
        restored = TwoTower(maps, 12, 16).to(device)
        restored.load_state_dict(
            torch.load(
                output / "probe_weights.pt", map_location=device, weights_only=True
            )
        )
        restored.eval()
        torch.testing.assert_close(
            expected,
            restored.item_vectors(codes, text, image, masks),
            rtol=1e-5,
            atol=1e-6,
        )
    if device == "cuda":
        torch.cuda.synchronize()
    result = {
        "status": "passed",
        "synthetic_only": True,
        "device": device,
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "cuda_runtime": torch.version.cuda,
        "optimizer_steps": 1,
        "parameter_change": changed,
        "loss": float(loss.detach().cpu()),
        "same_artifact_reload": "passed",
        "free_vram_before": free,
        "total_vram": total,
        "device_name": torch.cuda.get_device_name() if device == "cuda" else "CPU",
        "peak_allocated": torch.cuda.max_memory_allocated() if device == "cuda" else 0,
        "peak_reserved": torch.cuda.max_memory_reserved() if device == "cuda" else 0,
    }
    atomic_json(output / "device_probe.json", result)
    return result
