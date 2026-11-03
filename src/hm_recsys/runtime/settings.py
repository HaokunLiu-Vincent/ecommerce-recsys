"""Explicit compute backend and checkpoint cadence configuration.

Consumes a validated training profile and optional optimization configuration;
emits a copied profile with recorded execution settings. Uses existing JSON
contracts. Model objectives, target ordering, FP32 settings and training budgets
are not changed by optimization selection.
"""

from pathlib import Path

from hm_recsys.core.contracts import read_json


def apply(profile: dict, project: Path, name: str = "reference") -> dict:
    """Resolve a provenance-bearing overlay without changing execution profiles."""
    if name not in ("reference", "optimized"):
        raise ValueError("Unknown performance backend")
    # Select the explicitly configured compute path while preserving the same
    # model/data/metric contract.
    if name == "optimized":
        settings = read_json(project / "configs/compute_optimized.json")
        if set(settings) != {
            "backend",
            "prefetch_batches",
            "item_placement",
            "local_checkpoint_steps",
            "local_checkpoint_seconds",
            "remote_checkpoint_seconds",
        }:
            raise ValueError("Unknown performance setting")
        if (
            settings["backend"] != "optimized"
            or settings["prefetch_batches"] not in (0, 1, 2)
            or settings["item_placement"] not in ("auto", "host")
        ):
            raise ValueError("Invalid bounded input settings")
        if any(
            settings[k] <= 0
            for k in (
                "local_checkpoint_steps",
                "local_checkpoint_seconds",
                "remote_checkpoint_seconds",
            )
        ):
            raise ValueError("Checkpoint cadences must be positive")
        profile = {**profile, "performance": settings}
    return profile


# Complete-batch decisions; remote time advances only on verified success.
class Cadence:
    """Complete-batch decisions; remote time advances only on verified success."""

    #   init  .
    def __init__(self, settings: dict, now: float) -> None:
        self.settings = settings
        self.local_time = self.remote_time = now
        self.local_step = 0
        self.first_complete = True

    # due.
    def due(self, step: int, now: float, *, force: bool = False) -> tuple[bool, bool]:
        remote = (
            force
            or self.first_complete
            or step == 1
            or now - self.remote_time >= self.settings["remote_checkpoint_seconds"]
        )
        local = (
            remote
            or step - self.local_step >= self.settings["local_checkpoint_steps"]
            or now - self.local_time >= self.settings["local_checkpoint_seconds"]
        )
        return local, remote

    # completed.
    def completed(self, step: int, now: float, *, remote: bool) -> None:
        self.local_step = step
        self.local_time = now
        if remote:
            self.remote_time = now
            self.first_complete = False
