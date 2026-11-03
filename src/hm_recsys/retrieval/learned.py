"""Load frozen retrieval artifacts and score bounded learned-channel batches.

Inputs are canonical cutoff snapshots, immutable learned factors/vectors and
query encodings whose model/roster fingerprints are checked before use. Outputs
are channel-local exact inner-product results; absent representable histories
remain empty. No model fit, label lookup or content-index substitution occurs.
"""

from datetime import date
from pathlib import Path

import numpy as np

from hm_recsys.core.contracts import digest, read_json
from hm_recsys.retrieval.vectors import ActiveBlocks, BlockIndex, block_topk
from hm_recsys.retrieval.wals.model import WALS, confidence, fold_user, fold_users_batch


class TowerRetriever:
    """Load immutable full-catalog vectors without the smoke index constructor."""

    def __init__(
        self, run: Path, *, reloaded: bool = False, scope: str | None = None
    ) -> None:
        if scope not in (None, "final_test"):
            raise ValueError("Unknown query vector scope")
        folder = run / (scope or ("predictions" if reloaded else "queries"))
        name = "reloaded_query_vectors" if reloaded else "query_vectors"
        state = read_json(folder / f"{name}.json")
        if state["model_sha256"] != digest(run / "models/twotower.pt"):
            raise ValueError("Query-vector model mismatch")
        self.layout = state.get("layout", {})
        self.mapping = {
            (u, date.fromisoformat(d)): j
            for j, (u, d) in enumerate(state.get(("queries"), []))
        }
        self.vectors = np.load(
            folder / f"{name}.npy", mmap_mode="r", allow_pickle=False
        )
        self.has_history = (
            state["has_history"]
            if "has_history" in state
            else np.load(
                folder / f"{name}_history.npy", mmap_mode=("r"), allow_pickle=False
            )
        )
        self.index = BlockIndex.load(run / "indexes", "twotower")


def learned_batch(snapshot, users, wals, towers) -> dict:
    """Compose the two independently callable learned channel adapters."""
    wr = wals_batch(snapshot, users, wals)
    tr = tower_batch(snapshot, users, towers)
    return {user: {"wals": wr[user], "twotower": tr[user]} for user in users}


def wals_batch(snapshot, users: list[str], wals: WALS) -> dict:
    """Fold a bounded user batch once against fixed factors, then search exactly."""
    observed = [
        confidence(
            snapshot.by_user.get(user, []),
            cutoff=snapshot.cutoff,
            vocabulary=wals.mapping,
        )
        for user in users
    ]
    factors = (
        np.concatenate(
            [
                fold_users_batch(
                    wals.index.vectors, observed[start : start + 512], gram=wals.gram
                )
                for start in range(0, len(observed), 512)
            ],
            axis=0,
        )
        if snapshot.optimized and observed
        else np.zeros((0, wals.index.vectors.shape[1]), dtype=np.float32)
        if snapshot.optimized
        else np.stack(
            [fold_user(wals.index.vectors, row, gram=wals.gram) for row in observed]
        )
    )
    if snapshot.optimized:
        if not hasattr(snapshot, "_wals_search"):
            snapshot._wals_search = ActiveBlocks(wals.index, snapshot.active)
        results = snapshot._wals_search.search(factors, 50)
    else:
        results = block_topk(
            wals.index.ids, wals.index.vectors, factors, snapshot.active, 50
        )
    return {user: results[j] if observed[j] else [] for j, user in enumerate(users)}


def tower_batch(snapshot, users: list[str], tower: TowerRetriever) -> dict:
    """Search frozen learned catalog vectors while preserving empty histories."""
    rows = [tower.mapping[user, snapshot.before] for user in users]
    if snapshot.optimized:
        if not hasattr(snapshot, "_tower_search"):
            snapshot._tower_search = ActiveBlocks(tower.index, snapshot.active)
        results = snapshot._tower_search.search(tower.vectors[rows], 50)
    else:
        results = block_topk(
            tower.index.ids,
            tower.index.vectors,
            tower.vectors[rows],
            snapshot.active,
            50,
        )
    return {
        user: results[j] if tower.has_history[rows[j]] else []
        for j, user in enumerate(users)
    }
