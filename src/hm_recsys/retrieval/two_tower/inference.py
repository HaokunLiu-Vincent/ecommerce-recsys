"""Encode frozen Two-Tower catalog and cutoff-query vectors for native retrieval.

Inputs are selected weights, checked content/maps and independent fold rosters.
Encoding uses bounded [batch,128] output arrays, retains zero-history queries and
checks checkpoint/model identities. This entry point writes derived query arrays
only; it never reads outer future labels, updates weights or loads a source text
or image FAISS index as the learned index. Reload uses the same tensor schema.
"""

from datetime import date
from pathlib import Path

import numpy as np
import torch

from hm_recsys.core.contracts import (
    digest,
    read_json,
    write_json,
)
from hm_recsys.core.progress import emit
from hm_recsys.data.history import History
from hm_recsys.data.streaming import Partition
from hm_recsys.features.content import MappedContent
from hm_recsys.retrieval.two_tower.inputs import SqlEncoder
from hm_recsys.retrieval.two_tower.model import TwoTower
from hm_recsys.retrieval.two_tower.training import device_for, move


def encode(
    run: Path,
    profile: dict,
    *,
    reload: bool = False,
    reloaded: bool | None = None,
    partitions: list[str] | None = None,
    output_scope: str | None = None,
) -> None:
    """Fresh-process inference on shared frozen maps/weights and current histories."""
    if reloaded is not None:
        reload = reloaded
    device = device_for(profile)
    torch.set_num_threads(2)
    history = History(run)
    state = read_json(run / "models/twotower.json")
    encoder = SqlEncoder(history, MappedContent(run), state["maps"])
    model = TwoTower(state["maps"], state["text_dim"], state["image_dim"]).to(device)
    model.load_state_dict(
        torch.load(run / "models/twotower.pt", map_location=device, weights_only=True)
    )
    model.eval()
    names = (
        partitions
        if partitions is not None
        else (["Validation"] if reload else profile["folds"])
    )
    readers = {name: Partition(run, name, labels=False) for name in names}
    count = sum(p.count for p in readers.values())
    scope = output_scope or profile.get("prediction_scope")
    if scope is not None and scope != "final_test":
        raise ValueError("Unknown prediction output scope")
    if names == ["Test"] and scope != "final_test":
        raise ValueError("Test vectors require the isolated final_test output scope")
    folder = run / (scope or ("predictions" if reload else "queries"))
    folder.mkdir(parents=True, exist_ok=True)
    name = "reloaded_query_vectors" if reload else "query_vectors"
    if (folder / f"{name}.npy").exists():
        raise ValueError("Query vectors exist; reuse their verified completed stage")
    vectors = np.lib.format.open_memmap(
        folder / f"{name}.npy", mode="w+", dtype=np.float32, shape=(count, 128)
    )
    has_history = np.lib.format.open_memmap(
        folder / f"{name}_history.npy", mode="w+", dtype=np.bool_, shape=(count,)
    )
    layout = {}
    offset = 0
    legacy_queries = []
    emit(
        "tower_query_encoding",
        0,
        count,
        "queries",
        metrics={"representation_width": 128},
        status=("started"),
        force=True,
    )
    try:
        with torch.no_grad():
            for part_name, part in readers.items():
                layout[part_name] = {
                    "offset": offset,
                    "count": part.count,
                    ("roster_sha256"): digest(run / ("queries") / f"{part_name}.json"),
                }
                optimized = profile.get("performance", {}).get("backend") == "optimized"
                prepared_queries = None
                if optimized:
                    from hm_recsys.retrieval.two_tower.query_inputs import QueryInputs

                    prepared_queries = QueryInputs(encoder, part)
                part_offset = 0
                for batch in part.batches(512 if optimized else 32):
                    queries = [
                        (u, date.fromisoformat(part.meta["start"])) for u in batch
                    ]
                    values = (
                        prepared_queries.users(part_offset, len(batch))
                        if prepared_queries
                        else encoder.users(queries)
                    )
                    part_offset += len(batch)
                    has_history[offset : offset + len(batch)] = (
                        values[2].sum(dim=1).numpy() > 0
                    )
                    vectors[offset : offset + len(batch)] = (
                        model.user_vectors(*move(values, device)).cpu().numpy()
                    )
                    if profile["profile"] == "pilot":
                        legacy_queries.extend(((u, str(d)) for u, d in queries))
                    offset += len(batch)
                    emit(
                        "tower_query_encoding",
                        offset,
                        count,
                        "queries",
                        metrics={
                            ("partition"): part_name,
                            ("partition_queries_encoded"): part_offset,
                            ("partition_queries"): part.count,
                        },
                    )
        emit(
            "tower_query_encoding",
            offset,
            count,
            "queries",
            status="complete",
            force=True,
        )
        vectors.flush()
        has_history.flush()
        state = {
            "layout": layout,
            "model_sha256": digest(run / "models/twotower.pt"),
            ("device"): str(device),
            ("labels_read"): False,
        }
        if profile["profile"] == "pilot":
            state.update(queries=legacy_queries, has_history=has_history.tolist())
        write_json(folder / f"{name}.json", state)
    finally:
        history.close()
