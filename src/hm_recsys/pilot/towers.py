"""Device-aware shared TwoTower training with SQL histories and epoch checkpoints."""

from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np
import torch

from hm_recsys.colab.checkpoints import load_checkpoint, save_checkpoint
from hm_recsys.colab.files import canonical_hash
from hm_recsys.contracts import (
    BASE_END,
    TT_END,
    TT_START,
    digest,
    read_json,
    write_json,
)
from hm_recsys.pilot.data import History, MappedContent
from hm_recsys.pilot.vectors import block_topk
from hm_recsys.towers import (
    ITEM_KEYS,
    PROFILE_KEYS,
    TowerEncoder,
    TwoTower,
    logq_loss,
    positive_mask,
)


def device_for(profile: dict) -> torch.device:
    """Resolve the requested actual device; CUDA never silently becomes CPU."""
    requested = profile["device"]
    if requested not in ("cpu", "cuda"):
        raise ValueError("Unsupported tower device")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no CPU fallback")
    if requested == "cuda":
        free, _ = torch.cuda.mem_get_info()
        if free < 2 * 1024**3:
            raise RuntimeError("blocked_resource: less than 2 GiB free CUDA memory")
        torch.cuda.reset_peak_memory_stats()
    return torch.device(requested)


class SqlEncoder(TowerEncoder):
    """Reuse tensor/category semantics while fetching at most 20 lines per event."""

    def __init__(
        self, history: History, content: MappedContent, maps: dict | None = None
    ) -> None:
        self.history_store = history
        self.content = content
        users = [
            r[0]
            for r in history.con.execute(
                "SELECT customer_id FROM cohort ORDER BY customer_id"
            ).fetchall()
        ]
        self.sample = history.sample(users)
        self.histories = {}
        if maps is None:
            legal = [
                r[0]
                for r in history.con.execute(
                    (
                        "SELECT DISTINCT article_id FROM transactions JOIN cohort "
                        "USING(customer_id) WHERE t_dat>=? AND t_dat<? ORDER BY "
                        "article_id"
                    ),
                    [TT_START, TT_END],
                ).fetchall()
            ]
            maps = {}
            for j, key in enumerate(ITEM_KEYS):
                values = sorted({self._item_values(i)[j] for i in legal})
                maps[key] = {v: k + 1 for k, v in enumerate(values)}
            legal_users = [
                r[0]
                for r in history.con.execute(
                    (
                        "SELECT DISTINCT customer_id FROM transactions JOIN cohort "
                        "USING(customer_id) WHERE t_dat>=? AND t_dat<? ORDER BY "
                        "customer_id"
                    ),
                    [TT_START, TT_END],
                ).fetchall()
            ]
            for j, key in enumerate(PROFILE_KEYS):
                values = sorted({self._profile_values(u)[j] for u in legal_users})
                maps[key] = {v: k + 1 for k, v in enumerate(values)}
        self.maps = maps

    def users(
        self, queries: list[tuple[str, date]]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fetch one bounded SQL event batch and reuse the original padding/weights."""
        self.sample.customers = self.history_store.customers(
            sorted({u for u, _ in queries})
        )
        histories = self.history_store.event_histories(queries)
        blocks = []
        for query, rows in zip(queries, histories, strict=True):
            self.histories = {query[0]: rows}
            blocks.append(super().users([query]))
        self.histories = {}
        return (
            torch.cat([b[0] for b in blocks]),
            torch.cat([b[1] for b in blocks]),
            torch.cat([b[2] for b in blocks]),
        )


def move(values: tuple, device: torch.device) -> tuple:
    """Transfer only bounded input tensors to the selected compute device."""
    return tuple(value.to(device) for value in values)


def item_vectors(
    model: TwoTower, encoder: SqlEncoder, path: Path, device: torch.device
) -> np.ndarray:
    """Encode the full catalog in batches into a derived memory-mapped array."""
    ids = encoder.sample.catalog
    path.parent.mkdir(parents=True, exist_ok=True)
    vectors = np.lib.format.open_memmap(
        path, mode="w+", dtype=np.float32, shape=(len(ids), 128)
    )
    model.eval()
    with torch.no_grad():
        for start in range(0, len(ids), 512):
            vectors[start : start + 512] = (
                model.item_vectors(
                    *move(encoder.items(ids[start : start + 512]), device)
                )
                .cpu()
                .numpy()
            )
    vectors.flush()
    return vectors


def internal_recall(
    model: TwoTower, encoder: SqlEncoder, run: Path, device: torch.device
) -> tuple[float, int]:
    """Daily full-truth cohort Recall@50, using the globally computed active pool."""
    history = encoder.history_store
    vectors = item_vectors(model, encoder, run / "tmp/internal_vectors.npy", device)
    total = 0.0
    count = 0
    for day_index in range((BASE_END - TT_END).days):
        from datetime import timedelta

        day = TT_END + timedelta(days=day_index)
        users = [
            r[0]
            for r in history.con.execute(
                (
                    "SELECT DISTINCT customer_id FROM transactions JOIN cohort "
                    "USING(customer_id) WHERE t_dat=? ORDER BY customer_id"
                ),
                [day],
            ).fetchall()
        ]
        active = history.active(day)
        for start in range(0, len(users), 32):
            queries = [(u, day) for u in users[start : start + 32]]
            inputs = encoder.users(queries)
            has_history = inputs[2].sum(dim=1).numpy() > 0
            with torch.no_grad():
                uv = model.user_vectors(*move(inputs, device)).cpu().numpy()
            predictions = block_topk(history.catalog, vectors, uv, active, 50)
            truths = history.positives(queries)
            for query, prediction, present in zip(
                queries, predictions, has_history, strict=True
            ):
                truth = truths[query]
                total += (
                    len({i for i, _ in prediction} & truth) / len(truth)
                    if present
                    else 0
                )
                count += 1
    if not count:
        raise ValueError("No internal-validation cohort queries")
    return total / count, count


def fit(run: Path, profile: dict, options: dict) -> None:
    """Train actual shared towers, select internally, and checkpoint full progress.

    The fixed selected-event order and Q match the existing sampler. A real epoch
    loop repeats that order; no gradient accumulation claims equivalence to 2048.
    Every completed epoch is evaluated without updating on internal labels.
    """
    device = device_for(profile)
    torch.manual_seed(42)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    history = History(run)
    content = MappedContent(run)
    encoder = SqlEncoder(history, content)
    config = profile["compute"]
    targets = history.targets(config["tt_max_targets"])
    if len(targets) < 2:
        raise ValueError("Insufficient legal targets")
    counts = Counter(i for _, _, i in targets)
    probabilities = {i: n / len(targets) for i, n in counts.items()}
    model = TwoTower(
        encoder.maps,
        content.vectors["text"].shape[1],
        content.vectors["image"].shape[1],
    ).to(device)
    initial = model.text.weight.detach().cpu().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    batches = (len(targets) + config["tt_batch_size"] - 1) // config["tt_batch_size"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, min(config["tt_max_steps"], batches * config["tt_epochs"])),
    )
    identity = {
        "run": read_json(run / "run_manifest.json")["migration_signature"],
        "profile": profile,
        "maps": encoder.maps,
        "targets": canonical_hash([(u, str(d), i) for u, d, i in targets]),
    }
    epoch = offset = step = 0
    losses: list[float] = []
    scores: list[float] = []
    best_score = -1.0
    best_epoch = 0
    best = None
    processed = 0
    internal_queries = 0
    checkpoint = Path(options["persistent_checkpoint"])
    if (checkpoint / "latest.json").is_file():
        state = load_checkpoint(checkpoint, identity)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        epoch, offset, step = (
            state["epoch"],
            state["next_target_offset"],
            state["global_step"],
        )
        losses, scores, best_score, best_epoch, best = (
            state["losses"],
            state["scores"],
            state["best_score"],
            state["best_epoch"],
            state["best_model"],
        )
        processed, internal_queries = state["processed"], state["internal_queries"]

    def checkpoint_state(phase: str) -> None:
        save_checkpoint(
            run / "checkpoints",
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "next_target_offset": offset,
                "global_step": step,
                "losses": losses,
                "scores": scores,
                "best_score": best_score,
                "best_epoch": best_epoch,
                "best_model": best,
                "processed": processed,
                "internal_queries": internal_queries,
                "checkpoint_suffix": f"e{epoch}-{phase}",
            },
            identity,
            persistent=checkpoint,
            mount_root=Path(options["mount_root"])
            if options.get("mount_root")
            else None,
        )

    try:
        while epoch < config["tt_epochs"] and (
            step < config["tt_max_steps"] or offset > 0
        ):
            model.train()
            for start in range(offset, len(targets), config["tt_batch_size"]):
                if step >= config["tt_max_steps"]:
                    break
                batch = targets[start : start + config["tt_batch_size"]]
                queries = [(u, d) for u, d, _ in batch]
                candidates = sorted({i for _, _, i in batch})
                mask = positive_mask(
                    queries, candidates, history.positives(queries)
                ).to(device)
                valid = (~mask).any(dim=1)
                offset = start + len(batch)
                if not valid.any():
                    continue
                users = model.user_vectors(*move(encoder.users(queries), device))[valid]
                items = model.item_vectors(*move(encoder.items(candidates), device))
                loss = logq_loss(
                    users,
                    items,
                    torch.tensor([probabilities[i] for i in candidates], device=device),
                    mask[valid],
                )
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite pilot tower loss")
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                step += 1
                processed += len(batch)
                losses.append(float(loss.detach().cpu()))
                if (
                    step == 1
                    or step % profile["checkpoint_every_steps"] == 0
                    or offset >= len(targets)
                    or step == config["tt_max_steps"]
                ):
                    checkpoint_state("batch")
            if not step:
                raise ValueError("No actual tower optimization steps")
            score, internal_queries = internal_recall(model, encoder, run, device)
            scores.append(score)
            if score > best_score:
                best_score, best_epoch = score, epoch + 1
                best = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
            epoch += 1
            offset = 0
            checkpoint_state("selected")
            if step >= config["tt_max_steps"]:
                break
        if best is None:
            raise ValueError("No selected internal checkpoint")
        model.load_state_dict(best)
        changed = float((model.text.weight.detach().cpu() - initial).abs().max())
        if changed <= 0:
            raise ValueError("No trained content projection")
        torch.save(best, run / "models/twotower.pt")
        item_vectors(model, encoder, run / "models/twotower_items.npy", device)
        write_json(run / "models/twotower_items.json", {"article_ids": history.catalog})
        # Ensure a final-batch/final-epoch restore also publishes local evidence.
        if not (run / "checkpoints/latest.json").exists():
            from hm_recsys.colab.files import atomic_json, verified_copy

            record = read_json(checkpoint / "latest.json")
            verified_copy(
                checkpoint / record["file"],
                run / "checkpoints" / record["file"],
                expected_sha256=record["sha256"],
            )
            atomic_json(run / "checkpoints/latest.json", record)
        if device.type == "cuda":
            torch.cuda.synchronize()
        write_json(
            run / "models/twotower.json",
            {
                "maps": encoder.maps,
                "text_dim": content.vectors["text"].shape[1],
                "image_dim": content.vectors["image"].shape[1],
                "output_dim": 128,
                "requested_device": profile["device"],
                "device": str(device),
                "steps": step,
                "epochs": epoch,
                "selected_epoch": best_epoch,
                "internal_daily_recall50": best_score,
                "internal_validation_queries": internal_queries,
                "epoch_recalls": scores,
                "losses": losses,
                "processed_targets": processed,
                "selected_targets": len(targets),
                "projection_max_parameter_change": changed,
                "sampler": (
                    "fixed SHA256 unique cohort events; repeat order per epoch; Q"
                    " from selected events only"
                ),
                "target_identity": identity["targets"],
                "precision": "float32_no_amp",
                "peak_cuda_allocated": torch.cuda.max_memory_allocated()
                if device.type == "cuda"
                else 0,
                "peak_cuda_reserved": torch.cuda.max_memory_reserved()
                if device.type == "cuda"
                else 0,
            },
        )
    finally:
        history.close()


def encode(run: Path, profile: dict, *, reload: bool = False) -> None:
    """Fresh-process inference on shared frozen maps/weights and current histories."""
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
    queries = []
    for name in ["Validation"] if reload else profile["folds"]:
        roster = read_json(run / "queries" / f"{name}.json")
        queries.extend(
            (u, date.fromisoformat(roster["start"])) for u in roster["users"]
        )
    vectors = np.empty((len(queries), 128), dtype=np.float32)
    has_history = []
    try:
        with torch.no_grad():
            for start in range(0, len(queries), 32):
                values = encoder.users(queries[start : start + 32])
                has_history.extend((values[2].sum(dim=1).numpy() > 0).tolist())
                vectors[start : start + 32] = (
                    model.user_vectors(*move(values, device)).cpu().numpy()
                )
        folder = run / ("predictions" if reload else "queries")
        name = "reloaded_query_vectors" if reload else "query_vectors"
        np.save(folder / f"{name}.npy", vectors, allow_pickle=False)
        write_json(
            folder / f"{name}.json",
            {
                "queries": [(u, str(d)) for u, d in queries],
                "has_history": has_history,
                "model_sha256": digest(run / "models/twotower.pt"),
                "device": str(device),
                "labels_read": False,
            },
        )
    finally:
        history.close()
