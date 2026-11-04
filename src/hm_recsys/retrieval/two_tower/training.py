"""Train the shared production Two-Tower and select a frozen internal checkpoint.

Both smoke and full use this trainer with explicit population/resource settings.
Targets retain fixed SHA256 order and batch boundaries; complete daily positives
feed the unchanged multi-positive LogQ loss. Each epoch encodes the complete
candidate catalog and evaluates only the disjoint internal date window. Complete
model/optimizer/scheduler/RNG state is persisted at verified batch boundaries.
CUDA requests fail explicitly; raw input banks never freeze trainable branches.
"""

import gc
import time
from pathlib import Path

import numpy as np
import torch

from hm_recsys.core.contracts import (
    BASE_END,
    TT_END,
    read_json,
    write_json,
)
from hm_recsys.core.progress import emit
from hm_recsys.data.history import History
from hm_recsys.data.streaming import TargetStream, export_query, rows
from hm_recsys.evaluation.metrics import recall_at_k
from hm_recsys.features.content import MappedContent
from hm_recsys.retrieval.two_tower.execution import ItemBank, Measurements, prefetch
from hm_recsys.retrieval.two_tower.inputs import SqlEncoder
from hm_recsys.retrieval.two_tower.model import TwoTower, logq_loss, positive_mask
from hm_recsys.retrieval.two_tower.prepared import Prepared
from hm_recsys.retrieval.two_tower.prepared import identity as cache_identity
from hm_recsys.runtime.checkpoints import load_checkpoint, save_checkpoint
from hm_recsys.runtime.settings import Cadence


def device_for(profile: dict) -> torch.device:
    """Resolve the requested actual device; CUDA never silently becomes CPU."""
    requested = profile["device"]
    if requested not in ("cpu", "cuda"):
        raise ValueError("Unsupported tower device")
    if requested == "cuda" and (not torch.cuda.is_available()):
        raise RuntimeError("CUDA requested but unavailable; no CPU fallback")
    if requested == "cuda":
        free, _ = torch.cuda.mem_get_info()
        if free < 2 * 1024**3:
            raise RuntimeError("blocked_resource: less than 2 GiB free CUDA memory")
        torch.cuda.reset_peak_memory_stats()
    return torch.device(requested)


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
    emit(
        "tower_catalog",
        0,
        len(ids),
        "items",
        metrics={"representation_width": 128},
        status=("started"),
        force=True,
    )
    step = min(4096, encoder.bank.max_batch) if encoder.bank else 4096
    model.eval()
    with torch.no_grad():
        for start in range(0, len(ids), step):
            end = min(start + step, len(ids))
            vectors[start:end] = (
                model.item_vectors(
                    *(
                        encoder.bank.items(np.arange(start, end, dtype=np.int64))
                        if encoder.bank
                        else move(encoder.items(ids[start:end]), device)
                    )
                )
                .cpu()
                .numpy()
            )
            emit("tower_catalog", end, len(ids), "items")
    emit("tower_catalog", len(ids), len(ids), "items", status="complete", force=True)
    vectors.flush()
    return vectors


def _gpu_topk_indices(
    uv: torch.Tensor,
    active_tensor: torch.Tensor,
    active_idx: np.ndarray,
    k: int = 50,
) -> np.ndarray:
    """Compute exact Top-K inner-product indices on device with deterministic ties."""
    if len(active_idx) == 0 or uv.shape[0] == 0:
        return np.empty((uv.shape[0], 0), dtype=np.int64)
    k_eff = min(k, len(active_idx))
    scores = torch.matmul(uv, active_tensor.T)
    _, topk_pos = torch.topk(scores, k=k_eff, dim=1, largest=True, sorted=True)
    return active_idx[topk_pos.cpu().numpy()]


def internal_recall(
    model: TwoTower, encoder: SqlEncoder, run: Path, device: torch.device
) -> tuple[float, int]:
    """Daily full-truth cohort Recall@50, vectorized via torch.matmul + torch.topk."""
    history = encoder.history_store
    vectors = item_vectors(model, encoder, run / "tmp/internal_vectors.npy", device)
    item_tensor = torch.as_tensor(
        np.asarray(vectors), dtype=torch.float32, device=device
    )
    cat_to_idx = {aid: idx for idx, aid in enumerate(history.catalog)}
    emit(
        "tower_internal_validation",
        0,
        None,
        "queries",
        message=("Complete internal Recall@50 population"),
        force=True,
    )
    total = 0.0
    count = 0
    for day_index in range((BASE_END - TT_END).days):
        from datetime import timedelta

        day = TT_END + timedelta(days=day_index)
        path = run / "tmp/internal_queries.parquet"
        export_query(
            history.con,
            (
                """SELECT DISTINCT customer_id FROM transactions JOIN cohort
                USING(customer_id) WHERE t_dat=? ORDER BY customer_id"""
            ),
            [day],
            path,
        )
        active = history.active(day)
        active_idx = np.array(
            sorted(cat_to_idx[aid] for aid in active if aid in cat_to_idx),
            dtype=np.int64,
        )
        active_tensor = (
            item_tensor[torch.as_tensor(active_idx, dtype=torch.long, device=device)]
            if len(active_idx) > 0
            else torch.empty((0, 128), dtype=torch.float32, device=device)
        )
        for batch in rows(path, 512):
            queries = [(r["customer_id"], day) for r in batch]
            inputs = encoder.users(queries)
            has_history = inputs[2].sum(dim=1).numpy() > 0
            with torch.no_grad():
                uv = model.user_vectors(*move(inputs, device))
                topk_idx = _gpu_topk_indices(uv, active_tensor, active_idx, 50)
            truths = history.positives(queries)
            for query, pred_row, present in zip(
                queries, topk_idx, has_history, strict=True
            ):
                truth = truths[query]
                total += (
                    recall_at_k(
                        [history.catalog[j] for j in pred_row],
                        truth,
                        50,
                    )
                    if present
                    else 0.0
                )
                count += 1
            emit("tower_internal_validation", count, None, "queries")
    del item_tensor
    if not count:
        raise ValueError("No internal-validation cohort queries")
    emit(
        "tower_internal_validation",
        count,
        None,
        "queries",
        status="complete",
        metrics={("internal_recall50"): total / count},
        force=True,
    )
    return (total / count, count)


def cached_internal_recall(model, encoder, prepared, run, device, timing):
    """Same complete daily population; GPU vectorized via torch.matmul + torch.topk."""
    with timing.span("internal_catalog_encoding"):
        vectors = item_vectors(model, encoder, run / "tmp/internal_vectors.npy", device)
    item_tensor = torch.as_tensor(
        np.asarray(vectors), dtype=torch.float32, device=device
    )
    import pyarrow.parquet as pq

    internal_total = sum(
        pq.ParquetFile(prepared.folder / m["file"]).metadata.num_rows
        for m in prepared.meta[("daily")].values()
    )
    emit(
        "tower_internal_validation",
        0,
        internal_total,
        "queries",
        status="started",
        force=True,
    )
    total = 0.0
    count = 0
    eval_batch = min(4096, getattr(prepared, "max_batch", 512))
    offsets = prepared.arrays["positive_offsets"]
    positive = prepared.arrays["positive_items"]
    for meta in prepared.meta["daily"].values():
        active_idx = np.sort(np.asarray(meta["active"], dtype=np.int64))
        active_tensor = (
            item_tensor[torch.as_tensor(active_idx, dtype=torch.long, device=device)]
            if len(active_idx) > 0
            else torch.empty((0, 128), dtype=torch.float32, device=device)
        )
        for records in rows(prepared.folder / meta["file"], eval_batch):
            contexts = np.array([r["context_id"] for r in records], dtype=np.int64)
            inputs = prepared.users(contexts)
            present = inputs[2].sum(dim=1).numpy() > 0
            with torch.no_grad():
                uv = model.user_vectors(*encoder.bank.transfer("users", inputs))
                with timing.span("internal_retrieval"):
                    topk_idx = _gpu_topk_indices(uv, active_tensor, active_idx, 50)
            for context, pred_row, has in zip(contexts, topk_idx, present, strict=True):
                truth = {
                    prepared.catalog[j]
                    for j in positive[offsets[context] : offsets[context + 1]]
                }
                total += (
                    recall_at_k([prepared.catalog[j] for j in pred_row], truth, 50)
                    if has
                    else 0.0
                )
                count += 1
            emit("tower_internal_validation", count, internal_total, "queries")
    del item_tensor
    if not count:
        raise ValueError("No internal-validation queries")
    emit(
        "tower_internal_validation",
        count,
        internal_total,
        "queries",
        status=("complete"),
        metrics={("internal_recall50"): total / count},
        force=True,
    )
    return (total / count, count)


def fit(run: Path, profile: dict, options: dict) -> None:
    """Train actual shared towers, select internally, and checkpoint full progress.

    The fixed selected-event order and Q match the existing sampler. A real epoch
    loop repeats that order; no gradient accumulation claims equivalence to 2048.
    Every completed epoch is evaluated without updating on internal labels.
    """
    (run / "models").mkdir(parents=True, exist_ok=True)
    device = device_for(profile)
    torch.manual_seed(42)
    config = profile["compute"]
    torch.set_num_threads(max(2, int(config.get("threads", 12))))
    torch.set_num_interop_threads(max(1, int(config.get("interop_threads", 2))))
    history = History(run)
    content = MappedContent(run)
    performance = profile.get("performance", {})
    optimized = performance.get("backend") == "optimized"
    max_batch = max(512, int(config["tt_batch_size"]), 8192)
    if optimized:
        meta = read_json(run / "prepared/tower/manifest.json")
        encoder = SqlEncoder(history, content, meta["maps"])
        targets = Prepared(
            run / "prepared/tower",
            {
                **cache_identity(run, encoder.maps, meta[("identity")][("target")]),
                ("include_internal"): True,
            },
            max_batch=max_batch,
        )
    else:
        encoder = SqlEncoder(history, content)
        targets = TargetStream(history, config["tt_max_targets"])
    timing = Measurements(device)
    encoder.measurements = timing
    bank = (
        ItemBank(
            encoder,
            device,
            placement=performance.get("item_placement", "auto"),
            max_batch=max_batch,
        )
        if optimized
        else None
    )
    encoder.bank = bank
    cadence = Cadence(performance, time.perf_counter()) if optimized else None
    probabilities = targets.probabilities
    target_count = targets.count
    model = TwoTower(
        encoder.maps,
        content.vectors["text"].shape[1],
        content.vectors[("image")].shape[1],
    )
    from hm_recsys.runtime.capacity import check

    parameter_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    check(
        run,
        "twotower",
        memory=parameter_bytes * 12 + 512 * 1024**2,
        disk=parameter_bytes * 24 + len(history.catalog) * 128 * 4 * 6,
        details={
            ("parameter_bytes"): parameter_bytes,
            ("checkpoint_retention"): ("latest complete state including best model")
            if profile[("profile")] == ("full")
            else ("all pilot checkpoints"),
        },
    )
    if device.type == "cuda":
        free, _ = torch.cuda.mem_get_info()
        if parameter_bytes * 12 + 512 * 1024**2 > free:
            raise RuntimeError(
                "blocked_resource: tower optimizer and batch CUDA estimate"
            )
    model = model.to(device)
    initial = model.text.weight.detach().cpu().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    batches = (target_count + config["tt_batch_size"] - 1) // config["tt_batch_size"]
    step_limit = config["tt_max_steps"]
    scheduling_steps = batches * config["tt_epochs"]
    if step_limit is not None:
        scheduling_steps = min(step_limit, scheduling_steps)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, scheduling_steps)
    )
    identity = {
        "run": read_json(run / "run_manifest.json")["migration_signature"],
        ("profile"): profile,
        ("maps"): encoder.maps,
        ("targets"): targets.identity,
    }
    epoch = offset = step = 0
    losses: list[float] = []
    progress = (
        np.lib.format.open_memmap(
            run / "tmp/loss_progress.npy",
            mode="w+",
            dtype=np.float64,
            shape=(scheduling_steps,),
        )
        if optimized
        else None
    )
    scores: list[float] = []
    best_score = -1.0
    best_epoch = 0
    best = None
    processed = 0
    internal_queries = 0
    checkpoint = Path(options["persistent_checkpoint"])
    local_checkpoint = (
        Path(options.get("local_checkpoint", run / "checkpoints"))
        if optimized
        else run / ("checkpoints")
    )
    recovery = (
        local_checkpoint if (local_checkpoint / "latest.json").is_file() else checkpoint
    )
    if (recovery / "latest.json").is_file():
        state = load_checkpoint(recovery, identity)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        epoch, offset, step = (
            state["epoch"],
            state["next_target_offset"],
            state[("global_step")],
        )
        losses, scores, best_score, best_epoch, best = (
            state["losses"],
            state["scores"],
            state[("best_score")],
            state[("best_epoch")],
            state[("best_model")],
        )
        processed, internal_queries = (state["processed"], state["internal_queries"])
        if progress is not None:
            progress[:step] = np.asarray(losses)
            losses = []

    def checkpoint_state(phase: str, *, remote: bool = True) -> None:
        emit(
            "tower_checkpoint",
            step,
            None,
            "updates",
            message=("Saving complete continuation state"),
            metrics={
                ("remote_requested"): remote,
                ("epoch"): epoch,
                ("consumed_target_offset"): offset,
            },
            force=True,
        )
        save_checkpoint(
            local_checkpoint,
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                ("scheduler"): scheduler.state_dict(),
                ("epoch"): epoch,
                ("next_target_offset"): offset,
                ("global_step"): step,
                ("losses"): torch.from_numpy(np.array(progress[:step], copy=True))
                if progress is not None
                else losses,
                ("scores"): scores,
                ("best_score"): best_score,
                ("best_epoch"): best_epoch,
                ("best_model"): best,
                ("processed"): processed,
                ("internal_queries"): internal_queries,
                ("checkpoint_suffix"): f"e{epoch}-{phase}",
            },
            identity,
            persistent=checkpoint if remote else None,
            measurements=timing,
            retain_latest_only=profile[("profile")] == ("full"),
            mount_root=Path(options[("mount_root")])
            if options.get("mount_root")
            else None,
        )
        emit(
            "tower_checkpoint",
            step,
            None,
            "updates",
            status="complete",
            message=("Remote verified")
            if remote
            else ("Local saved; remote unchanged"),
            metrics={
                ("remote_verified"): remote,
                ("epoch"): epoch,
                ("consumed_target_offset"): offset,
            },
            force=True,
        )
        if cadence:
            cadence.completed(step, time.perf_counter(), remote=remote)

    complete_boundary = True
    try:
        while epoch < config["tt_epochs"] and (
            step_limit is None or step < step_limit or offset > 0
        ):
            epoch_initial_offset = offset
            emit(
                f"tower_epoch_{epoch + 1}",
                offset,
                target_count,
                "targets",
                metrics={
                    ("epoch"): epoch + 1,
                    ("epochs"): config[("tt_epochs")],
                    ("global_step"): step,
                    ("batch"): (offset + config[("tt_batch_size")] - 1)
                    // config[("tt_batch_size")],
                    ("batches"): batches,
                    ("device"): str(device),
                },
                force=True,
            )
            model.train()
            iterator = iter(
                prefetch(
                    iter(targets.batches(offset, config["tt_batch_size"])),
                    performance.get(("prefetch_batches"), 0),
                )
            )
            while True:
                if step_limit is not None and step >= step_limit:
                    break
                started = time.perf_counter()
                step_event = timing.begin_step()
                with timing.span("target_read_and_wait"):
                    batch = next(iterator, None)
                if batch is None:
                    break
                positions = np.empty(0, dtype=np.int64)
                if optimized:
                    contexts, positions, inputs, mask = batch
                    count = len(contexts)
                    candidates = [history.catalog[j] for j in positions]
                else:
                    count = len(batch)
                    queries = [(u, d) for u, d, _ in batch]
                    candidates = sorted({i for _, _, i in batch})
                    with timing.span("positive_sql"):
                        positives = history.positives(queries)
                    with timing.span("positive_mask"):
                        mask = positive_mask(queries, candidates, positives)
                    with timing.span("cpu_users_assembly_including_sql"):
                        inputs = encoder.users(queries)
                valid = (~mask).any(dim=1)
                complete_boundary = False
                offset += count
                if not valid.any():
                    complete_boundary = True
                    continue
                with timing.span("host_to_device", gpu=True):
                    moved = (
                        bank.transfer("users", inputs) if bank else move(inputs, device)
                    )
                    item_inputs = (
                        bank.items(positions)
                        if bank
                        else move(encoder.items(candidates), device)
                    )
                    q = torch.tensor(
                        [probabilities[i] for i in candidates], device=device
                    )
                    device_mask = mask[valid].to(device)
                    device_valid = valid.to(device)
                use_amp = device.type == "cuda"
                amp_dtype = (
                    torch.bfloat16
                    if (use_amp and torch.cuda.is_bf16_supported())
                    else torch.float16
                )
                with torch.amp.autocast(
                    device_type="cuda", dtype=amp_dtype, enabled=use_amp
                ):
                    with timing.span("user_forward", gpu=True):
                        users = model.user_vectors(*moved)[device_valid]
                    with timing.span("item_forward", gpu=True):
                        items = model.item_vectors(*item_inputs)
                with timing.span("loss", gpu=True):
                    loss = logq_loss(users.float(), items.float(), q, device_mask)
                optimizer.zero_grad()
                with timing.span("backward", gpu=True):
                    loss.backward()
                with timing.span("optimizer", gpu=True):
                    optimizer.step()
                    scheduler.step()
                with timing.span("finite_check_and_scalar"):
                    scalar = float(loss.detach().cpu())
                    if not np.isfinite(scalar):
                        raise ValueError("Nonfinite tower loss")
                step += 1
                processed += count
                if progress is not None:
                    progress[step - 1] = scalar
                else:
                    losses.append(scalar)
                complete_boundary = True
                timing.step(time.perf_counter() - started, count, step_event)
                gpu = (
                    {
                        "cuda_allocated_bytes": torch.cuda.memory_allocated(device),
                        ("cuda_reserved_bytes"): torch.cuda.memory_reserved(device),
                    }
                    if device.type == ("cuda")
                    else {("gpu_utilization"): ("unavailable")}
                )
                emit(
                    f"tower_epoch_{epoch + 1}",
                    offset,
                    target_count,
                    "targets",
                    metrics={
                        ("epoch"): epoch + 1,
                        ("epochs"): config[("tt_epochs")],
                        ("batch"): (offset + config[("tt_batch_size")] - 1)
                        // config[("tt_batch_size")],
                        ("batches"): batches,
                        ("global_step"): step,
                        ("examples_processed"): processed,
                        ("loss"): scalar,
                        ("step_seconds"): time.perf_counter() - started,
                        **gpu,
                    },
                    force=offset <= epoch_initial_offset + config[("tt_batch_size")]
                    or offset == target_count,
                )
                if cadence:
                    local, remote = cadence.due(step, time.perf_counter())
                    if local:
                        checkpoint_state("batch", remote=remote)
                elif (
                    step == 1
                    or step % profile["checkpoint_every_steps"] == 0
                    or offset >= target_count
                    or (step == step_limit)
                ):
                    checkpoint_state("batch")
            iterator.close()
            if not step:
                raise ValueError("No actual tower optimization steps")
            with timing.span("internal_validation_including_catalog"):
                score, internal_queries = (
                    cached_internal_recall(model, encoder, targets, run, device, timing)
                    if optimized
                    else internal_recall(model, encoder, run, device)
                )
            scores.append(score)
            if score > best_score:
                best_score, best_epoch = (score, epoch + 1)
                best = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
            emit(
                f"tower_epoch_{epoch + 1}",
                offset,
                target_count,
                "targets",
                status=("complete") if offset == target_count else ("stopped"),
                metrics={
                    ("epoch"): epoch + 1,
                    ("epochs"): config[("tt_epochs")],
                    ("global_step"): step,
                    ("internal_recall50"): score,
                    ("loss"): float(progress[step - 1])
                    if progress is not None
                    else losses[-1],
                },
                force=True,
            )
            epoch += 1
            offset = 0
            checkpoint_state("selected")
            if step_limit is not None and step >= step_limit:
                break
        if best is None:
            raise ValueError("No selected internal checkpoint")
        model.load_state_dict(best)
        changed = float((model.text.weight.detach().cpu() - initial).abs().max())
        if changed <= 0:
            raise ValueError("No trained content projection")
        torch.save(best, run / "models/twotower.pt")
        with timing.span("final_catalog_encoding"):
            item_vectors(model, encoder, run / "models/twotower_items.npy", device)
        write_json(run / "models/twotower_items.json", {"article_ids": history.catalog})
        if not (run / "checkpoints/latest.json").exists():
            from hm_recsys.runtime.files import atomic_json, verified_copy

            recovery = (
                local_checkpoint
                if (local_checkpoint / "latest.json").is_file()
                else checkpoint
            )
            record = read_json(recovery / "latest.json")
            verified_copy(
                recovery / record["file"],
                run / "checkpoints" / record["file"],
                expected_sha256=record[("sha256")],
            )
            atomic_json(run / "checkpoints/latest.json", record)
        if device.type == "cuda":
            torch.cuda.synchronize()
        write_json(
            run / "models/twotower.json",
            {
                "maps": encoder.maps,
                "text_dim": content.vectors[("text")].shape[1],
                ("image_dim"): content.vectors[("image")].shape[1],
                ("output_dim"): 128,
                ("requested_device"): profile[("device")],
                ("device"): str(device),
                ("steps"): step,
                ("epochs"): epoch,
                ("selected_epoch"): best_epoch,
                ("internal_daily_recall50"): best_score,
                ("internal_validation_queries"): internal_queries,
                ("epoch_recalls"): scores,
                ("losses"): progress[:step].tolist()
                if progress is not None
                else losses,
                ("processed_targets"): processed,
                ("selected_targets"): target_count,
                ("projection_max_parameter_change"): changed,
                ("sampler"): (
                    "fixed SHA256 unique cohort events; repeat or"
                    "der per epoch; Q from selected events only"
                ),
                ("target_identity"): identity[("targets")],
                ("performance"): performance or {("backend"): ("reference")},
                ("static_inputs"): bank.report
                if bank
                else {("placement"): ("reference_host")},
                ("precision"): ("amp_bf16_fp32_loss")
                if device.type == "cuda"
                else ("float32_no_amp"),
                ("peak_cuda_allocated"): torch.cuda.max_memory_allocated()
                if device.type == ("cuda")
                else 0,
                ("peak_cuda_reserved"): torch.cuda.max_memory_reserved()
                if device.type == ("cuda")
                else 0,
            },
        )
    except KeyboardInterrupt:
        if step and complete_boundary:
            checkpoint_state("stop")
        raise
    finally:
        write_json(run / "tower_performance.json", timing.summary())
        history.close()
        model.cpu()
        optimizer.zero_grad(set_to_none=True)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
