"""Bounded reference/optimized compute diagnostics using canonical model primitives.

Consumes an authenticated synthetic fixture or verified prior artifact view and
explicit benchmark limits; emits parity and performance JSON. Depends on canonical
Two-Tower encoders, loss, optimizer and complete checkpoint helpers. It is
diagnostic only and never substitutes a benchmark model for the training pipeline.
"""

import argparse
import os
import re
import subprocess
import sys
import time
from itertools import islice
from pathlib import Path

import numpy as np
import psutil
import torch

from hm_recsys.core.contracts import ROOT, Config, read_json, write_json
from hm_recsys.data.history import History, canonical
from hm_recsys.data.streaming import TargetStream
from hm_recsys.features.content import MappedContent, adapt_content
from hm_recsys.retrieval.two_tower.execution import ItemBank, Measurements, prefetch
from hm_recsys.retrieval.two_tower.inputs import SqlEncoder
from hm_recsys.retrieval.two_tower.model import TwoTower, logq_loss, positive_mask
from hm_recsys.retrieval.two_tower.prepared import Prepared, prepare
from hm_recsys.retrieval.two_tower.training import (
    cached_internal_recall,
    device_for,
    internal_recall,
    move,
)
from hm_recsys.runtime.checkpoints import save_checkpoint
from hm_recsys.runtime.files import canonical_hash, verified_copy
from hm_recsys.runtime.profiles import is_colab_runtime, load_profile, runtime_identity
from hm_recsys.runtime.resources import run_stage
from hm_recsys.runtime.runner import code_identity
from hm_recsys.runtime.testing import create_full_source


# Callable original batch contract, retaining SQL and row-wise tensor assembly.
def reference_batches(encoder, targets, timing):
    """Callable original batch contract, retaining SQL and row-wise tensor assembly."""
    source = iter(targets.batches(0, 512))
    # Continue this owned loop only under its stated condition, respecting the
    # consumed-offset/resource boundary.
    while True:
        # Time the named operation here; host dispatch and asynchronous CUDA duration
        # remain distinct measurements.
        with timing.span("target_read"):
            batch = next(source, None)
        if batch is None:
            # Stop at this declared completion/limit boundary; no extra rows or
            # optimization steps are consumed.
            break
        queries = [(u, d) for u, d, _ in batch]
        candidates = sorted({i for _, _, i in batch})
        # Time the named operation here; host dispatch and asynchronous CUDA duration
        # remain distinct measurements.
        with timing.span("positive_sql"):
            positives = encoder.history_store.positives(queries)
        # Time the named operation here; host dispatch and asynchronous CUDA duration
        # remain distinct measurements.
        with timing.span("positive_mask"):
            mask = positive_mask(queries, candidates, positives)
        # Time the named operation here; host dispatch and asynchronous CUDA duration
        # remain distinct measurements.
        with timing.span("cpu_users_assembly_including_sql"):
            inputs = encoder.users(queries)
        # Time the named operation here; host dispatch and asynchronous CUDA duration
        # remain distinct measurements.
        with timing.span("static_item_assembly"):
            items = encoder.items(candidates)
        q = torch.tensor([targets.probabilities[i] for i in candidates])
        # Yield this ordered result/context to the consumer without eagerly
        # materializing subsequent input rows.
        yield inputs, items, mask, q, len(batch)


# CPU prefetch performs bounded gathers and masks only, preserving order.
def optimized_batches(prepared, bank, timing):
    """CPU prefetch performs bounded gathers and masks only, preserving order."""
    source = prefetch(iter(prepared.batches(0, 512)), 2)
    # Protect this operation with explicit failure/cleanup handling so partial work
    # is never reported as complete.
    try:
        # Continue this owned loop only under its stated condition, respecting the
        # consumed-offset/resource boundary.
        while True:
            # Time the named operation here; host dispatch and asynchronous CUDA
            # duration remain distinct measurements.
            with timing.span("prepared_read_and_wait"):
                batch = next(source, None)
            if batch is None:
                # Stop at this declared completion/limit boundary; no extra rows or
                # optimization steps are consumed.
                break
            contexts, candidates, inputs, mask = batch
            q = torch.tensor(
                [prepared.probabilities[prepared.catalog[i]] for i in candidates]
            )
            # Yield this ordered result/context to the consumer without eagerly
            # materializing subsequent input rows.
            yield inputs, candidates, mask, q, len(contexts)
    finally:
        # Release this owned file/database/writer handle at the stage boundary.
        source.close()


# Identical FP32 update and RNG draw order; CPU-known valid mask is unchanged.
def update(model, optimizer, scheduler, batch, bank, device, timing):
    """Identical FP32 update and RNG draw order; CPU-known valid mask is unchanged."""
    users, items, mask, q, count = batch
    valid = (~mask).any(dim=1)
    if not valid.any():
        return None
    # Time the named operation here; host dispatch and asynchronous CUDA duration
    # remain distinct measurements.
    with timing.span("host_to_device", gpu=True):
        user_inputs = bank.transfer("users", users) if bank else move(users, device)
        item_inputs = bank.items(items) if bank else move(items, device)
        mask = mask[valid].to(device)
        valid = valid.to(device)
        q = q.to(device)
    # Time the named operation here; host dispatch and asynchronous CUDA duration
    # remain distinct measurements.
    with timing.span("user_forward", gpu=True):
        uv = model.user_vectors(*user_inputs)[valid]
    # Time the named operation here; host dispatch and asynchronous CUDA duration
    # remain distinct measurements.
    with timing.span("item_forward", gpu=True):
        iv = model.item_vectors(*item_inputs)
    # Time the named operation here; host dispatch and asynchronous CUDA duration
    # remain distinct measurements.
    with timing.span("loss", gpu=True):
        loss = logq_loss(uv, iv, q, mask)
    # Time the named operation here; host dispatch and asynchronous CUDA duration
    # remain distinct measurements.
    with timing.span("finite_check_and_scalar"):
        scalar = float(loss.detach().cpu())
        if not np.isfinite(scalar):
            raise ValueError("Nonfinite diagnostic loss")
    # Clear previous gradients before the current optimizer update.
    optimizer.zero_grad()
    # Time the named operation here; host dispatch and asynchronous CUDA duration
    # remain distinct measurements.
    with timing.span("backward", gpu=True):
        # Backpropagate the current loss through all trainable ID and content
        # projection branches.
        loss.backward()
    # Time the named operation here; host dispatch and asynchronous CUDA duration
    # remain distinct measurements.
    with timing.span("optimizer", gpu=True):
        # Apply the current gradients using the unchanged AdamW optimizer state.
        optimizer.step()
        # Advance the unchanged cosine schedule once for this completed optimizer
        # update.
        scheduler.step()
    return scalar


# Reset model/optimizer/dropout RNG after warmup, including on CUDA.
def new_model(encoder, device, steps):
    """Reset model/optimizer/dropout RNG after warmup, including on CUDA."""
    # Restore the recorded tensor RNG sequence, including training-only shared ID
    # dropout.
    torch.manual_seed(42)
    model = TwoTower(
        encoder.maps,
        encoder.content.vectors["text"].shape[1],
        encoder.content.vectors["image"].shape[1],
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, steps * 10)
    )
    return model, optimizer, scheduler


# Compare every prepared input and a short multi-step real gradient sequence.
def parity(encoder, targets, prepared, device, run, limit=3):
    """Compare every prepared input and a short multi-step real gradient sequence."""
    timing = Measurements(device)
    bank = ItemBank(encoder, device)
    baseline = new_model(encoder, device, 4)
    optimized = new_model(encoder, device, 4)
    checked = 0
    maximum = 0.0
    gradient_max = 0.0
    loss_max = 0.0
    updates = 0
    # Consume the next bounded ordered row block; keep the final partial batch and
    # avoid an all-table Python list.
    for reference, fast in zip(
        reference_batches(encoder, targets, timing),
        optimized_batches(prepared, bank, timing),
        strict=True,
    ):
        users, items, mask, q, n = reference
        fastusers, ids, fastmask, fastq, _ = fast
        for a, b in zip(users, fastusers, strict=True):
            # Verify numerical parity using the stated tolerance and aligned tensors.
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        for a, b in zip(items, bank.items(ids), strict=True):
            # Verify numerical parity using the stated tolerance and aligned tensors.
            torch.testing.assert_close(a, b.cpu(), rtol=0, atol=0)
        # Verify numerical parity using the stated tolerance and aligned tensors.
        torch.testing.assert_close(mask, fastmask, rtol=0, atol=0)
        # Verify numerical parity using the stated tolerance and aligned tensors.
        torch.testing.assert_close(q, fastq, rtol=0, atol=0)
        checked += n
        if updates >= limit:
            # Skip this explicitly inapplicable/empty record while retaining the
            # order of all remaining valid inputs.
            continue
        rng = torch.get_rng_state()
        cuda = torch.cuda.get_rng_state_all() if device.type == "cuda" else []
        before = baseline[0].text.weight.detach().clone()
        a = update(*baseline, reference, None, device, timing)
        torch.set_rng_state(rng)
        # Handle CUDA only under the explicit device condition; input placement
        # cannot silently move the model to CPU.
        if cuda:
            torch.cuda.set_rng_state_all(cuda)
        b = update(*optimized, fast, bank, device, timing)
        if a is None:
            if b is not None:
                raise ValueError("Skipped row mismatch")
            # Skip this explicitly inapplicable/empty record while retaining the
            # order of all remaining valid inputs.
            continue
        if b is None:
            raise ValueError("Optimized update unexpectedly skipped")
        loss_max = max(loss_max, abs(a - b))
        updates += 1
        for left, right in zip(
            baseline[0].parameters(), optimized[0].parameters(), strict=True
        ):
            # Verify numerical parity using the stated tolerance and aligned tensors.
            torch.testing.assert_close(left, right, rtol=1e-5, atol=1e-6)
            maximum = max(maximum, float((left - right).abs().max().detach().cpu()))
            if left.grad is not None:
                if right.grad is None:
                    raise ValueError("Optimized gradient missing")
                # Verify numerical parity using the stated tolerance and aligned
                # tensors.
                torch.testing.assert_close(left.grad, right.grad, rtol=1e-5, atol=1e-6)
                gradient_max = max(
                    gradient_max, float((left.grad - right.grad).abs().max().cpu())
                )
        if torch.equal(before, baseline[0].text.weight):
            raise ValueError("Content projection did not train")
    report = {
        "status": "passed",
        "device": str(device),
        "input_examples": checked,
        "gradient_updates": updates,
        "parameter_max_abs_difference": maximum,
        "gradient_max_abs_difference": gradient_max,
        "loss_max_abs_difference": loss_max,
        "input_mask_q_tolerance": 0,
        "continuous_rtol": 1e-5,
        "continuous_atol": 1e-6,
        "cuda": "measured" if device.type == "cuda" else "not_run",
    }
    # Write measured state/lineage to the owned run path without mutating source
    # manifests.
    write_json(run / "parity_report.json", report)
    return report


def input_view(source: Path, destination: Path) -> None:
    """Read verified current-release upstream data only, never copy fitted models."""
    from hm_recsys.runtime.modular import ALL_STAGES, GRAPH
    from hm_recsys.runtime.state import StageStore

    identity = read_json(source / "identity.json")
    if identity["code"] != code_identity(ROOT):
        raise ValueError("Benchmark requires current release upstream data")
    store = StageStore(
        source,
        source,
        identity,
        stages=ALL_STAGES,
        dependency_graph=GRAPH,
        optional_stages=("final_test",),
    )
    record = store.record("prepare")
    if record is None or record["dependencies"] != {}:
        raise ValueError("Missing verified canonical prepare stage for benchmark")
    records = [record]
    for name, path in store.view(records).items():
        if name.startswith(("canonical/", "content/")):
            target = destination / name
            # Create only the required owned directory; immutable inputs and other
            # runs remain untouched.
            target.parent.mkdir(parents=True, exist_ok=True)
            # Independent copy prevents any later consumer from modifying source.
            verified_copy(path, target)
    # Write measured state/lineage to the owned run path without mutating source
    # manifests.
    write_json(
        destination / "benchmark_dependency.json",
        {
            "identity": canonical_hash(identity),
            "stages": {r["stage"]: r["manifest_sha256"] for r in records},
        },
    )


# Measure cold preparation, data/replay/end-to-end work without full fitting.
def run_benchmark(run: Path, options: dict) -> dict:
    """Measure cold preparation, data/replay/end-to-end work without full fitting."""
    started = time.perf_counter()
    # Apply the existing native-thread budget before numerical execution.
    torch.set_num_threads(2)
    if torch.get_num_interop_threads() != 1:
        # Keep the explicitly configured inter-op budget for isolated PyTorch
        # execution.
        torch.set_num_interop_threads(1)
    profile = load_profile(ROOT, "full")
    profile["device"] = options["device"]
    device = device_for(profile)
    work = run / "input"
    # Create only the required owned directory; immutable inputs and other runs
    # remain untouched.
    work.mkdir()
    prepstart = time.perf_counter()
    if options["synthetic"]:
        source = run / "source"
        create_full_source(source)
        canonical(work, source, profile)
        adapt_content(work, source)
    else:
        input_view(Path(options["input_run"]), work)
    # Write measured state/lineage to the owned run path without mutating source
    # manifests.
    write_json(
        work / "run_manifest.json",
        {
            "migration_signature": canonical_hash(
                {"code": code_identity(ROOT), "options": options}
            )
        },
    )
    for folder in ("models", "tmp"):
        # Create only the required owned directory; immutable inputs and other runs
        # remain untouched.
        (work / folder).mkdir(exist_ok=True)
    h = History(work)
    encoder = SqlEncoder(h, MappedContent(work))
    targets = TargetStream(h, options["batches"] * 512)
    common_seconds = time.perf_counter() - prepstart
    cache_start = time.perf_counter()
    prepare(
        work, encoder, targets, include_internal=options["validation"] == "complete"
    )
    cache_seconds = time.perf_counter() - cache_start
    cache_start = time.perf_counter()
    prepared = Prepared(work / "prepared/tower")
    warm_open = time.perf_counter() - cache_start
    results = []
    parity_report = parity(encoder, targets, prepared, device, run)
    modes = (
        ("data", "replay", "end_to_end")
        if options["mode"] == "all"
        else (options["mode"],)
    )
    for mode in modes:
        for backend in ("reference", "optimized"):
            timing = Measurements(device)
            encoder.measurements = timing
            bank_start = time.perf_counter()
            bank = ItemBank(encoder, device) if backend == "optimized" else None
            bank_seconds = time.perf_counter() - bank_start
            encoder.bank = bank

            # batches.
            def batches(bank=bank, timing=timing):
                return (
                    optimized_batches(prepared, bank, timing)
                    if bank
                    else reference_batches(encoder, targets, timing)
                )

            # Mark replay as unavailable until the explicit initialization/recovery
            # branch fills it.
            replay = None
            if mode == "replay":
                replay = list(batches())
            warmstart = time.perf_counter()
            warm = new_model(encoder, device, options["batches"])
            # Consume the next bounded ordered row block; keep the final partial
            # batch and avoid an all-table Python list.
            for batch in islice(
                iter(replay) if replay is not None else batches(), options["warmup"]
            ):
                if mode != "data":
                    update(*warm, batch, bank, device, timing)
            # Handle CUDA only under the explicit device condition; input placement
            # cannot silently move the model to CPU.
            if device.type == "cuda":
                torch.cuda.synchronize()
            warm_seconds = time.perf_counter() - warmstart
            # Release the owned temporary reference after its last use without
            # deleting the source artifact.
            del warm
            timing = Measurements(device)
            encoder.measurements = timing
            model, optimizer, scheduler = new_model(encoder, device, options["batches"])
            # Enable the existing training behavior; no learned branch is frozen by
            # input caching.
            model.train()
            losses = []
            steps = 0
            examples = 0
            destination = run / f"{mode}_{backend}"
            # Create only the required owned directory; immutable inputs and other
            # runs remain untouched.
            destination.mkdir()
            checkpoint = (
                (
                    Path(options["persistent_root"])
                    / run.name
                    / destination.name
                    / "checkpoints"
                )
                if options.get("persistent_root")
                else destination / "remote_simulation"
            )
            # Create only the required owned directory; immutable inputs and other
            # runs remain untouched.
            checkpoint.parent.mkdir(parents=True, exist_ok=True)

            # measured batches.
            def measured_batches(bank=bank, timing=timing):
                return (
                    optimized_batches(prepared, bank, timing)
                    if bank
                    else reference_batches(encoder, targets, timing)
                )

            # Handle CUDA only under the explicit device condition; input placement
            # cannot silently move the model to CPU.
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            begin = time.perf_counter()
            iterator = iter(replay) if replay is not None else iter(measured_batches())
            # Consume the next bounded ordered row block; keep the final partial
            # batch and avoid an all-table Python list.
            for _ in range(options["batches"]):
                batch_start = time.perf_counter()
                step_event = timing.begin_step()
                batch = next(iterator, None)
                if batch is None:
                    # Stop at this declared completion/limit boundary; no extra rows
                    # or optimization steps are consumed.
                    break
                examples += batch[-1]
                if mode != "data":
                    loss = update(
                        model, optimizer, scheduler, batch, bank, device, timing
                    )
                    if loss is not None:
                        # Count this completed optimizer update for scheduling,
                        # reporting and checkpoint cadence.
                        steps += 1
                        losses.append(loss)
                # Record this profiling/progress boundary without advancing the
                # training RNG.
                timing.step(time.perf_counter() - batch_start, batch[-1], step_event)
                if mode == "end_to_end" and steps and (steps == 1 or steps % 10 == 0):
                    # Save complete model/optimizer/scheduler/RNG/progress state;
                    # publish remotely only when requested.
                    save_checkpoint(
                        destination / "local",
                        {
                            "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(),
                            "global_step": steps,
                            "next_target_offset": examples,
                            "epoch": 0,
                            "losses": losses,
                            "best_model": None,
                        },
                        {"scope": "benchmark", "backend": backend},
                        persistent=checkpoint,
                        mount_root=Path(options["mount_root"])
                        if options.get("mount_root")
                        else None,
                        retain_latest_only=True,
                        measurements=timing,
                    )
            close = getattr(iterator, "close", None)
            if close is not None:
                # Release this owned file/database/writer handle at the stage
                # boundary.
                close()
            # Handle CUDA only under the explicit device condition; input placement
            # cannot silently move the model to CPU.
            if device.type == "cuda":
                torch.cuda.synchronize()
            train_seconds = time.perf_counter() - begin
            # Mark validation as unavailable until the explicit
            # initialization/recovery branch fills it.
            validation = None
            if mode == "end_to_end" and options["validation"] == "complete":
                # Time the named operation here; host dispatch and asynchronous CUDA
                # duration remain distinct measurements.
                with timing.span("internal_validation_including_catalog"):
                    validation = (
                        cached_internal_recall(
                            model, encoder, prepared, work, device, timing
                        )
                        if bank
                        else internal_recall(model, encoder, work, device)
                    )
            if mode == "end_to_end" and steps:
                # Save complete model/optimizer/scheduler/RNG/progress state; publish
                # remotely only when requested.
                save_checkpoint(
                    destination / "local",
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "global_step": steps,
                        "next_target_offset": examples,
                        "epoch": 0,
                        "losses": losses,
                        "best_model": model.state_dict() if validation else None,
                        "checkpoint_suffix": "final",
                    },
                    {"scope": "benchmark", "backend": backend},
                    persistent=checkpoint,
                    mount_root=Path(options["mount_root"])
                    if options.get("mount_root")
                    else None,
                    retain_latest_only=True,
                    measurements=timing,
                )
            total = time.perf_counter() - begin
            result = {
                "backend": backend,
                "mode": mode,
                "examples": examples,
                "steps": steps,
                "batch_size": 512,
                "device": str(device),
                "precision": "float32_no_amp",
                "input_bank_setup_seconds": bank_seconds,
                "cold_total_seconds": total
                + (cache_seconds + warm_open + bank_seconds if bank else 0),
                "timing_scope": (
                    "Post-canonical bounded work; train loop includes "
                    "periodic checkpoint I/O; replay excludes I/O"
                ),
                "cache_cold_seconds": cache_seconds if bank else 0.0,
                "cache_verified_open_seconds": warm_open if bank else 0.0,
                "cache_amortized_ten_epochs_seconds": cache_seconds / 10
                if bank
                else 0.0,
                "warmup_seconds": warm_seconds,
                "train_seconds": train_seconds,
                "total_seconds": total,
                "examples_per_second": examples / train_seconds,
                "internal_validation": {
                    "recall": validation[0],
                    "queries": validation[1],
                }
                if validation
                else "not_run",
                "remote_filesystem": str(checkpoint)
                if options.get("persistent_root")
                else "local simulation, not Drive",
                "losses": losses,
                "telemetry": timing.summary(),
                "input_placement": bank.report
                if bank
                else {"placement": "reference_host"},
            }
            results.append(result)
    if options["trace"]:
        timing = Measurements(device)
        bank = ItemBank(encoder, device)
        model, optimizer, scheduler = new_model(encoder, device, options["batches"])
        trace_start = time.perf_counter()
        activities = [torch.profiler.ProfilerActivity.CPU]
        # Handle CUDA only under the explicit device condition; input placement
        # cannot silently move the model to CPU.
        if device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(
            activities=activities,
            schedule=torch.profiler.schedule(wait=0, warmup=1, active=2, repeat=1),
        ) as profiler:
            # Consume the next bounded ordered row block; keep the final partial
            # batch and avoid an all-table Python list.
            for j, batch in enumerate(optimized_batches(prepared, bank, timing)):
                if j >= 3:
                    # Stop at this declared completion/limit boundary; no extra rows
                    # or optimization steps are consumed.
                    break
                update(model, optimizer, scheduler, batch, bank, device, timing)
                # Record this profiling/progress boundary without advancing the
                # training RNG.
                profiler.step()
        profiler.export_chrome_trace(str(run / "bounded_trace.json"))
        trace_seconds = time.perf_counter() - trace_start
    else:
        # Mark trace seconds as unavailable until the explicit
        # initialization/recovery branch fills it.
        trace_seconds = None
    # Mark gpu sample as unavailable until the explicit initialization/recovery
    # branch fills it.
    gpu_sample = None
    # Handle CUDA only under the explicit device condition; input placement cannot
    # silently move the model to CPU.
    if device.type == "cuda":
        # Protect this operation with explicit failure/cleanup handling so partial
        # work is never reported as complete.
        try:
            probe = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,utilization.gpu,memory.used",
                    "--format=csv,noheader",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if probe.returncode == 0:
                gpu_sample = {
                    "sample_at": "end of bounded run",
                    "value": probe.stdout.strip(),
                }
        # Handle the declared exception type here and preserve the surrounding
        # failure/recovery contract.
        except (FileNotFoundError, subprocess.TimeoutExpired):
            # Preserve this intentionally empty handler; the surrounding branch
            # defines which failure is optional.
            pass
    summary = {
        "scope": "bounded diagnostic, not full training",
        "configuration": options,
        "environment": runtime_identity(ROOT),
        "common_input_preparation_seconds": common_seconds,
        "cache_build_seconds": cache_seconds,
        "whole_benchmark_seconds": time.perf_counter() - started,
        "whole_pipeline_seconds": "not_run",
        "traced_window_seconds_excluded_from_throughput": trace_seconds,
        "gpu_utilization_sample": gpu_sample,
        "cpu_affinity": getattr(psutil.Process(), "cpu_affinity", lambda: None)()
        if hasattr(psutil.Process(), "cpu_affinity")
        else None,
        "torch_cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name() if device.type == "cuda" else None,
        "cpu_quota": Path("/sys/fs/cgroup/cpu.max").read_text(encoding="utf-8").strip()
        if Path("/sys/fs/cgroup/cpu.max").is_file()
        else None,
        "threads": 2,
        "ram_available_bytes": psutil.virtual_memory().available,
        "variants": results,
        "parity": parity_report,
        "cache": {
            "build_misses": 1,
            "verified_reopens": 1,
            "contexts": prepared.meta["contexts"],
            "targets": prepared.count,
            "transaction_rows_exported": prepared.meta["transaction_rows_exported"],
            "history_query_rows_scanned": prepared.meta["history_query_rows_scanned"],
            "history_query_seconds": prepared.meta["history_query_seconds"],
        },
    }
    # Write measured state/lineage to the owned run path without mutating source
    # manifests.
    write_json(run / "performance_summary.json", summary)
    import zipfile

    with zipfile.ZipFile(
        run / "compute_review.zip", "x", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        for name in ("performance_summary.json", "parity_report.json"):
            # Serialize the explicitly selected metadata/records to this operation's
            # output; raw source inputs are not overwritten.
            archive.write(run / name, name)
    # Release this owned file/database/writer handle at the stage boundary.
    h.close()
    return summary


# Supervise a capped diagnostic; real inputs are allowed only on Colab.
def main() -> None:
    """Supervise a capped diagnostic; real inputs are allowed only on Colab."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--input-run", type=Path)
    parser.add_argument("--persistent-root", type=Path)
    parser.add_argument("--mount-root", type=Path)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument(
        "--mode", choices=("data", "replay", "end_to_end", "all"), default="all"
    )
    parser.add_argument("--validation", choices=("none", "complete"), default="none")
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch("[A-Za-z0-9_-]+", args.run_id):
        parser.error("Unsafe run ID")
    if not 1 <= args.batches <= 100 or not 0 <= args.warmup <= 10:
        parser.error("Bounded diagnostic budget exceeded")
    if args.synthetic == (args.input_run is not None):
        parser.error("Choose synthetic OR a verified input run")
    if args.plan:
        import json

        print(json.dumps(vars(args), default=str, indent=2))
        return
    if not is_colab_runtime() and (
        not args.synthetic or args.batches > 4 or args.warmup > 1
    ):
        parser.error("Mac allows only tiny synthetic benchmarks")
    if args.persistent_root is not None:
        if (
            not is_colab_runtime()
            or args.mount_root is None
            or not os.path.ismount(args.mount_root)
        ):
            parser.error("Remote benchmark checkpoints require a verified Colab mount")
        if not args.persistent_root.resolve().is_relative_to(args.mount_root.resolve()):
            parser.error("Persistent benchmark root must be beneath the mount")
        # Create only the required owned directory; immutable inputs and other runs
        # remain untouched.
        (args.persistent_root / args.run_id).mkdir(parents=True, exist_ok=False)
    run = ROOT / "data/runs/benchmark" / args.run_id
    # Create only the required owned directory; immutable inputs and other runs
    # remain untouched.
    run.mkdir(parents=True, exist_ok=False)
    options = vars(args)
    options["input_run"] = str(args.input_run) if args.input_run else None
    options["persistent_root"] = (
        str(args.persistent_root) if args.persistent_root else None
    )
    options["mount_root"] = str(args.mount_root) if args.mount_root else None
    # Write measured state/lineage to the owned run path without mutating source
    # manifests.
    write_json(run / "benchmark_options.json", options)
    os.environ["PYTHONPATH"] = (
        str(ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")
    )
    run_stage(
        "benchmark", run, Config.load(), module="hm_recsys.runtime.benchmark", cwd=ROOT
    )
    print(str(run / "performance_summary.json"))


if __name__ == "__main__":
    _, location = sys.argv[1:]
    directory = Path(location)
    run_benchmark(directory, read_json(directory / "benchmark_options.json"))
