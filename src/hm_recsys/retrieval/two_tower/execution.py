"""Measure actual tower work and transfer bounded raw inputs to the requested device.

Measurements separate host wall time from asynchronous CUDA events and retain
bounded step samples. ItemBank stores immutable categories/content only, using
an explicit device or pinned-host policy. Ordered prefetch owns its CPU producer
and closes it on interruption. No network, label lookup or model fit is started
at import, and no CUDA-to-CPU training fallback is performed."""

import time
from collections import defaultdict, deque
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

import numpy as np
import psutil
import torch


class Measurements:
    """Host wall times and asynchronous CUDA events, resolved at boundaries only."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.wall = defaultdict(float)
        self.calls = defaultdict(int)
        self.events = []
        self.gpu = defaultdict(float)
        self.steps = deque(maxlen=4096)
        self.cuda_steps = deque(maxlen=4096)
        self.examples = 0
        self.rss = 0

    @contextmanager
    def span(self, name: str, *, gpu: bool = False):
        start = time.perf_counter()
        pair = None
        if gpu and self.device.type == "cuda":
            pair = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            pair[0].record()
        with torch.profiler.record_function(name):
            yield
        if pair:
            pair[1].record()
            self.events.append((name, pair))
        self.wall[name] += time.perf_counter() - start
        self.calls[name] += 1
        if len(self.events) >= 128:
            self.resolve(wait=False)

    def resolve(self, *, wait: bool) -> None:
        if wait and self.device.type == "cuda":
            torch.cuda.synchronize()
        pending = []
        for name, (a, b) in self.events:
            if b.query():
                seconds = a.elapsed_time(b) / 1000
                self.gpu[name] += seconds
                if name == "batch_interval":
                    self.cuda_steps.append(seconds)
            else:
                pending.append((name, (a, b)))
        self.events = pending

    def begin_step(self):
        """Record a device boundary without synchronizing the compute stream."""
        if self.device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            start.record()
            return start
        return None

    def step(self, seconds: float, examples: int, event=None) -> None:
        if event is not None:
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            self.events.append(("batch_interval", (event, end)))
        self.steps.append(seconds)
        self.examples += examples
        self.rss = max(self.rss, psutil.Process().memory_info().rss)

    def summary(self) -> dict:
        self.resolve(wait=True)
        return {
            "host_seconds": dict(self.wall),
            "calls": dict(self.calls),
            ("cuda_event_seconds"): dict(self.gpu)
            if self.device.type == ("cuda")
            else ("not_run"),
            ("examples"): self.examples,
            ("host_step_p50_seconds"): float(np.percentile(self.steps, 50))
            if self.steps
            else None,
            ("host_step_p95_seconds"): float(np.percentile(self.steps, 95))
            if self.steps
            else None,
            ("cuda_step_p50_seconds"): float(np.percentile(self.cuda_steps, 50))
            if self.cuda_steps
            else None,
            ("cuda_step_p95_seconds"): float(np.percentile(self.cuda_steps, 95))
            if self.cuda_steps
            else None,
            ("host_latency_semantics"): (
                "Host dispatch/completion on CPU; not CUDA kernel time"
            ),
            ("latency_window"): ("last 4096 completed batches"),
            ("sampled_peak_process_rss_bytes"): self.rss,
            ("cuda_peak_allocated_bytes"): torch.cuda.max_memory_allocated()
            if self.device.type == ("cuda")
            else None,
            ("cuda_peak_reserved_bytes"): torch.cuda.max_memory_reserved()
            if self.device.type == ("cuda")
            else None,
            ("overlap_warning"): (
                "Host and device durations overlap; do not sum as percentages"
            ),
        }


def prefetch(source: Iterator, depth: int):
    """At most two prepared CPU batches; producer position never drives checkpoint."""
    if not 0 <= depth <= 2:
        raise ValueError("Prefetch depth must be 0, 1 or 2")
    if not depth:
        yield from source
        return
    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = deque()

        def take():
            return next(source, None)

        for _ in range(depth):
            pending.append(executor.submit(take))
        try:
            while pending:
                value = pending.popleft().result()
                if value is None:
                    break
                pending.append(executor.submit(take))
                yield value
        finally:
            for future in pending:
                future.cancel()


class ItemBank:
    """Frozen raw inputs, separate from model state and live trainable projections."""

    def __init__(
        self,
        encoder,
        device: torch.device,
        *,
        placement: str = "auto",
        max_batch: int = 8192,
    ) -> None:
        self.device = device
        self.max_batch = max_batch
        self.ids = encoder.sample.catalog
        self.position = {i: j for j, i in enumerate(self.ids)}
        content = encoder.content
        if content.ids != self.ids:
            raise ValueError("Static bank catalog/content order mismatch")
        path = encoder.history_store.run / "prepared/tower/item_codes.npy"
        if path.is_file():
            self.codes = np.load(path, mmap_mode="r", allow_pickle=False)
        else:
            self.codes = np.empty((len(self.ids), 6), dtype=np.int64)
            for start in range(0, len(self.ids), 512):
                self.codes[start : start + 512] = encoder.items(
                    self.ids[start : start + 512]
                )[0].numpy()
        self.host = (
            self.codes,
            content.vectors["text"],
            content.vectors["image"],
            np.column_stack([content.masks[m] for m in (("text"), ("image"))]).astype(
                np.float32
            ),
        )
        self.bytes = sum(a.nbytes for a in self.host)
        self.resident = None
        if placement not in ("auto", "host"):
            raise ValueError("Unknown item placement")
        if device.type == "cuda" and placement == "auto":
            free, _ = torch.cuda.mem_get_info()
            if self.bytes * 1.25 < min(free * 0.5, free - 2 * 1024**3):
                self.resident = tuple(
                    torch.from_numpy(np.array(a, copy=True)).to(device)
                    for a in self.host
                )
        self.report = {
            "placement": "device" if self.resident else "host",
            "bytes": self.bytes,
            ("model_device"): str(device),
            ("checkpoint_included"): False,
        }
        self.buffers = {}
        self.inflight = {}

    def transfer(self, name: str, values: tuple) -> tuple:
        """Reuse pinned staging only after its DMA event; no claimed compute overlap."""
        if self.device.type != "cuda":
            return values
        if name in self.inflight:
            self.inflight[name].synchronize()
        buffers = self.buffers.get(name)
        if buffers is None or any(
            len(b) < len(v) for b, v in zip(buffers, values, strict=True)
        ):
            cap = max(self.max_batch, max((len(v) for v in values), default=512))
            buffers = tuple(
                torch.empty((cap, *v.shape[1:]), dtype=v.dtype, pin_memory=True)
                for v in values
            )
            self.buffers[name] = buffers
            self.report["pinned_staging_bytes"] = sum(
                v.numel() * v.element_size()
                for group in self.buffers.values()
                for v in group
            )
        if any(len(v) > self.max_batch for v in values):
            raise ValueError("Input transfer exceeds batch budget")
        output = []
        for buffer, value in zip(buffers, values, strict=True):
            if buffer.shape[1:] != value.shape[1:] or buffer.dtype != value.dtype:
                raise ValueError("Transfer schema mismatch")
            view = buffer[: len(value)]
            view.copy_(value)
            output.append(view.to(self.device, non_blocking=True))
        event = torch.cuda.Event()
        event.record()
        self.inflight[name] = event
        return tuple(output)

    def items(self, positions: np.ndarray) -> tuple:
        if positions.ndim != 1 or len(positions) > self.max_batch:
            raise ValueError("Static item gather exceeds the batch contract")
        if self.resident is not None:
            indices = torch.from_numpy(positions.astype(np.int64, copy=False)).to(
                self.device
            )
            return tuple(a[indices] for a in self.resident)
        values = tuple(
            torch.from_numpy(np.array(a[positions], copy=True)) for a in self.host
        )
        return self.transfer("items", values)
