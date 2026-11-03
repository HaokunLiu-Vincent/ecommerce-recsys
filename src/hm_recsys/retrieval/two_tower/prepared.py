"""Prepare verified model-independent training contexts and complete positive sets.

A bounded sorted merge preserves twenty raw purchase lines, strict same-day
exclusion and the original SHA256 target order. Ragged positives and raw category
inputs are persisted as mmap arrays. Identity/file hashes reject stale or partial
caches. No learned projection, item vector or pooled user vector is cached."""

from collections.abc import Iterator
from datetime import timedelta
from itertools import groupby
from pathlib import Path

import numpy as np
import torch

from hm_recsys.core.contracts import BASE_END, TT_END, TT_START, digest, read_json
from hm_recsys.core.progress import emit
from hm_recsys.data.streaming import TargetStream, export_query, rows
from hm_recsys.retrieval.two_tower.model import ITEM_KEYS, PROFILE_KEYS
from hm_recsys.runtime.capacity import check
from hm_recsys.runtime.files import atomic_json


def identity(run: Path, maps: dict, target: str) -> dict:
    """Bind source, mappings, exact target order and the fixed history contract."""
    return {
        "version": 1,
        "run": read_json(run / "run_manifest.json")[("migration_signature")],
        ("maps"): maps,
        ("target"): target,
        ("history"): [
            str(TT_START),
            str(BASE_END),
            20,
            ("raw/date-item-price-channel-desc"),
        ],
        ("content"): {
            p.name: digest(p)
            for p in sorted((run / ("content")).glob("*"))
            if p.is_file()
        },
    }


def verify(folder: Path, expected: dict | None = None) -> dict:
    """Reject partial, changed or corrupt caches instead of rebuilding in place."""
    meta = read_json(folder / "manifest.json")
    if expected is not None and meta["identity"] != expected:
        raise ValueError("Prepared cache identity mismatch")
    for name, record in meta["files"].items():
        p = folder / name
        if (
            not p.is_file()
            or p.stat().st_size != record["bytes"]
            or digest(p) != record[("sha256")]
        ):
            raise ValueError("Prepared cache integrity failure")
    return meta


def daily_rows(path: Path, position: dict) -> Iterator[tuple]:
    """Retain only twenty raw lines and one day's unique positives at a time."""
    records = (r for batch in rows(path, 8192) for r in batch)
    for key, group in groupby(records, lambda r: (r["customer_id"], r["t_dat"])):
        latest, positives = ([], set())
        for row in group:
            item = position[row["article_id"]]
            if len(latest) < 20:
                latest.append((item, row["t_dat"]))
            positives.add(item)
            if len(positives) > 100000:
                raise RuntimeError("blocked_resource: context positive edges")
        yield (key, latest, sorted(positives))


def prepare(
    run: Path, encoder, targets: TargetStream, *, include_internal: bool = True
) -> dict:
    """Persist one context per user/day, complete positives, and ordered target IDs.

    The transaction pass is sorted by user/day with descending deterministic
    within-day ties. A rolling twenty-line history excludes the current day.
    Train targets and internal queries reference disjoint contexts; Q remains
    the selected Train-event distribution. No learned tensor enters this cache.
    """
    folder = run / "prepared/tower"
    expected = identity(run, encoder.maps, targets.identity)
    expected["include_internal"] = include_internal
    if folder.exists():
        emit(
            "prepared_inputs",
            0,
            None,
            unit="contexts",
            status="started",
            message=("Verifying existing model-independent training cache"),
            force=True,
        )
        cached = verify(folder, expected)
        emit(
            "prepared_inputs",
            cached["contexts"],
            cached["contexts"],
            unit="contexts",
            status=("restored"),
            message=("Raw training contexts verified and reused"),
            metrics={("targets"): cached[("count")]},
            force=True,
        )
        return cached
    folder.mkdir(parents=True)
    history, con = (encoder.history_store, encoder.history_store.con)
    catalog = history.catalog
    position = {item: j for j, item in enumerate(catalog)}
    con.execute(
        (
            """CREATE OR REPLACE TEMP TABLE selected_targets AS SELECT * FROM
            read_parquet(?)"""
        ),
        [str(targets.path)],
    )
    con.execute(
        (
            """CREATE OR REPLACE TEMP TABLE contexts AS SELECT *,
            row_number() OVER(ORDER BY customer_id,t_dat)-1 AS context_id FROM
            (SELECT DISTINCT customer_id,t_dat FROM selected_targets UNION
            SELECT DISTINCT customer_id,t_dat FROM transactions
            JOIN cohort USING(customer_id)
            WHERE t_dat>=? AND t_dat<?)"""
        ),
        [TT_END, BASE_END if include_internal else TT_END],
    )
    emit(
        "prepared_inputs",
        0,
        None,
        unit="contexts",
        status="started",
        message=(
            "Counting unique event contexts, complete positives and legal history"
        ),
        force=True,
    )
    context_count = con.execute("SELECT count(*) FROM contexts").fetchone()[0]
    edge_count = con.execute(
        """SELECT count(*) FROM (SELECT DISTINCT
        t.customer_id,t.t_dat,t.article_id
        FROM transactions t JOIN contexts c USING(customer_id,t_dat))"""
    ).fetchone()[0]
    raw_count = con.execute(
        (
            """SELECT count(*) FROM transactions JOIN (SELECT DISTINCT
            customer_id FROM
            contexts) c USING(customer_id) WHERE t_dat>=? AND t_dat<?"""
        ),
        [TT_START, BASE_END],
    ).fetchone()[0]
    required = (
        context_count * (3 * 8 + 20 * 3 * 8 + 20 * 4 + 8)
        + edge_count * 8
        + targets.count * 16
    )
    check(
        run,
        "prepared_contexts",
        memory=128 * 1024**2,
        disk=required * 3 + raw_count * 200,
        details={
            ("contexts"): context_count,
            ("targets"): targets.count,
            ("positive_edges"): edge_count,
            ("transaction_rows"): raw_count,
        },
    )
    export_query(
        con,
        "SELECT * FROM contexts ORDER BY context_id",
        [],
        folder / ("contexts.parquet"),
    )
    source = run / "tmp/context_transactions.parquet"
    con.execute("PRAGMA enable_profiling=json")
    con.execute("SET profiling_output=?", [str(folder / "history_query_profile.json")])
    try:
        export_query(
            con,
            (
                """SELECT t.* FROM transactions t
                JOIN (SELECT DISTINCT customer_id FROM contexts) c
                USING(customer_id)
                WHERE t_dat>=? AND t_dat<? ORDER BY customer_id,t_dat,
                article_id DESC,price DESC,sales_channel_id DESC"""
            ),
            [TT_START, BASE_END],
            source,
        )
    finally:
        con.execute("PRAGMA disable_profiling")
    query_profile = read_json(folder / "history_query_profile.json")

    def array(name, dtype, shape):
        return np.lib.format.open_memmap(
            folder / f"{name}.npy", mode="w+", dtype=dtype, shape=shape
        )

    profiles = array("profiles", np.int64, (context_count, 3))
    histories = array("histories", np.int64, (context_count, 20, 3))
    weights = array("weights", np.float32, (context_count, 20))
    offsets = array("positive_offsets", np.int64, (context_count + 1,))
    positives = array("positive_items", np.int64, (edge_count,))
    codes = array("item_codes", np.int64, (len(catalog), len(ITEM_KEYS)))
    emit(
        "prepared_item_codes",
        0,
        len(catalog),
        unit="items",
        status="started",
        message=("Encoding static item category IDs once"),
        force=True,
    )
    for start in range(0, len(catalog), 512):
        codes[start : start + 512] = encoder.items(catalog[start : start + 512])[
            0
        ].numpy()
        emit(
            "prepared_item_codes",
            min(start + 512, len(catalog)),
            len(catalog),
            unit=("items"),
            message=("Static item category batch written"),
        )
    emit(
        "prepared_item_codes",
        len(catalog),
        len(catalog),
        unit="items",
        status=("completed"),
        message=("Static item category IDs ready"),
        force=True,
    )
    histories[:] = 0
    weights[:] = 0
    iterator = iter(daily_rows(source, position))
    current = next(iterator, None)
    recent, previous_user, edge_offset = ([], None, 0)
    emit(
        "prepared_inputs",
        0,
        context_count,
        unit="contexts",
        status="running",
        message=("Writing profiles, latest-20 history and complete positives"),
        metrics={("targets"): targets.count, ("positive_edges"): edge_count},
        force=True,
    )
    completed_contexts = 0
    for batch in rows(folder / "contexts.parquet", 512):
        encoder.sample.customers = history.customers(
            sorted({r["customer_id"] for r in batch})
        )
        for record in batch:
            user, day, index = (
                record["customer_id"],
                record["t_dat"],
                record[("context_id")],
            )
            if previous_user != user:
                recent = []
                previous_user = user
            while current is not None and current[0] < (user, day):
                if current[0][0] == user:
                    recent = (current[1] + recent)[:20]
                current = next(iterator, None)
            if current is None or current[0] != (user, day):
                raise ValueError("Context lost its complete same-day positives")
            profiles[index] = [
                encoder.maps[k].get(v, 0)
                for k, v in zip(
                    PROFILE_KEYS, encoder._profile_values(user), strict=True
                )
            ]
            if recent:
                indices, dates = zip(*recent, strict=True)
                histories[index, : len(recent)] = codes[list(indices)][:, [0, 2, 4]]
                weights[index, : len(recent)] = [
                    np.exp(-((day - d).days - 1) / 14) for d in dates
                ]
            offsets[index] = edge_offset
            values = current[2]
            positives[edge_offset : edge_offset + len(values)] = values
            edge_offset += len(values)
        completed_contexts += len(batch)
        emit(
            "prepared_inputs",
            completed_contexts,
            context_count,
            unit="contexts",
            message=("Raw context batch written"),
            metrics={
                ("positive_edges_written"): edge_offset,
                ("targets"): targets.count,
            },
        )
    offsets[-1] = edge_offset
    if edge_offset != edge_count:
        raise ValueError("Prepared positive count mismatch")
    export_query(
        con,
        (
            """SELECT context_id,article_id FROM selected_targets
            JOIN contexts USING(customer_id,t_dat)
            ORDER BY sha256('42:' || customer_id || ':' ||
            cast(t_dat AS VARCHAR) || ':' || article_id),
            customer_id,t_dat,article_id"""
        ),
        [],
        folder / ("ordered.parquet"),
    )
    target_ids = array("target_context", np.int64, (targets.count,))
    item_ids = array("target_item", np.int64, (targets.count,))
    start = 0
    emit(
        "prepared_target_order",
        0,
        targets.count,
        unit="targets",
        status="started",
        message=("Copying stable ordered target coordinates"),
        force=True,
    )
    for batch in rows(folder / "ordered.parquet", 8192):
        target_ids[start : start + len(batch)] = [r["context_id"] for r in batch]
        item_ids[start : start + len(batch)] = [
            position[r["article_id"]] for r in batch
        ]
        start += len(batch)
        emit(
            "prepared_target_order",
            start,
            targets.count,
            unit="targets",
            message=("Ordered target-to-context rows written"),
        )
    emit(
        "prepared_target_order",
        start,
        targets.count,
        unit="targets",
        status=("completed"),
        message=("Ordered target coordinates ready"),
        force=True,
    )
    daily = {}
    for j in range((BASE_END - TT_END).days if include_internal else 0):
        day = TT_END + timedelta(days=j)
        path = folder / f"internal-{day}.parquet"
        export_query(
            con,
            (
                """SELECT context_id FROM contexts WHERE t_dat=? ORDER BY
             customer_id"""
            ),
            [day],
            path,
        )
        daily[str(day)] = {
            "file": path.name,
            "active": sorted(position[i] for i in history.active(day)),
        }
    for value in (
        profiles,
        histories,
        weights,
        offsets,
        positives,
        codes,
        target_ids,
        item_ids,
    ):
        value.flush()
    meta = {
        "identity": expected,
        "maps": encoder.maps,
        "catalog": catalog,
        ("probabilities"): targets.probabilities,
        ("count"): targets.count,
        ("contexts"): context_count,
        ("daily"): daily,
        ("history_sql_passes"): 1,
        ("transaction_rows_exported"): raw_count,
        ("history_query_rows_scanned"): query_profile[("cumulative_rows_scanned")],
        ("history_query_seconds"): query_profile[("latency")],
        ("files"): {
            p.name: {("bytes"): p.stat().st_size, ("sha256"): digest(p)}
            for p in sorted(folder.iterdir())
        },
    }
    atomic_json(folder / "manifest.json", meta)
    emit(
        "prepared_inputs",
        context_count,
        context_count,
        unit="contexts",
        status=("completed"),
        message=("Training raw-input cache built and sealed"),
        metrics={
            ("targets"): targets.count,
            ("positive_edges"): edge_count,
            ("transaction_rows_exported"): raw_count,
        },
        force=True,
    )
    return meta


class Prepared:
    """Read only bounded gathers; no SQL or per-row Tensor construction per batch."""

    def __init__(
        self, folder: Path, expected: dict | None = None, *, max_batch: int = 512
    ) -> None:
        self.folder = folder
        self.max_batch = max_batch
        self.meta = verify(folder, expected)
        self.arrays = {
            p.stem: np.load(p, mmap_mode="r", allow_pickle=False)
            for p in folder.glob("*.npy")
        }
        self.count = self.meta["count"]
        self.catalog = self.meta["catalog"]
        self.identity = self.meta["identity"]["target"]
        self.probabilities = self.meta["probabilities"]

    def users(self, contexts: np.ndarray) -> tuple:
        if contexts.ndim != 1 or len(contexts) > self.max_batch:
            raise ValueError("Prepared user gather exceeds the batch contract")
        return tuple(
            torch.from_numpy(np.array(self.arrays[n][contexts], copy=True))
            for n in (("profiles"), ("histories"), ("weights"))
        )

    def mask(self, contexts: np.ndarray, candidates: np.ndarray) -> torch.Tensor:
        """Vectorized membership over ragged full positives, bounded by this batch."""
        if len(contexts) > self.max_batch or len(candidates) > self.max_batch:
            raise ValueError("Prepared positive mask exceeds the batch contract")
        offsets = self.arrays["positive_offsets"]
        values = self.arrays["positive_items"]
        starts, ends = (offsets[contexts], offsets[contexts + 1])
        lengths = ends - starts
        if int(lengths.sum()) > 1000000:
            raise RuntimeError("blocked_resource: batch positive edges")
        row = np.repeat(np.arange(len(contexts)), lengths)
        flat = np.concatenate([values[a:b] for a, b in zip(starts, ends, strict=True)])
        columns = np.searchsorted(candidates, flat)
        valid = columns < len(candidates)
        valid[valid] &= candidates[columns[valid]] == flat[valid]
        mask = np.zeros((len(contexts), len(candidates)), dtype=bool)
        mask[row[valid], columns[valid]] = True
        return torch.from_numpy(mask)

    def batches(self, offset: int, size: int):
        if not 1 <= size <= self.max_batch or not 0 <= offset <= self.count:
            raise ValueError("Invalid prepared batch size or consumed offset")
        for start in range(offset, self.count, size):
            contexts = np.array(
                self.arrays["target_context"][start : start + size], copy=True
            )
            candidates = np.unique(self.arrays["target_item"][start : start + size])
            yield (
                contexts,
                candidates,
                self.users(contexts),
                self.mask(contexts, candidates),
            )
