"""Audit-driven optimized execution with unchanged temporal/model contracts."""

import os
from datetime import date
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pytest
import torch

from hm_recsys.core.contracts import ROOT, digest, write_json
from hm_recsys.data.history import History, canonical, rosters
from hm_recsys.data.streaming import Partition
from hm_recsys.features.content import MappedContent, adapt_content
from hm_recsys.features.storage import FeatureShards
from hm_recsys.retrieval.snapshot import build_snapshot
from hm_recsys.retrieval.two_tower.inputs import SqlEncoder
from hm_recsys.retrieval.two_tower.query_inputs import QueryInputs
from hm_recsys.retrieval.vectors import ActiveBlocks, BlockIndex, block_topk
from hm_recsys.retrieval.wals.model import fold_user, fold_users_batch
from hm_recsys.runtime.files import DigestCache
from hm_recsys.runtime.profiles import load_profile
from hm_recsys.runtime.testing import create_full_source


def test_query_inputs_full_roster_and_corrupt_cache(tmp_path: Path) -> None:
    """Reuse exact raw-line histories; inference never opens a future label file."""
    torch.set_num_threads(2)
    source, run = (tmp_path / "source", tmp_path / "run")
    create_full_source(source)
    run.mkdir()
    profile = load_profile(ROOT, "full")
    canonical(run, source, profile)
    adapt_content(run, source)
    rosters(run, profile)
    write_json(run / "run_manifest.json", {"migration_signature": "fixture"})
    history = History(run)
    encoder = SqlEncoder(history, MappedContent(run))
    part = Partition(run, "Validation", labels=False)
    cache = QueryInputs(encoder, part)
    before = date.fromisoformat(part.meta["start"])
    offset = 0
    for users in part.batches():
        reference = encoder.users([(u, before) for u in users])
        actual = cache.users(offset, len(users))
        for a, b in zip(reference, actual, strict=True):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        offset += len(users)
    assert offset == part.count
    assert QueryInputs(encoder, part).meta == cache.meta
    with pytest.raises(ValueError, match="slice"):
        cache.users(0, 513)
    path = run / "prepared/queries/Validation/weights.npy"
    with path.open("r+b") as handle:
        handle.seek(-1, 2)
        handle.write(b"X")
    with pytest.raises(ValueError, match="integrity"):
        QueryInputs(encoder, part)
    history.close()


def test_batched_wals_and_exact_ties() -> None:
    """Retain fixed-item equations and exact ID ties over physical block boundaries."""
    rng = np.random.default_rng(42)
    factors = rng.normal(size=(6001, 16)).astype(np.float32)
    gram = factors.astype(np.float64).T @ factors.astype(np.float64)
    observed = [{}, {4: 3.0}, {0: 1.01, 4: 7.5, 23: 20.0}]
    reference = np.stack([fold_user(factors, obs, gram=gram) for obs in observed])
    actual = fold_users_batch(factors, observed, gram=gram)
    np.testing.assert_array_equal(reference, actual)
    ids = [f"{i:010d}" for i in range(len(factors))]
    factors[:100] = 1
    factors[4090:4110] = 1
    queries = np.ones((3, 16), dtype=np.float32)
    queries[1] = 0
    active = set(ids[::2])
    index = BlockIndex(ids, factors)
    optimized = ActiveBlocks(index, active)
    assert optimized.search(queries, 50) == block_topk(
        ids, factors, queries, active, 50
    )
    assert ActiveBlocks(index, set()).search(queries, 50) == [[], [], []]
    with pytest.raises(ValueError, match="batch"):
        fold_users_batch(factors, [{}] * 513, gram=gram)


def test_itemcf_seed_layout_and_decay_parity(tmp_path: Path) -> None:
    """Keep global pair population, timestamp kernel and directed neighbor ordering."""
    source, run = (tmp_path / "source", tmp_path / "run")
    create_full_source(source)
    run.mkdir()
    canonical(run, source, load_profile(ROOT, "full"))
    cutoff = date(2020, 9, 8)
    build_snapshot(run, "reference", cutoff)
    build_snapshot(run, "optimized", cutoff, optimized=True)
    values = []
    for name in ("reference", "optimized"):
        con = duckdb.connect(
            str(run / "indexes" / f"itemcf_{name}.duckdb"), read_only=True
        )
        values.append(
            con.execute(
                "SELECT seed,article_id,score FROM neighbors "
                "ORDER BY seed,score DESC,article_id"
            ).fetchall()
        )
        con.close()
    assert [(s, i) for s, i, _ in values[0]] == [(s, i) for s, i, _ in values[1]]
    np.testing.assert_allclose(
        [r[2] for r in values[0]], [r[2] for r in values[1]], rtol=1e-12, atol=1e-14
    )


def test_incremental_shards_and_stat_receipt_invalidation(tmp_path: Path) -> None:
    """Keep whole/empty groups and rehash same-size changes despite restored mtime."""
    folder = tmp_path / "features"
    folder.mkdir()
    writer = FeatureShards(tmp_path, folder, None)
    table = pa.table({"customer_id": ["u"], "article_id": ["0000000001"], "x": [1.0]})
    groups = [{"user": "u", "rows": 1}, {"user": "empty", "rows": 0}]
    assert writer.append(table, groups, None) is None
    assert writer.append(table, groups, None) is None
    entry = writer.finish()
    assert entry is not None and entry["rows"] == 2 and (len(entry["groups"]) == 4)
    writer.close()
    path = folder / "part-00000.parquet"
    cache = DigestCache()
    first = cache.digest(path)
    assert cache.digest(path) == first and cache.hits == 1 and (cache.misses == 1)
    stat = path.stat()
    value = path.read_bytes()
    path.write_bytes(value[:-1] + bytes([value[-1] ^ 1]))
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert cache.digest(path) != first and cache.misses == 2
    with pytest.raises(ValueError, match="checksum"):
        cache.verify(folder, {path.name: {"bytes": len(value), "sha256": first}})
    fresh = DigestCache()
    assert fresh.digest(path) == digest(path) and fresh.misses == 1


def test_kernel_comparison_evidence(tmp_path: Path) -> None:
    """Measure identical bounded equations/search/hash work without speed assertions."""
    from time import perf_counter

    from hm_recsys.core.contracts import write_json

    rng = np.random.default_rng(17)
    factors = rng.normal(size=(6001, 64)).astype(np.float32)
    gram = factors.astype(np.float64).T @ factors.astype(np.float64)
    observed = [
        {int(j): float(j + 2) for j in rng.choice(6001, 20, replace=False)}
        for _ in range(32)
    ]
    ids = [f"{i:010d}" for i in range(len(factors))]
    active = set(ids[::2])
    queries = rng.normal(size=(32, 64)).astype(np.float32)
    started = perf_counter()
    prepared = ActiveBlocks(BlockIndex(ids, factors), active)
    cache_seconds = perf_counter() - started
    operations = {
        "wals_reference": lambda: np.stack(
            [fold_user(factors, o, gram=gram) for o in observed]
        ),
        "wals_optimized": lambda: fold_users_batch(factors, observed, gram=gram),
        "search_reference": lambda: block_topk(ids, factors, queries, active, 50),
        "search_optimized": lambda: prepared.search(queries, 50),
    }
    results = {}
    for name, operation in operations.items():
        operation()
        started = perf_counter()
        for _ in range(10):
            operation()
        results[name] = (perf_counter() - started) / 10
    write_json(
        tmp_path / "kernel_comparison.json",
        {
            "seconds_per_32_queries": results,
            "active_cache_seconds": cache_seconds,
            "items": 6001,
            "active": len(active),
            "dimensions": 64,
            "repeats": 10,
            "device": "cpu",
        },
    )
