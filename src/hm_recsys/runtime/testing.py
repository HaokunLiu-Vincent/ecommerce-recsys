"""Tiny deterministic fixture authorization, never a production CLI bypass."""

import copy
import tempfile
from pathlib import Path

from hm_recsys.core.contracts import digest
from hm_recsys.core.synthetic import create_fixture
from hm_recsys.runtime.files import INPUTS
from hm_recsys.runtime.profiles import load_profile


# Create the exact existing 42-item synthetic source plus opaque unused indexes.
def create_source(root: Path) -> None:
    """Create the exact existing 42-item synthetic source plus opaque unused indexes."""
    create_fixture(root)
    for modality in ("text", "image"):
        (root / f"{modality}_features/{modality}_faiss.index").write_bytes(
            b"synthetic unused content index"
        )


# Match every source byte to a generated fixture before allowing local fixture.
def assert_fixture(source: Path, *, full: bool = False) -> None:
    """Match every source byte to a generated fixture before allowing local fixture.

    Arbitrary tiny real data or a self-declared marker is not sufficient. There
    is no CLI flag disabling the production host gate.
    """
    with tempfile.TemporaryDirectory(prefix="hm-fixture-reference-") as directory:
        variants = (False, True) if full else (False,)
        for include_test in variants:
            reference = Path(directory) / str(include_test) / "data"
            if full:
                create_full_source(reference, include_test=include_test)
            else:
                create_source(reference)
            matches = all(
                (source / folder / name).is_file()
                and (source / folder / name).stat().st_size
                == (reference / folder / name).stat().st_size
                and digest(source / folder / name) == digest(reference / folder / name)
                for folder, names in INPUTS.items()
                for name in names
            )
            if matches:
                return
        raise ValueError("Local fixture requires exact generated synthetic input bytes")


# Reduce compute only for the authenticated fixture, including two real epochs.
def fixture_profile(project: Path) -> dict:
    """Reduce compute only for the authenticated fixture, including two real epochs."""
    profile = copy.deepcopy(load_profile(project, "smoke"))
    profile["device"] = "cpu"
    profile["training_subset"] = {"base_users": 12}
    profile["compute"].update(
        wals_factors=8,
        wals_iterations=2,
        tt_batch_size=16,
        tt_epochs=2,
        tt_max_targets=96,
        tt_max_steps=12,
        cluster_count=8,
        cluster_max_iter=10,
        max_queries_per_fold=24,
        ranker_rounds=10,
        ranker_num_leaves=7,
        ranker_min_child_samples=5,
        ranker_early_stopping=5,
    )
    profile["checkpoint_every_steps"] = 2
    profile["synthetic_fixture_only"] = True
    return profile


# Exercise the canonical stages on byte-authenticated bounded synthetic data.
def execute_fixture(
    project: Path, source: Path, local: Path, persistent: Path, *, until: str = "reload"
) -> dict:
    """Exercise the canonical stages on byte-authenticated bounded synthetic data."""
    from hm_recsys.runtime.runner import _execute_profile

    assert_fixture(source)
    return _execute_profile(
        project,
        source,
        local,
        persistent,
        profile=fixture_profile(project),
        until=until,
        synthetic_fixture=True,
    )


# Shrink only input data, retaining enough legal images for real K=128 fitting.
def create_full_source(root: Path, *, include_test: bool = False) -> None:
    """Shrink only input data, retaining enough legal images for real K=128 fitting."""
    import csv

    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    create_source(root)
    raw = root / "original data"
    with (raw / "articles.csv").open(encoding="utf-8", newline="") as handle:
        articles = list(csv.DictReader(handle))
    additional = [f"{j:010d}" for j in range(43, 171)]
    for j, item in enumerate(additional):
        articles.append(
            {
                **articles[j % 40],
                "article_id": item,
                "product_code": str(800000 + j // 3),
            }
        )
    with (raw / "articles.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(articles[0]))
        writer.writeheader()
        writer.writerows(articles)
    with (raw / "customers.csv").open(encoding="utf-8", newline="") as handle:
        users = [r["customer_id"] for r in csv.DictReader(handle)]
    with (raw / "transactions_train.csv").open(encoding="utf-8", newline="") as handle:
        transactions = list(csv.DictReader(handle))
    for j, item in enumerate(additional):
        for day in ("2020-07-15", "2020-07-16", "2020-08-04"):
            transactions.append(
                {
                    "t_dat": day,
                    "customer_id": users[j % len(users)],
                    "article_id": item,
                    "price": "0.02",
                    "sales_channel_id": "2",
                }
            )
    for row in transactions:
        if row["customer_id"] == users[-1] and row["t_dat"] == "2020-09-10":
            row["article_id"] = additional[-1]
    if include_test:
        for index, user in enumerate(users):
            transactions.append(
                {
                    "t_dat": "2020-09-17",
                    "customer_id": user,
                    "article_id": f"{1 + index % 40:010d}",
                    "price": "0.02",
                    "sales_channel_id": "2",
                }
            )
    with (raw / "transactions_train.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(transactions[0]))
        writer.writeheader()
        writer.writerows(transactions)
    rng = np.random.default_rng(231)
    for modality in ("text", "image"):
        folder = root / f"{modality}_features"
        path = folder / f"{modality}_embeddings.npy"
        vectors = np.load(path, allow_pickle=False)
        extra = rng.normal(size=(len(additional), vectors.shape[1])).astype(np.float32)
        extra /= np.linalg.norm(extra, axis=1, keepdims=True)
        np.save(path, np.concatenate([vectors, extra]), allow_pickle=False)
        mapping_path = folder / f"{modality}_article_ids.parquet"
        mapped = pq.read_table(mapping_path)["article_id"].to_pylist() + additional
        pq.write_table(
            pa.table({"article_id": mapped, "embedding_row": list(range(len(mapped)))}),
            mapping_path,
        )


# Run the unchanged full compute configuration on authenticated tiny inputs.
def execute_full_fixture(
    project: Path,
    source: Path,
    local: Path,
    persistent: Path,
    *,
    until: str = "reload",
    allow_full: bool = False,
    performance: str = "reference",
) -> dict:
    """Run the unchanged full compute configuration on authenticated tiny inputs."""
    from hm_recsys.runtime.runner import _execute_profile

    if not allow_full:
        raise ValueError("Full requires explicit authorization, including fixtures")
    assert_fixture(source, full=True)
    from hm_recsys.runtime.settings import apply

    profile = apply(load_profile(project, "full"), project, performance)
    profile["device"] = "cpu"
    return _execute_profile(
        project,
        source,
        local,
        persistent,
        profile=profile,
        until=until,
        synthetic_fixture=True,
    )
