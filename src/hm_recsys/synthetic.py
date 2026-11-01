"""Deterministic offline fixture with real learning signal and missing content."""

import csv
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from hm_recsys.contracts import write_json
from hm_recsys.data import ARTICLE_FIELDS


def create_fixture(root: Path) -> None:
    """Write tiny source-shaped inputs, including held-out and missing items.

    This fixture is never mixed with real inputs. Forty items have repeated
    historical support; the final item occurs only in future truth. Repeated
    purchase lines deliberately remain in the CSV.
    """
    raw = root / "original data"
    raw.mkdir(parents=True, exist_ok=False)
    ids = [f"{i:010d}" for i in range(1, 43)]
    with (raw / "articles.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["article_id", *ARTICLE_FIELDS])
        writer.writeheader()
        for i, item in enumerate(ids):
            writer.writerow(
                {
                    "article_id": item,
                    "product_code": str(i // 2),
                    "product_type_no": str(i % 5),
                    "graphical_appearance_no": str(i % 3),
                    "colour_group_code": str(i % 4),
                    "section_no": str(i % 3),
                    "garment_group_no": str(i % 4),
                    "index_group_no": str(i % 2),
                }
            )
    users = [f"fixture-user-{i:03d}" for i in range(24)]
    with (raw / "customers.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["customer_id", "age", "club_member_status", "fashion_news_frequency"]
        )
        for i, user in enumerate(users):
            writer.writerow([user, 22 + i, "ACTIVE", "Regularly"])
    rows = []
    begin = date(2020, 2, 1)
    for day_index in range((date(2020, 9, 16) - begin).days):
        day = begin + timedelta(days=day_index)
        if day_index % 3:
            continue
        for u, user in enumerate(users):
            # Persistent individual preference plus rotating variant support.
            item = ids[(u * 2 + day_index % 2) % 40]
            rows.append([str(day), user, item, 0.01 + (u % 5) * 0.01, 1 + u % 2])
            other = ids[(u * 2 + 2 + day_index // 3) % 40]
            rows.append([str(day), user, other, 0.02, 2])
            if u % 7 == 0:
                rows.append(rows[-1].copy())
    # Every ranker week has native positives and full-truth misses.
    for day in (
        date(2020, 8, 13),
        date(2020, 8, 20),
        date(2020, 8, 27),
        date(2020, 9, 3),
        date(2020, 9, 10),
    ):
        for u, user in enumerate(users):
            rows.append([str(day), user, ids[u * 2 % 40], 0.02, 2])
            rows.append([str(day), user, ids[-1], 0.02, 2])
    # A legal historically selected user whose Validation ground truth is all
    # outside the selected catalog. This user remains in the metric population.
    rows = [
        r
        for r in rows
        if not (r[1] == users[-1] and "2020-09-09" <= r[0] < "2020-09-16")
    ]
    rows.append(["2020-09-10", users[-1], ids[-1], 0.02, 2])
    with (raw / "transactions_train.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["t_dat", "customer_id", "article_id", "price", "sales_channel_id"]
        )
        writer.writerows(rows)
    rng = np.random.default_rng(42)
    for modality, dimension in (("text", 12), ("image", 16)):
        folder = root / f"{modality}_features"
        folder.mkdir()
        available = ids if modality == "text" else ids[1:]
        order = rng.permutation(len(available))
        mapped = [available[i] for i in order]
        vectors = rng.normal(size=(len(mapped), dimension)).astype(np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        np.save(folder / f"{modality}_embeddings.npy", vectors, allow_pickle=False)
        pq.write_table(
            pa.table({"article_id": mapped, "embedding_row": list(range(len(mapped)))}),
            folder / f"{modality}_article_ids.parquet",
        )
        write_json(
            folder / f"{modality}_features_manifest.json",
            {
                "embedding_dim": dimension,
                "embedding_dtype": "float32",
                "normalization": "L2",
                "model_id": "synthetic-fixture-only",
            },
        )
        if modality == "image":
            pq.write_table(
                pa.table({"article_id": [ids[0]], "status": ["fixture_missing"]}),
                folder / "image_embedding_failures.parquet",
            )
