"""Deterministic offline fixture with real learning signal and missing content."""

# Use literal CSV fields without losing zero-padded article IDs.
import csv

# Use calendar-day cutoffs and non-overlapping target intervals.
from datetime import date, timedelta

# Use portable paths resolved beneath project-owned roots.
from pathlib import Path

# Use typed vectorized arrays, stable indexing and numerical linear algebra.
import numpy as np

# Use bounded columnar records and portable Parquet/IPC schemas.
import pyarrow as pa

# Use bounded columnar records and portable Parquet/IPC schemas.
import pyarrow.parquet as pq

# Reuse write_json from hm_recsys.core.contracts so this path shares the project's
# data/model contract.
from hm_recsys.core.contracts import write_json

# Reuse ARTICLE_FIELDS from hm_recsys.data.sample so this path shares the project's
# data/model contract.
from hm_recsys.data.sample import ARTICLE_FIELDS


# Write tiny source-shaped inputs, including held-out and missing items.
def create_fixture(root: Path) -> None:
    """Write tiny source-shaped inputs, including held-out and missing items.

    This fixture is never mixed with real inputs. Forty items have repeated
    historical support; the final item occurs only in future truth. Repeated
    purchase lines deliberately remain in the CSV.
    """
    # Compute raw with the existing arithmetic, preserving calendar-day units and
    # numeric precision.
    raw = root / "original data"
    # Create only the required owned directory; immutable inputs and other runs
    # remain untouched.
    raw.mkdir(parents=True, exist_ok=False)
    # Collect article IDs kept as ten-character strings in row order using the
    # declared filter/order, preserving original key-to-row associations.
    ids = [f"{i:010d}" for i in range(1, 43)]
    # Keep the file handle scoped to this operation so it closes before later
    # validation/publication.
    with (raw / "articles.csv").open("w", newline="") as handle:
        # Resolve writer for write tiny source-shaped inputs, including held-out and
        # missing items before the downstream consumer uses it.
        writer = csv.DictWriter(handle, fieldnames=["article_id", *ARTICLE_FIELDS])
        # Invoke writer.writeheader for write tiny source-shaped inputs, including
        # held-out and missing items at the declared side-effect boundary.
        writer.writeheader()
        # Process each (i, item) from the declared enumerate(ids) sequence for write
        # tiny source-shaped inputs, including held-out and missing items.
        for i, item in enumerate(ids):
            # Invoke writer.writerow for write tiny source-shaped inputs, including
            # held-out and missing items at the declared side-effect boundary.
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
    # Collect customer IDs or user factors aligned with query order using the
    # declared filter/order, preserving original key-to-row associations.
    users = [f"fixture-user-{i:03d}" for i in range(24)]
    # Keep the file handle scoped to this operation so it closes before later
    # validation/publication.
    with (raw / "customers.csv").open("w", newline="") as handle:
        # Resolve writer for write tiny source-shaped inputs, including held-out and
        # missing items before the downstream consumer uses it.
        writer = csv.writer(handle)
        # Invoke writer.writerow for write tiny source-shaped inputs, including
        # held-out and missing items at the declared side-effect boundary.
        writer.writerow(
            ["customer_id", "age", "club_member_status", "fashion_news_frequency"]
        )
        # Process each (i, user) from the declared enumerate(users) sequence for
        # write tiny source-shaped inputs, including held-out and missing items.
        for i, user in enumerate(users):
            # Invoke writer.writerow for write tiny source-shaped inputs, including
            # held-out and missing items at the declared side-effect boundary.
            writer.writerow([user, 22 + i, "ACTIVE", "Regularly"])
    # Initialize bounded records or row positions consumed by the current operation
    # with the explicit structured fields/empty state used by this operation.
    rows = []
    # Resolve begin for write tiny source-shaped inputs, including held-out and
    # missing items before the downstream consumer uses it.
    begin = date(2020, 2, 1)
    # Process each day index from the declared days) sequence for write tiny
    # source-shaped inputs, including held-out and missing items.
    for day_index in range((date(2020, 9, 16) - begin).days):
        # Compute calendar purchase date without invented intraday ordering with the
        # existing arithmetic, preserving calendar-day units and numeric precision.
        day = begin + timedelta(days=day_index)
        # Choose the day_index % 3 case required by write tiny source-shaped inputs,
        # including held-out and missing items.
        if day_index % 3:
            # Skip this explicitly inapplicable/empty record while retaining the
            # order of all remaining valid inputs.
            continue
        # Process each (u, user) from the declared enumerate(users) sequence for
        # write tiny source-shaped inputs, including held-out and missing items.
        for u, user in enumerate(users):
            # Persistent individual preference plus rotating variant support.
            item = ids[(u * 2 + day_index % 2) % 40]
            # Retain the current records in bounded records or row positions consumed
            # by the current operation, preserving explicit row/key associations and
            # encounter order.
            rows.append([str(day), user, item, 0.01 + (u % 5) * 0.01, 1 + u % 2])
            # Read other from article IDs kept as ten-character strings in row order
            # at the explicit row/key coordinates.
            other = ids[(u * 2 + 2 + day_index // 3) % 40]
            # Retain the current records in bounded records or row positions consumed
            # by the current operation, preserving explicit row/key associations and
            # encounter order.
            rows.append([str(day), user, other, 0.02, 2])
            # Choose the u % 7 == 0 case required by write tiny source-shaped inputs,
            # including held-out and missing items.
            if u % 7 == 0:
                # Retain the current records in bounded records or row positions
                # consumed by the current operation, preserving explicit row/key
                # associations and encounter order.
                rows.append(rows[-1].copy())
    # Every ranker week has native positives and full-truth misses.
    for day in (
        date(2020, 8, 13),
        date(2020, 8, 20),
        date(2020, 8, 27),
        date(2020, 9, 3),
        date(2020, 9, 10),
    ):
        # Process each (u, user) from the declared enumerate(users) sequence for
        # write tiny source-shaped inputs, including held-out and missing items.
        for u, user in enumerate(users):
            # Retain the current records in bounded records or row positions consumed
            # by the current operation, preserving explicit row/key associations and
            # encounter order.
            rows.append([str(day), user, ids[u * 2 % 40], 0.02, 2])
            # Retain the current records in bounded records or row positions consumed
            # by the current operation, preserving explicit row/key associations and
            # encounter order.
            rows.append([str(day), user, ids[-1], 0.02, 2])
    # A legal historically selected user whose Validation ground truth is all
    # outside the selected catalog. This user remains in the metric population.
    rows = [
        r
        for r in rows
        if not (r[1] == users[-1] and "2020-09-09" <= r[0] < "2020-09-16")
    ]
    # Retain the current records in bounded records or row positions consumed by the
    # current operation, preserving explicit row/key associations and encounter
    # order.
    rows.append(["2020-09-10", users[-1], ids[-1], 0.02, 2])
    # Keep the file handle scoped to this operation so it closes before later
    # validation/publication.
    with (raw / "transactions_train.csv").open("w", newline="") as handle:
        # Resolve writer for write tiny source-shaped inputs, including held-out and
        # missing items before the downstream consumer uses it.
        writer = csv.writer(handle)
        # Invoke writer.writerow for write tiny source-shaped inputs, including
        # held-out and missing items at the declared side-effect boundary.
        writer.writerow(
            ["t_dat", "customer_id", "article_id", "price", "sales_channel_id"]
        )
        # Invoke writer.writerows for write tiny source-shaped inputs, including
        # held-out and missing items at the declared side-effect boundary.
        writer.writerows(rows)
    # Resolve rng for write tiny source-shaped inputs, including held-out and missing
    # items before the downstream consumer uses it.
    rng = np.random.default_rng(42)
    # Process each (modality, dimension) from the declared (('text', 12), ('image',
    # 16)) sequence for write tiny source-shaped inputs, including held-out and
    # missing items.
    for modality, dimension in (("text", 12), ("image", 16)):
        # Compute stage-owned directory for the current artifact family with the
        # existing arithmetic, preserving calendar-day units and numeric precision.
        folder = root / f"{modality}_features"
        # Create only the required owned directory; immutable inputs and other runs
        # remain untouched.
        folder.mkdir()
        # Resolve available through the explicit available/absent branch without
        # fabricating a replacement value.
        available = ids if modality == "text" else ids[1:]
        # Resolve order for write tiny source-shaped inputs, including held-out and
        # missing items before the downstream consumer uses it.
        order = rng.permutation(len(available))
        # Collect mapped using the declared filter/order, preserving original
        # key-to-row associations.
        mapped = [available[i] for i in order]
        # For content or learned vectors aligned with the corresponding explicit row
        # map: convert the numeric representation explicitly at the model/artifact
        # boundary.
        vectors = rng.normal(size=(len(mapped), dimension)).astype(np.float32)
        # Accumulate content or learned vectors aligned with the corresponding
        # explicit row map using the existing arithmetic and encounter order.
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        # Persist the complete requested state to its separate model/index artifact
        # path.
        np.save(folder / f"{modality}_embeddings.npy", vectors, allow_pickle=False)
        # Invoke pq.write_table for write tiny source-shaped inputs, including
        # held-out and missing items at the declared side-effect boundary.
        pq.write_table(
            pa.table({"article_id": mapped, "embedding_row": list(range(len(mapped)))}),
            folder / f"{modality}_article_ids.parquet",
        )
        # Write measured state/lineage to the owned run path without mutating source
        # manifests.
        write_json(
            folder / f"{modality}_features_manifest.json",
            {
                "embedding_dim": dimension,
                "embedding_dtype": "float32",
                "normalization": "L2",
                "model_id": "synthetic-fixture-only",
            },
        )
        # Choose the modality == 'image' case required by write tiny source-shaped
        # inputs, including held-out and missing items.
        if modality == "image":
            # Invoke pq.write_table for write tiny source-shaped inputs, including
            # held-out and missing items at the declared side-effect boundary.
            pq.write_table(
                pa.table({"article_id": [ids[0]], "status": ["fixture_missing"]}),
                folder / "image_embedding_failures.parquet",
            )
