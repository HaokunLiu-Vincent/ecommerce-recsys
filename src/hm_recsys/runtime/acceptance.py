"""Validate immutable source headers, ID maps and split contracts.

Inputs are the three existing source directories and a bounded execution config.
Outputs are acceptance and split JSON under the owning run. ContentSource checks
metadata without extraction; no model is fitted and no Test labels are built.
"""

import csv
import importlib.metadata
from pathlib import Path

from hm_recsys.core.contracts import (
    Config,
    article_id,
    source_identity,
    split_manifest,
    validate_splits,
    write_json,
)
from hm_recsys.data.sample import (
    ARTICLE_FIELDS,
    CUSTOMER_FIELDS,
    inspect_csv,
)
from hm_recsys.features.content import ContentSource


def catalog_ids(root: Path) -> set[str]:
    """Read static catalog identity only, retaining leading zeros."""
    with (root / "original data/articles.csv").open(newline="") as handle:
        ids = [article_id(row["article_id"]) for row in csv.DictReader(handle)]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate catalog IDs")
    return set(ids)


def preflight(run: Path, source: Path, config: Config) -> None:
    """Inspect input headers/maps and CPU dependencies without training models."""
    validate_splits(split_manifest())
    report: dict = {"status": "accepted_headers_and_maps", "csv": {}, "content": {}}
    for name, required in (
        ("articles.csv", {"article_id", *ARTICLE_FIELDS}),
        ("customers.csv", {"customer_id", *CUSTOMER_FIELDS}),
        (
            "transactions_train.csv",
            {"customer_id", "article_id", "t_dat", "price", "sales_channel_id"},
        ),
    ):
        report["csv"][name] = inspect_csv(source / "original data" / name, required)
    catalog = catalog_ids(source)
    for modality in ("text", "image"):
        content = ContentSource(source, modality, catalog)
        report["content"][modality] = content.report
    report["versions"] = {
        name: importlib.metadata.version(name)
        for name in (
            "duckdb",
            "numpy",
            "pyarrow",
            "scipy",
            "scikit-learn",
            "implicit",
            "torch",
            "faiss-cpu",
            "lightgbm",
            "psutil",
        )
    }
    report["inputs"] = {
        str(p.relative_to(source)): source_identity(p)
        for folder in ("original data", "text_features", "image_features")
        for p in (source / folder).iterdir()
        if p.is_file()
    }
    report["config"] = config.values
    # Write measured state/lineage to the owned run path without mutating source
    # manifests.
    write_json(run / "input_acceptance.json", report)
    # Write measured state/lineage to the owned run path without mutating source
    # manifests.
    write_json(
        run / "split_manifest.json", {"splits": split_manifest(), "test_enabled": False}
    )
