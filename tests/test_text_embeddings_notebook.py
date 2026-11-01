"""Offline tests for the standalone article text extraction notebook."""

import csv
import json
from pathlib import Path
from typing import Any

import faiss
import nbformat
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

NOTEBOOK = (
    Path(__file__).resolve().parents[1]
    / "notebooks/03_text_embeddings_extraction.ipynb"
)


@pytest.fixture
def definitions() -> dict[str, Any]:
    """Load definitions only; never install, mount, download or encode."""
    ns: dict[str, Any] = {}
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            compile(cell.source, str(NOTEBOOK), "exec")
            if "definitions" in cell.metadata.get("tags", []):
                exec(cell.source, ns)
    return ns


def test_text_contract(definitions: dict[str, Any]) -> None:
    """Verify exact fields, value order, null handling and excluded metadata."""
    columns = definitions["TEXT_COLUMNS"]
    assert columns == [
        "prod_name",
        "product_type_name",
        "product_group_name",
        "graphical_appearance_name",
        "colour_group_name",
        "perceived_colour_value_name",
        "perceived_colour_master_name",
        "department_name",
        "index_code",
        "index_name",
        "index_group_name",
        "section_name",
        "garment_group_name",
        "detail_desc",
    ]
    row = dict(zip(columns, [f"value{i}" for i in range(14)], strict=True))
    row.update(
        article_id="0000000001", product_code="excluded", colour_group_code="999"
    )
    assert definitions["joined_text"](row) == "|".join(f"value{i}" for i in range(14))
    cleaned = dict.fromkeys(columns)
    cleaned.update(
        prod_name="  same\t value ",
        product_type_name="same value",
        product_group_name="Unknown",
        graphical_appearance_name="Other",
        colour_group_name="None",
        detail_desc=float("nan"),
        section_name="  ",
    )
    assert (
        definitions["joined_text"](cleaned)
        == "same value|same value|Unknown|Other|None"
    )


def test_csv_ids_and_identical_text(
    definitions: dict[str, Any], tmp_path: Path
) -> None:
    """Preserve leading zeros and identical descriptions; reject invalid inputs."""
    path = tmp_path / "articles.csv"
    columns = ["article_id", *definitions["TEXT_COLUMNS"]]

    def write(ids: list[str], text: str = "None") -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for item in ids:
                writer.writerow({"article_id": item, "prod_name": text})

    write(["0000000002", "0000000001"])
    assert definitions["read_articles"](path) == (
        ["0000000001", "0000000002"],
        ["None", "None"],
    )
    for ids in (["0000000001"] * 2, [""], ["123"], [" 0000000001"]):
        write(ids)
        with pytest.raises(ValueError):
            definitions["read_articles"](path)
    write(["0000000001"], " ")
    with pytest.raises(ValueError, match="no usable"):
        definitions["read_articles"](path)


def test_artifact_reload_ties_and_publication(
    definitions: dict[str, Any], tmp_path: Path
) -> None:
    """Validate identical-vector ties, alignment, hashes and immutable publication."""
    faiss.omp_set_num_threads(1)
    local = tmp_path / "local"
    local.mkdir()
    ids = ["0000000001", "0000000002"]
    vectors = np.zeros((2, 384), dtype=np.float32)
    vectors[:, 0] = 1
    np.save(local / "text_embeddings.npy", vectors)
    table = pa.Table.from_arrays(
        [
            pa.array([0, 1], type=pa.int64()),
            pa.array(ids),
            pa.array(["same", "same"]),
            pa.array([3, 300], type=pa.int32()),
            pa.array([False, True]),
        ],
        schema=definitions["MAPPING_SCHEMA"],
    )
    pq.write_table(table, local / "text_article_ids.parquet")
    index = faiss.IndexFlatIP(384)
    index.add(vectors)
    faiss.write_index(index, str(local / "text_faiss.index"))
    result = definitions["validate_artifacts"](local, ids, ["same", "same"], 256)
    assert result["validation_status"] == "validated"
    manifest = {
        "validation_status": "validated",
        "files": {
            name: {
                "bytes": (local / name).stat().st_size,
                "sha256": definitions["sha256_file"](local / name),
            }
            for name in definitions["ARTIFACTS"]
        },
    }
    dest = tmp_path / "published"
    definitions["publish_verified"](local, dest, manifest)
    assert json.loads((dest / definitions["MANIFEST_NAME"]).read_text()) == manifest
    before = {p.name: p.read_bytes() for p in dest.iterdir()}
    with pytest.raises(ValueError, match="nonempty"):
        definitions["publish_verified"](local, dest, manifest)
    assert before == {p.name: p.read_bytes() for p in dest.iterdir()}
    with pytest.raises(ValueError, match="mapping mismatch"):
        definitions["validate_artifacts"](local, ids[::-1], ["same", "same"], 256)
    vectors[0] = 0
    np.save(local / "text_embeddings.npy", vectors)
    with pytest.raises(ValueError, match="embedding"):
        definitions["validate_artifacts"](local, ids, ["same", "same"], 256)
    with pytest.raises(ValueError, match="Checksum"):
        definitions["verify_files"](local, manifest)
