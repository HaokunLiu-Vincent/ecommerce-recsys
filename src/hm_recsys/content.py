"""Read-only separate article-to-row adapters for supplied content vectors."""

from pathlib import Path

import numpy as np
import numpy.typing as npt
import pyarrow.parquet as pq

from hm_recsys.contracts import article_id, digest, read_json, source_identity

FloatArray = npt.NDArray[np.float32]


class ContentSource:
    """Validate one mapping and header, then read only explicitly consumed rows."""

    def __init__(self, root: Path, modality: str, catalog: set[str]) -> None:
        folder = root / f"{modality}_features"
        self.modality = modality
        self.manifest = read_json(folder / f"{modality}_features_manifest.json")
        array_path = folder / f"{modality}_embeddings.npy"
        self.array = np.load(array_path, mmap_mode="r", allow_pickle=False)
        if self.array.ndim != 2 or self.array.dtype != np.float32:
            raise ValueError("Content must be a two-dimensional float32 array")
        expected = self.manifest.get(
            "embedding_dim", self.manifest.get("contract", {}).get("embedding_dim")
        )
        if expected is None or expected != self.array.shape[1]:
            raise ValueError("Content manifest dimension conflict")
        declared_shape = self.manifest.get("embedding_shape")
        if declared_shape and tuple(declared_shape) != self.array.shape:
            raise ValueError("Content manifest shape conflict")
        if self.manifest.get("embedding_dtype") != str(self.array.dtype):
            raise ValueError("Content manifest dtype conflict")
        mapping_path = folder / f"{modality}_article_ids.parquet"
        table = pq.read_table(mapping_path, columns=["article_id", "embedding_row"])
        ids = [article_id(v) for v in table["article_id"].to_pylist()]
        rows = table["embedding_row"].to_pylist()
        if len(ids) != len(self.array) or len(set(ids)) != len(ids):
            raise ValueError("Content mapping duplicate or row count mismatch")
        if sorted(rows) != list(range(len(self.array))):
            raise ValueError("Content row indexes are not a complete permutation")
        if not set(ids) <= catalog:
            raise ValueError("Content mapping has catalog join failures")
        self.mapping = dict(zip(ids, rows, strict=True))
        self.failed: set[str] = set()
        failure_schema = None
        if modality == "image":
            failures = pq.read_table(folder / "image_embedding_failures.parquet")
            failure_schema = str(failures.schema)
            self.failed = {article_id(v) for v in failures["article_id"].to_pylist()}
            if self.failed & self.mapping.keys():
                raise ValueError("Contradictory image failures and success mapping")
        declared = (
            self.manifest.get("files", {}).get(mapping_path.name, {}).get("sha256")
        )
        declared = declared or self.manifest.get("artifact_sha256", {}).get(
            mapping_path.name
        )
        actual = digest(mapping_path)
        if declared and declared != actual:
            raise ValueError("Content mapping checksum mismatch")
        self.report = {
            "array": source_identity(array_path),
            "shape": list(self.array.shape),
            "dtype": str(self.array.dtype),
            "read_only": not self.array.flags.writeable,
            "mapping_rows": len(ids),
            "mapping_sha256": actual,
            "mapping_checksum_status": "verified"
            if declared
            else "computed_no_reference",
            "manifest_sha256": digest(folder / f"{modality}_features_manifest.json"),
            "manifest": self.manifest,
            "failure_schema": failure_schema,
            "failed_items": len(self.failed),
            "raw_faiss_loaded": False,
            "numeric_validation_scope": "consumed rows only",
            "consumed_unique_rows": 0,
            "provenance_limitations": (
                "Static snapshots; upstream claims not independently reproduced"
            ),
        }
        self.consumed: set[str] = set()

    def fetch(self, ids: list[str]) -> tuple[FloatArray, npt.NDArray[np.bool_]]:
        """Return normalized copies and masks, validating every consumed vector.

        Missing rows are zeros with mask false. Present invalid rows block;
        normalization never writes through the read-only memory map.
        """
        if len(ids) > 200000:
            raise ValueError("Unbounded content fetch")
        vectors = np.zeros((len(ids), self.array.shape[1]), dtype=np.float32)
        mask = np.zeros(len(ids), dtype=bool)
        for start in range(0, len(ids), 256):
            pairs = [
                (j, self.mapping[item])
                for j, item in enumerate(ids[start : start + 256], start)
                if item in self.mapping
            ]
            if not pairs:
                continue
            positions, rows = zip(*pairs, strict=True)
            block = np.asarray(self.array[list(rows)], dtype=np.float32)
            norms = np.linalg.norm(block, axis=1)
            if not np.isfinite(block).all() or np.any(norms <= 0):
                raise ValueError("Consumed embedding has nonfinite or zero values")
            if self.manifest.get("normalization") == "L2" and not np.allclose(
                norms, 1, atol=1e-3
            ):
                raise ValueError("Consumed embedding violates declared normalization")
            vectors[list(positions)] = block / norms[:, None]
            mask[list(positions)] = True
            self.consumed.update(ids[j] for j in positions)
        self.report["consumed_unique_rows"] = len(self.consumed)
        return vectors, mask


class ContentBank:
    """Small validated content subset shared by features and the item tower."""

    def __init__(self, root: Path, ids: list[str], full_catalog: set[str]) -> None:
        self.ids = sorted(set(ids))
        self.position = {item: i for i, item in enumerate(self.ids)}
        self.vectors: dict[str, FloatArray] = {}
        self.masks: dict[str, npt.NDArray[np.bool_]] = {}
        self.reports: dict[str, dict] = {}
        for modality in ("text", "image"):
            source = ContentSource(root, modality, full_catalog)
            vectors, masks = source.fetch(self.ids)
            self.vectors[modality] = vectors
            self.masks[modality] = masks
            self.reports[modality] = source.report

    def vector(self, item: str, modality: str) -> FloatArray | None:
        """Return a unit vector or explicit unavailability."""
        position = self.position.get(item)
        if position is None or not self.masks[modality][position]:
            return None
        return self.vectors[modality][position]
