"""Write bounded Arrow feature blocks without breaking query boundaries.

feature_table places audit keys beside the frozen float32 schema. FeatureShards
appends small compute batches into at most 512-query Parquet files and publishes
hashes only after close. Empty candidate groups remain in the shard ledger;
optional provenance is audit data, never a ranker feature.
"""

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from hm_recsys.core.contracts import digest
from hm_recsys.features.builder import FEATURES


class FeatureShards:
    """Write small compute batches into bounded 512-query files without buffering."""

    def __init__(self, run: Path, folder: Path, audit: Path | None) -> None:
        self.run, self.folder, self.audit = (run, folder, audit)
        self.index = 0
        self.features = self.provenance = None
        self.groups = []
        self.rows = 0

    def append(self, table, groups, provenance) -> dict | None:
        """Append one whole-query batch; publish its digest only after file close."""
        if len(groups) > 32 or sum(g["rows"] for g in groups) != len(table):
            raise ValueError("Feature writer query-group batch contract mismatch")
        name = f"part-{self.index:05d}.parquet"
        if self.features is None:
            self.features = pq.ParquetWriter(self.folder / name, table.schema)
        self.features.write_table(table, row_group_size=max(1, len(table)))
        if self.audit is not None and provenance is not None:
            if self.provenance is None:
                self.provenance = pq.ParquetWriter(self.audit / name, provenance.schema)
            self.provenance.write_table(provenance)
        self.groups.extend(groups)
        self.rows += len(table)
        return self.finish() if len(self.groups) >= 512 else None

    def finish(self) -> dict | None:
        """Close a partial last shard without dropping zero-candidate query groups."""
        if self.features is None:
            return None
        self.features.close()
        if self.provenance is not None:
            self.provenance.close()
        path = self.folder / f"part-{self.index:05d}.parquet"
        entry = {
            "file": str(path.relative_to(self.run)),
            "rows": self.rows,
            "groups": self.groups,
            "sha256": digest(path),
        }
        self.index += 1
        self.features = self.provenance = None
        self.groups = []
        self.rows = 0
        return entry

    def close(self) -> None:
        """Close incomplete attempt handles; stage completion remains unpublished."""
        if self.features is not None:
            self.features.close()
        if self.provenance is not None:
            self.provenance.close()


def feature_table(
    users: list[str], ids: list[str], blocks: list[np.ndarray]
) -> pa.Table:
    """Construct a typed bounded table with separate audit keys and features."""
    values = (
        np.concatenate(blocks)
        if blocks
        else np.empty((0, len(FEATURES)), dtype=np.float32)
    )
    arrays = {
        "customer_id": pa.array(users, type=pa.string()),
        "article_id": pa.array(ids, type=pa.string()),
    }
    arrays.update(
        {
            name: pa.array(values[:, i], type=pa.float32())
            for i, name in enumerate(FEATURES)
        }
    )
    return pa.table(arrays)
