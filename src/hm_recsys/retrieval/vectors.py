"""Search and persist exact learned inner-product indexes with bounded memory.

Index row maps are checked against separate learned float32 vectors. Active
eligibility is applied before Top-K, ties use article IDs, and block searches
never assume oversampling recovers a filtered Top-K. FAISS artifacts are derived
learned indexes, never the supplied source content indexes. File writes happen
only in explicit save calls; loading validates vectors/metric/map alignment."""

from pathlib import Path

import numpy as np

from hm_recsys.core.contracts import read_json, write_json
from hm_recsys.data.history import allocation
from hm_recsys.features.content import FloatArray


def block_topk(
    ids: list[str],
    vectors: FloatArray,
    queries: FloatArray,
    active: set[str],
    k: int,
    *,
    block_size: int = 4096,
    active_rows: list[int] | None = None,
) -> list[list[tuple[str, float]]]:
    """Exact active-first top-K with bounded score blocks and stable ID ties.

    Sorting is limited to one vector block per query, never all catalog scores
    or a Python catalog-sized pair list. Equal scores survive with ascending IDs.
    """
    if len(queries) > 8192 or k < 1 or block_size < 1:
        raise ValueError("Invalid exact search batch")
    best: list[list[tuple[str, float]]] = [[] for _ in queries]
    cached = (
        np.asarray(active_rows, dtype=np.int64) if active_rows is not None else None
    )
    for start in range(0, len(ids), block_size):
        rows = (
            [
                j
                for j in range(start, min(start + block_size, len(ids)))
                if ids[j] in active
            ]
            if cached is None
            else cached[(cached >= start) & (cached < start + block_size)].tolist()
        )
        if not rows:
            continue
        names = np.array([ids[j] for j in rows])
        scores = np.asarray(queries @ vectors[rows].T, dtype=np.float32)
        if not np.isfinite(scores).all():
            raise ValueError("Nonfinite exact-search score")
        for q, values in enumerate(scores):
            selected = np.lexsort((names, -values))[:k]
            merged = best[q] + [(str(names[j]), float(values[j])) for j in selected]
            best[q] = sorted(merged, key=lambda p: (-p[1], p[0]))[:k]
    return best


class ActiveBlocks:
    """Cache cutoff eligibility, preserving reference physical GEMM block shapes."""

    def __init__(self, index, active: set[str]) -> None:
        self.index = index
        self.blocks = []
        for start in range(0, len(index.ids), 4096):
            rows = np.array(
                [
                    j
                    for j in range(start, min(start + 4096, len(index.ids)))
                    if index.ids[j] in active
                ],
                dtype=np.int64,
            )
            if len(rows):
                self.blocks.append((rows, np.array([index.ids[j] for j in rows])))

    def search(self, queries: FloatArray, k: int) -> list[list[tuple[str, float]]]:
        """Partition before lexsort, retaining all exact ties at each block boundary."""
        if len(queries) > 8192 or k < 1:
            raise ValueError("Invalid exact search batch")
        best = [[] for _ in queries]
        for rows, names in self.blocks:
            scores = np.asarray(queries @ self.index.vectors[rows].T, dtype=np.float32)
            if not np.isfinite(scores).all():
                raise ValueError("Nonfinite exact-search score")
            for q, values in enumerate(scores):
                take = min(k, len(values))
                threshold = np.partition(values, len(values) - take)[-take]
                eligible = np.flatnonzero(values >= threshold)
                chosen = eligible[
                    np.lexsort((names[eligible], -values[eligible]))[:take]
                ]
                best[q] = sorted(
                    best[q] + [(str(names[j]), float(values[j])) for j in chosen],
                    key=lambda p: (-p[1], p[0]),
                )[:k]
        return best


class BlockIndex:
    """Full vocabulary, bounded exact search and separate learned FAISS artifacts."""

    def __init__(self, ids: list[str], vectors: FloatArray) -> None:
        if not ids or len(ids) != len(set(ids)) or len(ids) != len(vectors):
            raise ValueError("Learned index mapping mismatch")
        if vectors.ndim != 2:
            raise ValueError("Invalid learned vector shape")
        allocation(vectors.nbytes * 2, "learned index and map")
        for start in range(0, len(ids), 4096):
            if not np.isfinite(vectors[start : start + 4096]).all():
                raise ValueError("Nonfinite learned vector")
        self.ids, self.vectors = (ids, vectors)

    def search(
        self, vector: FloatArray, active: set[str], k: int
    ) -> list[tuple[str, float]]:
        """Preserve all eligible top candidates without oversampling assumptions."""
        return block_topk(self.ids, self.vectors, vector.reshape(1, -1), active, k)[0]

    def save(self, folder: Path, name: str) -> None:
        """Build a separate learned FAISS IndexFlatIP with bounded additions."""
        import faiss

        folder.mkdir(exist_ok=True, parents=True)
        index = faiss.IndexFlatIP(self.vectors.shape[1])
        for start in range(0, len(self.ids), 4096):
            index.add(np.ascontiguousarray(self.vectors[start : start + 4096]))
        faiss.write_index(index, str(folder / f"{name}.faiss"))
        output = np.lib.format.open_memmap(
            folder / f"{name}.npy",
            mode="w+",
            dtype=np.float32,
            shape=self.vectors.shape,
        )
        for start in range(0, len(self.ids), 4096):
            output[start : start + 4096] = self.vectors[start : start + 4096]
        output.flush()
        write_json(
            folder / f"{name}.json",
            {
                "article_ids": self.ids,
                "dimension": index.d,
                ("kind"): ("learned_IndexFlatIP"),
                ("search"): ("blocked exact active-first inner product"),
                ("raw_content_index"): False,
            },
        )

    @classmethod
    def load(cls, folder: Path, name: str) -> "BlockIndex":
        """Verify persisted FAISS/vector/map alignment in bounded chunks."""
        import faiss

        state = read_json(folder / f"{name}.json")
        vectors = np.load(folder / f"{name}.npy", mmap_mode="r", allow_pickle=False)
        obj = cls(state["article_ids"], vectors)
        index = faiss.read_index(str(folder / f"{name}.faiss"))
        if (
            index.ntotal != len(obj.ids)
            or index.d != vectors.shape[1]
            or index.metric_type != faiss.METRIC_INNER_PRODUCT
        ):
            raise ValueError("Learned FAISS contract mismatch")
        for start in range(0, len(obj.ids), 4096):
            if not np.array_equal(
                index.reconstruct_n(start, min(4096, len(obj.ids) - start)),
                vectors[start : start + 4096],
            ):
                raise ValueError("Learned index/vector mismatch")
        return obj
