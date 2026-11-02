"""Small real WALS fits, fixed-item solves, and independent exact indexes."""

from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np

from hm_recsys.content import FloatArray
from hm_recsys.contracts import (
    BASE_END,
    START,
    VOCAB_START,
    Config,
    read_json,
    write_json,
)
from hm_recsys.data import Sample, Transaction
from hm_recsys.pilot.vectors import BlockIndex

REGULARIZATION = 0.05


def confidence(
    history: list[Transaction], *, cutoff: date, vocabulary: dict[str, int]
) -> dict[int, float]:
    """Construct full observed confidence once; unobserved entries have c=1."""
    weights: dict[int, float] = defaultdict(float)
    for row in history:
        if START <= row.day <= cutoff and row.article in vocabulary:
            weights[vocabulary[row.article]] += np.exp(-(cutoff - row.day).days / 30)
    return {i: float(1 + 20 * np.log1p(value)) for i, value in weights.items()}


def fold_user(
    factors: FloatArray, observed: dict[int, float], *, gram: np.ndarray | None = None
) -> FloatArray:
    """Solve regularized implicit least squares with fixed, unnormalized items."""
    if not observed:
        return np.zeros(factors.shape[1], dtype=np.float32)
    positions = list(observed)
    conf = np.array([observed[i] for i in positions], dtype=np.float64)
    selected = factors[positions].astype(np.float64)
    if gram is None:
        y = factors.astype(np.float64)
        gram = y.T @ y
    lhs = gram + (selected.T * (conf - 1)) @ selected
    lhs += np.eye(factors.shape[1]) * REGULARIZATION
    rhs = selected.T @ conf
    return np.linalg.solve(lhs, rhs).astype(np.float32)


class ExactIndex:
    """Own learned-vector IndexFlatIP; never opens supplied content indexes."""

    def __init__(self, ids: list[str], vectors: FloatArray) -> None:
        import faiss

        if len(ids) != len(set(ids)) or len(ids) != len(vectors):
            raise ValueError("Learned index mapping mismatch")
        if not len(ids) or len(ids) > 5000 or not np.isfinite(vectors).all():
            raise ValueError("Invalid or oversized learned index")
        self.ids = ids
        self.vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(self.vectors)

    def search(
        self, vector: FloatArray, active: set[str], k: int
    ) -> list[tuple[str, float]]:
        """Filter before quota truncation; search all tiny rows to resolve ties."""
        scores, rows = self.index.search(vector.reshape(1, -1), len(self.ids))
        matches = [
            (self.ids[int(i)], float(score))
            for i, score in zip(rows[0], scores[0], strict=True)
            if i >= 0 and self.ids[int(i)] in active
        ]
        return sorted(matches, key=lambda value: (-value[1], value[0]))[:k]

    def save(self, folder: Path, name: str) -> None:
        """Persist a separate learned index, vectors, and row map."""
        import faiss

        folder.mkdir(exist_ok=True, parents=True)
        faiss.write_index(self.index, str(folder / f"{name}.faiss"))
        np.save(folder / f"{name}.npy", self.vectors, allow_pickle=False)
        write_json(
            folder / f"{name}.json",
            {
                "article_ids": self.ids,
                "dimension": self.vectors.shape[1],
                "kind": "learned_IndexFlatIP",
                "raw_content_index": False,
            },
        )

    @classmethod
    def load(cls, folder: Path, name: str) -> "ExactIndex":
        """Verify the saved learned index against its explicit vectors and map."""
        import faiss

        mapping = read_json(folder / f"{name}.json")
        vectors = np.load(folder / f"{name}.npy", allow_pickle=False)
        obj = cls(mapping["article_ids"], vectors)
        index = faiss.read_index(str(folder / f"{name}.faiss"))
        if index.ntotal != len(obj.ids) or index.d != vectors.shape[1]:
            raise ValueError("Stored learned index shape mismatch")
        if mapping["kind"] != "learned_IndexFlatIP" or mapping["dimension"] != index.d:
            raise ValueError("Stored learned index contract mismatch")
        if index.metric_type != faiss.METRIC_INNER_PRODUCT:
            raise ValueError("Stored learned index metric mismatch")
        if not np.array_equal(index.reconstruct_n(0, index.ntotal), vectors):
            raise ValueError("Stored learned index vector mismatch")
        obj.index = index
        return obj


class WALS:
    """Frozen item factors with genuine current-history user folding-in."""

    def __init__(self, index: ExactIndex | BlockIndex) -> None:
        self.index = index
        self.mapping = {item: i for i, item in enumerate(index.ids)}
        self.gram = np.zeros((index.vectors.shape[1], index.vectors.shape[1]))
        for start in range(0, len(index.ids), 4096):
            block = index.vectors[start : start + 4096].astype(np.float64)
            self.gram += block.T @ block

    @classmethod
    def fit(cls, sample: Sample, config: Config, run: Path) -> "WALS":
        """Fit implicit ALS on bounded base-only full-confidence CSR data."""
        from implicit.cpu.als import AlternatingLeastSquares
        from scipy.sparse import csr_matrix

        sample.validate_training(config)
        base = [t for t in sample.transactions if START <= t.day < BASE_END]
        catalog = set(sample.catalog)
        ids = sorted(
            {t.article for t in base if t.day >= VOCAB_START and t.article in catalog}
        )
        if not ids or len(ids) > config.limit("max_catalog"):
            raise ValueError("Empty or oversized WALS vocabulary")
        mapping = {item: i for i, item in enumerate(ids)}
        histories: dict[str, list[Transaction]] = defaultdict(list)
        for row in base:
            if row.article in mapping:
                histories[row.customer].append(row)
        users = sorted(histories)
        row_ids, col_ids, values = [], [], []
        from datetime import timedelta

        for u, user in enumerate(users):
            obs = confidence(
                histories[user], cutoff=BASE_END - timedelta(days=1), vocabulary=mapping
            )
            for i, value in obs.items():
                row_ids.append(u)
                col_ids.append(i)
                values.append(value)
        matrix = csr_matrix(
            (np.array(values, dtype=np.float32), (row_ids, col_ids)),
            shape=(len(users), len(ids)),
        )
        model = AlternatingLeastSquares(
            factors=config.limit("wals_factors"),
            regularization=REGULARIZATION,
            alpha=1.0,
            iterations=config.limit("wals_iterations"),
            num_threads=config.limit("threads"),
            random_state=42,
            use_cg=True,
        )
        model.fit(matrix, show_progress=False)
        factors = np.asarray(model.item_factors, dtype=np.float32)
        if not np.isfinite(factors).all() or not np.any(factors):
            raise ValueError("WALS produced invalid factors")
        index = ExactIndex(ids, factors)
        index.save(run / "indexes", "wals")
        np.save(
            run / "models/wals_user_factors.npy", model.user_factors, allow_pickle=False
        )
        write_json(
            run / "models/wals.json",
            {
                "users": users,
                "items": ids,
                "user_to_row": {u: i for i, u in enumerate(users)},
                "item_to_row": mapping,
                "iterations": config.limit("wals_iterations"),
                "factors": config.limit("wals_factors"),
                "nnz": matrix.nnz,
                "confidence_alpha": 20,
                "epsilon": 1,
                "library_alpha": 1.0,
                "regularization": REGULARIZATION,
                "normalization": "none",
                "fit_start": str(START),
                "fit_end_exclusive": str(BASE_END),
                "vocabulary_start": str(VOCAB_START),
                "vocabulary_scope": (
                    "base-observed intersection smoke candidate catalog"
                ),
            },
        )
        return cls(index)

    def retrieve(
        self, history: list[Transaction], *, cutoff: date, active: set[str]
    ) -> list[tuple[str, float]]:
        """Return fixed-factor scores without suppressing previously bought items."""
        obs = confidence(history, cutoff=cutoff, vocabulary=self.mapping)
        if not obs:
            return []
        user = fold_user(self.index.vectors, obs, gram=self.gram)
        return self.index.search(user, active, 50)


class TowerRetriever:
    """FAISS retrieval from frozen user/item vectors encoded in another process."""

    def __init__(self, run: Path, *, reloaded: bool = False) -> None:
        folder = run / "predictions" if reloaded else run / "queries"
        name = "reloaded_query_vectors" if reloaded else "query_vectors"
        state = read_json(folder / f"{name}.json")
        self.vectors = np.load(folder / f"{name}.npy", allow_pickle=False)
        self.mapping = {
            (user, date.fromisoformat(day)): row
            for row, (user, day) in enumerate(state["queries"])
        }
        self.has_history = state["has_history"]
        if len(self.mapping) != len(self.vectors) or self.vectors.shape[1] != 128:
            raise ValueError("Two-Tower query vector mapping mismatch")
        from hm_recsys.contracts import digest

        if state["model_sha256"] != digest(run / "models/twotower.pt"):
            raise ValueError("Query vectors do not match the frozen tower weights")
        self.index: ExactIndex | BlockIndex = ExactIndex.load(
            run / "indexes", "twotower"
        )

    def retrieve(
        self, user: str, *, before: date, active: set[str]
    ) -> list[tuple[str, float]]:
        """Search real learned vectors; no-history users retain empty R6 output."""
        row = self.mapping.get((user, before))
        if row is None:
            raise ValueError("Query was not encoded by the frozen user tower")
        if not self.has_history[row]:
            return []
        return self.index.search(self.vectors[row], active, 50)


def build_tower_index(run: Path) -> None:
    """Build the real learned FAISS index without loading PyTorch's runtime."""
    vectors = np.load(run / "models/twotower_items.npy", allow_pickle=False)
    ids = read_json(run / "models/twotower_items.json")["article_ids"]
    ExactIndex(ids, vectors).save(run / "indexes", "twotower")
