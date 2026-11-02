"""Sparse cohort WALS and exact global legal-population visual preprocessing."""

from pathlib import Path

import numpy as np

from hm_recsys.contracts import BASE_END, START, VOCAB_START, stable_key, write_json
from hm_recsys.pilot.data import History, MappedContent, allocation
from hm_recsys.pilot.vectors import BlockIndex


def fit_wals(run: Path, profile: dict) -> None:
    """Fit cohort confidence CSR on represented columns of the global vocabulary."""
    from implicit.cpu.als import AlternatingLeastSquares
    from scipy.sparse import csr_matrix

    history = History(run)
    con = history.con
    try:
        con.execute(
            """CREATE TEMP TABLE vocabulary AS SELECT DISTINCT article_id
            FROM transactions WHERE t_dat>=? AND t_dat<?""",
            [VOCAB_START, BASE_END],
        )
        con.execute(
            """CREATE TEMP TABLE observed AS SELECT customer_id,article_id,
            1+20*ln(1+sum(exp(-date_diff('day',t_dat,DATE
            '2020-08-11')/30.0))) AS confidence
            FROM transactions JOIN cohort USING(customer_id) JOIN vocabulary
            USING(article_id)
            WHERE t_dat>=? AND t_dat<? GROUP BY customer_id,article_id""",
            [START, BASE_END],
        )
        global_ids = [
            r[0]
            for r in con.execute(
                "SELECT article_id FROM vocabulary ORDER BY article_id"
            ).fetchall()
        ]
        ids = [
            r[0]
            for r in con.execute(
                "SELECT DISTINCT article_id FROM observed ORDER BY article_id"
            ).fetchall()
        ]
        users = [
            r[0]
            for r in con.execute(
                "SELECT DISTINCT customer_id FROM observed ORDER BY customer_id"
            ).fetchall()
        ]
        nnz = con.execute("SELECT count(*) FROM observed").fetchall()[0][0]
        if not nnz:
            raise ValueError("No legal WALS observations")
        allocation(
            nnz * 32
            + (len(ids) + len(users)) * profile["compute"]["wals_factors"] * 16,
            "WALS sparse fit",
        )
        con.execute(
            "CREATE TEMP TABLE item_map AS SELECT article_id,row_number()"
            " OVER(ORDER BY article_id)-1 AS i FROM (SELECT DISTINCT "
            "article_id FROM observed)"
        )
        con.execute(
            "CREATE TEMP TABLE user_map AS SELECT "
            "customer_id,row_number() OVER(ORDER BY customer_id)-1 AS u "
            "FROM (SELECT DISTINCT customer_id FROM observed)"
        )
        table = con.execute(
            "SELECT u,i,confidence FROM observed JOIN item_map "
            "USING(article_id) JOIN user_map USING(customer_id) ORDER BY "
            "u,i"
        ).to_arrow_table()
        sparse = csr_matrix(
            (
                table["confidence"].to_numpy().astype(np.float32),
                (table["u"].to_numpy(), table["i"].to_numpy()),
            ),
            shape=(len(users), len(ids)),
        )
        c = profile["compute"]
        model = AlternatingLeastSquares(
            factors=c["wals_factors"],
            regularization=0.05,
            alpha=1.0,
            iterations=c["wals_iterations"],
            num_threads=2,
            random_state=42,
            use_cg=True,
        )
        model.fit(sparse, show_progress=False)
        BlockIndex(ids, np.asarray(model.item_factors, dtype=np.float32)).save(
            run / "indexes", "wals"
        )
        np.save(
            run / "models/wals_user_factors.npy", model.user_factors, allow_pickle=False
        )
        write_json(
            run / "models/wals.json",
            {
                "users": users,
                "items": ids,
                "global_vocabulary": global_ids,
                "unsupported_cohort_items": sorted(set(global_ids) - set(ids)),
                "normalization": "none",
                "iterations": c["wals_iterations"],
                "factors": c["wals_factors"],
                "nnz": nnz,
                "library_alpha": 1.0,
                "confidence_alpha": 20,
                "regularization": 0.05,
                "fit_start": str(START),
                "fit_end_exclusive": str(BASE_END),
                "vocabulary_start": str(VOCAB_START),
                "coverage": (
                    "only columns with legal cohort observations have trained factors"
                ),
            },
        )
    finally:
        history.close()


def preprocessing(run: Path, profile: dict) -> None:
    """Fit unchanged Lloyd KMeans on every valid globally pre-base observed image."""
    from sklearn.cluster import KMeans

    from hm_recsys.features import CATEGORICAL, FEATURES

    history = History(run)
    content = MappedContent(run)
    try:
        observed = {
            r[0]
            for r in history.con.execute(
                "SELECT DISTINCT article_id FROM transactions WHERE t_dat<?", [BASE_END]
            ).fetchall()
        }
        ids = sorted(
            (i for i in observed if content.masks["image"][content.position[i]]),
            key=lambda i: (stable_key(i), i),
        )
        c = profile["compute"]
        if len(ids) < c["cluster_count"]:
            raise ValueError("Insufficient legal visual-clustering population")
        shape = (len(ids), content.vectors["image"].shape[1])
        estimate = int(np.prod(shape)) * 4 * 4 + len(ids) * c["cluster_count"] * 8
        allocation(estimate, "exact KMeans fit and working copies")
        vectors = np.lib.format.open_memmap(
            run / "tmp/cluster_fit.npy", mode="w+", dtype=np.float32, shape=shape
        )
        for start in range(0, len(ids), 256):
            vectors[start : start + 256] = content.vectors["image"][
                [content.position[i] for i in ids[start : start + 256]]
            ]
        model = KMeans(
            n_clusters=c["cluster_count"],
            max_iter=c["cluster_max_iter"],
            random_state=42,
            copy_x=False,
            algorithm="lloyd",
        )
        model.set_params(n_init=c["cluster_n_init"])
        model.fit(vectors)
        if len(np.unique(model.cluster_centers_, axis=0)) != c["cluster_count"]:
            raise ValueError("Clustering did not produce distinct requested centers")
        np.save(
            run / "models/visual_centers.npy",
            model.cluster_centers_,
            allow_pickle=False,
        )
        maps = {
            key: {
                v: j
                for j, v in enumerate(
                    sorted(
                        {
                            str(meta[key])
                            for meta in history.articles.values()
                            if meta.get(key) is not None
                        }
                    )
                )
            }
            for key in ("product_type_no", "index_group_no")
        }
        write_json(
            run / "feature_schema.json",
            {
                "version": "v0.3.2-pilot-features-1",
                "names": list(FEATURES),
                "dtype": "float32",
                "categorical": list(CATEGORICAL),
                "category_maps": maps,
                "missing": "NaN with presence masks",
                "cluster_count": c["cluster_count"],
                "cluster_iterations": int(model.n_iter_),
                "cluster_fit_ids": ids,
                "cluster_fit_end_exclusive": str(BASE_END),
                "cluster_fit_population": "all valid global pre-base observed images",
                "allocation_estimate_bytes": estimate,
                "static_metadata_assumption": True,
            },
        )
    finally:
        history.close()
