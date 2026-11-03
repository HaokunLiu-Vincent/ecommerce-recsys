"""Fit frozen visual preprocessing and persist the shared ranking schema.

Inputs are canonical pre-base purchases and validated, immutable image adapters.
Exact Lloyd K-Means writes centers, category maps and frozen assignments beneath
one run. The fit never observes ranking or evaluation label windows. Memory is
estimated before materializing the bounded fit matrix; full mode never samples.
"""

from pathlib import Path

import numpy as np

from hm_recsys.core.contracts import BASE_END, stable_key, write_json
from hm_recsys.core.progress import emit
from hm_recsys.data.history import History, allocation
from hm_recsys.features.content import MappedContent


def preprocessing(run: Path, profile: dict) -> None:
    """Fit exact Lloyd KMeans on the recorded legal pre-base image population."""
    from sklearn.cluster import KMeans

    from hm_recsys.features.builder import CATEGORICAL, FEATURES

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
        if profile["profile"] == "full" and c.get("cluster_max_rows") is not None:
            raise ValueError("Full preprocessing cannot sample image fit rows")
        if c.get("cluster_max_rows") is not None:
            ids = ids[: c["cluster_max_rows"]]
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
        emit(
            "visual_clustering",
            0,
            None,
            "iterations",
            message="Fitting frozen pre-base visual centers",
            metrics={
                "valid_image_rows": len(ids),
                "clusters": c["cluster_count"],
                "max_iterations": c["cluster_max_iter"],
            },
            force=True,
        )
        model.fit(vectors)
        emit(
            "visual_clustering",
            int(model.n_iter_),
            None,
            "iterations",
            status="complete",
            metrics={"valid_image_rows": len(ids), "clusters": c["cluster_count"]},
            force=True,
        )
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
                "version": "v0.3.2-canonical-features-1",
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
        if profile.get("performance", {}).get("backend") == "optimized":
            from hm_recsys.features.builder import frozen_clusters

            frozen_clusters(content, run, model.cluster_centers_)
    finally:
        history.close()
