"""One named, past-only feature builder for training and reloaded inference."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from hm_recsys.content import ContentBank, FloatArray
from hm_recsys.contracts import (
    BASE_END,
    CHANNELS,
    Config,
    read_json,
    stable_key,
    write_json,
)
from hm_recsys.data import Sample

if TYPE_CHECKING:
    from hm_recsys.recall import Candidate, Snapshot
from hm_recsys.data import age_bucket

CORE = (
    "pop_score",
    "pop_rank",
    "repurchase_score",
    "repurchase_count",
    "days_since_last_buy",
    "sibling_score",
    "sibling_source_days",
    "itemcf_score",
    "itemcf_rank",
    "wals_score",
    "wals_rank",
    "twotower_score",
    "twotower_rank",
    "recall_channel_hit_count",
    "rrf_fused_score",
    "user_item_img_cosine_sim",
    "user_item_last1_img_sim",
    "user_item_text_cosine_sim",
    "item_visual_cluster_id",
    "item_sales_3d",
    "item_sales_7d",
    "item_sales_14d",
    "item_sales_28d",
    "sales_velocity_3d_vs_7d",
    "sales_velocity_7d_vs_28d",
    "item_repurchase_ratio_28d",
    "user_age",
    "sales_channel_preference",
    "user_tx_count_30d",
    "user_tx_count_90d",
    "user_avg_price_90d",
    "price_diff_from_user_avg",
    "price_ratio_to_user_avg",
    "user_garment_group_affinity",
    "user_section_affinity",
    "user_colour_affinity",
    "item_rank_in_user_age_bucket_7d",
    "product_type_no",
    "index_group_no",
)
MASKS = (
    *[f"hit_{c}" for c in CHANNELS],
    "has_item_image",
    "has_item_text",
    "has_user_image_history",
    "has_user_text_history",
)
FEATURES = (*CORE, *MASKS)
CATEGORICAL = ("item_visual_cluster_id", "product_type_no", "index_group_no")


def fit_preprocessing(
    sample: Sample, content: ContentBank, config: Config, run: Path
) -> None:
    """Fit K=8 on pre-base valid images only, and persist frozen category maps."""
    from sklearn.cluster import KMeans

    sample.validate_training(config)
    legal = {r.article for r in sample.transactions if r.day < BASE_END}
    ids = sorted(
        (i for i in legal if content.vector(i, "image") is not None),
        key=lambda i: (stable_key(i), i),
    )[: config.limit("cluster_max_rows")]
    if len(ids) < config.limit("cluster_count"):
        raise ValueError("Insufficient pre-base images for smoke clustering")
    vectors = np.stack(
        [content.vectors["image"][content.position[item]] for item in ids]
    )
    if len(np.unique(vectors, axis=0)) < config.limit("cluster_count"):
        raise ValueError("Insufficient distinct image vectors for clustering")
    model = KMeans(
        n_clusters=config.limit("cluster_count"),
        max_iter=config.limit("cluster_max_iter"),
        random_state=42,
    )
    # sklearn's constructor annotation accepts only str although its runtime
    # contract supports integer n_init. set_params preserves that public API.
    model.set_params(n_init=config.limit("cluster_n_init"))
    model.fit(vectors)
    np.save(
        run / "models/visual_centers.npy", model.cluster_centers_, allow_pickle=False
    )
    categories = {
        key: {
            value: i
            for i, value in enumerate(
                sorted(
                    {
                        str(meta[key])
                        for meta in sample.articles.values()
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
            "version": "v0.3.2-smoke-features-1",
            "names": list(FEATURES),
            "dtype": "float32",
            "categorical": list(CATEGORICAL),
            "category_maps": categories,
            "missing": "NaN with presence masks",
            "cluster_fit_ids": ids,
            "cluster_fit_end_exclusive": str(BASE_END),
            "cluster_count": config.limit("cluster_count"),
            "cluster_iterations": int(model.n_iter_),
            "static_metadata_assumption": True,
        },
    )


def _cosine(first: FloatArray | None, second: FloatArray | None) -> float:
    return (
        float(np.dot(first, second))
        if first is not None and second is not None
        else float("nan")
    )


class FeatureBuilder:
    """Frozen schema and clusters, rebuilt cutoff snapshots, shared at inference."""

    def __init__(self, content: ContentBank, run: Path) -> None:
        self.content = content
        self.schema = read_json(run / "feature_schema.json")
        if self.schema["names"] != list(FEATURES):
            raise ValueError("Unsupported feature schema/order")
        self.centers = np.load(run / "models/visual_centers.npy", allow_pickle=False)
        if self.centers.shape != (
            self.schema["cluster_count"],
            content.vectors["image"].shape[1],
        ):
            raise ValueError("Frozen cluster dimension mismatch")

    def build(
        self, snapshot: Snapshot, user: str, candidates: list[Candidate]
    ) -> FloatArray:
        """Build [native candidates,F] float32 features without accepting labels."""
        if len(candidates) > 270:
            raise ValueError("Unbounded query feature request")
        history = snapshot.by_user.get(user, [])
        latest = sorted(
            history, key=lambda r: (r.day, r.article, r.price, r.channel), reverse=True
        )[:5]
        profiles: dict[str, FloatArray | None] = {}
        for modality in ("image", "text"):
            valid = [
                vector
                for row in latest
                if (vector := self.content.vector(row.article, modality)) is not None
            ]
            mean = np.mean(np.stack(valid), axis=0) if valid else None
            norm = np.linalg.norm(mean) if mean is not None else 0
            profiles[modality] = (
                mean / norm if mean is not None and norm > 1e-12 else None
            )
        last_image = self.content.vector(latest[0].article, "image") if latest else None
        past90 = snapshot.window(history, 90)
        user_price = (
            float(np.mean([r.price for r in past90])) if past90 else float("nan")
        )
        channel_preference = (
            sum(r.channel == 2 for r in past90) / len(past90)
            if past90
            else float("nan")
        )
        _, rep_count, rep_last = snapshot.repurchase(user)
        _, sibling_days = snapshot.sibling(user)
        meta_user = snapshot.sample.customers.get(user, {})
        bucket = age_bucket(meta_user.get("age"))
        result = np.full(
            (len(candidates), len(self.schema["names"])), np.nan, dtype=np.float32
        )
        for row_index, candidate in enumerate(candidates):
            item = candidate.article
            values: dict[str, float] = {}
            for channel in CHANNELS:
                hit = candidate.hits.get(channel)
                prefix = "pop" if channel == "trend" else channel
                if hit is not None:
                    values[f"{prefix}_score"] = hit.score
                    if f"{prefix}_rank" in FEATURES:
                        values[f"{prefix}_rank"] = float(hit.rank)
                values[f"hit_{channel}"] = float(hit is not None)
            if "repurchase" in candidate.hits:
                values["repurchase_count"] = rep_count[item]
                values["days_since_last_buy"] = (snapshot.cutoff - rep_last[item]).days
            if "sibling" in candidate.hits:
                values["sibling_source_days"] = sibling_days[item]
            values["recall_channel_hit_count"] = len(candidate.hits)
            values["rrf_fused_score"] = candidate.rrf
            image = self.content.vector(item, "image")
            text = self.content.vector(item, "text")
            values.update(
                {
                    "user_item_img_cosine_sim": _cosine(profiles["image"], image),
                    "user_item_last1_img_sim": _cosine(last_image, image),
                    "user_item_text_cosine_sim": _cosine(profiles["text"], text),
                    "has_item_image": float(image is not None),
                    "has_item_text": float(text is not None),
                    "has_user_image_history": float(profiles["image"] is not None),
                    "has_user_text_history": float(profiles["text"] is not None),
                }
            )
            if image is not None:
                values["item_visual_cluster_id"] = int(
                    np.argmin(np.sum((self.centers - image) ** 2, axis=1))
                )
            for days in (3, 7, 14, 28):
                values[f"item_sales_{days}d"] = snapshot.sales[days][item]
            sales = snapshot.sales
            values["sales_velocity_3d_vs_7d"] = (sales[3][item] / 3) / (
                sales[7][item] / 7 + 0.1
            )
            values["sales_velocity_7d_vs_28d"] = (sales[7][item] / 7) / (
                sales[28][item] / 28 + 0.1
            )
            ratio, item_price = snapshot.item_statistics(item)
            values["item_repurchase_ratio_28d"] = ratio
            values["user_age"] = (
                float(meta_user["age"])
                if meta_user.get("age") is not None
                else float("nan")
            )
            values["sales_channel_preference"] = channel_preference
            values["user_tx_count_30d"] = len(snapshot.window(history, 30))
            values["user_tx_count_90d"] = len(past90)
            values["user_avg_price_90d"] = user_price
            values["price_diff_from_user_avg"] = item_price - user_price
            values["price_ratio_to_user_avg"] = (
                item_price / user_price if user_price > 0 else float("nan")
            )
            meta = snapshot.sample.articles[item]
            for field, name in (
                ("garment_group_no", "user_garment_group_affinity"),
                ("section_no", "user_section_affinity"),
                ("colour_group_code", "user_colour_affinity"),
            ):
                values[name] = (
                    sum(
                        snapshot.sample.articles[r.article].get(field)
                        == meta.get(field)
                        for r in past90
                    )
                    / len(past90)
                    if past90 and meta.get(field) is not None
                    else float("nan")
                )
            values["item_rank_in_user_age_bucket_7d"] = snapshot.bucket_ranks.get(
                bucket, {}
            ).get(item, float("nan"))
            for key, mapping in self.schema["category_maps"].items():
                values[key] = mapping.get(str(meta.get(key)), float("nan"))
            result[row_index] = [
                values.get(name, float("nan")) for name in self.schema["names"]
            ]
        if np.isinf(result).any():
            raise ValueError("Infinite feature values")
        return result
