"""Construct the sole shared ranking feature matrix for training and inference.

A frozen schema, category vocabulary and pre-base visual centers define every
column. Cutoff-safe snapshot histories and native candidate provenance produce
float32 [N, F] blocks, preserving candidate order, NaNs and availability masks.
Raw content stays in separate read-only banks; learned recall models are never
updated here. Frozen visual assignments and static fields are reused by row ID.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from hm_recsys.core.contracts import (
    CHANNELS,
    read_json,
)
from hm_recsys.features.content import ContentBank, FloatArray

if TYPE_CHECKING:
    from hm_recsys.fusion.native import Candidate
    from hm_recsys.retrieval.snapshot import GlobalSnapshot as Snapshot
from hm_recsys.data.sample import age_bucket

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
        self.cluster_ids = frozen_clusters(content, run, self.centers)
        self.static_categories: FloatArray | None = None

    def build(
        self, snapshot: Snapshot, user: str, candidates: list[Candidate]
    ) -> FloatArray:
        """Construct the frozen float32 [candidates, features] matrix in schema order.

        Candidate order and NaN/source-mask semantics are unchanged. All modes use
        this columnar implementation and frozen model-independent input caches.
        """
        return self._build_columns(snapshot, user, candidates)

    def _build_columns(self, snapshot, user, candidates):
        if len(candidates) > 270:
            raise ValueError("Unbounded query feature request")
        names = self.schema["names"]
        col = {name: j for j, name in enumerate(names)}
        result = np.full((len(candidates), len(names)), np.nan, dtype=np.float32)
        if not candidates:
            return result
        positions = np.fromiter(
            (self.content.position[c.article] for c in candidates), dtype=np.int64
        )
        if getattr(self, "_snapshot_key", None) != (id(snapshot), snapshot.cutoff):
            table = np.full(
                (len(self.content.ids), len(names)), np.nan, dtype=np.float32
            )
            prices = np.empty(len(table), dtype=np.float64)
            for j, item in enumerate(self.content.ids):
                for days in (3, 7, 14, 28):
                    table[j, col[f"item_sales_{days}d"]] = snapshot.sales[days][item]
                sales = snapshot.sales
                table[j, col["sales_velocity_3d_vs_7d"]] = (
                    sales[3][item] / 3 / (sales[7][item] / 7 + 0.1)
                )
                table[j, col["sales_velocity_7d_vs_28d"]] = (
                    sales[7][item] / 7 / (sales[28][item] / 28 + 0.1)
                )
                ratio, prices[j] = snapshot.item_statistics(item)
                table[j, col["item_repurchase_ratio_28d"]] = ratio
                for key, mapping in self.schema["category_maps"].items():
                    table[j, col[key]] = mapping.get(
                        str(snapshot.sample.articles[item].get(key)), np.nan
                    )
            table[:, col["has_item_image"]] = self.content.masks["image"]
            table[:, col["has_item_text"]] = self.content.masks["text"]
            if self.cluster_ids is not None:
                valid = self.content.masks["image"]
                table[valid, col["item_visual_cluster_id"]] = self.cluster_ids[valid]
            self._item_table, self._item_prices = (table, prices)
            self._snapshot_key = (id(snapshot), snapshot.cutoff)
        result[:] = self._item_table[positions]
        history = snapshot.by_user.get(user, [])
        latest = sorted(
            history, key=lambda r: (r.day, r.article, r.price, r.channel), reverse=True
        )[:5]
        for modality, name in (
            ("image", "user_item_img_cosine_sim"),
            ("text", "user_item_text_cosine_sim"),
        ):
            valid = [
                v
                for r in latest
                if (v := self.content.vector(r.article, modality)) is not None
            ]
            mean = np.mean(np.stack(valid), axis=0) if valid else None
            norm = np.linalg.norm(mean) if mean is not None else 0
            profile = mean / norm if mean is not None and norm > 1e-12 else None
            result[:, col[f"has_user_{modality}_history"]] = float(profile is not None)
            if profile is not None:
                mask = self.content.masks[modality][positions]
                result[mask, col[name]] = (
                    self.content.vectors[modality][positions[mask]] @ profile
                )
        last = self.content.vector(latest[0].article, "image") if latest else None
        if last is not None:
            mask = self.content.masks["image"][positions]
            result[mask, col["user_item_last1_img_sim"]] = (
                self.content.vectors["image"][positions[mask]] @ last
            )
        past90 = snapshot.window(history, 90)
        user_price = float(np.mean([r.price for r in past90])) if past90 else np.nan
        meta = snapshot.sample.customers.get(user, {})
        result[:, col["user_age"]] = (
            float(meta["age"]) if meta.get("age") is not None else np.nan
        )
        result[:, col["sales_channel_preference"]] = (
            sum(r.channel == 2 for r in past90) / len(past90) if past90 else np.nan
        )
        result[:, col["user_tx_count_30d"]] = len(snapshot.window(history, 30))
        result[:, col["user_tx_count_90d"]] = len(past90)
        result[:, col["user_avg_price_90d"]] = user_price
        result[:, col["price_diff_from_user_avg"]] = (
            self._item_prices[positions] - user_price
        )
        if user_price > 0:
            result[:, col["price_ratio_to_user_avg"]] = (
                self._item_prices[positions] / user_price
            )
        for field, name in (
            ("garment_group_no", "user_garment_group_affinity"),
            ("section_no", "user_section_affinity"),
            ("colour_group_code", "user_colour_affinity"),
        ):
            counts = Counter(
                snapshot.sample.articles[r.article].get(field) for r in past90
            )
            result[:, col[name]] = [
                counts[v] / len(past90)
                if past90
                and (v := snapshot.sample.articles[c.article].get(field)) is not None
                else np.nan
                for c in candidates
            ]
        ranks = snapshot.bucket_ranks.get(age_bucket(meta.get("age")), {})
        result[:, col["item_rank_in_user_age_bucket_7d"]] = [
            ranks.get(c.article, np.nan) for c in candidates
        ]
        _, rep_count, rep_last = snapshot.repurchase(user)
        _, sibling_days = snapshot.sibling(user)
        for channel in CHANNELS:
            prefix = "pop" if channel == "trend" else channel
            result[:, col[f"hit_{channel}"]] = 0
            for j, candidate in enumerate(candidates):
                if (hit := candidate.hits.get(channel)) is None:
                    continue
                result[j, col[f"hit_{channel}"]] = 1
                result[j, col[f"{prefix}_score"]] = hit.score
                if f"{prefix}_rank" in col:
                    result[j, col[f"{prefix}_rank"]] = hit.rank
                if channel == "repurchase":
                    result[j, col["repurchase_count"]] = rep_count[candidate.article]
                    result[j, col["days_since_last_buy"]] = (
                        snapshot.cutoff - rep_last[candidate.article]
                    ).days
                elif channel == "sibling":
                    result[j, col["sibling_source_days"]] = sibling_days[
                        candidate.article
                    ]
        result[:, col["recall_channel_hit_count"]] = [len(c.hits) for c in candidates]
        result[:, col["rrf_fused_score"]] = [c.rrf for c in candidates]
        if np.isinf(result).any():
            raise ValueError("Infinite feature values")
        return result


def frozen_clusters(content: ContentBank, run: Path, centers: FloatArray) -> np.ndarray:
    """Assign exact squared-distance/first-argmin clusters once per frozen input."""
    import hashlib

    from hm_recsys.core.contracts import digest
    from hm_recsys.runtime.files import atomic_json, canonical_hash

    hasher = hashlib.sha256()
    for start in range(0, len(content.ids), 256):
        hasher.update(
            np.ascontiguousarray(
                content.vectors["image"][start : start + 256]
            ).tobytes()
        )
    expected = {
        "centers": digest(run / "models/visual_centers.npy"),
        "images": hasher.hexdigest(),
        "mapping": canonical_hash(content.ids),
        "mask": hashlib.sha256(content.masks["image"].tobytes()).hexdigest(),
    }
    path = run / "models/visual_assignments.npy"
    manifest = path.with_suffix(".json")
    if path.exists() or manifest.exists():
        meta = read_json(manifest)
        if meta["identity"] != expected or digest(path) != meta["sha256"]:
            raise ValueError("Frozen cluster cache identity/integrity mismatch")
        return np.load(path, mmap_mode="r", allow_pickle=False)
    temporary = path.with_suffix(".partial.npy")
    output = np.lib.format.open_memmap(
        temporary, mode="w+", dtype=np.int32, shape=(len(content.ids),)
    )
    output[:] = -1
    for start in range(0, len(content.ids), 4096):
        indices = np.arange(start, min(start + 4096, len(content.ids)))
        valid = indices[content.masks["image"][indices]]
        if len(valid):
            distance = np.sum(
                (centers[None, :, :] - content.vectors["image"][valid, None, :]) ** 2,
                axis=2,
            )
            output[valid] = np.argmin(distance, axis=1)
    output.flush()
    del output
    temporary.replace(path)
    atomic_json(manifest, {"identity": expected, "sha256": digest(path)})
    return np.load(path, mmap_mode="r", allow_pickle=False)
