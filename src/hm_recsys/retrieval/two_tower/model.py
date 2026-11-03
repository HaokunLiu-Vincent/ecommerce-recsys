"""Define the shared multimodal user/item networks and multi-positive LogQ loss.

Inputs are integer category/history tensors, immutable raw content vectors and
availability masks. Both towers return normalized [batch,128] vectors. Only raw
inputs may be cached: embeddings, projections and history pooling stay trainable.
This module has no filesystem, orchestration, label-query or device-fallback
side effects. Same-query positives share softmax mass over unique batch items.
"""

from datetime import date

import torch
from torch import nn
from torch.nn import functional as functional

ITEM_KEYS = (
    "article_id",
    "product_code",
    "product_type_no",
    "graphical_appearance_no",
    ("colour_group_code"),
    ("section_no"),
)

PROFILE_KEYS = ("age_bucket", "club_member_status", "fashion_news_frequency")

ITEM_DIMS = (64, 32, 16, 8, 8, 8)

PROFILE_DIMS = (8, 4, 4)


class TwoTower(nn.Module):
    """Independent user/item MLPs with frozen external content and 128d output."""

    def __init__(self, maps: dict, text_dim: int, image_dim: int) -> None:
        super().__init__()
        self.item_embeddings = nn.ModuleList(
            [
                nn.Embedding(len(maps[key]) + 1, dim, padding_idx=0)
                for key, dim in zip(ITEM_KEYS, ITEM_DIMS, strict=True)
            ]
        )
        self.profile_embeddings = nn.ModuleList(
            [
                nn.Embedding(len(maps[key]) + 1, dim, padding_idx=0)
                for key, dim in zip(PROFILE_KEYS, PROFILE_DIMS, strict=True)
            ]
        )
        self.history_embeddings = nn.ModuleList(
            [
                nn.Embedding(len(maps[key]) + 1, dim, padding_idx=0)
                for key, dim in zip(
                    (("article_id"), ("product_type_no"), ("colour_group_code")),
                    (64, 16, 8),
                    strict=True,
                )
            ]
        )
        self.text = nn.Linear(text_dim, 32)
        self.image = nn.Linear(image_dim, 32)
        self.item_mlp = nn.Sequential(
            nn.Linear(sum(ITEM_DIMS) + 66, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
        )
        self.user_mlp = nn.Sequential(
            nn.Linear(sum(PROFILE_DIMS) + 88, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
        )

    @staticmethod
    def shared_id_mask(
        batch: int, training: bool, *, device: torch.device | str = "cpu"
    ) -> torch.Tensor:
        """One Bernoulli mask shared by article and product IDs during training."""
        return (
            (torch.rand(batch, 1, device=device) >= 0.25).float()
            if training
            else torch.ones(batch, 1, device=device)
        )

    def item_vectors(
        self,
        codes: torch.Tensor,
        text: torch.Tensor,
        image: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        """Encode user-independent item vectors; missing content branches zeroed."""
        branches = [layer(codes[:, i]) for i, layer in enumerate(self.item_embeddings)]
        drop = self.shared_id_mask(len(codes), self.training, device=codes.device)
        branches[0] = branches[0] * drop
        branches[1] = branches[1] * drop
        branches.extend(
            [self.text(text) * masks[:, :1], self.image(image) * masks[:, 1:2], masks]
        )
        return functional.normalize(self.item_mlp(torch.cat(branches, dim=1)), dim=1)

    def user_vectors(
        self, profiles: torch.Tensor, history: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        """Pool valid history lines; padding has zero weight, UNK remains valid."""
        profile = [
            layer(profiles[:, i]) for i, layer in enumerate(self.profile_embeddings)
        ]
        parts = [
            layer(history[:, :, i]) for i, layer in enumerate(self.history_embeddings)
        ]
        embedded = torch.cat(parts, dim=2)
        pooled = (embedded * weights[:, :, None]).sum(dim=1) / weights.sum(
            dim=1, keepdim=True
        ).clamp_min(1e-12)
        return functional.normalize(
            self.user_mlp(torch.cat([*profile, pooled], dim=1)), dim=1
        )


def positive_mask(
    queries: list[tuple[str, date]],
    candidates: list[str],
    positives: dict[tuple[str, date], set[str]],
) -> torch.Tensor:
    """Mark all same-query positives, preventing duplicate/multi-positive negatives."""
    return torch.tensor(
        [[item in positives[q] for item in candidates] for q in queries],
        dtype=torch.bool,
    )


def logq_loss(
    users: torch.Tensor,
    items: torch.Tensor,
    probabilities: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Multi-positive softmax mass loss on unique in-batch items, temperature .07."""
    if not mask.any(dim=1).all() or not (~mask).any(dim=1).all():
        raise ValueError("Each contrastive row needs a positive and a negative")
    logits = users @ items.T / 0.07 - probabilities.log()[None, :]
    positive_logits = logits.masked_fill(~mask, -torch.inf)
    return (
        torch.logsumexp(logits, dim=1) - torch.logsumexp(positive_logits, dim=1)
    ).mean()
