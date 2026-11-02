"""CPU multimodal towers with past-only events and multi-positive LogQ loss."""

from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as functional

from hm_recsys.content import ContentBank, FloatArray
from hm_recsys.contracts import (
    BASE_END,
    TT_END,
    TT_START,
    Config,
    read_json,
    stable_key,
    write_json,
)
from hm_recsys.data import Sample, Transaction, age_bucket

ITEM_KEYS = (
    "article_id",
    "product_code",
    "product_type_no",
    "graphical_appearance_no",
    "colour_group_code",
    "section_no",
)
PROFILE_KEYS = ("age_bucket", "club_member_status", "fashion_news_frequency")
ITEM_DIMS = (64, 32, 16, 8, 8, 8)
PROFILE_DIMS = (8, 4, 4)


def event_history(rows: list[Transaction], *, before: date) -> list[Transaction]:
    """Latest 20 purchase lines before the event date; same-day rows excluded."""
    legal = [r for r in rows if TT_START <= r.day < before]
    return sorted(
        legal, key=lambda r: (r.day, r.article, r.price, r.channel), reverse=True
    )[:20]


def sample_targets(
    sample: Sample, config: Config
) -> tuple[list[tuple[str, date, str]], dict[str, float]]:
    """Uniform fixed-seed sampling of unique (user,date,item) training events.

    Same-day repeats are one supervision event, while raw histories retain
    lines. Q is the exact item distribution of the selected event population;
    no smoothing or future counts. All known same-query positives are retained
    separately by the loss, even when an event is outside the target cap.
    """
    config.check_sample(sample.manifest)
    events = {
        (t.customer, t.day, t.article)
        for t in sample.transactions
        if TT_START <= t.day < TT_END
    }
    ordered = sorted(events, key=lambda e: (stable_key(f"{e[0]}:{e[1]}:{e[2]}"), e))
    targets = ordered[: config.limit("tt_max_targets")]
    if len(targets) < 2:
        raise ValueError("Insufficient Two-Tower training targets")
    counts = Counter(item for _, _, item in targets)
    return targets, {item: count / len(targets) for item, count in counts.items()}


class TowerEncoder:
    """Frozen categorical maps and bounded tensors; ID maps stop before Aug 5."""

    def __init__(
        self, sample: Sample, content: ContentBank, maps: dict | None = None
    ) -> None:
        self.sample = sample
        self.content = content
        self.histories: dict[str, list[Transaction]] = defaultdict(list)
        for row in sample.transactions:
            self.histories[row.customer].append(row)
        self.maps = maps if maps is not None else self._make_maps()

    def _item_values(self, item: str) -> list[str]:
        meta = self.sample.articles[item]
        return [item, *[str(meta.get(key) or "UNKNOWN") for key in ITEM_KEYS[1:]]]

    def _profile_values(self, user: str) -> list[str]:
        meta = self.sample.customers.get(user, {})
        return [
            age_bucket(meta.get("age")),
            *[str(meta.get(key) or "UNKNOWN") for key in PROFILE_KEYS[1:]],
        ]

    def _make_maps(self) -> dict:
        legal = sorted(
            {t.article for t in self.sample.transactions if TT_START <= t.day < TT_END}
        )
        maps: dict[str, dict[str, int]] = {}
        for i, key in enumerate(ITEM_KEYS):
            # Attribute/category vocabularies are frozen at the same boundary.
            values = sorted({self._item_values(item)[i] for item in legal})
            maps[key] = {value: j + 1 for j, value in enumerate(values)}
        legal_users = sorted(
            {t.customer for t in self.sample.transactions if TT_START <= t.day < TT_END}
        )
        for i, key in enumerate(PROFILE_KEYS):
            values = sorted({self._profile_values(user)[i] for user in legal_users})
            maps[key] = {value: j + 1 for j, value in enumerate(values)}
        return maps

    def items(
        self, ids: list[str]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode IDs, frozen content, and availability for [B] article IDs."""
        codes = [
            [
                self.maps[key].get(value, 0)
                for key, value in zip(ITEM_KEYS, self._item_values(item), strict=True)
            ]
            for item in ids
        ]
        rows = [self.content.position[item] for item in ids]
        masks = np.column_stack(
            [self.content.masks[m][rows] for m in ("text", "image")]
        ).astype(np.float32)
        return (
            torch.tensor(codes, dtype=torch.long),
            torch.from_numpy(self.content.vectors["text"][rows]),
            torch.from_numpy(self.content.vectors["image"][rows]),
            torch.from_numpy(masks),
        )

    def users(
        self, queries: list[tuple[str, date]]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode profile plus explicitly weighted/padded [B,20,3] history IDs."""
        profile = []
        history = np.zeros((len(queries), 20, 3), dtype=np.int64)
        weights = np.zeros((len(queries), 20), dtype=np.float32)
        for b, (user, before) in enumerate(queries):
            profile.append(
                [
                    self.maps[key].get(value, 0)
                    for key, value in zip(
                        PROFILE_KEYS, self._profile_values(user), strict=True
                    )
                ]
            )
            rows = event_history(self.histories.get(user, []), before=before)
            for j, row in enumerate(rows):
                values = self._item_values(row.article)
                for c, field in enumerate((0, 2, 4)):
                    history[b, j, c] = self.maps[ITEM_KEYS[field]].get(values[field], 0)
                # T is the last observable date, one day before prediction.
                weights[b, j] = np.exp(-((before - row.day).days - 1) / 14)
        return (
            torch.tensor(profile, dtype=torch.long),
            torch.from_numpy(history),
            torch.from_numpy(weights),
        )


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
                    ("article_id", "product_type_no", "colour_group_code"),
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
        batch: int,
        training: bool,
        *,
        device: torch.device | str = "cpu",
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


def fit_towers(
    sample: Sample,
    content: ContentBank,
    config: Config,
    run: Path,
    *,
    checkpoint_folder: Path | None = None,
    persistent_checkpoint: Path | None = None,
    persistent_mount_root: Path | None = None,
    resume_folder: Path | None = None,
    checkpoint_every_steps: int = 10,
) -> dict:
    """Perform bounded real CPU updates, then freeze weights and learned indexes."""
    sample.validate_training(config)
    torch.manual_seed(42)
    torch.set_num_threads(config.limit("threads"))
    if torch.get_num_interop_threads() != config.limit("interop_threads"):
        torch.set_num_interop_threads(config.limit("interop_threads"))
    encoder = TowerEncoder(sample, content)
    targets, probabilities = sample_targets(sample, config)
    positives: dict[tuple[str, date], set[str]] = defaultdict(set)
    for row in sample.transactions:
        if TT_START <= row.day < TT_END:
            positives[(row.customer, row.day)].add(row.article)
    model = TwoTower(
        encoder.maps,
        content.vectors["text"].shape[1],
        content.vectors["image"].shape[1],
    )
    initial = model.text.weight.detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    batches = min(
        config.limit("tt_max_steps"),
        (len(targets) + config.limit("tt_batch_size") - 1)
        // config.limit("tt_batch_size"),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, batches)
    )
    losses: list[float] = []
    used: list[tuple[str, str, str]] = []
    skipped = 0
    offset = 0
    checkpoint_identity = (
        {
            "sample_files": sample.manifest["file_sha256"],
            "config": config.values,
            "maps": encoder.maps,
            "target_order": [(u, str(d), i) for u, d, i in targets],
            "build": read_json(run / "run_manifest.json").get("migration_signature"),
        }
        if checkpoint_folder is not None
        else {}
    )
    if resume_folder is not None:
        from hm_recsys.colab.checkpoints import load_checkpoint

        state = load_checkpoint(resume_folder, checkpoint_identity)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        offset = state["next_target_offset"]
        losses, used, skipped = state["losses"], state["used"], state["skipped"]
        if checkpoint_folder is not None:
            from hm_recsys.colab.files import atomic_json, verified_copy

            record = read_json(resume_folder / "latest.json")
            verified_copy(
                resume_folder / record["file"],
                checkpoint_folder / record["file"],
                expected_sha256=record["sha256"],
            )
            atomic_json(checkpoint_folder / "latest.json", record)
    model.train()
    for start in range(offset, len(targets), config.limit("tt_batch_size")):
        if len(losses) >= config.limit("tt_max_steps"):
            break
        batch = targets[start : start + config.limit("tt_batch_size")]
        candidates = sorted({item for _, _, item in batch})
        queries = [(user, day) for user, day, _ in batch]
        mask = positive_mask(queries, candidates, positives)
        valid = (~mask).any(dim=1)
        if not valid.any():
            skipped += len(batch)
            continue
        user_vectors = model.user_vectors(*encoder.users(queries))[valid]
        item_vectors = model.item_vectors(*encoder.items(candidates))
        loss = logq_loss(
            user_vectors,
            item_vectors,
            torch.tensor([probabilities[item] for item in candidates]),
            mask[valid],
        )
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite Two-Tower training loss")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(float(loss.detach()))
        used.extend((u, str(d), i) for u, d, i in batch)
        if checkpoint_folder is not None and (
            len(losses) == 1
            or len(losses) % checkpoint_every_steps == 0
            or len(losses) == batches
            or start + len(batch) >= len(targets)
        ):
            from hm_recsys.colab.checkpoints import save_checkpoint

            save_checkpoint(
                checkpoint_folder,
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": 0,
                    "global_step": len(losses),
                    "next_target_offset": start + len(batch),
                    "losses": losses,
                    "used": used,
                    "skipped": skipped,
                    "selected_checkpoint": "only bounded epoch; selection pending",
                },
                checkpoint_identity,
                persistent=persistent_checkpoint,
                mount_root=persistent_mount_root,
            )
    changed = float(torch.max(torch.abs(initial - model.text.weight.detach())))
    if not losses or changed == 0:
        raise ValueError("Two-Tower did not update its content projection")
    model.eval()
    vectors = encode_items(model, encoder, sample.catalog)
    np.save(run / "models/twotower_items.npy", vectors, allow_pickle=False)
    write_json(run / "models/twotower_items.json", {"article_ids": sample.catalog})
    # A single bounded epoch/checkpoint: evaluate daily internal targets without
    # backpropagation. No repeated checkpoint or hyperparameter search.
    internal: dict[tuple[str, date], set[str]] = defaultdict(set)
    for row in sample.transactions:
        if TT_END <= row.day < BASE_END:
            internal[(row.customer, row.day)].add(row.article)
    if not internal:
        raise ValueError("Empty Two-Tower internal-validation partition")
    recalls = []
    with torch.no_grad():
        for (user, day), truth in sorted(internal.items()):
            counts = Counter(
                t.article for t in sample.transactions if 0 < (day - t.day).days <= 28
            )
            pool = {item for item in sample.catalog if counts[item] >= 3}
            if not event_history(encoder.histories.get(user, []), before=day):
                prediction: list[tuple[str, float]] = []
            else:
                vector = model.user_vectors(*encoder.users([(user, day)])).numpy()[0]
                scores = vectors @ vector
                prediction = sorted(
                    (
                        (item, float(scores[j]))
                        for j, item in enumerate(sample.catalog)
                        if item in pool
                    ),
                    key=lambda pair: (-pair[1], pair[0]),
                )[:50]
            recalls.append(len({i for i, _ in prediction} & truth) / len(truth))
    torch.save(model.state_dict(), run / "models/twotower.pt")
    report = {
        "maps": encoder.maps,
        "text_dim": content.vectors["text"].shape[1],
        "image_dim": content.vectors["image"].shape[1],
        "output_dim": 128,
        "device": "cpu",
        "steps": len(losses),
        "epochs": 1,
        "losses": losses,
        "projection_max_parameter_change": changed,
        "selected_targets": len(targets),
        "processed_targets": len(used),
        "skipped_no_contrast_rows": skipped,
        "target_events": [(u, str(d), i) for u, d, i in targets],
        "processed_events": used,
        "q": probabilities,
        "sampler": (
            "SHA256 unique user/date/item; uniform selected events; no smoothing"
        ),
        "loss": "unique in-batch items; all same-query positives in numerator",
        "fit_start": str(TT_START),
        "fit_end_exclusive": str(TT_END),
        "internal_validation_queries": len(recalls),
        "internal_validation_search": (
            "bounded exact dot-product diagnostic; native R6 uses FAISS "
            "in an isolated stage"
        ),
        "internal_daily_recall50": float(np.mean(recalls)),
        "checkpoint_selection": "only one bounded epoch/checkpoint",
        "inference_dropout": False,
        "unknown_id_policy": (
            "frozen zero row; padding independently masked by history weights"
        ),
    }
    write_json(run / "models/twotower.json", report)
    return report


def encode_items(model: TwoTower, encoder: TowerEncoder, ids: list[str]) -> FloatArray:
    """Build learned vectors in CPU batches, with dropout disabled by caller."""
    if model.training:
        raise ValueError("Learned item indexing requires inference mode")
    blocks = []
    with torch.no_grad():
        for start in range(0, len(ids), 32):
            blocks.append(
                model.item_vectors(*encoder.items(ids[start : start + 32])).numpy()
            )
    return np.concatenate(blocks).astype(np.float32)


def encode_queries(
    sample: Sample, content: ContentBank, run: Path, *, reload: bool
) -> None:
    """Reload frozen towers and encode query histories without labels or FAISS.

    A separate process owns FAISS. This stage persists only the tiny query
    vectors needed to avoid incompatible macOS OpenMP runtimes in one process.
    """
    from hm_recsys.contracts import digest, split_manifest
    from hm_recsys.data import query_users

    config = Config.load(run / "config.resolved.json")
    config.check_sample(sample.manifest)
    state = read_json(run / "models/twotower.json")
    encoder = TowerEncoder(sample, content, state["maps"])
    model = TwoTower(state["maps"], state["text_dim"], state["image_dim"])
    model.load_state_dict(
        torch.load(run / "models/twotower.pt", map_location="cpu", weights_only=True)
    )
    model.eval()
    queries: list[tuple[str, date]] = []
    if reload:
        manifest = read_json(run / "queries/Validation.json")
        queries = [
            (user, date.fromisoformat(manifest["start"]))
            for user in sorted(manifest["users"])
        ]
    else:
        for partition in split_manifest()[:5]:
            start = date.fromisoformat(partition["start"])
            users, _ = query_users(
                sample, start=start, cap=config.limit("max_queries_per_fold")
            )
            queries.extend((user, start) for user in sorted(users))
    vectors = []
    with torch.no_grad():
        for start in range(0, len(queries), 32):
            vectors.append(
                model.user_vectors(*encoder.users(queries[start : start + 32])).numpy()
            )
    if not vectors:
        raise ValueError("No queries to encode")
    name = "reloaded_query_vectors" if reload else "query_vectors"
    destination = run / "predictions" if reload else run / "queries"
    destination.mkdir(exist_ok=True)
    np.save(destination / f"{name}.npy", np.concatenate(vectors), allow_pickle=False)
    write_json(
        destination / f"{name}.json",
        {
            "queries": [[user, str(day)] for user, day in queries],
            "has_history": [
                bool(event_history(encoder.histories.get(user, []), before=day))
                for user, day in queries
            ],
            "model_sha256": digest(run / "models/twotower.pt"),
            "labels_read": False
            if reload
            else "purchase-activity query selection only",
        },
    )
