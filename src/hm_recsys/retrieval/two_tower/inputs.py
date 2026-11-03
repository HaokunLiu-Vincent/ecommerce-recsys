"""Assemble the one shared category, history, padding and raw-content input contract.

TowerEncoder defines deterministic tensor assembly; SqlEncoder supplies bounded
canonical histories instead of materializing all events. Histories contain the
latest twenty raw purchase lines strictly before each target date, with fixed
same-date ties, zero-weight padding and fourteen-day decay. Text/image row maps
remain independent. No encoder fit or learned-vector cache is performed here.
"""

from collections import defaultdict
from datetime import date

import numpy as np
import torch
from torch.nn import functional as functional

from hm_recsys.core.contracts import TT_END, TT_START
from hm_recsys.data.history import History
from hm_recsys.data.sample import Sample, Transaction, age_bucket
from hm_recsys.data.streaming import export_query, rows
from hm_recsys.features.content import ContentBank, MappedContent
from hm_recsys.retrieval.two_tower.execution import ItemBank, Measurements
from hm_recsys.retrieval.two_tower.model import ITEM_KEYS, PROFILE_KEYS


def event_history(rows: list[Transaction], *, before: date) -> list[Transaction]:
    """Latest 20 purchase lines before the event date; same-day rows excluded."""
    legal = [r for r in rows if TT_START <= r.day < before]
    return sorted(
        legal, key=lambda r: (r.day, r.article, r.price, r.channel), reverse=True
    )[:20]


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
            torch.from_numpy(self.content.vectors[("text")][rows]),
            torch.from_numpy(self.content.vectors[("image")][rows]),
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
                weights[b, j] = np.exp(-((before - row.day).days - 1) / 14)
        return (
            torch.tensor(profile, dtype=torch.long),
            torch.from_numpy(history),
            torch.from_numpy(weights),
        )


class SqlEncoder(TowerEncoder):
    """Reuse tensor/category semantics while fetching at most 20 lines per event."""

    def __init__(
        self, history: History, content: MappedContent, maps: dict | None = None
    ) -> None:
        self.history_store = history
        self.content = content
        self.sample = history.sample([])
        self.histories = {}
        if maps is None:
            legal = [
                r[0]
                for r in history.con.execute(
                    (
                        """SELECT DISTINCT article_id FROM transactions JOIN
                        cohort
                        USING(customer_id) WHERE t_dat>=? AND t_dat<? ORDER BY
                        article_id"""
                    ),
                    [TT_START, TT_END],
                ).fetchall()
            ]
            maps = {}
            for j, key in enumerate(ITEM_KEYS):
                values = sorted({self._item_values(i)[j] for i in legal})
                maps[key] = {v: k + 1 for k, v in enumerate(values)}
            export_query(
                history.con,
                (
                    """SELECT DISTINCT c.* FROM customers c JOIN cohort
                    USING(customer_id) JOIN
                    transactions t USING(customer_id) WHERE t.t_dat>=? AND
                    t.t_dat<? ORDER
                    BY c.customer_id"""
                ),
                [TT_START, TT_END],
                history.run / ("tmp/profile_categories.parquet"),
            )
            categories = [set() for _ in PROFILE_KEYS]
            for batch in rows(history.run / "tmp/profile_categories.parquet", 512):
                for record in batch:
                    user = record["customer_id"]
                    record["age"] = float(record["age"]) if record.get("age") else None
                    self.sample.customers = {user: record}
                    for j, value in enumerate(self._profile_values(user)):
                        categories[j].add(value)
            for key, values in zip(PROFILE_KEYS, categories, strict=True):
                maps[key] = {v: k + 1 for k, v in enumerate(sorted(values))}
            self.sample.customers = {}
        self.maps = maps
        self.measurements: Measurements | None = None
        self.bank: ItemBank | None = None

    def users(
        self, queries: list[tuple[str, date]]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fetch one bounded SQL event batch and reuse the original padding/weights."""
        from contextlib import nullcontext

        timing = self.measurements
        with timing.span("profile_lookup") if timing else nullcontext():
            self.sample.customers = self.history_store.customers(
                sorted({u for u, _ in queries})
            )
        with timing.span("history_sql") if timing else nullcontext():
            histories = self.history_store.event_histories(queries)
        blocks = []
        for query, history_rows in zip(queries, histories, strict=True):
            self.histories = {query[0]: history_rows}
            blocks.append(super().users([query]))
        self.histories = {}
        return (
            torch.cat([b[0] for b in blocks]),
            torch.cat([b[1] for b in blocks]),
            torch.cat([b[2] for b in blocks]),
        )
