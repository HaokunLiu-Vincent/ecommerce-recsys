"""Past-only dynamic channels and an untruncated native provenance union."""

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, timedelta

from hm_recsys.contracts import CHANNELS, QUOTAS, Config
from hm_recsys.data import Sample, Transaction, age_bucket
from hm_recsys.learned import WALS, TowerRetriever


@dataclass(frozen=True)
class Hit:
    """One source contribution; raw scores retain their channel-local scale."""

    score: float
    rank: int
    fallback: bool = False


@dataclass
class Candidate:
    """One native item with every contributing channel, without future labels."""

    article: str
    hits: dict[str, Hit]

    @property
    def rrf(self) -> float:
        """Sum contributions from actual ranks only, without truncating union."""
        return sum(1 / (60 + hit.rank) for hit in self.hits.values())


def top(scores: dict[str, float], active: set[str], k: int) -> list[tuple[str, float]]:
    """Apply eligibility first and stable article-ID ties before the quota."""
    return sorted(
        ((i, s) for i, s in scores.items() if i in active),
        key=lambda pair: (-pair[1], pair[0]),
    )[:k]


class Snapshot:
    """Bounded cutoff aggregates shared by dynamic retrieval and features."""

    def __init__(self, sample: Sample, *, cutoff: date, config: Config) -> None:
        config.check_sample(sample.manifest)
        self.sample = sample
        self.cutoff = cutoff
        self.before = cutoff + timedelta(days=1)
        self.history = [row for row in sample.transactions if row.day < self.before]
        self.by_user: dict[str, list[Transaction]] = defaultdict(list)
        self.by_item: dict[str, list[Transaction]] = defaultdict(list)
        self.sales = {d: Counter() for d in (3, 7, 14, 28)}
        self.pop: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        bucket_sales: dict[str, dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        self.edges: dict[str, dict[str, date]] = defaultdict(dict)
        for row in self.history:
            self.by_user[row.customer].append(row)
            self.by_item[row.article].append(row)
            age = (self.cutoff - row.day).days
            for d, counts in self.sales.items():
                if age < d:
                    counts[row.article] += 1
            if age < 7:
                bucket = age_bucket(sample.customers[row.customer].get("age"))
                bucket_sales[bucket][row.article] += 1
                score = math.exp(-age / 3)
                self.pop["GLOBAL"][row.article] += score
                self.pop[bucket][row.article] += score
            if age < 28:
                last = self.edges[row.customer].get(row.article)
                if last is None or row.day > last:
                    self.edges[row.customer][row.article] = row.day
        self.active = {i for i in sample.catalog if self.sales[28][i] >= 3}
        self.bucket_ranks = {
            bucket: {
                item: rank
                for rank, (item, _) in enumerate(
                    top(scores, self.active, len(self.active)), 1
                )
            }
            for bucket, scores in bucket_sales.items()
        }
        self.itemcf, self.pair_estimate = self._itemcf(config)
        self.products: dict[str, list[str]] = defaultdict(list)
        for item in sample.catalog:
            product = sample.articles[item].get("product_code")
            if product is not None:
                self.products[str(product)].append(item)

    def window(self, rows: list[Transaction], days: int) -> list[Transaction]:
        """Return the exact inclusive-T trailing window from past-only rows."""
        lower = self.before - timedelta(days=days)
        return [row for row in rows if lower <= row.day < self.before]

    def _itemcf(self, config: Config) -> tuple[dict[str, list[tuple[str, float]]], int]:
        item_users: dict[str, dict[str, date]] = defaultdict(dict)
        pairs = sum(len(edges) * (len(edges) - 1) for edges in self.edges.values())
        if pairs > config.limit("max_itemcf_pairs"):
            raise ValueError("ItemCF estimated pair budget exceeded before enumeration")
        for user, edges in self.edges.items():
            for item, day in edges.items():
                item_users[item][user] = day
        neighbors: dict[str, list[tuple[str, float]]] = {}
        # At most one seed's sparse accumulator is live. No I-by-I matrix or
        # full pair snapshot: each user list is traversed in bounded seed blocks.
        for seed in sorted(item_users):
            score: dict[str, float] = defaultdict(float)
            for user, seed_day in item_users[seed].items():
                edges = self.edges[user]
                for other, other_day in edges.items():
                    if other != seed and other in self.active:
                        score[other] += math.exp(
                            -abs((seed_day - other_day).days) / 7
                        ) / math.log1p(len(edges))
            normalized = {
                other: value
                / math.sqrt(len(item_users[seed]) * len(item_users[other]) ** 0.75)
                for other, value in score.items()
            }
            neighbors[seed] = top(normalized, self.active, 100)
        return neighbors, pairs

    def item_statistics(self, item: str) -> tuple[float, float]:
        """Return the shared 28-day distinct-date ratio and purchase-line mean price."""
        import numpy as np

        rows = self.window(self.by_item.get(item, []), 28)
        dates: dict[str, set] = defaultdict(set)
        for row in rows:
            dates[row.customer].add(row.day)
        ratio = (
            sum(len(value) >= 2 for value in dates.values()) / len(dates)
            if dates
            else float("nan")
        )
        price = float(np.mean([row.price for row in rows])) if rows else float("nan")
        return ratio, price

    def repurchase(
        self, user: str
    ) -> tuple[dict[str, float], Counter, dict[str, date]]:
        """Aggregate 60-day line counts, last dates, and 14-day decays."""
        score: dict[str, float] = defaultdict(float)
        count: Counter = Counter()
        last: dict[str, date] = {}
        for row in self.window(self.by_user.get(user, []), 60):
            score[row.article] += math.exp(-(self.cutoff - row.day).days / 14)
            count[row.article] += 1
            last[row.article] = max(last.get(row.article, row.day), row.day)
        return score, count, last

    def sibling(self, user: str) -> tuple[dict[str, float], dict[str, int]]:
        """Use real product-code joins and maximum nomination, excluding self."""
        seeds: dict[str, date] = {}
        for row in self.window(self.by_user.get(user, []), 30):
            seeds[row.article] = max(seeds.get(row.article, row.day), row.day)
        scores: dict[str, float] = {}
        days: dict[str, int] = {}
        for seed, day in sorted(seeds.items()):
            product = str(self.sample.articles[seed].get("product_code"))
            age = (self.cutoff - day).days
            for item in self.products.get(product, []):
                if item == seed or item not in self.active:
                    continue
                score = math.exp(-age / 14) * math.log1p(self.sales[7][item])
                if item not in scores or score > scores[item]:
                    scores[item] = score
                    days[item] = age
        return scores, days

    def retrieve(
        self,
        user: str,
        wals: WALS,
        tower: TowerRetriever,
        *,
        learned: dict[str, list[tuple[str, float]]] | None = None,
    ) -> tuple[list[Candidate], dict[str, list[tuple[str, float]]]]:
        """Generate all six real channels, then retain the complete native union."""
        bucket = age_bucket(self.sample.customers.get(user, {}).get("age"))
        primary = top(
            self.pop.get(bucket, {}) if bucket != "UNKNOWN" else self.pop["GLOBAL"],
            self.active,
            60,
        )
        primary_ids = {i for i, _ in primary}
        backfill = [
            pair
            for pair in top(self.pop["GLOBAL"], self.active, len(self.active))
            if pair[0] not in primary_ids
        ][: 60 - len(primary)]
        repurchase, _, _ = self.repurchase(user)
        sibling, _ = self.sibling(user)
        cf: dict[str, float] = defaultdict(float)
        seeds = sorted(
            self.edges.get(user, {}).items(),
            key=lambda pair: (-pair[1].toordinal(), pair[0]),
        )[:15]
        for seed, day in seeds:
            for item, value in self.itemcf.get(seed, []):
                cf[item] += math.exp(-(self.cutoff - day).days / 10) * value
        outputs = {
            "trend": primary + backfill,
            "repurchase": top(repurchase, self.active, 30),
            "sibling": top(sibling, self.active, 30),
            "itemcf": top(cf, self.active, 50),
            "wals": learned["wals"]
            if learned is not None
            else wals.retrieve(
                self.by_user.get(user, []), cutoff=self.cutoff, active=self.active
            ),
            "twotower": learned["twotower"]
            if learned is not None
            else tower.retrieve(user, before=self.before, active=self.active),
        }
        union: dict[str, Candidate] = {}
        for channel in CHANNELS:
            values = outputs[channel]
            if len(values) > QUOTAS[channel] or len({i for i, _ in values}) != len(
                values
            ):
                raise ValueError("Channel quota or uniqueness violation")
            for rank, (item, score) in enumerate(values, 1):
                if item not in self.active or not math.isfinite(score):
                    raise ValueError("Invalid channel eligibility or score")
                candidate = union.setdefault(item, Candidate(item, {}))
                candidate.hits[channel] = Hit(
                    score, rank, channel == "trend" and item not in primary_ids
                )
        if len(union) > sum(QUOTAS.values()):
            raise ValueError("Native union exceeds budget")
        return [union[item] for item in sorted(union)], outputs
