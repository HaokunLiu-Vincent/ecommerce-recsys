# H&M Evaluation Protocol v2.0

**Status:** Proposed implementation contract; not executed or frozen as data artifacts.  
**Updated:** 2026-11-02 · documentation revision v0.3.2  
**Companion:** [proposal.md](../proposal.md)  
**Primary metric:** Native end-to-end macro MAP@12.

This protocol follows the newly supplied six-channel architecture, not the
archived v1.0 protocol. Explicit clarifications are recorded in
[implementation_decisions.md](implementation_decisions.md). The latest user
decisions and approved new architecture take precedence
over older reports and archived notebooks. References in `docs/achive/` are
not an alternative evaluation specification. The source alone cannot establish
leakage-free execution or measured recommendation quality.

## Contents

- [1. Units and time](#1-units-and-time)
- [2. Frozen and dynamic state](#2-frozen-and-dynamic-state)
- [3. Catalog, candidate eligibility, and ground truth](#3-catalog-candidate-eligibility-and-ground-truth)
- [4. Native candidate generation](#4-native-candidate-generation)
- [5. Ranker training and validation behavior](#5-ranker-training-and-validation-behavior)
- [6. Metrics](#6-metrics)
- [7. Ablations and selection](#7-ablations-and-selection)
- [8. Minimum audit artifacts and tests](#8-minimum-audit-artifacts-and-tests)
- [9. Limitations](#9-limitations)

## 1. Units and time

A query is `(customer_id, cutoff_date)`. The cutoff T is the last observable
calendar date, inclusive. The prediction date is `a = T + 1 day`.

```text
History:       t_dat < a
Trailing d:    [a - d days, a)
Future labels: [a, a + 7 days)
```

Dates are source calendar dates. There is no intraday ordering. Preserve raw
purchase lines; deduplicate article IDs only when creating a query's label set.

| Role | Data range or cutoff | Target range |
|---|---|---|
| WALS training | `[2020-02-01, 2020-08-12)` | Implicit preference/confidence |
| WALS item vocabulary | Purchases in `[2020-06-16, 2020-08-12)` | Fixed column map |
| Two-Tower training | Target dates in `[2020-05-19, 2020-08-05)` | History strictly before each target date |
| Two-Tower internal validation | `[2020-08-05, 2020-08-12)` | Daily past-only query Recall@50 |
| Ranker train F1 | `2020-08-11` | `[2020-08-12, 2020-08-19)` |
| Ranker train F2 | `2020-08-18` | `[2020-08-19, 2020-08-26)` |
| Ranker train F3 | `2020-08-25` | `[2020-08-26, 2020-09-02)` |
| Ranker train F4 | `2020-09-01` | `[2020-09-02, 2020-09-09)` |
| Global validation | `2020-09-08` | `[2020-09-09, 2020-09-16)` |
| Final test | `2020-09-15` | `[2020-09-16, 2020-09-23)` |

**D06 is mandatory:** All six target intervals (F1–F4, global Validation, and
Test) are pairwise disjoint, each exactly seven days, and jointly assign the
42 dates from 2020-08-12 through 2020-09-22 exactly once. Two-Tower training
and internal-validation target dates are also disjoint. Reject overlapping
weekly anchors, random row splits, and the old three-day anchor stride.

Persist the exact table in a split manifest. Assert the declared dates,
seven-day lengths, pairwise-empty intersections, and that each ranker cutoff
is one day before its label start. Fail before candidates/features are built
on a conflicting manifest; do not merely issue a warning.

Users and past-only lookback histories may overlap. A completed earlier week
may be observed history for a later query, but must not become a second target
week through overlapping windows. Shared base-model history is intentional;
statistical independence is not asserted.

## 2. Frozen and dynamic state

WALS item factors/mappings and selected Two-Tower weights are frozen before the
first ranker query. Two-Tower gradient training stops before its internal
validation interval. Clustering and learned preprocessing have documented
pre-base fit populations and remain frozen. Supplied text/image artifacts are
read from `data/text_features/` and `data/image_features/` after mapping,
dimension, and provenance checks. Do not regenerate them by default or infer
future-interaction safety from filenames. Missing content never removes a
future positive from the main metric.

At a later cutoff, R1-R4, the active pool, user history, feature statistics, and
WALS user folding-in may use newly observed transactions through that cutoff.
This is permitted sequential observation, not refitting on the current query's
future labels. Item factors and Two-Tower parameters do not change. Model
selection uses global Validation only; no Test-driven refit is permitted.

## 3. Catalog, candidate eligibility, and ground truth

Let `C_catalog` be the supplied article catalog. The dynamic candidate pool is:

```text
A(T) = catalog articles with >= 3 purchase lines in [a - 28 days, a)
```

This is an activity filter, not stock verification. Each channel selects only
from `A(T)`; WALS further requires a frozen item factor. Missing content disables
that content representation, not the label or other channels.

For query q, define the **full ground truth**:

```text
G_q = unique articles purchased by that customer in [a, a + 7 days)
      that match C_catalog
```

Do **not** intersect `G_q` with `A(T)`, WALS vocabulary, images, or retrieved
candidates for the primary metric. A future purchase of an inactive or
unrepresented item remains a miss. Report any unexpected catalog join failures
rather than silently dropping them.

The main offline population is all customer/cutoff queries with `|G_q| >= 1`.
There is no minimum historical-purchase count. This is a purchase-active-query
benchmark, not an estimate across all online requests. Keep excluded
zero-future-purchase counts where a broader query universe is constructed.

Create an independent query manifest before joining candidates. A query with
zero candidates or zero retrieved positives must remain in Validation/Test and
contribute zero to end-to-end metrics.

## 4. Native candidate generation

Default channel quotas are:

| R1 Trend | R2 Repurchase | R3 Sibling | R4 ItemCF | R5 WALS | R6 Two-Tower |
|---:|---:|---:|---:|---:|---:|
| 60 | 30 | 30 | 50 | 50 | 50 |

Filter candidates before channel truncation. Keep unique article IDs within each
channel, one-based ranks, deterministic article-ID tie breaks, original scores,
and model/cutoff versions. Union by query/article and preserve all provenance.
The pre-deduplication bound is 270; actual sizes must be measured.

No additional RRF truncation is applied by default. RRF is a ranker feature and
may be reported as a non-learned comparison. No future-positive injection is
allowed in Train, Validation, Test, or a disguised candidate backfill.

## 5. Ranker training and validation behavior

Attach binary labels only after native candidates and past-only features have
been built. Repeat purchases of the same article within the target week remain
one relevant item, not relevance grade 2.

Train on F1-F4. Drop training groups with no positive or no proxy-negative pair;
log those counts. Preserve all native positives. Set the number of negatives
to `min(available_negatives, 20 * native_positives)`. The default hard quota is
`floor(0.5 * selected_negatives)` using descending native RRF and article-ID tie
breaks. Sample the remainder without replacement from the remaining negatives
using a stable query-local seed derived from the recorded run seed and query
key. Do not use process-randomized Python `hash()` for persistent seed identity.

Validation and Test use complete native candidate sets, no negative sampling,
and no positive-hit filtering. Feature schemas and categorical encodings must
match training. Groups must be contiguous; assert row alignment and group-size
sums after every transformation.

LightGBM candidate NDCG@12 can drive the source proposal's optimization-stage
early stopping (50 rounds), but is recorded as a diagnostic. Compare models and
quotas using external, full-ground-truth native MAP@12. Do not claim that
candidate-only NDCG or candidate-only AP is the pipeline's main metric.

## 6. Metrics

For each query, let `P_q` be the final unique ranked article list and `G_q` the
full label set. Ties use article ID ascending. Recommendations shorter than K
are valid for offline evaluation; unfilled positions earn no credit.

```text
rel_q(j) = 1 if P_q[j] is in G_q, otherwise 0
Precision@j(q) = relevant recommendations through position j / j
AP@12(q) = sum_{j=1..min(12, len(P_q))}(Precision@j(q) * rel_q(j))
           / min(12, |G_q|)
MAP@12 = arithmetic mean of AP@12 over all eligible queries
Recall@K(q) = |P_q[:K] intersect G_q| / |G_q|
DCG@K(q) = sum_j rel_q(j) / log2(j + 1)
IDCG@K(q) = sum_{j=1..min(K, |G_q|)} 1 / log2(j + 1)
NDCG@K(q) = DCG@K(q) / IDCG@K(q)
```

Ranks j are one-based. Evaluate final macro Recall@12 and full-ground-truth
macro NDCG@12 as secondary metrics. Aggregate by query, not transaction line,
article, or unique customer across the four training cutoffs.

Also report:

- Per-channel macro Recall at configured K and selected smaller K values.
  Do not generate extra candidates merely to report a higher K.
- Native union recall `|C_q intersect G_q| / |G_q|`, actual candidate counts,
  and queries with no retrieved positives.
- Active-pool coverage `|A(T) intersect G_q| / |G_q|`, with excluded-positive
  counts. This separates a catalog-policy loss from a channel-ranking loss.
- An optional conditional ranking diagnostic uses `G_q intersect C_q` and
  excludes zero-hit queries only from that diagnostic, reporting its denominator.

The metric implementation must either reject duplicate recommendation IDs or
apply an explicit, tested stable-deduplication rule. Never grant extra relevance
credit to duplicate articles.

## 7. Ablations and selection

Keep the same date, query, label, pool, feature, and cutoff semantics across a
comparison. For default leave-one-channel-out recall, remove the channel and
rebuild the union without redistributing its quota. Budget redistribution is a
separate configuration and must be labeled accordingly. If comparing complete
retrieval/ranking pipelines, retrain the ranker consistently and record that
choice rather than conflating it with a simple retrieval ablation.

For ranking-feature ablations, freeze candidate sets. For Two-Tower content
ablations, retain training sampler and model-selection rules and vary only the
stated branches. No observed uplift is claimed until recorded runs support it.

Global Validation may select quotas, hyperparameters, and features. Save the
complete chosen configuration before evaluating Test once. Repeated manual
Test peeking is test-driven tuning even if model weights are not updated.

## 8. Minimum audit artifacts and tests

Save versioned manifests for splits, catalog, active pools, query populations,
unique labels, channel quotas, source ranks/scores, sampling, feature schemas,
models, indexes, encoder identities, code revision, and runtime. Keep them under
`data/`; store readable explanations in `docs/archi/`.

Tests must cover exact split dates and pairwise-disjoint target windows,
seven-day window length, same-day leakage, raw duplicates vs
unique labels, inactive-item positives, vocabulary misses, query preservation,
empty candidates, short rankings, duplicate outputs, hand-calculated AP/NDCG,
and train-only sampling. A fixture that retrieves nothing must produce zero
main score without losing any eligible query.

## 9. Limitations

There are no reliable historical stock, impression, order, or metadata
availability logs. The active mask is imperfect; static content/customer
snapshots are an explicit assumption. Purchase zeros are unobserved outcomes,
not verified dislikes. Model staleness differs across frozen and dynamic
channels. The benchmark population is conditioned on future purchasing.

The final offline Test ends on 2020-09-22. It is not an actual Kaggle submission
for 2020-09-23 onward, whose labels are not in the supplied training data. A later
competition submission or deployment refit needs its own documented artifact
and schedule; do not claim the offline split executed such a refit.
