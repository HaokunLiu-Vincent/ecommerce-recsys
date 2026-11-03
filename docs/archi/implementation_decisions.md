# Implementation Decisions and Source Clarifications

**Version:** v0.3.2 · Evaluation Protocol v2.0  
**Updated:** 2026-11-03  
**Status:** Current design decisions, not new experimental results.

## Contents

- [1. Sources and revision boundary](#1-sources-and-revision-boundary)
- [2. Time, data, and architecture](#2-time-data-and-architecture)
- [3. Retrieval and content artifacts](#3-retrieval-and-content-artifacts)
- [4. Ranking, features, and metrics](#4-ranking-features-and-metrics)
- [5. Remaining implementation gates](#5-remaining-implementation-gates)
- [6. Single-source execution layout](#6-single-source-execution-layout)

## 1. Sources and revision boundary

The latest explicit user decisions and the approved new architecture govern
[proposal.md](../proposal.md) and all current architecture documents.
Preserve the six channels, `60/30/30/50/50/50` quotas, exact dates, base/dynamic
update policies, single LambdaMART, and previously recorded algorithm choices.

This revision resolves D06/D07 and the supplied-content implications of D14.
It does not restart embedding extraction, alter model objectives, or claim
new benchmark results. The user reports prepared artifacts in
`data/text_features/` and `data/image_features/`; this documentation task did
not inspect their contents.

The historical [Phase 0 report](<../phase0-data download & EDA report.md>)
remains evidence for its measured facts. The listed MVP report and
`docs/achive/` contents are limited references, not controlling specifications.
Their availability in the user's directory is not proof they have been reviewed
or are compatible with the new architecture.

## 2. Time, data, and architecture

| ID | Source ambiguity or superseded assumption | Current decision |
|---|---|---|
| D01 | A seven-day lookback was also written as inclusive `[T-7,T]`. | `T` is the last observable date; `a=T+1`; trailing d days are `[a-d,a)` and labels are `[a,a+7)`. |
| D02 | Approximate week counts accompanied exact dates. | Use the listed calendar dates, not a recomputation from approximate weeks. |
| D03 | Active items were described as verified stock. | Preserve the three-purchase-lines/28-day rule as an activity proxy only. It does not establish stock or sizes. |
| D04 | Estimated active pool and candidate counts appeared as definite quantities. | Keep them as unmeasured estimates. Quotas sum to 270; actual pool and union counts require execution. |
| D05 | Siblings were explained by ID prefixes or size variants. | Join the real `product_code`. Exclude the seed itself, not all purchased items; do not assert size/order semantics. |
| D06 | Old and new temporal protocols conflict. | The new schedule is mandatory. F1–F4, global Validation, and Test target weeks are pairwise disjoint; Two-Tower train/internal-validation targets are disjoint. Reject overlapping anchors and incorrect boundaries before execution. Preserve exact dates below. |
| D07 | Prior documents assumed `data/raw/`, `data/embeddings/`, and `docs/references/`. | Read the user's actual inputs in place. Preserve `data/original data/`, `data/image_features/`, `data/text_features/`, and the literal `docs/achive/` name. Add current documents under `docs/archi/`; use old reports/notebooks only as limited references. New architecture wins every core-requirement conflict. |

### 2.1 D06 acceptance contract

| Target partition | Start, inclusive | End, exclusive | Inclusive feature cutoff |
|---|---|---|---|
| Ranker train F1 | 2020-08-12 | 2020-08-19 | 2020-08-11 |
| Ranker train F2 | 2020-08-19 | 2020-08-26 | 2020-08-18 |
| Ranker train F3 | 2020-08-26 | 2020-09-02 | 2020-08-25 |
| Ranker train F4 | 2020-09-02 | 2020-09-09 | 2020-09-01 |
| Global Validation | 2020-09-09 | 2020-09-16 | 2020-09-08 |
| Final Test | 2020-09-16 | 2020-09-23 | 2020-09-15 |

Assert exact dates, seven-day lengths, pairwise-empty intersections, cutoffs
one day before starts, and single coverage of all 42 dates. The old
three-day-stride fifteen-anchor protocol must fail validation. Two-Tower
training targets are `[2020-05-19,2020-08-05)` and internal-validation targets
are `[2020-08-05,2020-08-12)`.

Past-only histories may overlap and earlier purchases may become later observed
history; these are not overlapping target splits. Do not reinterpret this
permission as authorization to reuse a purchase in multiple target weeks.

### 2.2 D07 reference and filesystem contract

The exact initial trees and artifact roles are in proposal Sections 5.2–5.3.
They supersede the old generic source-path examples. No directory renaming,
source copying, archive extraction, or notebook execution is part of this
revision.

Existing reports and notebooks may inform data facts or an individually reviewed
helper. They cannot select dates, channels, quotas, labels, candidate universes,
feature semantics, or metric denominators for the new pipeline. Do not run old
notebooks as implementation entry points or transplant their entire pipeline.
Document any limited reuse and retest it under current contracts.

All newly generated machine artifacts remain under `data/`; current design
files remain under `docs/archi/`. Existing archived files remain untouched.

## 3. Retrieval and content artifacts

| ID | Source ambiguity or missing definition | Current decision |
|---|---|---|
| D08 | Age buckets overlap at boundaries. | Use `<25`, `[25,35)`, `[35,50]`, `>50`; unknown ages use UNKNOWN/global fallback. All backfill counts within R1's quota. |
| D09 | ItemCF repeated-interaction timestamps are undefined. | Use distinct user-item edges and the last date in the 28-day window; latest 15 distinct seeds at inference. Do not delete raw purchase lines. |
| D10 | ItemCF `alpha=0.75` was called stronger popularity suppression. | Keep the formula, document its directionality and weaker candidate denominator relative to alpha=1; version any formula change separately. |
| D11 | Explicit confidence and library `alpha=20` could both scale the data. | Construct full confidence once with `confidence_alpha=20`, `epsilon=1`; library alpha is 1. |
| D12 | Vector pooling was conflated with ALS folding-in. | Use the fixed-item regularized ALS user solve. History begins 2020-02-01; decay is relative to the current cutoff. The normalized mixture is an unselected alternative. |
| D13 | New users and WALS vocabulary boundaries were unclear. | Fold in only representable item history; unknown items have no factors. Only WALS intersects its fixed vocabulary with the current active pool. |
| D14 | The original encoder examples and older status assumed content extraction was pending. | Superseded by the user's prepared artifacts. Load the two arrays, separate ID maps, manifests, and image failure log in place. Inspect actual dimensions/provenance; use `Linear(D_text,32)` and `Linear(D_image,32)`. Never choose or rerun an encoder by default. Existing content FAISS indexes are not standalone channels or substitutes for learned indexes. |
| D15 | User-history aggregation and same-day order were unclear. | Use time-decayed mean pooling, 14-day decay, history from 2020-05-19, latest 20 purchase lines. Same-day tie handling is deterministic, not observed chronology. |
| D16 | Zeroing IDs was conflated with learned UNK embeddings. | Share a 0.25 training-time mask across article/product-code ID branches and zero them. Unknown IDs have separate UNK semantics; no inference dropout. Train vocabularies stop on 2020-08-04. |
| D17 | InfoNCE omitted duplicate and multi-positive treatment. | Lock/test the sampler, duplicate handling, and multi-positive masks. Q comes only from the training sampler, not future frequency. |
| D18 | All FAISS usage was described as ANN. | The new learned `IndexFlatIP` is exact inner-product search. The supplied content indexes' type/metric are unknown until inspected; filenames are not evidence. |

### 3.1 Content acceptance instead of re-extraction

Validate shapes/dtypes, finite/nonzero usable vectors, separate row/ID mappings,
catalog joins, failure semantics, normalization, provenance, and available
fingerprints. Do not assume text/image row orders match or that a text encoder
used only `detail_desc`. Record unknown manifest fields honestly.

Do not mutate the original manifests or arrays. A required mapping/dimension
conflict blocks its consumer; ordinary missing modalities use masks. Optional
content indexes are not necessary for vector lookup. Only audit/use them when
needed; do not silently rebuild them or replace them with learned indexes.

Create acceptance reports in `data/manifests/`. Later tower vectors, factor
indexes, and K-Means are separate derived artifacts with their own lineage.
No raw-image decoding or source-embedding generation is a default phase gate.

## 4. Ranking, features, and metrics

| ID | Source ambiguity or missing definition | Current decision |
|---|---|---|
| D19 | Binary labels also suggested grade 2 for repeats. | Use unique-article binary relevance 0/1 for MAP@12; do not promote repeated purchase lines. |
| D20 | A 1:20 negative limit also said retain all hard negatives. | Limit negatives to `min(N,20P)`; take floor(half) by RRF and sample the remainder with a recorded stable seed. Keep all native positives. |
| D21 | Zero-positive, all-positive, and zero-candidate groups were unspecified. | Filter no-pair groups in Train only. Validation/Test retain all purchase-active queries, including zero hits/candidates. |
| D22 | Missing scores could be -999 or zero; ranks could be 999. | Use NaN plus separate presence masks with one shared feature schema. |
| D23 | The listed feature total said 38. | Preserve the 39 enumerated core names; add six channel and four content masks for a proposed 49 columns. Code reads F from schema. |
| D24 | An order-based repurchase ratio lacked order data. | Define it as the share of item buyers purchasing on at least two distinct dates within 28 days. Preserve the name, document the clarified meaning. |
| D25 | Some lookbacks and missing policies were unspecified. | Item mean price uses 28 days; user mean price, affinities, and channel preference use 90. Missing stays missing, without future filling. |
| D26 | Visual clustering used all items without a fit boundary. | Fit 128 clusters on valid supplied image vectors of pre-base observed items; freeze and only transform later items. Check any upstream learned transformation's provenance separately. |
| D27 | NDCG@12 was equated with MAP@12. | Distinguish the LambdaRank objective, candidate NDCG diagnostics, and full-ground-truth native MAP@12 selection. Equal cutoffs do not imply equal metrics. |
| D28 | The active pool's effect on labels was unclear. | Primary ground truth includes all catalog-matched future positives; pool/vocabulary/content misses remain system misses. Pool-conditioned metrics are diagnostic only. |
| D29 | RRF might imply another candidate truncation. | Send the complete native six-channel union to the ranker, at most 270 initially. No pre-ranker; tune quotas only on Validation. |
| D30 | Production readiness, cold-start quality, and zero leakage were asserted. | Treat them as goals requiring actual tests/runs. Prepared content alone does not validate quality, temporal provenance, or end-to-end capability. |

## 5. Remaining implementation gates

**Existing artifact evidence:** Inspect local arrays, mappings, manifests,
failures, and any optional index actually used. The user's preparation status
is accepted; the documentation revision does not invent its dimensions or audit
outcome. Unknown behavioral/postprocessing provenance remains a limitation.

**Two-Tower sampling:** Specify user/date/positive sampling, same-day repeats,
duplicate in-batch items, multi-positive masks, and the resulting Q. The loss
formula is not a complete sampler implementation.

**Environment and scale:** Verify actual package versions and resources before
large ItemCF pairing, batch=2048 training, or full ranker matrices. Keep bounded
local checks separate from real H&M execution.

**Source limitations:** Static snapshots do not provide historical stock,
exposure, orders, or guaranteed point-in-time attributes. No new configuration
can manufacture those missing observations.

Record resolved choices, tests, data paths, and immutable identities in current
configs/manifests. Do not edit archives to erase the design history.

## 6. Single-source execution layout

D30 (2026-11-03): `src/hm_recsys/` is the sole active implementation,
organized into core, data, features, retrieval, fusion, ranking, evaluation and
runtime layers. Smoke and full use the same SQL histories, model trainers,
feature builder, sampler and evaluator; only explicit budgets differ. The
former in-memory smoke model path is archived. The current optimized SQL backend
was captured on a tiny synthetic input before refactoring for artifact comparison.

D31: Functional Notebook stages are independently executable and require
verified completed dependencies. Six recall stages share prepared cutoff
statistics but never implicitly train another channel. RRF persists the full
native union, source ranks/scores and deterministic provenance. The new stage
graph has a new immutable identity; old runs cannot be renamed or resumed as
new-layout runs.

D32: Smoke's candidate eligibility cap does not truncate its model input
vocabulary or discard legitimate selected training targets. Complete selected
history metadata is represented separately from the capped candidate catalog.
Global full-mode statistics, article scope, temporal windows and denominators
are unchanged. Existing resource gates apply to both representations.

D33: Final Test is an optional last stage, absent from all default runs. It
requires full profile, two explicit approvals, the completed Validation/reload
lock and matching model/schema hashes. Test labels are prepared only inside
that authorized stage; no refitting or configuration selection is permitted.
These are execution guards, not changes to the evaluation protocol.

See the [migration map](refactor_migration_map.md) and
[verification report](code_architecture_refactor_report.md) for file provenance
and the distinction between synthetic verification and unexecuted cloud work.
