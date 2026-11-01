# H&M Cascade Architecture

**Version:** v0.3.2  
**Status:** Current implementation contract; no new pipeline execution is claimed.

## Contents

- [1. Authority and data flow](#1-authority-and-data-flow)
- [2. Module contracts](#2-module-contracts)
- [3. Storage and artifact ownership](#3-storage-and-artifact-ownership)
- [4. Temporal boundaries](#4-temporal-boundaries)
- [5. Execution boundaries and failure handling](#5-execution-boundaries-and-failure-handling)

## 1. Authority and data flow

[proposal.md](../../proposal.md) defines the new architecture;
[evaluation_protocol.md](evaluation_protocol.md) defines the exact dates and
metric denominators. The latest user decisions take precedence over historical
reports and `docs/achive/` notebooks. Clarifications are recorded in
[implementation_decisions.md](implementation_decisions.md).

Prepared text/image artifacts are **inputs**, not a pending extraction job.
The six channels are Trend, Repurchase, Sibling, ItemCF, WALS, and Two-Tower.
The existing raw-content FAISS files are not additional retrieval channels.

The following dependency table preserves the architecture without implying
that a source input or output has already been validated or produced:

| From | To | Allowed dependency |
|---|---|---|
| `data/original data/` CSVs | Canonical tables | Typed, non-destructive source conversion |
| Supplied text/image arrays, maps, manifests | Content loader | Validate mappings, dimensions, available provenance, and masks |
| Pre-base canonical history | WALS and Two-Tower fitting | Obey separate base/internal-validation boundaries |
| Verified content loader | Two-Tower and frozen visual clustering | Consume prepared vectors; train permitted downstream components only |
| Past-only cutoff snapshot | Active pool and R1–R4 | Recompute dynamic statistics and ItemCF |
| Fixed WALS factors plus current history | R5 | Fold in the user; keep item factors fixed |
| Fixed Two-Tower plus current history/content | R6 | Encode user and search learned item vectors |
| R1–R6 | Native candidate union | Deduplicate articles; preserve source scores/ranks/masks |
| Union plus past-only features/content | Shared FeatureBuilder | One schema for training and inference |
| Training features plus separate future labels | LambdaMART training | Train-only group filtering and negative sampling |
| Native inference features plus trained ranker | Stable Top-12 | Score and order within query |
| Top-12 plus full independent ground truth | Evaluation | Native macro MAP@12 and supporting metrics |

Future labels have no dependency into retrieval or feature generation. Original
content indexes have no default dependency into candidate generation; any
optional diagnostic use is separate and must not change the six-channel plan.

## 2. Module contracts

| Module | Build/training input | Persisted output | Inference contract |
|---|---|---|---|
| Content loader | Existing NPY/Parquet/JSON files; image failure log | Acceptance/lineage report and optional derived adapter cache | Article ID to verified vector and availability, separately per modality |
| Trend | Seven-day purchases plus age buckets | Global/bucket ranks per cutoff | User bucket/pool to at most 60 articles |
| Repurchase | Sixty-day user purchases | Decay/count/last-date aggregates | User/pool to at most 30 prior articles |
| Sibling | Actual product-code mapping, 30-day seeds, seven-day sales | Mappings and cutoff statistics | User seeds/pool to at most 30 variant articles |
| ItemCF | 28-day distinct edges and last dates | Sparse top-100 neighbors per seed/cutoff | Up to 15 recent seeds to at most 50 articles |
| WALS | Base-window confidence CSR | 64d factors and fixed maps | Current representable history to folded-in factor and at most 50 articles |
| Two-Tower | Past-only targets/history, metadata, existing frozen content | Tower weights, maps, learned 128d item vectors and independent index | Current profile/history to query vector and at most 50 articles |
| FeatureBuilder | Union, cutoff snapshots, content, frozen models | Schema and cutoff feature tables | Same columns and semantics for Train/Validation/Test/serving |
| LambdaMART | `X[N,F]`, `y[N]`, contiguous query sizes | Model, best iteration, schema and lineage | `X[N,F]` to scores; outer query grouping to Top-12 |

Every channel returns query/customer/cutoff/article identity, channel, raw score,
source rank, and retriever version. Candidate identity is
`(customer_id, cutoff_date, article_id)`. The initial union bound is 270.

## 3. Storage and artifact ownership

Read source CSVs from `data/original data/`. Read existing arrays/maps/manifests
from `data/text_features/` and `data/image_features/` without relocation or
mutation. The source manifests may have different actual schemas; inspect them
rather than imposing imaginary encoder metadata.

Use `data/canonical/` for derived tables, `data/processed/` for cutoff-specific
pools/candidates/features/labels, `data/models/` for fitted models, and
`data/indexes/` for new ItemCF and learned-factor search indexes. Store input
acceptance, split, mapping, and schema reports in `data/manifests/`, metrics in
`data/metrics/`, and local experiment state in `data/runs/`.

The prepared text/image indexes remain original content artifacts. A WALS
factor index or Two-Tower index must have a separate path and identity, even
when dimensions coincide. No loader may replace `image_faiss.index` or
`text_faiss.index` with a learned model's index.

Keep the initial reference layout unchanged: `docs/achive/` and the two reports
at the `docs/` root. Add current design documents under `docs/archi/`. Do not
rename archives, migrate references, run archived notebooks by default, or write
new runtime arrays/models into documentation directories.

## 4. Temporal boundaries

The base cutoff is inclusive 2020-08-11. Two-Tower gradient targets end on
2020-08-04, and internal validation ends on 2020-08-11. Preserve the exact
WALS training/vocabulary ranges in the proposal.

Ranker training target weeks are Aug 12–18, Aug 19–25, Aug 26–Sep 1, and
Sep 2–8. Global Validation is Sep 9–15; Test is Sep 16–22, all in 2020.
Target windows are strictly disjoint. Validate exact dates and seven-day lengths
before generating candidates. Past-only lookback reuse is permitted; overlapping
target windows are not.

Weights remain frozen while permitted user histories, active pools, statistics,
ItemCF, and WALS user folding-in advance with the cutoff. Do not let historical
notebook defaults override this policy.

## 5. Execution boundaries and failure handling

Logical channels need not be separate services. Reusable implementation belongs
under `src/hm_recsys/`; new notebooks are exploratory clients. A later batch or
HTTP endpoint must call the same candidate and feature contracts.

Missing user history/factors may yield an empty channel; R1 provides the defined
fallback within its quota and pool. A missing modality yields documented masks,
not removal of ground truth. A blocking artifact mapping/dimension/integrity
mismatch stops the affected consumer rather than triggering regeneration.

Do not normalize zero vectors or mix row mappings because arrays have equal
lengths. An optional source-index mismatch need not block direct vector lookup;
a model's required learned-index mismatch must block its retrieval path.

Return fewer than 12 unique articles when the permitted pipeline cannot supply
12; missing positions earn no credit. Preserve zero-candidate queries in the
independent evaluation manifest. Source preparation, successful loading, model
training, and validated recommendation quality are distinct statuses.
