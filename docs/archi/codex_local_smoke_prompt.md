# Codex Task: Resource-Bounded Local Smoke Implementation

## Contents

- [1. Objective and authority](#1-objective-and-authority)
- [2. Scope and prohibited actions](#2-scope-and-prohibited-actions)
- [3. Resource profile and execution guards](#3-resource-profile-and-execution-guards)
- [4. Existing inputs and output isolation](#4-existing-inputs-and-output-isolation)
- [5. Deterministic real-data smoke sample](#5-deterministic-real-data-smoke-sample)
- [6. Temporal and evaluation contracts](#6-temporal-and-evaluation-contracts)
- [7. Minimum end-to-end implementation](#7-minimum-end-to-end-implementation)
- [8. Synthetic and real-data verification](#8-synthetic-and-real-data-verification)
- [9. Entry points and deliverables](#9-entry-points-and-deliverables)
- [10. Completion report and stopping conditions](#10-completion-report-and-stopping-conditions)

## 1. Objective and authority

Implement and execute a first local, resource-bounded vertical slice of this
H&M recommendation project. The goal is working interfaces and verified smoke
tests, not competitive recommendation quality or production readiness.

Target machine: Apple M1 Max with only approximately 7–8 GB currently available
memory. Do not interpret the machine's installed RAM as the available budget.

Read the complete root `AGENTS.md`, applicable nested instructions,
`proposal.md` v0.3.2, and these current documents before editing:

- `docs/archi/architecture.md`
- `docs/archi/evaluation_protocol.md`
- `docs/archi/implementation_decisions.md`

Inspect existing code, configuration, dependencies, and Git changes. Extend
working modules rather than replacing the repository. Preserve unrelated work.
If current documents are missing, report that fact; do not substitute an older
proposal from `docs/achive/` or invent missing decisions.

This task explicitly authorizes one bounded smoke milestone spanning the minimal
P1–P5 interfaces. It does not authorize full-scale completion of those phases,
P6 final evaluation, or a redesign. Follow current architecture and engineering
rules. The resource reductions below are smoke-only configuration overrides,
not changes to production defaults. Record every deviation in the run manifest.

Proceed from inspection to implementation, tests, and a bounded real-data run
without waiting for approval of each ordinary local step. Stop the affected
step for missing permissions, unsafe resource use, or corrupt required inputs.

## 2. Scope and prohibited actions

Implement these six real retrieval paths and only one learned ranker:

| Channel | Maximum candidates per query |
|---|---:|
| 7-Day Trend Popularity | 60 |
| Repurchase | 30 |
| Product_Code Sibling Recall | 30 |
| Time-Decayed ItemCF | 50 |
| Time-Decayed WALS | 50 |
| Multi-modal Two-Tower | 50 |

Merge and deduplicate the complete native union, at most 270 candidates per
query. Preserve all channel provenance. RRF is a feature, not a pre-ranker.
Use LightGBM LambdaMART as the only learned ranker and return up to 12 unique
articles. A channel can legitimately return fewer than its quota or no items.

Do not:

- Train any model on the complete transaction dataset, construct full-scale
  training matrices, run hyperparameter searches, or launch the full pipeline.
- Regenerate text/image embeddings, download encoders, decode original images,
  replace source embeddings, or add standalone text/image retrieval.
- Load a full transaction CSV into pandas, materialize dense user-item or
  item-item matrices, or build an unbounded candidate cross product.
- Replace unimplemented channels with random scores, popularity aliases,
  constant embeddings, or silent no-op adapters and claim completion.
- Relax active-pool rules, change labels, inject positives, or alter split dates
  merely to obtain nonzero metrics on a small sample.
- Run actual final-Test evaluation, tune against Test, or claim final results.
- Start Docker, cloud jobs, HTTP services, remote MLflow/DVC infrastructure, or
  external uploads. Local manifests are sufficient for this smoke milestone.
- Rewrite architecture documents or archived notebooks to match shortcuts.

Use reusable `src/hm_recsys/` modules and thin CLI entry points. A new notebook
is not required. Any notebook added must follow `AGENTS.md` and must call shared
modules rather than contain a second pipeline.

## 3. Resource profile and execution guards

Create an explicit `smoke` profile, such as `configs/smoke.yaml`. Keep these
limits separate from default/full-scale settings:

| Setting | Initial smoke limit |
|---|---|
| Device | CPU only; no CUDA or automatic MPS selection |
| Concurrent compute stages | One; no parallel folds or parallel training |
| Threads | At most 2 per numerical library; PyTorch inter-op 1 |
| DataLoader workers | 0 |
| DuckDB memory limit | 1 GB |
| DuckDB temporary disk limit | 2 GiB, under this smoke run's `tmp/` |
| Process-tree RSS target | At most 4 GiB |
| Abort threshold | Process-tree RSS at or above 5 GiB, or system available memory below 2 GiB |
| Sampled customer cohort | At most 500 |
| Smoke candidate catalog | At most 5,000 articles |
| Materialized sampled transactions | At most 200,000 rows |
| Query cap | At most 50 queries per ranker fold and 50 global-Validation queries |
| Candidate-feature rows | At most 100,000 across the run |
| WALS | 16 factors, 2 iterations, at most 2 threads |
| Two-Tower | Batch size 32; at most 5,000 training targets; 1 epoch and at most 50 optimizer steps, whichever stops first |
| Two-Tower representation | Keep the current 128d output and model branches; read actual content dimensions |
| Visual clustering | Smoke-only K=8, at most 1,000 legal pre-base image vectors, n_init=1, max_iter=10 |
| LambdaMART | At most 30 boosting rounds, num_leaves=7, max_depth=3, min_child_samples=5, n_jobs=2 |
| Seed | 42, including stable query-local sampling |

These are conservative starting limits, not a measured memory guarantee.
DuckDB's memory setting is not a cap on the whole Python process.

Implement a lightweight parent/child-stage runner or equivalent supervisor.
Check memory before stages, monitor the worker process tree during execution,
record sampled peak RSS and sampling frequency, and stop the owned stage when a
threshold is crossed. Never kill unrelated user processes. Use actual available
memory, not total RAM. Explain that monitoring is not an OS-enforced hard limit.

Set thread limits before numerical libraries initialize. Disable unnecessary
multiprocessing. Use sequential stage execution and release large objects or
exit stage processes between ETL, training, and inference. Do not run the
integration pipeline repeatedly as part of every unit test.

Validate size caps before allocating arrays, generating pairs/candidates, and
calling `.fit()`. Every training entry point must require validated sampled-input
manifests and reject raw full-data training in smoke mode. Do not fall back to
full-scale defaults if a smoke setting is missing. Resource exhaustion must
produce an honest non-success result, not an automatic larger run.

Use the existing isolated project environment when possible. Do not modify
system Python, run privileged installs, or change machine settings. Check
Apple Silicon dependency availability. Report installation/environment blockers
rather than substituting algorithms. Ordinary project-local dependency setup
must respect the environment's permission and network policies.

## 4. Existing inputs and output isolation

Read these supplied inputs in place, without renaming or overwriting them:

```text
data/
├── original data/
│   ├── articles.csv
│   ├── customers.csv
│   └── transactions_train.csv
├── image_features/
│   ├── image_article_ids.parquet
│   ├── image_embedding_failures.parquet
│   ├── image_embeddings.npy
│   ├── image_faiss.index
│   └── image_features_manifest.json
└── text_features/
    ├── text_article_ids.parquet
    ├── text_embeddings.npy
    ├── text_faiss.index
    └── text_features_manifest.json
```

Keep `article_id` as a ten-character string, preserving leading zeros. Use real
`product_code` joins rather than an inferred ID prefix. Paths containing spaces
must work. Resolve the repository root explicitly; do not use personal absolute
paths or `/content/` fallbacks.

Inspect manifests, array headers, and mapping schemas. Open arrays read-only
using memory mapping where supported, for example
`np.load(path, mmap_mode="r", allow_pickle=False)`. Fetch only needed rows in
bounded batches. Never normalize or convert an entire source array into a new
in-memory copy. Text and image have separate article-to-row maps; do not assume
matching row order or sort maps independently of vectors.

Validate every embedding row actually consumed by this run: shape, dtype,
finite values, nonzero usable vectors, and map consistency. Report validation
scope; a subset check is not a full-artifact audit. Inspect the image failure
schema and missing modalities without inventing replacement vectors. Ordinary
missing content uses masks; contradictory required mappings block the consumer.

Do not load supplied raw-content FAISS indexes merely because they exist.
They are not required for embedding lookup or interchangeable with learned
Two-Tower/WALS indexes. Build only tiny derived indexes for the smoke catalog.

Write all new machine artifacts beneath:

```text
data/runs/smoke/<run_id>/
```

This includes sampled Parquet, manifests, reports in JSON, models, learned
indexes, feature tables, prediction files, logs, and scratch space. It is an
explicit smoke-only isolation convention. Never overwrite production artifacts
or mix smoke checkpoints with full runs. Tests use `tmp_path` fixtures.

A concise human-readable report may go to
`docs/archi/local_smoke_report.md`; do not publish raw customer histories or
customer-level prediction tables there. Preserve `docs/achive/` unchanged.

## 5. Deterministic real-data smoke sample

A bounded sequential scan of the original CSV to select/extract a tiny sample
is permitted. Scanning input bytes is not permission to train on all rows.
Use projected, typed DuckDB/Arrow reads with the resource limits above; reuse
small Parquet extracts rather than repeatedly scanning the CSV for each fold.

Construct the sample as follows:

1. Select at most 500 customers deterministically using only history before
   `2020-08-05`. A documented historical eligibility condition, such as purchases
   on at least two distinct dates during the permitted pre-training history,
   is acceptable for smoke viability. Select using a stable seeded hash, not
   Python's process-randomized `hash()` or global random sorting.
2. Preserve the selected customers' purchase lines needed from `2020-02-01`
   through `2020-09-15`. Do not remove legitimate identical purchase lines.
   Keep histories and labels separately accessible. Never select customers
   based on later relevance hits, Validation scores, or Test behavior.
3. Choose the bounded smoke candidate catalog using only pre-`2020-08-05`
   history and static catalog relations, optionally including product-code
   siblings. Cap it at 5,000 with a deterministic documented rule. Do not select
   items from future labels. Preserve full catalog-matched future positives
   for sampled users even when they fall outside this smoke catalog.
4. Apply every model's own fit boundary and vocabulary rules after sampling.
   A shared sample must not give the Two-Tower a vocabulary fitted on WALS's
   later cutoff or allow dynamic features to see the end of the extract.
5. Compute cutoff-specific statistics and active pools from the documented
   sampled interaction universe. Keep the three-lines-in-28-days threshold.
   Label these as sample statistics, never global H&M popularity or coverage.
6. Build bounded query manifests with a stable rule. Apply purchase-active
   eligibility as defined in the protocol. Filter no-pair groups for ranker
   training only; do not choose Validation queries based on retrieval hits.

If the sample exceeds a cap, stop or deterministically reduce complete sampled
customers before training. Do not take an arbitrary first-N transaction prefix,
slice away tail dates, silently lower thresholds, or expand to the full dataset.

If real data yield no valid ranker pairs, an empty required training partition,
or insufficient valid clustering rows, report the component as blocked. A
successful synthetic fixture does not turn that into a real-data pass. Do not
keep resampling seeds until a favorable metric appears.

Persist source identity, selected IDs locally, date coverage, row counts,
selection rule, seed, exclusions, and `sampling_scope="smoke"`. Reuse existing
source checksums when available and record their verification status. A
size/mtime fingerprint is not a verified cryptographic checksum. Do not require
expensive full-source checksum recomputation just to run this smoke test.

## 6. Temporal and evaluation contracts

`T` is the last observable date. `a = T + 1 day`. Histories satisfy `t_dat < a`,
trailing d-day windows are `[a-d, a)`, and labels are `[a, a+7)`.

Preserve these exact, non-overlapping target intervals:

| Partition | Start inclusive | End exclusive |
|---|---|---|
| Ranker F1 | 2020-08-12 | 2020-08-19 |
| Ranker F2 | 2020-08-19 | 2020-08-26 |
| Ranker F3 | 2020-08-26 | 2020-09-02 |
| Ranker F4 | 2020-09-02 | 2020-09-09 |
| Global Validation | 2020-09-09 | 2020-09-16 |
| Final Test, validation of boundaries only | 2020-09-16 | 2020-09-23 |

Validate exact dates, seven-day lengths, disjointness, and cutoff/start alignment
before candidate generation. Shared past history is allowed; overlapping label
weeks are not. Include a test that rejects the old overlapping-anchor schedule.

WALS training uses `[2020-02-01,2020-08-12)` and its vocabulary interval from
the current proposal. Two-Tower training targets use
`[2020-05-19,2020-08-05)`, with internal validation
`[2020-08-05,2020-08-12)`. Event history must precede the target date, including
no same-day target leakage. Frozen base parameters stay frozen at later cutoffs.

Execute the four tiny ranker folds and sampled global Validation when inputs
permit. Keep actual final Test disabled; only verify its date contracts with
synthetic data. Real Validation metrics are diagnostics for plumbing, not a
model-selection campaign. Do not tune sample construction or hyperparameters
until the smoke metric improves.

Ground truth remains all catalog-matched unique future purchases for each
selected query, including items outside the active pool, sampled candidate
catalog, WALS vocabulary, embedding coverage, or native candidate union.
Unretrieved positives remain misses. Never narrow the denominator to make a
sampled run look successful. Keep zero-hit and zero-candidate purchase-active
Validation queries in the metric denominator.

## 7. Minimum end-to-end implementation

Use shared code paths for smoke and future larger runs. Resource profiles can
change sizes, not secretly select a different recommendation algorithm.

1. **Input and feature acceptance:** Implement read-only CSV/embedding adapters,
   separate row maps, masks, bounded canonical sample generation, split and
   query manifests, and preflight resource checks.
2. **R1–R4:** Implement the actual current formulas for age/global trend,
   time-decayed repurchase, product-code siblings, and directional time-decayed
   ItemCF. Respect date windows and the common active pool. Generate ItemCF
   pairs in bounded batches and keep sparse top neighbors. Recompute dynamic
   channels at each cutoff; do not silently freeze them at the base date.
3. **WALS:** Fit a real small implicit ALS model. Apply the confidence transform
   once with library alpha=1. Use a genuine fixed-item regularized user solve
   for folding-in, not normalized vector pooling. Keep item factors fixed and
   disable already-liked filtering. Unknown items do not acquire factors.
4. **Two-Tower:** Use both existing content modalities, structured features,
   history/profile branches, trainable projections, training-time shared ID
   dropout, and LogQ-corrected InfoNCE. Implement and test duplicate/multi-positive
   handling, padding, and train-only Q estimation. Preserve 128d output; cap
   examples/steps instead of copying the production batch size of 2048. Show
   finite loss, at least one real parameter update, and inference with dropout
   disabled. A network that was initialized but never trained is not a pass.
5. **Learned retrieval:** Build small learned-vector `IndexFlatIP` indexes with
   explicit ID maps. If FAISS installation is the only blocker, a documented
   smoke-only bounded exact NumPy inner-product backend is permitted, with
   parity tests. Report FAISS verification as not run, not passed. Do not
   substitute the supplied content indexes or change dot products to cosine
   for WALS.
6. **Shared features:** Implement the current named feature schema, including
   content cross-features, frozen visual clusters, missing masks, and all
   channel provenance. Fit smoke clustering on legal sampled pre-base items
   only. Read F from the schema, not a literal 38/49. Do not fake expensive
   features as constants or silently remove columns only at inference.
7. **LambdaMART:** Generate native candidates before consulting labels. Use
   binary unique-article labels, contiguous query groups, the specified
   Train-only 1:20 negative cap and hard/random policy, and full candidates for
   selected Validation queries. Verify a genuine trained model with finite
   predictions and nontrivial splits when the fixture supplies learning signal.
8. **Inference and reload:** Persist immutable model/maps/schema/config/content
   references. In a fresh process, load them and generate native Top-12 using
   the same FeatureBuilder. Verify IDs, uniqueness, deterministic ties,
   score/feature parity within documented tolerance, and short-list behavior.
   Inference must not require labels or refit any models.

No-history users use the specified rule fallback. Missing content and missing
WALS vocabulary are normal cases, not reasons to delete ground truth. Report
empty channel outputs without disguising them as successful personalized recall.

## 8. Synthetic and real-data verification

Maintain separate statuses for:

- `synthetic_contract_tests`
- `sampled_real_end_to_end`
- `fresh_process_reload_parity`
- `quality_checks`

Use a small deterministic synthetic fixture to exercise all six channels,
positive and negative pairs, sibling overlaps, future-only items, missing
content, unknown users, zero-hit queries, and empty candidates. Synthetic
vectors and planted relevance are allowed only in tests and must never appear
as real H&M results. The fixture should be large enough for its configured
cluster count and valid in-batch contrasts.

Required tests include:

- Leading-zero IDs, paths with spaces, independent shuffled text/image maps,
  dimension conflicts, missing modalities, and read-only source behavior.
- Exact date windows, no same-day event leakage, disjoint target weeks, and
  failure on an overlapping split configuration.
- Future-label perturbations cannot change that query's candidates or features;
  base artifacts do not change from post-base transactions. Later cutoffs may
  use newly elapsed history, as specified.
- Correct active-pool filtering, sibling seed exclusion, source masks, native
  union deduplication, quotas, and RRF over actual hits only.
- WALS confidence applied once and folding-in checked against a tiny reference
  solve; Two-Tower false-negative masks, Q provenance, and real updates.
- Train-only negative sampling, no-pair Train filtering, contiguous groups,
  `sum(group_sizes) == len(X)`, and no label/ID/oracle feature leakage.
- Hand-computed MAP@12/Recall/NDCG cases, duplicate predictions, short lists,
  missed positives outside all candidate pools, and zero-candidate denominators.
- Same-code-path features before/after reload; no training during inference.
- Resource guards rejecting oversized manifests before fit; original inputs
  unchanged; all generated runtime artifacts isolated from normal runs.

Run Ruff formatting/lint checks, Pyright, and pytest according to `AGENTS.md`.
Default tests must use fixtures, not start real-data training. Make the real-data
smoke test an explicit opt-in command/marker. Do not broadly suppress lint/type
errors or mark failed tests as skipped just to obtain a green result.

## 9. Entry points and deliverables

Implement or adapt one reproducible CLI workflow with these modes:

```text
preflight: inspect environment, inputs, configuration, and budgets; no training
synthetic: run the deterministic integration fixture
real-smoke: bounded real-data extraction, training, Validation diagnostics
reload: fresh-process inference from an explicit smoke run ID
```

The concrete command names can follow existing repository conventions. Document
and execute the actual commands you implement; do not report example commands
as if they already existed. The default mode must not run full training or Test.

Deliver source modules, a smoke configuration, appropriate tests, dependency
changes only as needed, and a concise report. Store these run artifacts where
applicable:

```text
data/runs/smoke/<run_id>/
├── config.resolved.yaml
├── run_manifest.json
├── input_acceptance.json
├── sample_manifest.json
├── split_manifest.json
├── feature_schema.json
├── smoke_results.json
├── resource_usage.json
├── commands.log
├── sample/
├── models/
├── indexes/
├── predictions/
└── tmp/
```

Persist checkpoint/map fingerprints, sampled-source identities, fit cutoffs,
package versions, seeds, actual step counts, row/query/candidate counts, missing
content counts, and exclusion reasons. Mark `is_smoke=true` and
`is_benchmark=false`. Do not register a smoke model as a production/champion
release. Do not Git-add private data, vectors, models, or customer-level outputs.

## 10. Completion report and stopping conditions

The final response must state:

1. Implemented files and the meaningful changes.
2. Exact executed commands and exit statuses.
3. Separate synthetic, sampled-real, reload, and quality-check results.
4. Actual sampled sizes, trained iterations/steps, and per-channel output counts.
5. Measured memory evidence and limitations, including any guard termination.
6. Locations of manifests, models, predictions, and diagnostic metrics.
7. Any deviation from the current architecture, dependency blocker, or work not
   executed; distinguish proposed commands from verified commands.

A successful real end-to-end smoke requires actual sampled H&M input, real small
WALS/Two-Tower/LambdaMART fits, all six implemented retrieval paths, saved
artifacts, and verified fresh-process inference. Sparse channels need not all
return results for every real user; synthetic tests must exercise their logic.

Do not require a good MAP@12 or improvement over a baseline to pass a plumbing
smoke. Conversely, a positive metric does not excuse leakage, broken grouping,
placeholder models, or missing components.

If input files or dependencies are unavailable, or the sample cannot support a
required fit, finish all safe code and synthetic work and explicitly report the
real run as blocked/not run. Never fabricate success. Do not increase budgets,
change core semantics, launch full training, or consume the final Test merely
to complete the task.

Stop when the bounded smoke milestone has been verified and reported. Do not
continue into full training, benchmarking, cloud work, or architectural expansion.
