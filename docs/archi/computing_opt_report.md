# Audit-Driven Computing Optimization Report

Date: 2026-11-03. Scope: the current working checkout, with preceding uncommitted
full/compute work and the user's Notebook RUN_ID fix preserved. No old ZIP was
restored. Inputs, existing jobs/checkpoints and earlier runs were not modified.

## Contents

- [Summary and evidence boundary](#summary-and-evidence-boundary)
- [Audit-to-code action matrix](#audit-to-code-action-matrix)
- [Execution changes and equivalence](#execution-changes-and-equivalence)
- [Measured before and after](#measured-before-and-after)
- [Per-file change reference](#per-file-change-reference)
- [Statement comments and README coverage](#statement-comments-and-readme-coverage)
- [Verification commands and results](#verification-commands-and-results)
- [Release, reproduction and remaining work](#release-reproduction-and-remaining-work)

## Summary and evidence boundary

This task adds actual execution changes on top of the prior compute adapter:
partition-wide label-free Two-Tower query inputs, batched fixed-item WALS solves,
cutoff-scoped active row/name caching with thresholded exact Top-K, column-oriented
49-feature construction, incremental larger Parquet files, direct sampled numeric
ranker scratch, seed-clustered exact ItemCF SQL with a double-precision decay
lookup, and coordinator-scoped checksum receipts. Existing reference paths remain
callable for bounded comparisons. The README now describes the actual maintained
code rather than the earlier tooling-only repository.

The six quotas, native union, time windows, full truth, source masks, single
LambdaMART, full hyperparameters, batch 512/FP32, multi-positive LogQ, Train-only
1:20 sampling and complete per-epoch internal selection are unchanged. No AMP,
TF32, compile, ANN substitution, extra ranker, positive injection, Test evaluation
or data reduction was introduced. Only inference query batching changes from 32
to at most 512 on the optimized partition encoding path; training boundaries and
in-batch negatives do not change. Float32 inference/feature operations may differ
within the declared tolerance; candidate identities/order and Top-12 are checked.

All new executions in this report are bounded synthetic CPU checks on the M1 Max.
The audit's cloud attribution percentages, assumed hardware and 20–30x estimates
are not measurements reproduced here. Actual Colab/CUDA/full performance and
real Drive recovery remain **not run** for this release.

## Audit-to-code action matrix

The authority is [the supplied audit](../achive/single_source_refactor_20261103_01/historical_docs/docs/computing_performance_architecture_audit_report.md).
References below use its section numbers and actual source symbols rather than
stale audit line numbers.

| Audit finding / classification | Current-code check and action | Status / verification boundary |
|---|---|---|
| §3.1: 32-user repeated scans; compute/I/O | History.histories retains the safe 100,000-line/32-user guard but adds requested-user min/max predicates for Parquet row-group pruning. Partition QueryInputs removes repeated encoding window SQL entirely. | Implemented; bounds and exact input parity tested. No claim of full scans per SQL call without a plan. Larger all-history batches deferred pending actual cloud memory. |
| §3.1: candidate dictionaries/cluster/affinity repetition | FeatureBuilder._build_columns gathers cutoff item tables and batch content products; the prior frozen cluster, popularity and Repurchase/Sibling caches remain. | Implemented; reference rows and all 49 columns compared. |
| §3.1 / §5.2: per-user WALS solves | fold_users_batch preserves float64 equations and empty-user coordinates; learned_batch uses one stacked solve. | Implemented; exact fixture factors and retrieval ordering checked. |
| §3.1 / §5.2: active filtering and repeated full lexsort | ActiveBlocks stores physical-block eligibility/names once per cutoff/model object and uses score-threshold partitioning, including every tie before ID lexsort. | Implemented; >5,000-item, boundary-tie, zero-query/empty-pool paths tested. No GEMM regrouping or ANN. |
| §3.2 / §5.3: encode/reload_encode still per-user SQL/tensor concatenation | QueryInputs exports one partition-wide history window and bounded profile rows, stores typed mmaps and gathers 512-query input slices. Reload verifies and reuses raw inputs. | Implemented; complete roster, history/padding/weight parity and corrupt cache rejection tested. |
| §3.2: per-step scalar synchronization | The finite loss read before backward remains. Removing it would allow a nonfinite update before detection or change failure/checkpoint semantics. | Deferred; explicit correctness protection. CPU time is measured; no unverified CUDA optimization claim. |
| §3.2: CPU internal validation | cached_internal_recall uses cached active blocks/threshold Top-K but recomputes learned catalog/query vectors every epoch. | Implemented CPU optimization. Moving exact search to GPU deferred without CUDA tie/numerical evidence. |
| §3.2: checkpoint every 10 steps | Existing optimized Cadence already separates 100-step/60-second local saves from 300-second publication, with forced safe boundaries. Full snapshots and failure retention remain. | Already implemented in current baseline; regression retained, not credited as new work. |
| §3.2: Python context prepack | Training Prepared was already bounded and persistent. New partition query preparation uses columnar history slots/gathers and a raw item-code bank. The context merge remains for exact training event histories. | Partially optimized; full training prepack rewrite deferred without a safe large-data benchmark. No learned-vector caching. |
| §3.3 / §5.1: repeated ItemCF seed joins / exponentials | Exact SQL uses a seed-clustered/indexed edge layout, the same double-precision SQL exp lookup for 28 integer day gaps and an indexed persisted neighbor table. | Implemented safe alternative; exact fixture neighbor IDs/order and near-exact scores checked. The supplied float32 CSR/dense block blueprint is rejected for default parity/memory safety. |
| §3.4: repeated dependency SHA256 | DigestCache retains cryptographic receipts only within one coordinator and invalidates on dev/inode/mode/size/mtime/ctime changes. A new process fully rehashes. Copies and completion publication still verify bytes. | Implemented; same-size/restored-mtime corruption, fresh-cache misses, stage mutation/failure/recovery tests. No persistent path-only trust. |
| §3.4: small feature/provenance files | FeatureShards appends 32-query compute batches to at most 512-query files with bounded whole-group metadata; both provenance and feature row groups remain inspectable. | Implemented; zero-candidate groups and partial final files retained. No all-candidate Python materialization. |
| §3.4: triple ranker materialization | Optimized Train sampling appends float32 X/int32 y directly to two numeric scratch files and maps them after the independent fit estimate. | Implemented; sampling/group/label and actual single fit/reload tested. LightGBM remains memory-bound, not arbitrary out-of-core training. |
| §4.1: split compliance | Current F3 cutoff is Aug 25. The audit table incorrectly says Sep 1 for F3; executable split_manifest and protocol are authoritative. | Verified; no temporal changes. Exact disjoint windows/future perturbations remain tested. |
| §4.2: pilot WALS vocabulary | Global legal vocabulary and cohort-represented trained columns are already separately reported. Padding unsupported columns with random/zero pseudo-factors would change candidates/objective. | No silent algorithm change. Preserve the existing documented pilot limitation; full uses all legal base users. |
| §4.3: repeated average divisions | Candidate diagnostics accumulate per-query recall numerators and divide once by the complete partition count. Native ranker MAP/NDCG definitions remain unchanged. | Implemented; zero-hit queries retained; small summation-order drift is allowed within tolerance. |
| §4.4: threads/memory caps | Conservative two-thread/1 GB/8 GiB settings are confirmed. Automatically copying assumed A100 12-thread/32 GB settings conflicts with local constraints and lacks measured cloud evidence. | Deferred as an explicit resource experiment. No training configuration was silently widened. |
| §5.4: vector feature blueprint | Implemented by schema-name column mapping rather than hard-coded 49 offsets; prices remain float64 until final cast and NaNs/masks remain explicit. | Implemented with continuous/discrete parity checks. |
| §6: predicted 10–18-minute full run / §7 limits | No new full/cloud run is authorized locally and no actual cloud trace was supplied. | Unverified estimates; never used as measured speedup or acceptance evidence. |

## Execution changes and equivalence

### Two-Tower input and device flow

Training already used Prepared contexts and ItemBank. These remain model-independent:
category IDs, profile IDs, latest-20 raw-line history IDs/weights, ragged complete
positive IDs, stable SHA256 target order and separate internal daily rosters.
No learned item vectors, projection outputs or pooled user vectors are cached
across optimizer steps. Raw-bank content dimensions are read from real adapter
artifacts; category zero does not replace an article's separate content row.
GPU placement is budgeted and otherwise uses bounded pinned host lookup while
the model remains on requested CUDA. No real CUDA timing was collected locally.

The new QueryInputs cache is keyed by run/source/code identity, frozen maps,
roster manifest hash, before-date and exact raw20 ordering. It publishes a manifest
last and verifies every declared file on reuse. Profiles/histories/weights use
int64/int64/float32 with shapes [Q,3], [Q,20,3], [Q,20]. Arrow reads and gathers
are bounded; full histories/transactions are never converted to one Python list.
Current and reloaded query vectors are always recomputed using the saved model.

### Retrieval, features and I/O

Batched WALS retains the exact reference matrix products and ridge ordering.
ActiveBlocks retains reference physical 4,096-item blocks, active rows and
float32 GEMM shapes, avoiding the numerical regrouping in the audit's single
full-active GEMM. Threshold partitioning includes all score ties; final sorting
still uses descending score then ascending article ID.

FeatureBuilder's cutoff table contains only static/past-derived item fields and
is replaced whenever snapshot identity/cutoff changes. User latest-five content,
30/90-day statistics and affinities use their original windows. Source masks
start at zero, absent scores/ranks remain NaN, and final float32 column order is
loaded from the schema. Full labels never enter these APIs.

ItemCF retains distinct user-item last dates, user/item degrees, asymmetric
normalization and all legal pairs; high-degree users are not truncated. The
28-value decay table is calculated by DuckDB in double precision, avoiding the
float32 accumulation in the proposed dense CSR block. A seed-sorted temporary
edge copy costs additional disk; exact pair/disk rejection remains in force.

FeatureShards streams writes without buffering all 512-query feature tables.
A ranker read remains bounded by that file's whole-query groups. Partition.truth
chunks up to 512 requested users into guarded 32-query reads, keeping full truth.
Two direct numeric scratch streams avoid sampled-Parquet encoding/decoding, but
single-fit memory estimation still counts LightGBM copies, bins and histograms.

Checksum receipts are not a security claim against an actor able to forge all
kernel metadata. Ordinary writes, replacements, chmod and same-size modifications
invalidate them; atime alone does not. Fresh status/resume processes rehash all
files, and remote copy/destination verification and atomic completion are intact.
There is no shared global checksum cache or forced reuse of an old experiment.

## Measured before and after

### Scope and tower components

The final diagnostic is `data/runs/benchmark/computing_refactor_after_20261103_01/`;
its immediate before run is `computing_refactor_before_20261103_01`. Both use
Python 3.12.16, PyTorch 2.15.0, NumPy 2.5.4, DuckDB 1.5.6, CPU FP32 and two native
threads. Each end-to-end variant consumes exactly 1,584 events in batches
512/512/512/48, four actual updates, 72 complete internal daily queries and 170
catalog items. Warmup uses one disposable update, then model/optimizer/RNG reset.
Both variants perform two full local saves and two verified publications to a
local simulation; no Drive latency is measured. Four batches are a small
correctness diagnosis, not a stable full-epoch throughput estimate.

The prior optimized code already removed most training-input SQL. This table
therefore separates incremental changes from the older SQL reference:

| End-to-end scope | Before optimized | New optimized | Current SQL reference |
|---|---:|---:|---:|
| Training loop including periodic checkpoint (s) | 0.03040 | 0.03131 | 0.15732 |
| Training + full internal validation + final publication (s) | 0.05619 | 0.05388 | 0.23111 |
| Cold cache + verified reopen + bank + total (s) | 0.20262 | 0.18992 | 0.23111 |
| Training-loop examples/s | 52108.97588 | 50589.18189 | 10068.71920 |
| Host step p50 (ms) | 5.07600 | 5.16317 | 44.39742 |
| Host step p95 (ms) | 8.04089 | 8.44766 | 45.23230 |
| Sampled process RSS (MiB) | 422.90625 | 418.92188 | 416.48438 |

New cache build took 0.09103 s; verified reopen 0.03829 s; bank setup 0.00673 s.
Common canonical/input preparation was 0.19095 s. These are reported separately;
replay's prepared working set is diagnostic, not coverage of a full training
population. No substantial incremental tower-training throughput gain is
established: old/new optimized loop variation is in the millisecond range.
The SQL reference comparison includes previous-release cache improvements and
must not be credited entirely to this task.

| Final component (host milliseconds) | SQL reference | New optimized |
|---|---:|---:|
| Target read / prepared wait | 2.048 | 0.982 |
| History SQL including internal inputs | 56.772 | prepared (included in wait/build) |
| Positive-set SQL | 11.928 | prepared (included in wait/build) |
| Positive mask | 18.432 | prepared (included in wait/build) |
| Profile lookup including internal inputs | 6.927 | prepared (included in wait/build) |
| CPU user assembly including training SQL | 92.831 | prepared (included in wait/build) |
| Static item assembly | 0.866 | prepared (included in wait/build) |
| User forward | 4.223 | 3.491 |
| Item forward | 1.173 | 1.016 |
| Loss | 1.743 | 1.547 |
| Backward | 8.339 | 7.625 |
| Optimizer/scheduler | 3.650 | 3.609 |
| Finite check/scalar read | 0.107 | 0.092 |
| Internal catalog + retrieval/metrics | 61.918 | 11.298 |
| Checkpoint serialization/local write | 7.867 | 7.051 |
| Local checkpoint hash | 2.520 | 2.295 |
| Copy with source hash (simulation) | 3.519 | 3.381 |
| Destination verify/file publish (simulation) | 2.719 | 2.944 |
| Remote latest pointer (simulation) | 0.723 | 0.856 |
| H2D / CUDA kernels / allocated/reserved VRAM / GPU utilization | not_run | not_run |

Nested host regions overlap and must not be summed. CPU transfer spans are
no-op/dispatch overhead, not H2D evidence. CUDA events/profiler support remains
available but no CUDA device is present. The separate profiler window took
0.03372 s, excluded from throughput. The independent tower process-tree monitor
observed 655.02 MiB; its 0.2-second sampling may miss peaks and shared pages can
be double-counted. This is below the unchanged 5 GiB local abort limit.

### Retrieval, features, storage and supervised stages

The standalone exact-search kernel uses 6,001 items, 3,001 active items, 32
queries and 64 dimensions, ten repeats after equivalent warmup. Reference and
optimized return identical ordered items/scores, including ties across physical
blocks. The cache-build cost is 0.886 ms; warm search is 6.653 → 2.658 ms per
32 queries. WALS folding is 1.270 → 1.225 ms for the same 32 users/observations;
that small difference is not a robust speedup claim.

The full-fixture feature comparison (48 F1/Validation queries, 3,015 candidate
rows, all 49 columns) changed from the prior optimized 0.05742 s to new 0.03275 s.
Within the final process, original row reference took 0.16253 s versus new column
path 0.03275 s. This includes building the first cutoff item lookup table but
excludes frozen-cluster cold construction, learned recall and model fitting;
it is a shared rule-retrieval/feature assembly result, not whole-pipeline time.

Local shard I/O compared the same 544 synthetic query rows and 49 feature
columns, alternating order over three fresh-output trials. Median wall time
was 0.03747 → 0.02878 s; feature/provenance files changed from 17+17 to 2+2. Zero-row
groups and the final partial file are checked separately. This is local Parquet
writing/hashing, not FUSE/Drive throughput. Reference/new output bytes were
399132/308251 in each trial. A one-MiB checksum fixture cost 0.000673/0.000654 s for the first
verified pass; 20 verified reuses cost 0.012064 → 0.002847 s. The scoped cache recorded
20 hits, one miss and 1048570 actual hashed bytes; restored-mtime writes invalidate it.

The complete cutoff statistics/ItemCF microbenchmark retains the exact same
pair population and output neighbors over three alternated fresh outputs.
Median time was **0.03546 → 0.04156 s**, a small-fixture regression due to the extra
seed layout/index. No ItemCF speedup is claimed. Its global-scale scan/exp
reduction still needs real Colab evidence; the reference remains available.
Database bytes and exact counts are recorded in `io_kernel_performance.json`.

Whole supervised stage observations include interpreter import, output checks
and polling overhead; 0.2-second monitoring granularity dominates this fixture:

| Stage | Before optimized (s) | New optimized (s) |
|---|---:|---:|
| encode | 1.08447 | 1.30057 |
| stats_F1 | 0.43008 | 0.43566 |
| features_F1 | 0.43483 | 0.43714 |
| features_Validation | 0.43188 | 0.43834 |
| ranker | 1.27575 | 1.28285 |
| reload_encode | 1.07559 | 1.07668 |
| reload_features | 0.43313 | 0.43029 |

The tiny encoding stage is slower cold because it now constructs a durable input
cache. Its benefit is fewer repeated window scans and verified reuse at larger
rosters; it is not demonstrated as a latency win on 24 users. Ranker whole-stage
time is effectively unchanged; direct scratch removes intermediate serialization
work but does not prove a faster LightGBM fit. One actual fit remains. Formal
full-data feasibility and whole-pipeline speedup are not established.

### Numerical and downstream parity

All 1,584 diagnostic input/profile/history/mask/Q values matched exactly. The
three-update loss/gradient/parameter comparison had maximum absolute difference
zero; all four final A/B update loss values match. Learned projections changed,
confirming they were not frozen by input caching. Complete internal Recall@50
was 0.8402777778 on the same 72 queries.

Old optimized versus new optimized full-fixture comparison covered 120 queries
and 5,967 candidate rows across F1–F4/Validation: candidate keys/order and all
feature values were exact on this fixture. The persisted `ranker_groups.jsonl`
and `lambdamart.txt` bytes were identical. Reference and reloaded Top-12 Parquet
values and full Validation metrics were exact, including 24 eligible queries
and one zero-hit query. The full-dispatch fixture made 40 real Two-Tower updates
across ten epochs and processed 15,840 targets. These are synthetic results,
not H&M quality measurements. The feature row reference differs by at most
1.1920929e-7 from vectorized continuous columns; allowed tolerance is 1e-5
relative/1e-6 absolute, with exact discrete/mask/order/Top-12 checks. ItemCF
score comparison uses 1e-12 relative/1e-14 absolute with exact neighbor ordering.

Detailed evidence: `before_after_pipeline_parity.json`, final `feature_parity.json`,
`io_kernel_performance.json`, `kernel_comparison.json`, the two tower summaries,
`parity_report.json` and resource files. The exact one-off measurement programs
are preserved as generated text evidence under `reproduction/`; maintained
reference kernels and their bounded fixture tests remain in the source ZIP.
Final Test labels were never read.

The initial baseline and command provenance are retained as follows. Paths are under `data/runs/benchmark/computing_refactor_20261103_01/`.
The baseline was captured before core edits, including a source snapshot and
`before_files.json`; the prior working changes were preserved in
`initial_worktree.diff`. The before full fixture made the real unchanged 40
Two-Tower updates over ten epochs on 1,584 synthetic targets, four Train folds,
complete 24-query Validation and fresh-process inference. The tower diagnostic
separately compares four identical batches and complete 72-query internal
validation, not a full H&M epoch. No fixed speed multiplier is an assertion.

## Per-file change reference

The table covers every modified/added maintained Python file. Scope names are
searchable actual functions/classes; tests use offline tmp_path fixtures.
Comment-only files preserve their executable AST during annotation, including
constants, SQL, sampling, exception and serialization behavior. This task also
updates README, the current Notebook's code comments and fresh default smoke/benchmark IDs, the release document
allowlist and the architecture's previously broken proposal link.

| File | Functions/classes or scope | Before → change | Correctness / evidence |
|---|---|---|---|
| `scripts/benchmark_compute.py` | `Namespace/module statements` | Docstring-only namespace inspected; no execution statement requires an added comment (unchanged) | Annotation executable-AST equivalence; maintained Ruff/Pyright and existing regression suite |
| `scripts/build_release.py` | `main` | Existing execution retained; adjacent English notes added for semantic statements, data flow and guards | Annotation executable-AST equivalence; maintained Ruff/Pyright and existing regression suite |
| `scripts/local_smoke.py` | `Namespace/module statements` | Docstring-only namespace inspected; no execution statement requires an added comment (unchanged) | Annotation executable-AST equivalence; maintained Ruff/Pyright and existing regression suite |
| `scripts/probe_device.py` | `build_parser, main` | Existing execution retained; adjacent English notes added for semantic statements, data flow and guards | Annotation executable-AST equivalence; maintained Ruff/Pyright and existing regression suite |
| `scripts/run_pipeline.py` | `Namespace/module statements` | Docstring-only namespace inspected; no execution statement requires an added comment (unchanged) | Annotation executable-AST equivalence; maintained Ruff/Pyright and existing regression suite |
| `src/hm_recsys/__init__.py` | `Namespace/module statements` | Docstring-only namespace inspected; no execution statement requires an added comment (unchanged) | Annotation executable-AST equivalence; maintained Ruff/Pyright and existing regression suite |
| `src/hm_recsys/cli.py` | `execute, main` | Existing execution retained; adjacent English notes added for semantic statements, data flow and guards | Annotation executable-AST equivalence; maintained Ruff/Pyright and existing regression suite |
| `src/hm_recsys/colab/__init__.py` | `Namespace/module statements` | Docstring-only namespace inspected; no execution statement requires an added comment (unchanged) | Annotation executable-AST equivalence; maintained Ruff/Pyright and existing regression suite |
| `src/hm_recsys/colab/bootstrap.py` | `sha256_file, safe_member, unpack` | Existing execution retained; adjacent English notes added for semantic statements, data flow and guards | Annotation executable-AST equivalence; maintained Ruff/Pyright and existing regression suite |
| `src/hm_recsys/colab/checkpoints.py` | `save_checkpoint, read_previous, load_checkpoint` | Existing execution retained; adjacent English notes added for semantic statements, data flow and guards | Annotation executable-AST equivalence; maintained Ruff/Pyright and existing regression suite |
| `src/hm_recsys/colab/cli.py` | `main` | Existing execution retained; adjacent English notes added for semantic statements, data flow and guards | Annotation executable-AST equivalence; maintained Ruff/Pyright and existing regression suite |
| `src/hm_recsys/colab/device.py` | `probe` | Existing execution retained; adjacent English notes added for semantic statements, data flow and guards | Annotation executable-AST equivalence; maintained Ruff/Pyright and existing regression suite |
| `src/hm_recsys/colab/files.py` | `DigestCache.digest/verify` | Repeated unchanged SHA reads → process-scoped metadata-invalidated receipts | Restored-mtime corruption, fresh process and copy-failure tests |
| `src/hm_recsys/colab/notebook.py` | `command` | Existing execution retained; adjacent English notes added for semantic statements, data flow and guards | Annotation executable-AST equivalence; maintained Ruff/Pyright and existing regression suite |
| `src/hm_recsys/colab/profiles.py` | `stages_for, load_profile, is_colab_runtime, execution_gate, runtime_identity, resource_plan, plan` | Existing execution retained; adjacent English notes added for semantic statements, data flow and guards | Annotation executable-AST equivalence; maintained Ruff/Pyright and existing regression suite |
| `src/hm_recsys/colab/release.py` | `DOCS/build_release` | Old documentation allowlist → include new audit/task/current report | Clean extraction, source member hashes and Notebook equality |
| `src/hm_recsys/colab/review.py` | `publish_review` | Existing execution retained; adjacent English notes added for semantic statements, data flow and guards | Annotation executable-AST equivalence; maintained Ruff/Pyright and existing regression suite |
| `src/hm_recsys/colab/runner.py` | `code_identity, execution_identity, validate_stage, execute, _execute_profile, _validate_profile_stage` | Existing execution retained; adjacent English notes added for semantic statements, data flow and guards | Annotation executable-AST equivalence; maintained Ruff/Pyright and existing regression suite |
| `src/hm_recsys/colab/state.py` | `StageStore.record/persist/execute` | Repeated dependency hash passes/chmod → scoped receipts and conditional read-only chmod | Worker mutation, checksum failure, restore and persistence failure tests |
| `src/hm_recsys/colab/worker.py` | `main` | Existing execution retained; adjacent English notes added for semantic statements, data flow and guards | Annotation executable-AST equivalence; maintained Ruff/Pyright and existing regression suite |
| `src/hm_recsys/compute/__init__.py` | `Namespace/module statements` | Docstring-only namespace inspected; no execution statement requires an added comment (unchanged) | Annotation executable-AST equivalence; maintained Ruff/Pyright and existing regression suite |
| `src/hm_recsys/compute/benchmark.py` | `reference_batches, optimized_batches, update, new_model, parity, input_view, run_benchmark, main` | Existing execution retained; adjacent English notes added for semantic statements, data flow and guards | Annotation executable-AST equivalence; maintained Ruff/Pyright and existing regression suite |
| `src/hm_recsys/compute/feature_benchmark.py` | `compare` | Existing execution retained; adjacent English notes added for semantic statements, data flow and guards | Annotation executable-AST equivalence; maintained Ruff/Pyright and existing regression suite |
| `src/hm_recsys/compute/prepared.py` | `identity, verify, daily_rows, prepare, Prepared` | Existing execution retained; adjacent English notes added for semantic statements, data flow and guards | Annotation executable-AST equivalence; maintained Ruff/Pyright and existing regression suite |
| `src/hm_recsys/compute/query_inputs.py` | `QueryInputs.__init__/users` | New real optimized execution primitive; one partition history/profile export and hash-verified mmap reuse | Exact profiles/raw history/weights and corrupt-cache rejection |
| `src/hm_recsys/compute/runtime.py` | `Measurements, prefetch, ItemBank` | Existing execution retained; adjacent English notes added for semantic statements, data flow and guards | Annotation executable-AST equivalence; maintained Ruff/Pyright and existing regression suite |
| `src/hm_recsys/compute/settings.py` | `apply, Cadence` | Existing execution retained; adjacent English notes added for semantic statements, data flow and guards | Annotation executable-AST equivalence; maintained Ruff/Pyright and existing regression suite |
| `src/hm_recsys/compute/shards.py` | `FeatureShards.append/finish/close` | New bounded incremental writer preserving candidate audit file alignment | Whole/zero-row groups and partial final shard test |
| `src/hm_recsys/content.py` | `ContentSource, ContentBank` | Existing execution retained; adjacent English notes added for semantic statements, data flow and guards | Annotation executable-AST equivalence; maintained Ruff/Pyright and existing regression suite |
| `src/hm_recsys/contracts.py` | `write_json, read_json, digest, stable_key, article_id, split_manifest, validate_splits, Config, source_identity` | Existing execution retained; adjacent English notes added for semantic statements, data flow and guards | Annotation executable-AST equivalence; maintained Ruff/Pyright and existing regression suite |
| `src/hm_recsys/data.py` | `Transaction, Sample, connection, inspect_csv, _csv_view, extract_sample, query_users, age_bucket` | Existing execution retained; adjacent English notes added for semantic statements, data flow and guards | Annotation executable-AST equivalence; maintained Ruff/Pyright and existing regression suite |
| `src/hm_recsys/evaluation.py` | `metrics, union_recall` | Existing execution retained; adjacent English notes added for semantic statements, data flow and guards | Annotation executable-AST equivalence; maintained Ruff/Pyright and existing regression suite |
| `src/hm_recsys/features.py` | `FeatureBuilder.build/_build_columns/_build_rows` | Per-candidate dictionaries → schema-name column gathers and one cutoff item table | Reference 49-column/Top-12/metric comparison |
| `src/hm_recsys/learned.py` | `fold_user/fold_users_batch` | Per-user LAPACK calls → identical stacked float64 normal equations | Exact factors, empty histories, unchanged WALS confidence tests |
| `src/hm_recsys/pilot/__init__.py` | `Namespace/module statements` | Docstring-only namespace inspected; no execution statement requires an added comment (unchanged) | Annotation executable-AST equivalence; maintained Ruff/Pyright and existing regression suite |
| `src/hm_recsys/pilot/capacity.py` | `check, feature_capacity` | Existing execution retained; adjacent English notes added for semantic statements, data flow and guards | Annotation executable-AST equivalence; maintained Ruff/Pyright and existing regression suite |
| `src/hm_recsys/pilot/data.py` | `History.histories` | Join-only request filter → explicit requested-user min/max pruning with unchanged joins | Full history contents/size and future perturbation checks |
| `src/hm_recsys/pilot/features.py` | `learned_batch/generate` | Repeated active setup/single solves/tiny files → cached blocks, batch solve, streaming shard writer, numerator aggregation | Native channel order, feature/schema, Train sampling and full/reload tests |
| `src/hm_recsys/pilot/models.py` | `fit_wals, preprocessing` | Existing execution retained; adjacent English notes added for semantic statements, data flow and guards | Annotation executable-AST equivalence; maintained Ruff/Pyright and existing regression suite |
| `src/hm_recsys/pilot/ranking.py` | `train` | Sampled Parquet round trip → direct float32/int32 scratch streams after same sampling | One actual LambdaMART fit, native group/label counts and reload |
| `src/hm_recsys/pilot/snapshot.py` | `build_snapshot/GlobalSnapshot` | Unclustered seed filtering/repeated exp → indexed seed layout, double decay LUT, indexed neighbors | Exact ItemCF ordered neighbor fixture and time-cutoff tests |
| `src/hm_recsys/pilot/streaming.py` | `Partition.truth` | 32-query-only API → up to 512 whole-query shard truth via guarded chunks | Full-ground-truth preservation, group and negative-sampling tests |
| `src/hm_recsys/pilot/testing.py` | `create_source, assert_fixture, fixture_profile, execute_fixture, create_full_source, execute_full_fixture` | Existing execution retained; adjacent English notes added for semantic statements, data flow and guards | Annotation executable-AST equivalence; maintained Ruff/Pyright and existing regression suite |
| `src/hm_recsys/pilot/towers.py` | `encode/cached_internal_recall` | Per-batch window SQL/row tensors → partition inputs; repeated full lexsort → cached active threshold search | Raw input parity, actual ten epochs, internal score and fresh reload |
| `src/hm_recsys/pilot/vectors.py` | `ActiveBlocks.search/block_topk` | Repeated active filtering/full lexsort → cached physical-block indices/names and exact threshold sorting | 6,001-item tie/empty-pool search parity |
| `src/hm_recsys/pilot/worker.py` | `run_stage` | Statistics dispatch unaware of new path → explicit optimized snapshot switch | Real full/pilot stage and authorization contracts |
| `src/hm_recsys/pipeline.py` | `catalog_ids, preflight, load_content, feature_table, build_query_features, candidates_and_features, prepare_inference_input, seal_run, compatible_source, verify_frozen, reload_features, reload_inference` | Existing execution retained; adjacent English notes added for semantic statements, data flow and guards | Annotation executable-AST equivalence; maintained Ruff/Pyright and existing regression suite |
| `src/hm_recsys/ranking.py` | `select_training_rows, matrix, ordered_predictions, train_ranker` | Existing execution retained; adjacent English notes added for semantic statements, data flow and guards | Annotation executable-AST equivalence; maintained Ruff/Pyright and existing regression suite |
| `src/hm_recsys/recall.py` | `Hit, Candidate, top, Snapshot` | Existing execution retained; adjacent English notes added for semantic statements, data flow and guards | Annotation executable-AST equivalence; maintained Ruff/Pyright and existing regression suite |
| `src/hm_recsys/resources.py` | `constrained_env, memory_violation, tree_rss, stop_owned, run_stage` | Existing execution retained; adjacent English notes added for semantic statements, data flow and guards | Annotation executable-AST equivalence; maintained Ruff/Pyright and existing regression suite |
| `src/hm_recsys/synthetic.py` | `create_fixture` | Existing execution retained; adjacent English notes added for semantic statements, data flow and guards | Annotation executable-AST equivalence; maintained Ruff/Pyright and existing regression suite |
| `src/hm_recsys/towers.py` | `event_history, sample_targets, TowerEncoder, TwoTower, positive_mask, logq_loss, fit_towers, encode_items, encode_queries` | Existing execution retained; adjacent English notes added for semantic statements, data flow and guards | Annotation executable-AST equivalence; maintained Ruff/Pyright and existing regression suite |
| `src/hm_recsys/worker.py` | `main` | Existing execution retained; adjacent English notes added for semantic statements, data flow and guards | Annotation executable-AST equivalence; maintained Ruff/Pyright and existing regression suite |
| `tests/conftest.py` | `Namespace/module statements` | Docstring-only namespace inspected; no execution statement requires an added comment (unchanged) | Annotation executable-AST equivalence; maintained Ruff/Pyright and existing regression suite |
| `tests/test_colab_checkpoints.py` | `test_real_tower_complete_batch_resume` | Existing execution retained; adjacent English notes added for semantic statements, data flow and guards | Annotation executable-AST equivalence; maintained Ruff/Pyright and existing regression suite |
| `tests/test_colab_migration.py` | `test_input_whitelist_reuse_corruption_and_source_integrity, test_copy_interruption_checksum_missing_and_space, test_stage_failure_resume_fresh_runtime_and_incompatible_identity, test_local_completion_survives_failed_persistence, test_profiles_full_opt_in_local_gate_and_no_smoke_caps, test_notebook_structure_syntax_and_safe_defaults, test_release_clean_extract_and_hostile_members, test_resumable_existing_pipeline_on_synthetic_data, test_disconnected_mount_rejects_publication, write_text, check_text` | Existing execution retained; adjacent English notes added for semantic statements, data flow and guards | Annotation executable-AST equivalence; maintained Ruff/Pyright and existing regression suite |
| `tests/test_compute.py` | `test_cadence_failure_clock_and_overlay, test_prepared_inputs_updates_invalidation_and_prefetch, test_remote_publication_failure_preserves_last_valid, test_optimized_full_stage_restore, test_cli_contracts_and_guards, test_cuda_inputs_are_not_checkpoint_parameters, test_notebook_benchmark_command_contract` | Existing execution retained; adjacent English notes added for semantic statements, data flow and guards | Annotation executable-AST equivalence; maintained Ruff/Pyright and existing regression suite |
| `tests/test_computing_refactor.py` | `Five focused fixtures` | New kernel/query/cache/ItemCF/shard tests and timing evidence | No fixed-speed assertions; authentic algorithms and synthetic inputs |
| `tests/test_documentation.py` | `_missing/test_maintained_statement_comments_and_readme_inventory` | New meaningful statement/inventory coverage acceptance | Every maintained file and configuration/Notebook path checked |
| `tests/test_full_backend.py` | `test_full_none_population_targets_and_truth, test_full_authorization_and_resource_rejection, test_full_actual_training_restore_and_complete_validation, test_parquet_resume_crosses_row_groups, test_notebook_full_command_contract` | Existing execution retained; adjacent English notes added for semantic statements, data flow and guards | Annotation executable-AST equivalence; maintained Ruff/Pyright and existing regression suite |
| `tests/test_full_checkpoints.py` | `test_full_checkpoint_exact_continuation` | Existing execution retained; adjacent English notes added for semantic statements, data flow and guards | Annotation executable-AST equivalence; maintained Ruff/Pyright and existing regression suite |
| `tests/test_pilot_backend.py` | `canonical_fixture, test_global_snapshot_and_bounded_histories, test_large_exact_index_and_cached_wals, _verify_large_index_and_cached_wals, test_pilot_staged_fixture_and_restore, test_notebook_probe_uses_actual_parser, test_pilot_guards_and_complete_truth, test_cuda_request_does_not_fallback, test_cuda_probe_explicitly_not_run, test_future_partition_cannot_change_earlier_snapshot` | Existing execution retained; adjacent English notes added for semantic statements, data flow and guards | Annotation executable-AST equivalence; maintained Ruff/Pyright and existing regression suite |
| `tests/test_pilot_checkpoints.py` | `test_pilot_epoch_checkpoint_continuation` | Existing execution retained; adjacent English notes added for semantic statements, data flow and guards | Annotation executable-AST equivalence; maintained Ruff/Pyright and existing regression suite |
| `tests/test_smoke_contracts.py` | `fixture_data, test_ids_splits_and_age_boundaries, test_content_mapping_readonly_and_missing, test_content_invalid_consumed_vector, test_sample_preserves_lines_labels_and_model_boundaries, test_windows_itemcf_and_sibling_formula, test_wals_confidence_and_reference_solve, test_two_tower_loss_history_dropout_and_padding, test_sampling_metrics_and_denominators, test_limits_reject_before_training, test_exact_faiss_scores_ties_and_reload, test_guarded_integration_and_future_perturbation, test_supervisor_stops_only_owned_stage, test_training_rejects_misdeclared_sample` | Existing execution retained; adjacent English notes added for semantic statements, data flow and guards | Annotation executable-AST equivalence; maintained Ruff/Pyright and existing regression suite |
| `tests/test_text_embeddings_notebook.py` | `definitions, test_text_contract, test_csv_ids_and_identical_text, test_artifact_reload_ties_and_publication` | Existing execution retained; adjacent English notes added for semantic statements, data flow and guards | Annotation executable-AST equivalence; maintained Ruff/Pyright and existing regression suite |

| Documentation / client | Actual change | Validation |
|---|---|---|
| `README.md` | Replace obsolete setup-only content with actual environments, file-level APIs/I/O/stages, model/time contracts and operation/recovery | Exact inventory test for all maintained Python, five configs and four Notebooks |
| `notebooks/01_Colab_Training.ipynb` | Adjacent code-cell explanations and new default smoke/benchmark IDs; preserve user RUN_ID assignment/handling, smoke mode, safety defaults and CLI options | Cell executable AST parity, nbformat/compile and actual command-contract tests |
| `docs/archi/computing_opt_report.md` | This audit matrix, per-file changes, measurements and unresolved risks | Evidence linked to real run paths and exit codes |
| `docs/archi/compute_performance_runbook.md` | Document this revision and distinct new preparation/benchmark/full IDs | Copyable CLI and Notebook contracts; new release companion |
| `docs/archi/architecture.md` | Correct current proposal relative link from missing root copy to docs/proposal.md | Target file exists; no architecture semantics changed |

## Statement comments and README coverage

Scope is all 65 maintained Python files in src, scripts and tests, plus all nine
code cells of the current Colab training Notebook. Adjacent front comments cover
each AST execution/declaration statement; multi-line literals/expressions and
pure closing parentheses share their statement's note. Imports, assignments,
branches, loops, SQL calls, tensor/index operations, errors, returns and side
effects are included. Google-style docstrings remain in place. Operation-specific
review notes explain fixed-item equations, live gradients, input shapes,
temporal windows, missingness and I/O boundaries near the corresponding code.

The annotation pass records executable-AST equivalence against its immediate
pre-comment snapshot. Formatting/import checks follow; explicit functional
changes are the ones listed separately above. `annotation_ast_parity.json`
contains the original per-unit checks. `before_annotations/` preserves the
reviewable code before comment expansion. `tests/test_documentation.py` checks
statement adjacency and the README inventory rather than raw comment totals.
Final coverage is 6,749 semantic statements/exception handlers across 65 Python
files and nine Notebook cells, with zero missing adjacent notes; see
`documentation_coverage.json`. This is a coverage check, not an assertion that
comment counts alone establish clarity. All five JSON configs and four current/
preserved Notebook roles are indexed in the README.

Excluded: docs/achive references, preserved old EDA/image/text Notebook cells,
third-party .venv/cache code, generated worker strings/artifact files and JSON
configs. These are not incomplete maintained Python coverage. The README still
indexes all four Notebook roles and every JSON config. No illegal comments were
inserted into JSON, source embeddings, Parquet, index or lock files.

## Verification commands and results

Final results are recorded from actual commands in the evidence directory;
local checks use the existing locked uv environment, two native threads and the
unchanged safety monitor. No failing check is reclassified as passed, and no
unrelated old Notebook is reformatted to improve the repository-wide result.


Final repository suite: **54 passed, 2 CUDA skips in 156.41 seconds**, exit 0.
After the explicit new default smoke/benchmark ID update, the actual Notebook/CLI
contract selection was rerun separately; its log is `notebook_final.log` (**5 passed, 25 deselected in 3.02 s**, exit 0).
The last source edit after the full suite polished comment wording and context-sensitive
file-stat/visual-fit notes; executable AST was unchanged. Final static and
annotation checks follow that edit. No numerical/model code changed after the
passing full suite.

| Actual command | Exit | Inspected outcome |
|---|---:|---|
| `uv run --locked pytest -q --basetemp=.../final_tests --junitxml=.../pytest.xml` | 0 | 54 passed, 2 CUDA skips |
| `uv run --locked pytest -q tests/test_documentation.py tests/test_computing_refactor.py` | 0 | Six targeted checks passed |
| `uv run --locked ruff check src scripts tests notebooks/01_Colab_Training.ipynb` | 0 | Maintained scope passed |
| `uv run --locked ruff format --check src scripts tests notebooks/01_Colab_Training.ipynb` | 0 | Maintained scope passed |
| `uv run --locked pyright` | 0 | No errors/warnings |
| `uv run --locked ruff check .` | 1 | Same 122 pre-existing old/archive Notebook errors |
| `uv run --locked ruff format --check .` | 1 | Same eight old/archive Notebooks need formatting |
| `uv run --locked python -c ...` (nbformat and compile) | 0 | Nine code cells structurally/syntactically valid; not executed in Colab |
| `uv run --locked python scripts/benchmark_compute.py --run-id computing_refactor_before_20261103_01 --synthetic --batches 4 --warmup 1 --validation complete` | 0 | Immediate before baseline |
| Same command, `--run-id computing_refactor_after_20261103_01 --trace` | 0 | Final three A/B modes; separate bounded profiler window |
| `uv run --locked python /tmp/hm_compare_refactor.py` | 0 | Exact old/new candidate/features/model/Top-12/metric evidence |
| `uv run --locked python /tmp/hm_refactor_io_benchmark.py` | 0 | Bounded local shard/checksum/ItemCF measurements |
| `git diff --check` | 0 | No whitespace errors; no Git write operations |

All commands reuse `/Users/vincent/.local/bin/uv`, with
`UV_CACHE_DIR=/tmp/hm-smoke-uv-cache` and two native threads for numerical checks.
Exact argv/status records are in `static_checks.json` and
`verification_commands.json`. Early development checks caught Arrow date
conversion requiring `zero_copy_only=False` and an atime-only checksum false
positive; both were corrected and the failed logs retained. Those earlier
failures are not represented as passes. No checks or warnings were disabled.

Packaging checks completed successfully: 96 allowlisted source members, five
CLI help contracts, standalone/packaged Notebook byte equality, matching
SHA256/manifest and nine focused packaged tests passed. These checks use a clean
temporary extraction and execute no real Colab training. Actual build and
clean-extraction records are written to
`build_release.log`, `release_verification.json` and `packaged_tests.log` in the
evidence directory. The release manifest records final source/document bytes;
README and this report have no circular self-hash placeholder.

## Release, reproduction and remaining work

Build with `uv run --locked python scripts/build_release.py`; matching source ZIP,
SHA256, manifest and resolved runbooks remain under `data/releases/`. Package
verification extracts into a clean temporary directory, checks all required
members/entry points and the standalone Notebook bytes, then runs focused
packaged contracts. The content-addressed release name is recorded after build.

Use the resolved `.compute_runbook.md` and a new preparation/benchmark RUN_ID.
Start this release's smoke, prepare a current-code pilot only through content,
then manually run at most 100 CUDA diagnostic batches per variant with identical
population and Q. Inspect performance_summary.json/parity_report.json and actual
resource evidence before starting any new optimized full run. Keep default
smoke/full opt-in off and final Test False. Never rename an old pilot/full job
or edit its identity to continue changed code.

Remaining unverified: real full prepack and ItemCF scalability, large-cutoff item
table cost, full shard disk/memory, CUDA transfer/residency throughput and exact
near-tie behavior on that GPU, actual Drive/FUSE receipt metadata semantics,
failure/recovery in a replacement Colab runtime and one full LightGBM fit on the
actual candidate population. The current CPU exact-block path is retained;
GPU matmul validation and automatic thread/memory widening remain separate,
unselected experiments. Performance measurements here cannot guarantee a cloud
speed multiplier or that full fits the runtime.
