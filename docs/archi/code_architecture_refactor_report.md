# Single-Source Architecture Refactor Report

Updated: 2026-11-03. Scope: the current dirty working tree, not a historical
release ZIP. This report describes actual local synthetic verification; it is
not a full H&M or Colab benchmark.

## Contents

- [Outcome and boundaries](#outcome-and-boundaries)
- [Architecture and migration](#architecture-and-migration)
- [Canonical file responsibilities](#canonical-file-responsibilities)
- [Preserved model and evaluation contracts](#preserved-model-and-evaluation-contracts)
- [Notebook and recovery](#notebook-and-recovery)
- [Measured parity and performance](#measured-parity-and-performance)
- [Verification commands and results](#verification-commands-and-results)
- [Archive and privacy audit](#archive-and-privacy-audit)
- [Remaining execution limits](#remaining-execution-limits)

## Outcome and boundaries

The eight canonical packages under `src/hm_recsys/` replace the parallel root,
`pilot/`, `colab/` and `compute/` namespaces. The sole maintained execution
Notebook is [`colab_training_val.ipynb`](../../colab_training_val.ipynb). Smoke
and full share the same SQL data preparation, six retrievers, native fusion,
feature builder, grouped LambdaMART and complete-ground-truth evaluator.

No real full training, real final-Test evaluation, CUDA execution or cloud job
was started. Existing source CSVs, embeddings, row maps, original indexes,
checkpoints and previous run directories were preserved. No commit, push, reset, initialization or history rewrite was performed on
the project repository. Temporary Git fixtures were used for offline tests. Explicit index-only
archive untracking is recorded separately below.

Pre-refactor inventory, original bytes and actual baseline outputs were saved
under `data/runs/refactor/single_source_20261103_01/` before implementation.
The initial Git HEAD was `54c24e54b543490b334b68daeece91c29bfe3689`; that SHA
alone does not describe the pre-existing uncommitted changes or this delivery.

## Architecture and migration

See the exhaustive [migration map](refactor_migration_map.md), the updated
[architecture](architecture.md), [README](../../README.md) and
[runbook](github_colab_runbook.md).

- `core/`: IDs, inclusive cutoffs, disjoint label windows and live-event contract.
- `data/`: bounded SQL histories, independent rosters, raw-line target streams.
- `features/`: immutable content adapters, frozen preprocessors, one ordered
  feature builder and incremental columnar shards.
- `retrieval/`: four rule channels plus one WALS and one Two-Tower implementation.
- `fusion/`: native union, complete provenance and exact RRF contributions.
- `ranking/`: one Train-only sampler, one LambdaMART fit and frozen inference.
- `evaluation/`: one macro metric implementation with full truth denominators.
- `runtime/`: CLI, dependency-aware stage graph, identities, isolated workers,
  live progress, resource checks and atomic persistence/recovery.

There are no import-time cycles or imports from the retired namespaces or
archive. Function-local orchestration dependencies remain; this is not a claim
that every possible static call graph is acyclic. The Two-Tower model math does
not depend on runtime orchestration. WALS persistence uses shared content/index
adapters; those adapters depend on bounded file utilities.

Functional stages are `prepare`, the six channel names, `fusion`, `features`,
`ranker`, `validation`, and `reload`. `final_test` is optional and excluded from
the default stage order. A channel requires verified preparation, not another
model. Fusion requires all six channels. Missing dependencies fail with their
names; a cell never silently trains them. Shared cutoff statistics are built
once, while ItemCF indexes are separate immutable artifacts. Features consume
persisted fusion and do not rerun recall or read future labels.

New stage identities use version 3 and `single-source-modular-v1`, binding the
actual code hashes, config, staged inputs, schema, splits and runtime. Old
pilot/full artifacts cannot be relabelled as a new-layout run. Start a new run
ID and pinned commit; previous real jobs and checkpoints remain untouched.

## Canonical file responsibilities

The table lists every active source and thin entry point. Test responsibilities
are inventoried in the README and the migration map. Every active Python file
has a leading module contract; documentation coverage no longer requires
mechanical per-statement narration.

| File | Responsibility | Main definitions |
|---|---|---|
| `src/hm_recsys/__init__.py` | Canonical H&M next-seven-day recommendation implementation. | Import-safe namespace |
| `src/hm_recsys/core/__init__.py` | Shared temporal, identifier and progress contracts without training side effects. | Import-safe namespace |
| `src/hm_recsys/core/contracts.py` | Immutable smoke configuration, date contracts, and artifact helpers. | `write_json`, `read_json`, `digest`, `stable_key`, `article_id`, `split_manifest`, `validate_splits`, `Config`, `source_identity` |
| `src/hm_recsys/core/progress.py` | Local append-only worker progress; counters never drive model execution. | `scope`, `emit`, `Lines`, `describe` |
| `src/hm_recsys/core/synthetic.py` | Deterministic offline fixture with real learning signal and missing content. | `create_fixture` |
| `src/hm_recsys/data/__init__.py` | Bounded canonical history, independent query rosters and target streams. | Import-safe namespace |
| `src/hm_recsys/data/history.py` | Build canonical purchase storage and bounded, cutoff-safe query histories. | `allocation`, `connect`, `transaction_view`, `canonical`, `History`, `rosters`, `full_rosters`, `prepare_test_roster` |
| `src/hm_recsys/data/sample.py` | Define immutable purchase-line and bounded metadata contracts. | `Transaction`, `Sample`, `inspect_csv`, `age_bucket` |
| `src/hm_recsys/data/streaming.py` | Stream canonical targets, query rosters and feature ledgers in bounded batches. | `export_query`, `rows`, `TargetStream`, `Partition`, `Shards`, `load_manifest` |
| `src/hm_recsys/evaluation/__init__.py` | Full-ground-truth macro metrics preserving failed and empty queries. | Import-safe namespace |
| `src/hm_recsys/evaluation/metrics.py` | Compute native recommendation metrics from complete, independent ground truth. | `recall_at_k`, `metrics`, `union_recall`, `MetricAccumulator` |
| `src/hm_recsys/features/__init__.py` | Frozen content adaptation and the shared ordered ranking feature schema. | Import-safe namespace |
| `src/hm_recsys/features/benchmark.py` | Replay frozen feature artifacts through the canonical builder on bounded runs. | `compare` |
| `src/hm_recsys/features/builder.py` | Construct the sole shared ranking feature matrix for training and inference. | `FeatureBuilder`, `frozen_clusters` |
| `src/hm_recsys/features/content.py` | Validate and adapt immutable text/image vectors with independent row maps. | `ContentSource`, `ContentBank`, `adapt_content`, `MappedContent` |
| `src/hm_recsys/features/pipeline.py` | Build shared ranking features from persisted native candidate unions. | `generate` |
| `src/hm_recsys/features/preprocessing.py` | Fit frozen visual preprocessing and persist the shared ranking schema. | `preprocessing` |
| `src/hm_recsys/features/storage.py` | Write bounded Arrow feature blocks without breaking query boundaries. | `FeatureShards`, `feature_table` |
| `src/hm_recsys/fusion/__init__.py` | Native candidate deduplication, provenance and exact reciprocal rank fusion. | Import-safe namespace |
| `src/hm_recsys/fusion/native.py` | Merge six native lists without labels and compute reciprocal-rank features. | `Candidate`, `merge` |
| `src/hm_recsys/fusion/pipeline.py` | Persist the complete six-source native union and nonlearned RRF diagnostics. | `fuse` |
| `src/hm_recsys/fusion/storage.py` | Read bounded native source/candidate batches with verified columnar contracts. | `CandidateReader`, `load_candidates` |
| `src/hm_recsys/ranking/__init__.py` | One LambdaMART with Train-only sampling and schema-validated inference. | Import-safe namespace |
| `src/hm_recsys/ranking/model.py` | Define the sole native ranker sampling and prediction contracts. | `BoostProgress`, `select_training_rows`, `matrix`, `ordered_predictions` |
| `src/hm_recsys/ranking/pipeline.py` | Train one grouped LambdaMART and verify fresh-process ranking parity. | `read_shard`, `feature_manifest`, `train`, `ExpectedRows`, `reload`, `predict_partition` |
| `src/hm_recsys/retrieval/__init__.py` | Six canonical recall channels, frozen models and bounded learned indexes. | Import-safe namespace |
| `src/hm_recsys/retrieval/base.py` | Define native source-hit records and the shared eligibility/Top-K contract. | `Hit`, `top` |
| `src/hm_recsys/retrieval/itemcf.py` | Build R4 sparse asymmetric neighborhoods and retrieve from recent seed items. | `build`, `retrieve` |
| `src/hm_recsys/retrieval/learned.py` | Load frozen retrieval artifacts and score bounded learned-channel batches. | `TowerRetriever`, `learned_batch`, `wals_batch`, `tower_batch` |
| `src/hm_recsys/retrieval/pipeline.py` | Generate one explicitly selected native retrieval channel for saved rosters. | `run_channel` |
| `src/hm_recsys/retrieval/repurchase.py` | Build R2 repeat-purchase scores from legal sixty-day raw purchase lines. | `reference_scores`, `scores`, `retrieve` |
| `src/hm_recsys/retrieval/sibling.py` | Build R3 variant nominations using the actual product_code relationship. | `reference_scores`, `scores`, `retrieve` |
| `src/hm_recsys/retrieval/snapshot.py` | Expose bounded cutoff snapshots backed by complete legal SQL statistics. | `build_statistics`, `build_snapshot`, `Snapshot`, `GlobalSnapshot` |
| `src/hm_recsys/retrieval/trend.py` | Build R1 seven-day cutoff popularity with age-bucket/global fallback. | `popularity`, `retrieve` |
| `src/hm_recsys/retrieval/two_tower/__init__.py` | Shared multimodal networks, raw input preparation, training and frozen inference. | Import-safe namespace |
| `src/hm_recsys/retrieval/two_tower/execution.py` | Measure actual tower work and transfer bounded raw inputs to the requested device. | `Measurements`, `prefetch`, `ItemBank` |
| `src/hm_recsys/retrieval/two_tower/inference.py` | Encode frozen Two-Tower catalog and cutoff-query vectors for native retrieval. | `encode` |
| `src/hm_recsys/retrieval/two_tower/inputs.py` | Assemble the one shared category, history, padding and raw-content input contract. | `event_history`, `TowerEncoder`, `SqlEncoder` |
| `src/hm_recsys/retrieval/two_tower/model.py` | Define the shared multimodal user/item networks and multi-positive LogQ loss. | `TwoTower`, `positive_mask`, `logq_loss` |
| `src/hm_recsys/retrieval/two_tower/prepared.py` | Prepare verified model-independent training contexts and complete positive sets. | `identity`, `verify`, `daily_rows`, `prepare`, `Prepared` |
| `src/hm_recsys/retrieval/two_tower/query_inputs.py` | Prepare bounded raw cutoff-query inputs for frozen Two-Tower inference. | `QueryInputs` |
| `src/hm_recsys/retrieval/two_tower/training.py` | Train the shared production Two-Tower and select a frozen internal checkpoint. | `device_for`, `move`, `item_vectors`, `internal_recall`, `cached_internal_recall`, `fit` |
| `src/hm_recsys/retrieval/vectors.py` | Search and persist exact learned inner-product indexes with bounded memory. | `block_topk`, `ActiveBlocks`, `BlockIndex` |
| `src/hm_recsys/retrieval/wals/__init__.py` | Sparse WALS fitting and fixed-factor user folding-in on legal purchase histories. | Import-safe namespace |
| `src/hm_recsys/retrieval/wals/model.py` | Fold current users into frozen WALS item factors under one confidence transform. | `confidence`, `fold_user`, `fold_users_batch`, `WALS` |
| `src/hm_recsys/retrieval/wals/training.py` | Fit the one production sparse, time-bounded WALS base for every run profile. | `fit_wals` |
| `src/hm_recsys/runtime/__init__.py` | Explicit stage orchestration, immutable identities and guarded recovery. | Import-safe namespace |
| `src/hm_recsys/runtime/acceptance.py` | Validate immutable source headers, ID maps and split contracts. | `catalog_ids`, `preflight` |
| `src/hm_recsys/runtime/benchmark.py` | Bounded reference/optimized compute diagnostics using canonical model primitives. | `reference_batches`, `optimized_batches`, `update`, `new_model`, `parity`, `input_view`, `run_benchmark`, `main` |
| `src/hm_recsys/runtime/capacity.py` | Resource estimates for complete feature, training and Validation workloads. | `check`, `feature_capacity` |
| `src/hm_recsys/runtime/checkpoints.py` | Complete, atomically published Two-Tower training checkpoints. | `save_checkpoint`, `read_previous`, `load_checkpoint` |
| `src/hm_recsys/runtime/cli.py` | Expose canonical stages, explicit permissions and immutable run identities. | `main` |
| `src/hm_recsys/runtime/device.py` | Probe the explicitly requested execution device with a real tensor update. | `probe` |
| `src/hm_recsys/runtime/files.py` | Bounded immutable input staging and verified atomic artifact transfer. | `canonical_hash`, `atomic_json`, `relative_path`, `safe_destination`, `require_space`, `file_identity`, `verified_copy`, `stage_inputs`, `_declared_hash`, `verify_files`, `DigestCache` |
| `src/hm_recsys/runtime/git_checkout.py` | Pinned Git checkout and source provenance without automatic code updates. | `repository_url`, `validate_ref`, `validate_commit`, `git_environment`, `_git`, `_destination`, `inspect_checkout`, `checkout`, `development_identity`, `main` |
| `src/hm_recsys/runtime/git_safety.py` | Read-only Git privacy auditing for source, artifacts and credentials. | `_git`, `_display_path`, `path_indicators`, `content_indicators`, `_tree`, `audit_repository` |
| `src/hm_recsys/runtime/local_cli.py` | Run bounded local smoke through the same canonical stages as Colab full. | `main` |
| `src/hm_recsys/runtime/modular.py` | Execute the dependency-checked, resumable single-source stage graph. | `prerequisites`, `resolved_profile`, `_safe_value`, `safe_report`, `stage_summary`, `Session`, `run_all`, `run_one` |
| `src/hm_recsys/runtime/notebook.py` | Live worker event rendering and explicit Notebook command execution. | `safe_text`, `_stop_group`, `command` |
| `src/hm_recsys/runtime/profiles.py` | Validated smoke/full budgets, host gates and measured resource planning. | `stages_for`, `load_profile`, `is_colab_runtime`, `execution_gate`, `runtime_identity`, `resource_plan`, `plan` |
| `src/hm_recsys/runtime/resources.py` | Bound and observe independent native-library worker processes. | `constrained_env`, `memory_violation`, `tree_rss`, `stop_owned`, `run_stage` |
| `src/hm_recsys/runtime/review.py` | Aggregate-only diagnostic exports for verified completed runs. | `publish_review` |
| `src/hm_recsys/runtime/runner.py` | Bind modular runs to immutable source, data, schema and runtime identities. | `code_identity`, `execution_identity`, `execute`, `_execute_profile` |
| `src/hm_recsys/runtime/settings.py` | Explicit compute backend and checkpoint cadence configuration. | `apply`, `Cadence` |
| `src/hm_recsys/runtime/stages.py` | Dispatch canonical numerical phases without duplicating model implementations. | `cutoff_for`, `run_stage`, `validate` |
| `src/hm_recsys/runtime/state.py` | Checksummed dependency-aware stage publication and recovery. | `writer_lease`, `StageStore` |
| `src/hm_recsys/runtime/testing.py` | Tiny deterministic fixture authorization, never a production CLI bypass. | `create_source`, `assert_fixture`, `fixture_profile`, `execute_fixture`, `create_full_source`, `execute_full_fixture` |
| `src/hm_recsys/runtime/worker.py` | Run one canonical numerical phase in an independent Python process. | `main` |
| `scripts/audit_git.py` | Inspect current source/index/history privacy locally before a reviewed Git push. | `main` |
| `scripts/benchmark_compute.py` | Run a capped reference/optimized compute diagnostic, never a full training job. | Import-safe namespace |
| `scripts/local_smoke.py` | Run the repository-local guarded smoke CLI from any working directory. | Import-safe namespace |
| `scripts/probe_device.py` | Run an isolated bounded shared-tower device update and reload probe. | `build_parser`, `main` |
| `scripts/run_pipeline.py` | Call shared Colab plan, staging, stage, status, and recovery interfaces. | Import-safe namespace |

## Preserved model and evaluation contracts

| Contract | Result |
|---|---|
| Channels and quotas | Trend / Repurchase / Sibling / ItemCF / WALS / Two-Tower; 60/30/30/50/50/50 |
| Fusion | Native union at most 270 before overlaps; sum of `1/(60 + one_based_rank)`; no positive injection |
| Feature schema | Same 49 names/order, float32 values, categorical codes, NaNs and modality/source masks |
| Temporal scope | Inclusive cutoffs; strict same-day exclusion for tower histories; F1–F4, Validation and Test windows unchanged and disjoint |
| WALS | One confidence convention; frozen item factors; actual per-cutoff folding-in; raw inner product |
| Two-Tower | Same sorted unique batch candidates, multi-positive mask, Q, FP32 objective, optimizer, scheduler, target order and epoch selection |
| Ranker | Four folds jointly train one LambdaMART; all native positives and up to 20 hard/random negatives per positive; no Validation sampling |
| Evaluation | Full unique catalog-matched truth; zero-candidate and zero-hit queries retained; macro MAP/Recall/NDCG and signed LOO unchanged |
| Persistence | Complete model/optimizer/scheduler/RNG/progress checkpoints; atomic verified stage publication and compatible recovery |

Smoke no longer has a second in-memory model trainer. Its explicit limits are
still 500 selected users, 5,000 eligible candidate articles, 200,000 purchase
lines and 50 queries per fold. The complete selected history is represented in
the model input vocabulary independently of the capped candidate pool. This
fixes targets/history outside the pool without dropping legitimate events;
resource guards still apply. Full retains `None`/null unlimited semantics.

## Notebook and recovery

The Notebook is thin: Git checkout, existing lockfile environment, whitelisted
Drive-to-local input copy, CLI calls and measured reports. Train, Validation and
Test planning have explicit sections; Test preparation is metadata only until
the final authorized stage. Each of the six retrievers, RRF and LambdaMART has
its own executable cell. Structured worker events display real row counts,
percentages when totals are known, rates, epoch/step/loss, memory and checkpoint
status. Unknown totals remain unknown.

The last code cell defaults to skipped. It requires full profile, both explicit
Test flags, the locked completed Validation/reload signature and matching
model/schema hashes. It builds Test-only history/candidates/features and scores
with frozen models; it does not refit, sample negatives or tune on Test.
The synthetic guard test uses generated labels only.

Use `RUN_ACTION="resume"` with the original run ID, commit, configuration and
inputs. A new runtime verifies and restores dependency manifests and checksums;
directory existence alone is insufficient. Configuration/code/data/environment
drift is rejected. Per-stage reports are retained separately; the final merged
artifact view's `resource_usage.json` alone is not an all-branch timing ledger.
For all-channel audit, inspect each `reports/<stage>.json` and its phase records.

## Measured parity and performance

The actual before/after input is generated, not H&M customer data: 170 catalog
articles, 24 customers and 4,571 purchase lines. All five partitions have 24
queries. The unchanged full model configuration ran on this tiny input only:
WALS 15 iterations, Two-Tower 10 epochs / 40 updates / 15,840 processed target
examples, and one LambdaMART with 96 groups / 4,938 sampled rows / 480 positives.
LightGBM evaluated 52 rounds and selected round 2; the saved model has two trees.
These values are measurements, not a reduced production full configuration.

| Comparison | Actual result |
|---|---|
| All six source lists across five partitions | 24,766 rows; identical keys, scores, ranks and fallback provenance |
| Shared feature rows | 5,967; all 49 columns exactly equal including NaNs after key alignment |
| Source cutoff representation | Old string converted to canonical Arrow date32; same ISO cutoff values |
| Independent query manifests and complete labels | Identical across all five partitions |
| WALS factors / trained tower item vectors | Maximum absolute numeric delta 0 |
| Tower losses, internal recalls, selected epoch and step counts | Identical |
| Training group sizes/order, sampling and selected ranker | Identical |
| Trained tower state and saved Validation score arrays | Exactly identical |
| Validation and reloaded Top-12 | Identical |
| Synthetic Validation MAP@12 / Recall@12 / NDCG@12 | 0.8746437590 / 0.9583333333 / 0.9292306962 in both runs |
| Evaluation population | 24 queries, including one zero-hit query; no filtered denominator |

Evidence: `parity_report.json`, `performance_summary.json`, `compare_refactor.py`,
`before/` and `after_08/` below the evidence root. The comparison joins candidates
by stable keys; Parquet row placement and artifact partition layout may differ.
The preserved reference comes from the inspected optimized current backend,
not from importing the archive.

| Matched synthetic measurement | Before | After |
|---|---:|---:|
| Entire cold build/train/Validation/reload, seconds | 24.597 | 34.647 |
| End-to-end feature rows / second, including all work | 242.59 | 172.22 |
| Supervised worker phases | 26 | 45 |
| Tower CPU step p50, milliseconds | 5.011 | 4.939 |
| Tower CPU step p95, milliseconds | 8.975 | 8.373 |
| Sampled process-tree peak, MiB | 619.34 | 445.95 |

Total wall time increased approximately 40.9% in this one small run. More independent
workers, separate channel artifacts and verified stage publication add fixed
cost. Tower update timings are close and subject to noise; this is not proof of
a training speedup or full-data memory improvement. Both runs construct their
own caches and perform the same epochs, validation and checkpoint policy;
filesystem cache state was not controlled. RSS is sampled and includes shared
pages; it is not a hard OS cap. CUDA events/utilization, real Drive I/O and
full-data throughput were not measured.

The retained diagnostic CLI was also executed on the final canonical code with
170 generated items, batch 512, one warmup and two measured updates per path.
It compared reference and optimized inputs, multi-positive masks, Q, losses,
gradients and updated parameters: all numeric maximum differences were zero.
The same 72 internal selection queries and complete local checkpoint state
were used by both paths. This is not a Colab or Drive benchmark.

| CPU diagnostic scope | Reference seconds | Optimized seconds |
|---|---:|---:|
| Two-update replay window | 0.011431 | 0.010181 |
| Train loop including periodic checkpoint work | 0.148165 | 0.025247 |
| Post-canonical work including internal validation and persistence | 0.223690 | 0.052943 |
| Cold post-canonical total including cache build/open and bank setup | 0.223690 | 0.191809 |

Common input preparation cost 0.192598 seconds for each path; optimized context
cache build cost 0.092040 seconds, verified open 0.040586 seconds and input-bank
setup 0.006240 seconds. The warm-cache loop timing alone would exaggerate the
cold improvement. These tiny windows are noisy and justify no fixed speedup
promise. Full benchmark telemetry and exact configuration are in
`data/runs/benchmark/single_source_benchmark_20261103_01/`.

Final integration repairs were limited to stage boundaries: each worker owns
its output/scratch directories; reload/Test phases have unique log names;
Test scoring allows its existing query-vector directory while rejecting actual
prediction replacement. The diagnostic input reader now verifies the canonical
`prepare` record instead of looking for retired low-level stage records. CLI
`review` again publishes the aggregate review bundle after identity validation.
None of these changes modifies model math, targets, feature values or metrics.

## Verification commands and results

Commands used the existing Python 3.12 uv environment, with
`UV_CACHE_DIR=/tmp/hm-smoke-uv-cache` and `/Users/vincent/.local/bin/uv run --locked`.
Native workers used two CPU threads. Tests requiring macOS process-tree RSS or
owned-child termination ran with the required host monitoring permission; no
resource guard was disabled.

| Actual command/check | Exit | Result |
|---|---:|---|
| `uv run --locked ruff check .` | 0 | All checks passed; archive excluded |
| `uv run --locked ruff format --check .` | 0 | 121 files already formatted |
| `uv run --locked pyright` | 0 | 0 errors, warnings or information messages |
| First repository-wide `uv run --locked pytest -q` | 1 | 133 passed, 5 failed, 2 skipped; genuine optional-Test directory collision plus adapters/log names and a source-identity drift were investigated |
| Repeated whole suite after fixes | 0 | 139 passed, 2 CUDA checks skipped in 344.04 seconds |
| New canonical diagnostic-input regression | 0 | 1 passed; verified prepare-stage copy excludes fitted models |
| Final delivery `uv run --locked pytest -q` | 0 | **140 passed, 2 CUDA checks skipped in 350.11 seconds** |
| Follow-up `uv run --locked pytest -q tests/test_git_safety.py` | 0 | 6 passed after adding explicit JSONL/NDJSON path protection; final Ruff/format/Pyright rechecked |
| `nbformat.validate` and compile all code cells | 0 | 45 cells / 23 code cells; outputs clear; final Test is the last cell |
| Maintained Markdown heading/link checks | 0 | Eight current documents, no missing local links or heading jumps |
| Changed Mermaid | Static only | Flowchart directive, nodes/edges and balanced delimiters reviewed; rendered preview not run |
| Read-only Git privacy audit | 1 | Intentional blocked-push result; 36 HEAD/history findings, no current index/worktree artifact findings |

Actual smoke command, exit 0:

```bash
UV_CACHE_DIR=/tmp/hm-smoke-uv-cache /Users/vincent/.local/bin/uv run --locked python scripts/local_smoke.py --synthetic --run --run-id single_source_smoke_20261103_01
```

This ran the ordinary unchanged smoke budget, not a fake model: 24 selected
users, 40 eligible candidate items / 41 model-metadata items, 4,187 retained
purchase lines, 24 queries in each fold/Validation; WALS two iterations,
Two-Tower one epoch / 38 updates / 1,200 processed targets, and LambdaMART
30 rounds / 96 groups / 3,840 rows. Fresh-process score delta was zero and
unique Top-12 matched. Artifacts are under
`data/runs/smoke/single_source_smoke_20261103_01/{local,persistent}/`.

Actual bounded diagnostic command, exit 0:

```bash
UV_CACHE_DIR=/tmp/hm-smoke-uv-cache /Users/vincent/.local/bin/uv run --locked python scripts/benchmark_compute.py --synthetic --run-id single_source_benchmark_20261103_01 --device cpu --batches 2 --warmup 1 --mode all --validation complete
```

Parity comparison used the saved generated baseline and its independent
outputs, with `PYTHONPATH=src uv run --locked python
 data/runs/refactor/single_source_20261103_01/compare_refactor.py after_08`.
The comparison normalized only the source cutoff representation (ISO string
versus Arrow date), compared NaNs explicitly and joined by stable keys.

The Notebook model-stage cells were executed sequentially in a fresh fixture
namespace through the real canonical workers, with a local input/command
adapter and Test disabled. Git bootstrap and argument contracts were tested
against local bare remotes. A real Colab kernel Restart + Run All, Drive mount
and runtime installation were not executed locally. Earlier failing logs and
all successful evidence remain under the evidence root; failed attempts are
not relabelled as passing runs.

## Archive and privacy audit

The archive is ignored by Git and excluded from Ruff, Pyright and pytest.
Original current-source snapshots, obsolete packaging clients, four prior
Notebooks and retired test references remain local under
`docs/achive/single_source_refactor_20261103_01/`. The migration map identifies
original versus intermediate copies and records hashes; no active code imports
these files. Existing runtime datasets and real run directories were not moved.

JSONL and NDJSON exports are explicitly ignored and rejected by the independent
artifact-path audit, even outside the runtime data root. The canonical
`src/hm_recsys/data/*.py` source exception does not exempt datasets, vectors or
content/credential checks.

Ten previously tracked archive paths and the four superseded Notebook paths
were removed from the index using explicit `git rm --cached` only. Worktree
archive copies remain. Current `git ls-files docs/achive notebooks` is empty;
source data and embeddings remain ignored, not staged. `archive_untracking.json`
and its Notebook supplement record the affected paths.

The read-only Git privacy audit reports **push blocked**: 36 findings remain in
HEAD/history (10 archive paths, seven populated Notebook outputs and one ZIP,
counted across HEAD and historical scope). The current index/worktree has no
reported private artifact findings and no ignored-but-tracked paths. This is
not evidence that committed history is public-safe. No push or history cleanup
was attempted; review historical private outputs before publishing.

## Remaining execution limits

Real H&M smoke training in this refactor, real full training, real final Test,
Colab restart/Drive persistence, CUDA throughput and a public GitHub clone are
not claimed as verified here. Local Git bare-remote fixtures exercise pinned
checkout and identity behavior without network access. The earlier real Colab
results remain historical/user-reported evidence; they were not reproduced.

The code requires a new manually reviewed commit and a new run ID before cloud
execution. After privacy review and the user's manual commit/push, open the root
Notebook from GitHub, set the repository/ref or immutable SHA, retain default
smoke and disabled Test, run all cells in order, then explicitly authorize full
with a new run ID. Follow the runbook for exact paths and recovery controls.
No new ZIP release is required by the Git-driven workflow.

For the first cloud smoke keep `RUN_PROFILE="smoke"`, `DEVICE="cpu"`,
`ALLOW_FULL_RUN=False`, `RUN_FINAL_TEST=False`, `ALLOW_FINAL_TEST=False` and
choose a new `RUN_ID`. To start full only after that smoke and resource review,
use the following configuration in the first cell (placeholders are not run
IDs or commits that already exist):

```python
GIT_COMMIT = "PASTE_THE_NEW_REVIEWED_40_CHARACTER_COMMIT"
RUN_PROFILE = "full"
RUN_ID = "YOUR_NEW_FULL_RUN_ID"
RUN_ACTION = "run"
DEVICE = "cuda"
ALLOW_FULL_RUN = True
RUN_FINAL_TEST = False
ALLOW_FINAL_TEST = False
LOCKED_RUN_SIGNATURE = None
```

Use `RUN_ACTION="resume"` with that exact original commit/run/config/input
identity after interruption; do not update code mid-run. Stage results persist
to `/content/drive/MyDrive/projects/e_commerce_recsys/data/runs/colab/<RUN_ID>/`.
This delivery stops at local verification; the above cloud commands were not
executed. The Git-history privacy findings must be reviewed before a public push.
