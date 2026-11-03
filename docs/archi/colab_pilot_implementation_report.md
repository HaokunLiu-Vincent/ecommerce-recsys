# Colab Pilot Implementation Report

This report records the preceding pilot milestone. The user subsequently
reported successful real Colab CUDA pilot. The current full adapter and its
separate verification are described in [Full Implementation Report](colab_full_implementation_report.md).

## Contents

- [Delivery and evidence](#delivery-and-evidence)
- [Implementation](#implementation)
- [Validation](#validation)
- [Compatibility and limits](#compatibility-and-limits)
- [Handoff](#handoff)

## Delivery and evidence

The pilot backend implements real WALS, Two-Tower, six-channel native retrieval,
shared features, one LambdaMART and fresh-process Top-12 reload. Local tests use
small authenticated synthetic fixtures, including the actual model update and
checkpoint paths. New-release real Colab, CUDA and real H&M pilot execution are
**not run**. Full and final Test remain disabled.

The user reports successful real Colab CPU smoke for the earlier migration:
500 users, 5,000 items, 53,642 transaction lines, all six channels and three
models, and reload parity for 5,487 Validation rows with zero maximum score
difference. These are user-supplied historical measurements, not measurements
reproduced during this task. They do not validate the new pilot backend.

## Implementation

- Added `src/hm_recsys/pilot/` adapters around the existing model, rule recall,
  loss, sampler, feature and evaluator functions. The existing smoke interface
  and regressions remain intact. No training implementation is in the notebook.
- Canonical transactions are projected with DuckDB into portable monthly
  Parquet partitions. CSV input never becomes a full Python transaction list.
  Global statistics use all legal history; gradient users and query rosters
  are distinct deterministic subsets. Catalog join exclusions are counted.
- Global cutoff snapshots retain duplicate purchase lines, date semantics,
  age-bucket popularity and exact asymmetric ItemCF. Seed-block SQL joins build
  active-first top-100 neighborhoods. Resource rejection never truncates user
  degree or changes the formula.
- WALS uses the frozen global item vocabulary and explicitly records its
  cohort-supported subset. Cached fixed-item Gram matrices preserve folding-in.
  Complete-catalog Two-Tower encoding and learned FAISS indexes use bounded
  batches. Search uses blocked exact inner products with active filtering and
  deterministic ID ties; persisted IndexFlatIP contents are checked. Original
  text/image indexes remain read-only inputs, never learned indexes.
- Actual Two-Tower model, tensors, masks and LogQ computations use the requested
  CPU/CUDA device. Unavailable CUDA raises. Epoch selection uses existing daily
  internal Recall@50 and full truth, not outer Validation. Checkpoints restore
  optimizer, scheduler, RNG, progress and best-epoch state. Float32 uses no AMP.
- Feature shards hold at most 32 complete query groups. Full truth and an
  independent roster retain zero-hit and zero-candidate queries. One shared
  feature implementation serves training and reload. There is no injection.
- One LambdaMART fit consumes estimated, columnar/memory-mapped matrices after
  Train-only 1:20 hard/random sampling. Validation is unsampled. This is still
  an in-memory LightGBM fit, protected by an explicit estimate and RSS monitor;
  it is not repeated `init_model` fitting or claimed out-of-core training.
- Existing staging, stage-delta persistence, identity gates, leases and recovery
  are reused. Pilot selects 18 stages and only F1/Validation. Full retains its
  explicit gate. Aggregate review bundles exclude customer-level information.
- The notebook's actual CUDA command includes the interpreter, script path,
  `--device cuda`, and a fresh UUID output path, with no unsupported `--run-id`.
  Tests execute the generated command through a recorder and the real parser.
  Smoke remains the default; pilot has separate opt-in and never requires full.

Production pilot caps remain unchanged: 5,000 base users; full catalog/global
history; at most 1,000 queries per F1/Validation; WALS 64 factors/3 iterations;
Two-Tower batch 512, 20,000 targets, 1 epoch/100 steps; LambdaMART 100 rounds.
The existing explicit batch-512 deviation from proposed 2048 is retained.
Synthetic-only tests override dimensions/steps/K for a tiny exact fixture;
that hook verifies all input bytes and is not exposed as a real-data CLI bypass.

## Validation

Local commands use the unchanged locked Python 3.12 environment with
`UV_CACHE_DIR=/tmp/hm-smoke-uv-cache /Users/vincent/.local/bin/uv run --locked`.
The final results below were measured on 2026-11-02.
No real H&M pilot/full scan, training or feature construction was run locally.

Verification evidence is stored beneath
`data/runs/colab/pilot_local_validation_20261102_01/`. The synthetic fixture is
24 users and 42 catalog items; its gradient cohort is 12 users. It exercises
actual WALS, two epochs of Two-Tower, K-Means, one LambdaMART, F1/Validation,
new-process inference and local-directory deletion/restoration. A separate
6,001-item exact-index test demonstrates removal of the smoke catalog bound.
These are correctness/resource checks, not H&M quality measurements.

SQL aggregates and ItemCF match the original raw-row reference within 1e-12;
shared float32 features use 1e-6 relative/absolute tolerance. Cached WALS
folding-in uses 1e-6 relative/1e-7 absolute tolerance for floating-point summation
order. Mid-epoch two-epoch CPU checkpoint continuation compares final weights,
vectors, losses and selected epoch at exact zero tolerance. Reload checks
candidate IDs/order exactly, features at 1e-5/1e-6 and scores at 1e-6/1e-7, and
requires exact Top-12 parity. Tests add future partition rows and confirm that
earlier cutoff snapshots remain unchanged.

The existing regression suite covers interrupted/corrupt copies, failed stage
publication, recovery, incompatible identities, full/Test gates, temporal
boundaries, groups and metric denominators. New tests add real pilot dispatch,
complete-catalog search, actual epoch continuation, zero-candidate Validation,
CUDA rejection and notebook CLI parity. CUDA computation tests are explicitly
skipped locally, not passed. Dependency declarations and the lockfile are
unchanged; unrelated old notebook lint findings are preserved.

Final verification commands and outcomes (the uv prefix above applies):

| Command/check | Exit | Result |
|---|---:|---|
| `pytest -q --basetemp=data/runs/colab/pilot_local_validation_20261102_01/pytest_final --junitxml=data/runs/colab/pilot_local_validation_20261102_01/pytest_final.xml` | 0 | 34 passed, 1 CUDA test skipped; 74.03 seconds |
| `ruff check src scripts tests notebooks/01_Colab_Training.ipynb` | 0 | Passed |
| `ruff format --check src scripts tests notebooks/01_Colab_Training.ipynb` | 0 | 50 files formatted |
| `pyright` | 0 | Zero errors, warnings and information messages |
| `nbformat.validate` and `compile` on notebook code cells | 0 | 9 cells validated, none executed |
| `ruff check .` | 1 | 122 pre-existing notebook findings, unchanged |
| `ruff format --check .` | 1 | 8 existing files: 6 old notebooks and 2 user task prompts, unchanged |
| `git diff --check` | 0 | Passed |
| `python scripts/run_pipeline.py plan --profile pilot --persistent-root data/runs/colab/pilot_local_validation_20261102_01/unused_persistence` | 0 | Implemented, not cloud verified; no training |
| Four-test native-library isolation regression | 0 | 4 passed in 2.50 seconds |

The first two full-suite attempts terminated with exits 134 and 139 during a
PyTorch LayerNorm operation. Isolating index computation alone was insufficient:
shared recall imports still loaded FAISS/ALS before the Torch tests. Native
imports are now deferred until actual index/fit operations, and the new index
test uses its own process. The four-test ordering regression and complete suite
then passed. No checks were weakened or disabled. Both failed full-suite logs
are retained as `pytest.log` and `pytest_isolated.log`; final output is
`pytest_final.log`. Earlier fixture issues included a SQL placeholder count,
assuming an old smoke catalog miss remained a miss in the full catalog,
matching field names instead of customer values in the privacy test, and
copying test logs into a deliberately fresh test attempt. These were corrected.

Measured synthetic pilot results:

- 4,187 raw transaction lines, 24 global users, 42 catalog items; 12 base users.
- WALS: 8 dimensions, 2 actual iterations, 480 nonzero confidence entries.
- Two-Tower: 96 selected targets, 192 processed across 2 epochs, 12 actual
  optimizer steps. Interrupted continuation matched uninterrupted weights,
  vectors, losses and selected epoch exactly on CPU.
- F1 and Validation: 24 queries each; 960 and 984 native feature rows.
- LambdaMART: 960 training rows in 24 groups, 96 positives; 10 diagnostic
  evaluation iterations and 9 retained model rounds, with 50 tree splits.
- Fresh-process reload: 984 candidate rows, 24 queries, exact Top-12 and zero
  maximum score difference. It reads no labels and refits nothing.
- All 24 queries survive the separate empty-pool Validation fixture; MAP@12,
  NDCG@12 and Recall@12 are zero, with no exclusions for missing recall.
- The 18-stage pilot observed 0.754 GiB maximum sampled process-tree RSS and
  13.61 seconds summed worker time on this tiny fixture. The monitor includes
  its test supervisor, can double-count shared pages and can miss short peaks.
  These measurements are not real pilot resource estimates.

| Channel | F1 rows | Validation rows |
|---|---:|---:|
| Trend popularity | 960 | 984 |
| Repurchase | 528 | 552 |
| Product-code sibling | 288 | 288 |
| ItemCF | 960 | 984 |
| WALS | 960 | 960 |
| Two-Tower | 960 | 984 |

The existing staged smoke versus original smoke parity test also passed,
including original four-fold execution, features, sampling, predictions and
metrics. No real H&M smoke/pilot training was rerun in this task.

## Compatibility and limits

Use a new release and a new run ID. Existing smoke artifacts remain unchanged
and are not relabelled as pilot stages. Source, config, input, schema, cutoff,
device and numerical-runtime identities reject incompatible reuse. Frozen
source embeddings, mappings, manifests, failure logs and indexes remain read-only.

Full-only acceptance is outside this milestone. Full execution stays gated;
no final Test labels are evaluated. Pilot has no placeholder model or missing
stage backend, but real-data feasibility is not established. Full-global ItemCF,
K=128 fitting, content adaptation and input hashing can be costly. Memory/disk
checks stop rather than silently sample. Sampled RSS can miss brief peaks and
is not an OS-enforced cap. GPU wheel installation, CUDA training/continuation,
actual Drive FUSE failures and new-runtime recovery require Colab verification.

## Handoff

Implementation files are the new `pilot/{data,snapshot,vectors,models,towers,
features,ranking,worker,testing}.py` modules and their package initializer;
shared `features.py`, `recall.py`, `learned.py`; Colab
`profiles.py`, `runner.py`, `worker.py`, `state.py`, `checkpoints.py`, `cli.py`,
`release.py` and new `review.py`; `scripts/probe_device.py` and
`scripts/build_release.py`; the thin training notebook; two new pilot test
modules and extended migration identity tests. The current pilot profile caps
were reused. The existing runbook/report now distinguish historical migration
status from this implementation. Other pre-existing dirty files were preserved.


See `colab_pilot_runbook.md` and its generated resolved companion under
`data/releases/` for exact upload/configuration/execution/recovery instructions.
The ZIP, checksum, manifest and standalone notebook come from one build.
Content-addressed names are resolved outside this packaged report to avoid a
self-referential archive hash. The final verification JSON records clean
extraction, entry-point discovery and notebook equality. Stop after this pilot
implementation milestone; do not launch full, final Test or deployment.
