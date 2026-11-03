# Compute Performance Implementation Report

## Contents

- [Scope and preserved experiment](#scope-and-preserved-experiment)
- [Implementation](#implementation)
- [Measured evidence](#measured-evidence)
- [Verification](#verification)
- [Limits and delivery](#limits-and-delivery)

## Scope and preserved experiment

This work extends the current uncommitted full adapter, preserving the user's
Notebook run-ID handling and preceding changes. No older release was restored.
The locked environment, six channels/quotas, native candidates, single ranker,
49-column schema, target populations, cutoffs, complete ground truth and
Train-only negative sampling remain unchanged. Existing source inputs, running
jobs and old models/checkpoints were not edited. No real H&M training, full
prepack, final Test, cloud upload, Git initialization, commit or push ran locally.

The first experiment retains batch 512, FP32, loss/LogQ/multi-positive masks,
AdamW/learning rate/scheduler, target ordering, partial batches, history lines
and weights, and complete per-epoch internal selection. Static item inputs are
cached; learned embeddings, projections and pooled user representations are
recomputed with live gradients after every update.

## Implementation

- `compute/prepared.py`: one persistent user/day context, contiguous profiles,
  latest-20 raw-line category IDs/weights, ragged complete positives, ordered
  integer targets and separate internal daily rosters. A bounded sorted merge
  replaces repeated history SQL; source/model-independent identity, capacity
  estimates, file hashes and atomic manifests protect reuse. The new
  `prepare_towers` stage participates in existing publication/recovery. Its
  DuckDB JSON profile records the executed operator tree and scanned rows
  separately from exported transaction rows; these are not physical disk bytes.
- `compute/runtime.py`: ordered prefetch limited to two CPU batches; no SQL,
  CUDA or RNG in the producer. A raw item bank uses actual dimensions and
  separate catalog positions. CUDA placement reserves memory; otherwise bounded
  pinned host lookup keeps the model on CUDA. Static banks are outside model
  state. No non-blocking/overlap speed claim is made without measurement.
- `pilot/towers.py`: optimized batch consumption, complete internal input reuse,
  fresh learned vectors per epoch, bounded numerical progress storage and
  separate local/remote cadence. Last completed offsets, not prefetched offsets,
  drive continuation. Reference SQL/row assembly remains callable.
- `colab/checkpoints.py`: verified complete snapshots and independent latest
  pointers, local fsync, progress-pointer validation, timing boundaries, and
  preservation of the previous remote checkpoint on failed publication.
- `features.py`, `recall.py` and pilot adapters: persisted frozen cluster
  assignments, cutoff/bucket popularity ordering, bounded Repurchase/Sibling
  reuse, per-query affinity/count preparation and batched content dot products.
  Static category codes are gathered from one catalog table. Category values,
  masks, order and NaNs retain the existing schema.
- Benchmark CLI: data, model replay and end-to-end pairs with reset model/RNG,
  separate cold cache and warmup costs, short profiler window, CUDA events,
  host latencies, RSS, placement and environment reporting. The real-input
  diagnostic is Colab-only and capped; it cannot launch full training.
- Thin Notebook/config/CLI integration: explicit performance overlay, bounded
  benchmark-only gate and independent run ID. Legacy smoke remains unchanged;
  pilot/full support both reference and optimized execution with existing gates.

## Measured evidence

The initial pre-edit CPU fixture measured history/profile plus tensor assembly
at approximately 9–59 ms per batch and positive SQL at 4–5 ms. This supports
input preparation as a bottleneck on this fixture only. The user's observation
of about 0.4 GB CUDA memory is not evidence of a utilization percentage or VM type.

The final run is `data/runs/benchmark/compute_ab_20261103_04/` on macOS
arm64, Python 3.12.16, locked PyTorch 2.15.0, CPU, two native threads and FP32.
Both variants consumed 1,584 identical events in batches 512/512/512/48; replay
and end-to-end each made four updates. Each warmup used one disposable batch.
Complete synthetic internal validation used 72 daily queries and 170 catalog
items in both variants. Both performed two complete local saves and two verified
local-filesystem publications; no Drive latency was measured. Four batches are
adequate for bounded correctness diagnosis, not stable performance estimation.

| Measured scope | Reference (s) | Optimized (s) | Interpretation |
|---|---:|---:|---|
| Input-only, prepared steady state | 0.13255 | 0.00328 | Excludes one-time cache; no model updates |
| Model-only replay, four updates | 0.01878 | 0.01995 | No demonstrated model-compute gain |
| Training loop, including periodic checkpoint | 0.16307 | 0.03514 | 9,714 vs 45,083 examples/s on this fixture |
| Complete internal validation/catalog | 0.06501 | 0.01220 | Same 72 queries; Recall@50 = 0.84027778 |
| Warm train + validation + final publication | 0.24072 | 0.06066 | Same work; 3.97 measured ratio |
| Cold total, including cache/open/bank | 0.24072 | 0.21485 | Only 1.12 measured ratio |
| Cold total plus common upstream preparation | 0.42628 | 0.40040 | Includes 0.18555 s common preparation |

Cold context construction cost 0.10918 s, hash-verified reopen 0.03949 s,
and end-to-end item-bank setup 0.00551 s. Dividing cache construction by ten
configured epochs gives 0.01092 s/epoch arithmetically; ten-epoch throughput was
not benchmarked. The prepared cache has 744 contexts, 1,584 targets and 510,426
file bytes. The bank uses 28,560 bytes and is absent from model checkpoints.
The executed history plan exported 1,892 rows; DuckDB reported 16,000 Parquet
scan rows plus 744 context scan rows. Operator scan counts are not unique
transaction counts, physical disk bytes, or proof of scanning every source row.

| End-to-end component | Reference (ms) | Optimized (ms) |
|---|---:|---:|
| Target read / prepared read and wait | 2.234 | 1.094 |
| Positive-set SQL | 12.632 | 0 (prepared) |
| Positive mask construction | 18.944 | Included in prepared wait |
| History SQL, including internal input queries | 57.517 | 0 (prepared) |
| Profile lookup, including internal queries | 8.149 | 0 (prepared) |
| CPU user assembly including training SQL | 96.239 | Included in prepared wait |
| Static item assembly | 0.889 | Included in bank setup/lookup |
| User + item forward host time | 5.841 | 6.451 |
| Loss / backward / optimizer host time | 1.873 / 8.421 / 3.998 | 1.469 / 8.909 / 3.906 |
| Serialization and local writing | 7.887 | 7.294 |
| Local checkpoint hashing | 2.406 | 2.623 |
| Copy with source hash | 3.765 | 4.017 |
| Remote simulation verify/file publication | 2.799 | 3.264 |
| Remote simulation pointer publication | 0.785 | 0.876 |
| CUDA transfer/kernel/utilization | not_run | not_run |

Some boundaries are nested; the component rows must not be summed. CPU
`host_to_device` spans are dispatch/no-op overhead, not GPU transfer measurements.
The data-only path diagnoses repeated input SQL and Python assembly as the
fixture's main overhead. It does not establish the Colab bottleneck. The
optimized model-only replay was slightly slower, so there is no claim that the
network itself became faster. A separate three-update profiler window took
0.03554 s and was excluded from all throughput comparisons.

End-to-end host step p50/p95 were 46.30/47.72 ms reference and 5.64/9.53 ms
optimized, excluding checkpoint time after the step. Sampled per-process RSS
was 419.59/421.70 MiB. The independent 0.2-second process-tree monitor observed
658.52 MiB, below the unchanged 5 GiB abort ceiling. The full regression's
largest monitored stage reached 889.98 MiB. Sampled RSS can miss short peaks
and can double-count shared pages; it is not an OS-enforced memory limit.

The shared-feature comparison used 48 F1/Validation queries and 3,015 native
rows. Rule-channel retrieval plus feature assembly took 0.17089 s reference
and 0.06157 s optimized. This is a warm frozen-cluster comparison, excluding
model training, learned retrieval and cluster-cache construction; it is not
whole-pipeline speedup. All 49 columns matched within 1e-5 relative/1e-6 absolute
tolerance (maximum difference 1.1920929e-7). Discrete values, masks, candidate
order/provenance, training samples, Top-12 and complete-truth metrics were exact.

Every local measurement uses synthetic inputs. GPU performance, real full
prepack and real Drive I/O are not run. No fixed multiplier or full-data runtime
is promised.

## Verification

Final bounded regression: **48 passed, 2 CUDA skips in 160.84 seconds**, exit 0.
The optimized full-dispatch fixture executed F1–F4/Validation, real WALS,
40 Two-Tower updates over ten epochs (15,840 consumed targets), one LambdaMART
and fresh-process reload. Validation retained 24 queries including a zero-hit
query. It restored the prepared stage into a replacement local filesystem;
this is not evidence of new-runtime Colab/Drive recovery. Reference and optimized
interrupted complete-batch continuation tests preserve exact CPU state.

All 1,584 prepared inputs, masks and Q values matched exactly. The explicit
three-update gradient comparison found zero maximum loss, gradient and updated
parameter differences; all four final benchmark losses also matched exactly.
The learned projection changed, so static caching did not freeze learning.

Commands used the existing `uv` environment with
`UV_CACHE_DIR=/tmp/hm-smoke-uv-cache`; native thread limits were two for tests
and benchmarks. Exact argv/status records are in the evidence directory.

| Actual command/check | Exit | Result |
|---|---:|---|
| `uv run --locked pytest -q --basetemp=…/pytest_release --junitxml=…/pytest_release.xml` | 0 | 48 passed, 2 skipped |
| `uv run --locked ruff check src scripts tests notebooks/01_Colab_Training.ipynb` | 0 | Passed |
| `uv run --locked ruff format --check src scripts tests notebooks/01_Colab_Training.ipynb` | 0 | 62 files formatted |
| `uv run --locked pyright` | 0 | 0 errors, 0 warnings |
| `uv run --locked ruff check .` | 1 | Same 122 pre-existing legacy notebook lint errors |
| `uv run --locked ruff format --check .` | 1 | Same eight unrelated old notebooks need formatting |
| `uv run --locked python -c …` (`nbformat.validate` and `compile`) | 0 | Nine new Notebook code cells; none executed |
| `uv run --locked python scripts/run_pipeline.py plan --profile full …` | 0 | Plan only; no training |
| `uv run --locked python scripts/benchmark_compute.py --run-id compute_ab_20261103_04 --synthetic --batches 4 --warmup 1 --validation complete --trace` | 0 | All three reference/optimized pairs, bounded CPU trace |
| `git diff --check` | 0 | Passed; no Git writes |

Evidence is under `data/runs/benchmark/compute_20261103_01/`: initial worktree
snapshot and pre-edit baseline, `static_checks.json`, final pytest log/XML,
`pytest_release/test_optimized_full_stage_rest0/feature_parity.json` and
`release_verification.json` (generated after building the final archive).
The latter records clean extraction, manifest/checksum verification, matching
standalone/packaged Notebook, CLI help contracts and packaged focused tests.
The build emits an identity-addressed ZIP, SHA256 and external manifest under
`data/releases/`, plus a resolved `.compute_runbook.md` naming that exact release.

The focused tests exercise exact context/target/mask/Q inputs, live gradients,
future-only perturbations, cache integrity and identity changes, interrupted
prefetch, complete-batch continuation, independent cadence with fake clocks,
remote upload/pointer failures, immutable stage restoration, candidate/feature/
Top-12 parity and actual generated Notebook/CLI argument contracts.

Source API references consulted on 2026-11-03 match the locked PyTorch 2.15
version: [CUDA timing semantics](https://docs.pytorch.org/docs/2.15/notes/cuda.html),
[profiler scheduling](https://docs.pytorch.org/docs/2.15/profiler.html), and
[pinned transfer lifetimes](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html).
CUDA events resolve at boundaries; host times are not presented as kernel times.
The separate traced run is excluded from throughput and its elapsed overhead is
reported. No full per-epoch CUDA timing or utilization average was measured.

## Limits and delivery

The bounded benchmark is not a full H&M epoch or quality benchmark. Real-data
canonical sorting/context build, device-bank placement, full internal validation,
Drive publication cost and cloud recovery still need user-run evidence. Cold
cache construction scans/sorts the legal histories of requested context users;
its small-fixture cost must not be extrapolated. Complete internal validation
is opt-in in the bounded diagnostic, always retained in formal training.

Full-data ItemCF, feature disk space and a single LightGBM fit retain independent
resource guards. A faster input path does not establish full-run feasibility.
The CPU continuation check is exact; no cross-device bitwise promise is made.
GPU utilization support currently captures an end-of-run sample, not a sustained
training utilization trace. Checkpoint serialization/local writing and remote
copy/source hashing are combined measurements where the I/O API combines them.

See [compute runbook](compute_performance_runbook.md) and its resolved release
companion for exact upload, capped benchmark, new optimized run, status and
resume instructions. Standalone Notebook, ZIP, SHA256 and manifest are rebuilt
together. New code/config requires a new run identity; old evidence is preserved.
