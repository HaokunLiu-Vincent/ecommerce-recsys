# Compute Performance Runbook

## Contents

- [Scope and release](#scope-and-release)
- [Bounded Colab comparison](#bounded-colab-comparison)
- [New optimized training](#new-optimized-training)
- [Checkpoint and cache recovery](#checkpoint-and-cache-recovery)
- [Interpretation and limits](#interpretation-and-limits)

## Scope and release

The current training Notebook now exposes only smoke/full and live progress.
Its old benchmark/Pilot switches below describe the preceding release and must
not be pasted into the current client. Use the standalone capped benchmark CLI
in this document; use [the live runbook](colab_live_progress_runbook.md) for the
current Notebook. Historical CLI profiles remain supported independently.

This is a compute adapter for the existing pilot/full backend. The default
notebook remains smoke with reference compute, full opt-in off, and final Test
off. The local Mac must not preprocess or train real full data. Local acceptance
uses small authenticated synthetic inputs and the existing 5 GiB RSS ceiling.

Upload these three matching files from `data/releases/` to
`MyDrive/projects/e_commerce_recsys/releases/`:

- `{{RELEASE_NAME}}`
- Its matching `.sha256` file.
- Its matching `.manifest.json` file.

Independently upload `notebooks/01_Colab_Training.ipynb` to Colab. The packaged
and standalone notebooks have identical bytes. Use a fresh runtime for this
release. Leave existing full runs, models and checkpoints untouched; this
release cannot force-resume their optimizer state.

Only the existing three input directories are staged from
`/content/drive/MyDrive/projects/e_commerce_recsys/data/` into
`/content/e_commerce_recsys/data/`. Training and benchmarks run locally under
`/content`; embeddings and original content indexes remain read-only inputs.

The current audit-driven revision adds partition query caches, exact active-block
search, column feature gathers, incremental larger shards and direct numeric
ranker scratch. See [the current report](computing_opt_report.md) and the
[root code reference](../../README.md). Its new examples below use distinct run
IDs; do not force-resume the preceding compute release or an active full job.
Conservative native thread/RAM caps and model hyperparameters remain unchanged.

## Bounded Colab comparison

After checking this release's ordinary smoke, use the configuration below in
section 2. It prepares a new pilot population's canonical/content artifacts
and runs a diagnostic. It does not train WALS, ItemCF or a full epoch.

```python
RELEASE_NAME = "{{RELEASE_NAME}}"
RUN_PROFILE = "pilot"
RUN_ID = "computing_refactor_prepare_20261103_01"
RUN_ACTION = "run"
ALLOW_PILOT_RUN = True
ALLOW_FULL_RUN = False
PILOT_DEVICE = "cuda"
RUN_CUDA_PROBE = True
RUN_FINAL_TEST = False
PERFORMANCE_BACKEND = "reference"
RUN_BENCHMARK = True
BENCHMARK_ONLY = True
BENCHMARK_RUN_ID = "computing_refactor_ab_colab_20261103_01"
BENCHMARK_BATCHES = 100
BENCHMARK_WARMUP = 10
BENCHMARK_TRACE = False
```

Run in order. Section 6 prepares only through `content`, then invokes the capped
benchmark. Sections 8–9 stop at diagnostic reports when `BENCHMARK_ONLY` is true.
The diagnostic selects at most `100 * 512` stable Train events from that
prepared cohort, and calculates Q from exactly that selected diagnostic
population. It is not coverage of a full epoch, a full-population loss estimate,
or a quality benchmark. Both variants use that identical population and Q.

The three modes are `data`, `replay`, and `end_to_end`; `all` runs each pair.
Replay is model-only diagnosis on prepared batches, not a new training method.
Warmup uses disposable state; measured models/optimizers/RNG are reset. The
reference retains the original SQL/row-wise encoder; the optimized path uses
verified contexts and a budgeted static input bank.

Equivalent CLI after environment preparation and input staging:

```bash
cd /content/e_commerce_recsys
PYTHON=.venv/bin/python
PERSIST=/content/drive/MyDrive/projects/e_commerce_recsys/data/runs/colab

$PYTHON scripts/run_pipeline.py run --profile pilot --allow-pilot --device cuda \
  --performance reference --run-id computing_refactor_prepare_20261103_01 \
  --persistent-root "$PERSIST" --mount-root /content/drive --until content

$PYTHON scripts/benchmark_compute.py \
  --run-id computing_refactor_ab_colab_20261103_01 \
  --input-run data/runs/colab/computing_refactor_prepare_20261103_01 \
  --device cuda --batches 100 --warmup 10 --mode all --validation none \
  --persistent-root /content/drive/MyDrive/projects/e_commerce_recsys/data/runs/benchmark \
  --mount-root /content/drive
```

Use a new benchmark ID for every invocation. `--plan` parses and reports options
without running. Budget maxima are 100 measured batches and 10 warmup updates;
Mac allows only `--synthetic`, at most 4 batches and 1 warmup. A short profiler
window is separate from throughput timing: set `BENCHMARK_TRACE = True` or add
`--trace` (one warmup and two traced updates). It may fail if the runtime lacks
profiler support; do not reinterpret a trace failure as a successful measurement.

`--validation none` is the default bounded cloud diagnosis; its Validation
measurement is `not_run`. `--validation complete` explicitly processes every
internal daily query in the upstream cohort and the full catalog; this can be
expensive on full upstream data and is not capped by `--batches`. The local
synthetic comparison includes complete internal validation in both variants.
Formal training always keeps complete internal validation every epoch.

Reports live under `data/runs/benchmark/<benchmark_id>/`:
`performance_summary.json`, `parity_report.json`, `resource_usage.json`,
`compute_review.zip` and optional `bounded_trace.json`. The notebook verifies
and copies only the three compact report/bundle files to Drive
`data/runs/benchmark/<benchmark_id>/`. CLI users copy these three reports
explicitly; the benchmark does not copy its canonical/input/model working tree.
The Notebook and command above explicitly publish diagnostic checkpoints to
Drive, allowing its actual latency to be measured. Both variants use the same
publication work for comparison. Omitting `--persistent-root` uses local
`remote_simulation` directories; those timings are not Drive measurements.
Only this local simulation was exercised on the Mac.

## New optimized training

After inspecting parity and resource evidence, start a new optimized run. Keep
all model/sampling settings unchanged; performance settings are a separate,
identity-bound overlay in `configs/compute_optimized.json`.

```python
RUN_PROFILE = "full"
RUN_ID = "colab_full_computing_refactor_20261103_01"
RUN_ACTION = "run"
ALLOW_FULL_RUN = True
ALLOW_PILOT_RUN = False
FULL_DEVICE = "cuda"
RUN_CUDA_PROBE = True
PERFORMANCE_BACKEND = "optimized"
RUN_BENCHMARK = False
RUN_FINAL_TEST = False
```

Use a new unused ID; do not reuse the preparation or previous full ID. Full
executes the existing folds and one LambdaMART, with a new `prepare_towers`
stage before `towers`. The input cache is independently committed and restored.
A resource rejection stops the run; it never shrinks data or changes algorithms.

```bash
$PYTHON scripts/run_pipeline.py plan --profile full --device cuda \
  --performance optimized --run-id colab_full_computing_refactor_20261103_01 \
  --persistent-root "$PERSIST"
$PYTHON scripts/run_pipeline.py run --profile full --device cuda --allow-full \
  --performance optimized --run-id colab_full_computing_refactor_20261103_01 \
  --persistent-root "$PERSIST" --mount-root /content/drive --until reload
$PYTHON scripts/run_pipeline.py status --profile full --device cuda \
  --performance optimized --run-id colab_full_computing_refactor_20261103_01 \
  --persistent-root "$PERSIST" --mount-root /content/drive
```

Inspect logical `tower_performance.json`, `feature_performance/*.json`,
`ranker_performance.json` and resource estimates through `artifacts.json`.
Formal training still uses batch 512, existing FP32, optimizer/scheduler,
epoch count, full target order and per-epoch selection. No AMP, TF32, compile,
large batch, extra model or altered negative sampler is enabled.

## Checkpoint and cache recovery

The optimized starting cadence is:

- Local complete-batch save after 100 updates **or** 60 seconds.
- Verified remote publication after 300 seconds since the last successful one.
- Forced save/publication on first update, epoch selection and safe explicit stop.
  Final selection is published before normal model finalization.

These are configurable thresholds, checked at complete boundaries, not timing
guarantees. Runtime destruction loses all progress since the last **verified
remote** checkpoint, including upload latency. Hard OOM, SIGKILL and runtime
loss cannot promise a final save. An interruption during an incomplete update
keeps the previous complete checkpoint rather than publishing partial state.

Local and remote latest pointers are separate. A remote failure raises and
preserves the previous verified remote state. On the same runtime, a verified
newer local checkpoint can resume; a fresh runtime falls back to the verified
remote checkpoint. Full snapshots retain model, optimizer, scheduler, RNG,
consumed offset, epoch, selection and progress. Frozen input banks are rebuilt
from verified source/cache files and never embedded in model checkpoints.

Resume with the same release, device, overlay and run ID, using
`RUN_ACTION = "resume"` or the same CLI with `resume`. Restore still checks
code/config/input/schema/cutoff/runtime identities and all artifact hashes.
Recover a stale lease only after confirming its old writer stopped. Do not
alter identities or copy checkpoints from another experiment. Partial, corrupt
or stale input caches fail validation; they are never treated as completed by
path existence alone.

## Interpretation and limits

Read work counts before ratios. Separate shared input preparation, cold cache
construction, verified reopen, bank placement, data/transfer/model work,
internal validation, persistence, and whole-pipeline time. The latter remains
`not_run` in the bounded tower benchmark. Cold totals include the optimized
cache cost; ten-epoch amortization is arithmetic, not a measured full run.

CUDA kernel times use events, resolved at measurement boundaries. Host dispatch
latencies are labelled separately; overlapping host/device durations are not
additive percentages. The finite-loss check still synchronizes once per update.
Pinned staging is reused only after its copy event completes; no transfer/compute
overlap claim is made. GPU utilization, when available, is a labelled end-of-run
sample, not a training average. Missing measurements are unavailable/not run.

No real CUDA throughput, full-data prepack feasibility, Drive publication speed,
or new-runtime cloud recovery was established on the Mac. Inspect the detailed
[performance report](compute_performance_report.md) and run capped CUDA parity
before authorizing another full job. Final Test and deployment remain disabled.
