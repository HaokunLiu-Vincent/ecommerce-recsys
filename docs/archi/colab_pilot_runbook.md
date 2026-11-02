# Colab Pilot Runbook

## Contents

- [Scope and evidence](#scope-and-evidence)
- [Release and upload](#release-and-upload)
- [Notebook configuration](#notebook-configuration)
- [Execution order and commands](#execution-order-and-commands)
- [Status and recovery](#status-and-recovery)
- [Results and review bundle](#results-and-review-bundle)
- [Budgets and remaining risks](#budgets-and-remaining-risks)

## Scope and evidence

The pilot backend is executable. Its actual models are exercised on small,
authenticated synthetic CPU fixtures. This new release has **not run on real
Colab, CUDA or real H&M pilot data**. The user reports that the earlier release's
real Colab CPU smoke passed all six channels, WALS, Two-Tower, LambdaMART and
fresh-process inference. That historical success does not establish this
release's Linux environment, GPU execution or fresh-runtime Drive recovery.

Pilot is a bounded expansion experiment: F1 and a sampled Validation roster,
not a full Validation benchmark. Full, final Test, tuning and deployment remain
disabled. The smoke entry point and four-fold smoke regression remain supported.

## Release and upload

Build from the existing locked development environment:

```bash
uv run --locked python scripts/build_release.py
```

The build prints the release name and SHA256. It writes ZIP, `.sha256`,
`.manifest.json`, and a **resolved companion pilot runbook** under
`data/releases/`. The companion fills this document's `{{RELEASE_NAME}}` token
with the actual content-addressed filename. Use its copyable configuration;
the token avoids making an archive contain its own hash. The companion is
outside the archive, while this source runbook is inside it.

Upload the matching ZIP, SHA256 and manifest to this Drive folder:

```text
MyDrive/projects/e_commerce_recsys/releases/
```

Open the standalone `notebooks/01_Colab_Training.ipynb` through Colab's
**Upload notebook** action. It is also inside the matching ZIP; both copies
must come from the same build. Keep the resolved companion runbook for reference.
Do not reuse the earlier `hm_recsys_e0d2dd9e00e0daf1.zip` for this pilot.

Existing input files remain under:

```text
/content/drive/MyDrive/projects/e_commerce_recsys/data/
```

Only `original data/`, `text_features/`, and `image_features/` are copied to
`/content/e_commerce_recsys/data/`. The 12-file whitelist, checksums and immutable
source rules are unchanged. No old runs are copied as training inputs.

## Notebook configuration

First run this release with the notebook's smoke defaults and a fresh smoke ID.
Inspect completion, six channels and reload before starting pilot. Smoke uses
CPU; a CUDA probe is optional for smoke and mandatory inside a CUDA pilot.

For pilot, select a Colab GPU runtime if requesting CUDA, then replace the
configuration cell with these variable values (keep its path definitions and
validation code):

```python
RELEASE_NAME = "{{RELEASE_NAME}}"
RUN_PROFILE = "pilot"
RUN_ID = "colab_pilot_20261102_01"  # Choose an unused ID for a new experiment.
RUN_ACTION = "run"
ALLOW_PILOT_RUN = True
PILOT_DEVICE = "cuda"
RUN_CUDA_PROBE = True
ALLOW_FULL_RUN = False
RUN_FINAL_TEST = False
RECOVER_LOCK = False
```

CPU is a separate explicit experiment: set `PILOT_DEVICE = "cpu"`,
`RUN_CUDA_PROBE = False`, and use a new run ID. An unavailable requested CUDA
device raises; the training backend never falls back. Probe success alone is
not evidence that formal training used CUDA: inspect `models/twotower.json`
through `artifacts.json` for actual device, steps and peak CUDA memory.

## Execution order and commands

Run notebook sections 1–9 in order: configuration; Drive mount; verified code
extraction; locked environment and plan; input copying and preflight; optional
independent CUDA probe and execution through learned indexes; profile review;
remaining stages through Validation and fresh-process reload; final status.
The notebook invokes the selected profile's real stages in sections 6 and 8.
It contains no copied training implementation. Pilot never needs `ALLOW_FULL_RUN`.

Equivalent terminal commands, after the notebook prepares the environment:

```bash
cd /content/e_commerce_recsys
PYTHON=.venv/bin/python
PERSIST=/content/drive/MyDrive/projects/e_commerce_recsys/data/runs/colab
PILOT_ID=colab_pilot_20261102_01

$PYTHON scripts/run_pipeline.py stage-inputs \
  --source-root /content/drive/MyDrive/projects/e_commerce_recsys/data \
  --data-root /content/e_commerce_recsys/data
$PYTHON scripts/run_pipeline.py plan --profile pilot --device cuda \
  --run-id "$PILOT_ID" --persistent-root "$PERSIST"
$PYTHON scripts/run_pipeline.py preflight --profile pilot --device cuda \
  --allow-pilot --run-id "$PILOT_ID" --mount-root /content/drive \
  --persistent-root "$PERSIST"
PROBE_ID=$($PYTHON -c 'import uuid; print(uuid.uuid4().hex)')
$PYTHON scripts/probe_device.py --device cuda \
  --output "data/runs/colab/probe_$PROBE_ID"
$PYTHON scripts/run_pipeline.py run --profile pilot --device cuda \
  --allow-pilot --run-id "$PILOT_ID" --mount-root /content/drive \
  --persistent-root "$PERSIST" --until reload
```

The probe command includes the interpreter, script path, `--device` and a fresh
`--output`; it does not accept `--run-id`. The notebook generates UUID output
paths automatically. Do not point the probe at an existing run directory.

Pilot's 18 boundaries are `preflight`, `canonical`, `content`, `rosters`,
`device_probe`, `wals`, `towers`, `index_towers`, `encode`, `preprocessing`,
`stats_F1`, `features_F1`, `stats_Validation`, `features_Validation`, `ranker`,
`reload_encode`, `reload_features`, `reload`. There are no F2–F4 dependencies.
Each uses a new process; Torch, FAISS and LightGBM lifetimes remain separate.

## Status and recovery

```bash
$PYTHON scripts/run_pipeline.py status --profile pilot --device cuda \
  --run-id "$PILOT_ID" --mount-root /content/drive --persistent-root "$PERSIST"
$PYTHON scripts/run_pipeline.py resume --profile pilot --device cuda \
  --allow-pilot --run-id "$PILOT_ID" --mount-root /content/drive \
  --persistent-root "$PERSIST" --until reload
```

For a new runtime, open the same notebook and release, retain the exact profile,
device and run ID, set `RUN_ACTION = "resume"`, and run from the top. Verified
inputs and completed stage deltas are restored. Monthly Parquet paths are bound
at runtime, so they do not retain the old machine's absolute paths. Source,
code, config, schema, cutoffs, device and numerical-runtime mismatches reject
reuse. Changed releases require a new run ID.

Only after confirming the previous runtime has stopped, set
`RECOVER_LOCK = True` once, or add `--recover-lock`. Do not steal a live lease.
Directory existence is never completion: manifests, dependency signatures and
file hashes are checked. Failed WALS, statistics, encoding, feature or ranker
stages restart from their preceding committed boundary. A locally complete
stage with failed upload can finish persistence without retraining.

Two-Tower resumes at a saved complete batch with model, optimizer, scheduler,
RNG, order identity, epoch, offset and best-epoch state. Float32 uses no AMP.
Same-runtime synthetic CPU continuation has been tested; CUDA continuation
and actual Drive disconnection/restore remain unverified. A runtime/library
mismatch is rejected, not silently treated as numerically equivalent.

## Results and review bundle

Local work and logs stay beneath:
`/content/e_commerce_recsys/data/runs/colab/<run_id>/`.
Verified stage deltas, checkpoints and final summaries are persisted beneath:
`/content/drive/MyDrive/projects/e_commerce_recsys/data/runs/colab/<run_id>/`.

Inspect `status.json`, `run_summary.json`, `artifacts.json` and
`review_manifest.json`. `artifacts.json` resolves logical paths into immutable
stage-attempt files; it is not a replacement for completion manifests. The
logical `resource_usage.json` records sampled process-tree RSS and stage timing;
`ranker_memory_estimate.json` records the independent pre-fit estimate. Each
candidate partition records row/query counts, channel recall, native-union
recall, zero-candidate/zero-hit queries and active-pool coverage.

Successful pilot execution publishes a small `pilot_review_<hash>.zip` and its
checksum record. To collect partial diagnostics after a stage failure:

```bash
$PYTHON scripts/run_pipeline.py review --profile pilot --device cuda \
  --run-id "$PILOT_ID" --mount-root /content/drive --persistent-root "$PERSIST"
```

Download the ZIP named by `review_manifest.json` from the persistent run folder.
It includes aggregate config/identities/counts, status, training and resource
measurements, MAP@12/NDCG@12, per-channel diagnostics, reload checks and failure
types. It excludes raw customer IDs, histories, feature tables and arbitrary
logs. Detailed private stage logs remain available on Drive.

## Budgets and remaining risks

The production pilot caps are unchanged: at most 5,000 stable-hash pre-Aug-5
users for base gradients; complete legal catalog; global legal history for
popularity, active pool, ItemCF and dynamic features; F1/Validation at most
1,000 queries each; WALS 64 dimensions and 3 iterations; Two-Tower batch 512,
20,000 targets, 1 epoch and 100 steps; LambdaMART at most 100 rounds. Actual
steps may be lower. The 512 batch changes in-batch negatives relative to the
proposal's 2048 and is explicitly recorded. No activity-minimum cohort filter,
5,000-item cap, label injection or removal of recall-failure queries is applied.

WALS reports its full globally fixed vocabulary and the subset with cohort
training observations; unknown columns do not receive invented factors.
Two-Tower encodes the full legal catalog. K=128 preprocessing fits all legal
pre-base valid-image rows; no hidden sampling. The one LambdaMART fit uses
Train-only native 1:20 hard/random sampling and unsampled native Validation.
Candidate NDCG is an early-stopping diagnostic, not end-to-end MAP@12.

DuckDB uses 2 threads, a 1 GiB memory limit and 8 GiB local spill. Feature
batches contain at most 32 whole queries; histories over 100,000 lines per
batch reject rather than truncate. Exact ItemCF rejects estimates over one
billion directed pairs or insufficient disk, rather than changing its formula.
K-Means and LambdaMART have separate memory checks. Outer RSS monitoring
reserves available RAM and stops on resource limits. Sampling every 0.2 seconds
is not an OS hard cap and can miss peaks. Full catalog/global history can
still be expensive despite a small training cohort.

No Colab GPU type, memory size, elapsed time or affordability is guaranteed.
A resource rejection is a blocked run, not permission to shrink features or
change algorithms. New-release Linux/CUDA, real pilot and actual fresh-runtime
Drive recovery are **not run**. Full and final Test remain closed.
