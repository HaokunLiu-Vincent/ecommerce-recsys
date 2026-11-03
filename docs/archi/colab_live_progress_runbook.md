# Colab Smoke and Full Runs with Live Progress

The current primary workflow is [GitHub checkout](github_colab_runbook.md):
open the checked-in Notebook from GitHub, pin its actual Commit, and stage only
validated Drive inputs. No source archive upload is required. Existing archive
run identities and the archive setup examples below are historical compatibility
instructions; they do not apply to new Git runs. Keep the original Commit and
Git identity for new-run recovery. Model, resource and progress details below
remain applicable unless explicitly identified as historical.


## Contents

- [Scope and release files](#scope-and-release-files)
- [Start with Smoke](#start-with-smoke)
- [Start Full explicitly](#start-full-explicitly)
- [Read the live display](#read-the-live-display)
- [Logs, results and failure handling](#logs-results-and-failure-handling)
- [Resume the same experiment](#resume-the-same-experiment)
- [Progress protocol and limits](#progress-protocol-and-limits)

## Scope and release files

Open `notebooks/01_Colab_Training.ipynb`. Its only operator modes are `smoke`
and `full`. Existing CLI capabilities and historical runs remain available;
standalone benchmarking stays in the separate compute runbook, outside this
training Notebook. Model code remains in `src/hm_recsys/`; the Notebook prepares
configuration/environment and invokes the actual shared pipeline.

Build with the existing locked development environment:

```bash
uv run --locked python scripts/build_release.py
```

Use the matching files emitted under `data/releases/`. Upload these three files
to `MyDrive/projects/e_commerce_recsys/releases/`:

- `{{RELEASE_NAME}}`
- Its `.sha256` sidecar.
- Its `.manifest.json` sidecar.

Upload the standalone `notebooks/01_Colab_Training.ipynb` through Colab's
**Upload notebook** action. It must match the copy in the ZIP. The resolved
release companion runbook fills `{{RELEASE_NAME}}` with the exact archive name.
Use a fresh runtime for a new release; do not overwrite a checkout or job that
is already running. Leave earlier models, checkpoints and source inputs intact.

The fixed Drive input root is:

```text
/content/drive/MyDrive/projects/e_commerce_recsys/data/
```

The local project is `/content/e_commerce_recsys/`. Only `original data/`,
`text_features/` and `image_features/` are copied into its local `data/`.
Embeddings, row maps, manifests, failure records and original content indexes
remain read-only. No source extraction or encoder download occurs. Large
training/intermediate writes stay in local `/content`, followed by verified
stage publication to Drive `data/runs/colab/<run_id>/`.

## Start with Smoke

Select these values in Notebook section 1, retaining the path definitions and
validation code already in that cell:

```python
RELEASE_NAME = "{{RELEASE_NAME}}"
RUN_PROFILE = "smoke"
RUN_ID = "colab_smoke_live_20261103_01"
RUN_ACTION = "run"
ALLOW_FULL_RUN = False
RUN_CUDA_PROBE = False
RUN_FINAL_TEST = False
RECOVER_LOCK = False
SHOW_STAGE_TABLE = True
PROGRESS_POLL_SECONDS = 0.1
HEARTBEAT_SECONDS = 15
```

Choose an unused ID. Smoke always uses CPU and the existing bounded limits.
It performs real small WALS, Two-Tower and LambdaMART fits and fresh-process
reload, but its sampled Validation is not a full H&M benchmark.

Run the Notebook from the top. The sections are sequential:

1. Validate configuration and mount Drive.
2. Check archive/sidecar hashes and verify source extraction.
3. Prepare the locked project Python environment and display actual versions,
   CUDA availability, RAM and disk. Installation stdout streams live.
4. Stage the 12 required source files with real file/byte and verification status.
5. Display the exact selected CLI plan and applicable resource gates.
6. Optionally run the independent CUDA probe, then invoke one real pipeline
   command through Validation and reload.
7. Check persisted stage status and display aggregate results and locations.

Each command's argv and log location are shown before execution. The live
worker stream and stage table start as events arrive; there is no outer timer
pretending that a long training command is a completed percentage.

## Start Full explicitly

After inspecting this release's Smoke evidence and accepting the resource
requirements, choose an appropriate Colab runtime and set:

```python
RELEASE_NAME = "{{RELEASE_NAME}}"
RUN_PROFILE = "full"
RUN_ID = "colab_full_live_20261103_01"
RUN_ACTION = "run"
ALLOW_FULL_RUN = True
FULL_DEVICE = "cuda"
FULL_PERFORMANCE_BACKEND = "optimized"
RUN_CUDA_PROBE = True
RUN_FINAL_TEST = False
RECOVER_LOCK = False
SHOW_STAGE_TABLE = True
PROGRESS_POLL_SECONDS = 0.1
HEARTBEAT_SECONDS = 15
```

Use a new ID for this code/config identity. Full executes F1, F2, F3, F4 and
complete native Validation, jointly trains one LambdaMART and verifies
new-process inference. The six quotas, past-only histories, full ground truth,
zero-hit queries, Train-only negative sampling and model settings remain fixed.
`null` population limits mean uncapped legal Full data, not fallback to a smaller
profile. The existing Full `capacity` stage runs before expensive model fitting;
independent WALS, preprocessing, feature/disk and single-ranker checks can stop
the job. No runtime/GPU size or completion time is promised.

`FULL_DEVICE="cpu"` is an explicit separate experiment; a requested unavailable
CUDA device fails instead of changing the model device silently. The existing
`FULL_PERFORMANCE_BACKEND` value is recorded in identity and must remain the
same on resume. The optional CUDA probe uses an independent fresh output path;
a successful probe is not proof of complete training or cloud recovery.

Equivalent execution after the Notebook prepares/stages the environment:

```bash
cd /content/e_commerce_recsys
PYTHON=.venv/bin/python
PERSIST=/content/drive/MyDrive/projects/e_commerce_recsys/data/runs/colab
FULL_ID=colab_full_live_20261103_01

$PYTHON scripts/run_pipeline.py plan --profile full --device cuda \
  --performance optimized --run-id "$FULL_ID" \
  --persistent-root "$PERSIST" --mount-root /content/drive
$PYTHON scripts/run_pipeline.py run --profile full --device cuda --allow-full \
  --performance optimized --run-id "$FULL_ID" \
  --persistent-root "$PERSIST" --mount-root /content/drive --until reload
```

These are manual cloud operations. The local Mac must not start real Full
training or full-data preprocessing.

## Read the live display

The table retains the latest real task for each stage. Its status applies to
that task; stage completion requires validated/persisted completion evidence.
The compact text stream remains available if the IPython display is disabled
or unavailable. Use `SHOW_STAGE_TABLE=False` for text-only operation.

| Display | Meaning |
|---|---|
| Stage and task | Actual worker/runner phase, such as WALS fit, item encoding, feature generation or persistence |
| Completed / total and unit | Measured iterations, batches, targets, queries, rows, seeds, files or bytes; never interchangeable |
| Percentage | Only a grounded completed/total denominator; missing totals show indeterminate |
| Elapsed, throughput and estimated ETA | Timing for that task; ETA is absent when the denominator/rate is not meaningful |
| Epoch, batch and loss | Actual Two-Tower training updates; internal selection is separate from training |
| GPU allocated/reserved | Actual Torch samples when CUDA is used; low allocation alone says nothing about utilization |
| Checkpoint status | Local save or verified remote publication; an upload start is not durable completion |
| RESTORED / SKIPPED (verified) | Existing compatible committed work, not training performed now |
| Heartbeat / last event time | The process remains supervised while a library/SQL call has no finer counter; no fabricated percentage |

Fine-grained progress comes from the actual loops/callbacks. WALS shows ALS
iterations and sparse shape/nnz. Two-Tower shows epochs, consumed examples,
steps/loss, catalog encoding and complete internal Recall@50. ItemCF reports
actual seed-block progress; estimated directed pair work is labelled as an
estimate. Each F1–F4/Validation feature stage reports query groups, native rows
and shards. LambdaMART reports actual training rows/groups/positive rows, feature
count and boosting iterations; early stopping may finish below the configured
maximum without implying failure. Candidate-list NDCG is diagnostic. Native
MAP@12/NDCG@12/Recall@12 use full independent eligible truth.

Some SQL conversion and clustering calls do not expose defensible within-call
row/iteration counts. They show named phases, elapsed time and a heartbeat,
then their measured output counts. A phase with unknown total never becomes an
invented 99% bar based on file size. Exact finer coverage and tested limitations
are listed in the implementation report.

## Logs, results and failure handling

Notebook command logs are appended locally at:

```text
/content/e_commerce_recsys/data/runs/colab/<run_id>/notebook_logs/
```

`pipeline.log` contains the pipeline command's live stream. The coordinator
keeps local `live_progress.jsonl`; each worker writes its own
`progress/<stage>.jsonl` inside its stage attempt. Stage attempts keep
their detailed inner logs, resource reports and progress evidence beneath the
same local run root. `artifacts.json` resolves committed logical artifacts into
immutable attempt paths. Failed attempts remain available for inspection;
they are not published as successful stages.

On failure, the shared helper shows the nonzero exit and relevant bounded inner
log tail, then raises. The Notebook stops; later result cells do not fabricate
completion. Inspect the failing phase, last real counter, resource/capacity
report and complete inner log. Do not rerun with a reduced population while
still labelling the job Full. Do not clear a live lease or overwrite a checkpoint.

Completed output is verified under:

```text
/content/drive/MyDrive/projects/e_commerce_recsys/data/runs/colab/<run_id>/
```

The result cell displays aggregate population counts, actual steps/rounds,
feature count, native Validation metrics, zero-hit counts and reload tolerance
results. It intentionally omits customer IDs, selected-user lists, raw purchase
histories and per-query training order. Full also publishes a compact aggregate
review bundle referenced by `review_manifest.json`.

## Resume the same experiment

1. Confirm the old runtime/writer has stopped.
2. Open the identical release in the new runtime. Keep the original profile,
   RUN_ID, device and Full backend; set `RUN_ACTION="resume"`.
3. Run from the top. Input copies and committed stages are verified/restored;
   the stream labels these as restored rather than newly trained.
4. Use `RECOVER_LOCK=True` only for a confirmed stale lease, then return it to
   False after recovery. Do not take the lease from a live writer.

Equivalent Full command:

```bash
$PYTHON scripts/run_pipeline.py resume --profile full --device cuda --allow-full \
  --performance optimized --run-id "$FULL_ID" \
  --persistent-root "$PERSIST" --mount-root /content/drive --until reload
$PYTHON scripts/run_pipeline.py status --profile full --device cuda \
  --performance optimized --run-id "$FULL_ID" \
  --persistent-root "$PERSIST" --mount-root /content/drive
```

Model/optimizer/scheduler/RNG, epoch, consumed target offset and internal best
selection remain checkpointed. Prefetched-but-unconsumed rows are not progress.
A new runtime can lose all training since the last **verified remote** checkpoint,
including upload latency. Hard OOM, SIGKILL or runtime destruction cannot promise
a final save. Non-checkpointed failed stages restart from their preceding
committed boundary. Input/code/config/schema/cutoff/device/runtime mismatches
reject reuse; use a new experiment ID rather than editing identity files.

## Progress protocol and limits

Workers emit `HM_PROGRESS `-prefixed JSON records to flushed stdout and ordinary
JSON Lines to their local evidence files. Records have protocol `version=1`.
They include timestamp, run ID, profile,
stage, task, actual completed work, nullable total, unit, elapsed time, status,
message, optional aggregate metrics, nullable percentage/rate/ETA fields.
A total of zero has no invented ratio or throughput. The shared transport forwards nested
worker events to the Notebook immediately and flushes output. Detailed progress
is local during execution; normal validated stage publication handles durable
Drive evidence. Events do not write to mounted Drive on every batch.

Progress is rate limited. Counts reflect actual processed work at the emission
boundary; display cadence is not the training batch size. The stage table is
bounded and the text helper retains complete logs on disk rather than storing
unbounded histories in the Notebook. A progress event is never a substitute for
hash/identity checks, resource gates or atomic completion manifests.

No new-release real Colab/CUDA/full run or real replacement-runtime Drive
recovery is claimed by this runbook. Local child-process/real-model synthetic
checks and their exact outcomes belong in
[colab_progress_implementation_report.md](colab_progress_implementation_report.md).
Standalone compute diagnostics remain documented separately in
[compute_performance_runbook.md](compute_performance_runbook.md).
