# Colab Full Training and Validation Runbook

The current primary workflow is [GitHub checkout](github_colab_runbook.md):
open the checked-in Notebook from GitHub, pin its actual Commit, and stage only
validated Drive inputs. No source archive upload is required. Existing archive
run identities and the archive setup examples below are historical compatibility
instructions; they do not apply to new Git runs. Keep the original Commit and
Git identity for new-run recovery. Model, resource and progress details below
remain applicable unless explicitly identified as historical.


## Contents

- [Scope and evidence](#scope-and-evidence)
- [Upload and configuration](#upload-and-configuration)
- [Execution and inspection](#execution-and-inspection)
- [Recovery and compatibility](#recovery-and-compatibility)
- [Resource boundaries](#resource-boundaries)
- [Outputs and verification limits](#outputs-and-verification-limits)

## Scope and evidence

This release executes full training, F1–F4 jointly through one LambdaMART,
complete native Validation, and fresh-process inference parity. It reuses the
pilot backend and its six quotas (60/30/30/50/50/50), frozen base models,
dynamic histories, shared features and verified stage persistence.

The user reports successful real Colab CUDA pilot on the preceding implementation.
That evidence does not establish this release's real full-data capacity or CUDA
continuation. Local acceptance uses small synthetic input data with unchanged
full compute settings and explicit CPU selection. No real full training has
been executed locally. Final Test, tuning search and deployment remain disabled.

The current Notebook has only smoke/full and streams real worker counters; see
[the live progress runbook](colab_live_progress_runbook.md) for current setup,
unknown-total semantics, private log locations and failure recovery. Pilot and
benchmark interfaces remain separate backward-compatible CLI workflows.

## Upload and configuration

Build with the existing locked environment:

```bash
uv run --locked python scripts/build_release.py
```

The ZIP, `.sha256` and `.manifest.json` are written under `data/releases/`.
The command also writes a resolved `.full_runbook.md` with the exact release
name replacing `{{RELEASE_NAME}}` below. Keeping the resolved name outside the
archive avoids a self-referential hash. The standalone notebook and its packaged
copy have identical bytes.

Upload the three matching release files to:

```text
MyDrive/projects/e_commerce_recsys/releases/
```

Open `notebooks/01_Colab_Training.ipynb` independently through Colab's upload
interface. Use a fresh runtime for a new release: the verifier refuses to
replace an incompatible existing checkout. Do not rename a pilot run or reuse
its ID. Default settings remain smoke; validate the new release's smoke before
launching full. Then set these values in section 2, before execution cells:

```python
RELEASE_NAME = "{{RELEASE_NAME}}"
RUN_PROFILE = "full"
RUN_ID = "colab_full_live_20261103_01"  # Must be a new, unused full run ID.
RUN_ACTION = "run"
ALLOW_FULL_RUN = True
FULL_DEVICE = "cuda"
FULL_PERFORMANCE_BACKEND = "optimized"
RUN_CUDA_PROBE = True
RUN_FINAL_TEST = False
RECOVER_LOCK = False
```

Select a GPU runtime when requesting CUDA. Explicit CUDA requests fail when
unavailable; the backend does not fall back. Full CPU is an explicit alternative
configuration, never an automatic substitution. Do not infer formal training
uses CUDA merely from probe success: inspect the actual tower device report.

The fixed Drive input root is:
`/content/drive/MyDrive/projects/e_commerce_recsys/data/`.
Only the 12 files in `original data/`, `text_features/` and `image_features/`
are copied into `/content/e_commerce_recsys/data/`. Inputs, existing embeddings,
mappings, manifests, failures and original indexes remain read-only. All fitting
and large feature generation occur on local `/content`, never on Drive.

## Execution and inspection

Run notebook sections in order. They mount Drive, verify/extract the release,
prepare the locked environment, stage inputs, preflight, probe the device,
train frozen bases, generate F1–F4/Validation features, train one ranker and
reload in new processes. Torch, FAISS and LightGBM retain process isolation.
The kernel's existing CUDA PyTorch installation is not replaced.

Equivalent commands after notebook environment/input preparation:

```bash
cd /content/e_commerce_recsys
PYTHON=.venv/bin/python
PERSIST=/content/drive/MyDrive/projects/e_commerce_recsys/data/runs/colab
FULL_ID=colab_full_live_20261103_01

$PYTHON scripts/run_pipeline.py plan --profile full --device cuda \
  --run-id "$FULL_ID" --persistent-root "$PERSIST"
$PYTHON scripts/run_pipeline.py preflight --profile full --device cuda \
  --allow-full --run-id "$FULL_ID" --mount-root /content/drive \
  --persistent-root "$PERSIST"
$PYTHON scripts/run_pipeline.py run --profile full --device cuda \
  --allow-full --run-id "$FULL_ID" --mount-root /content/drive \
  --persistent-root "$PERSIST" --until reload
$PYTHON scripts/run_pipeline.py status --profile full --device cuda \
  --run-id "$FULL_ID" --mount-root /content/drive --persistent-root "$PERSIST"
```

`plan` does not train or certify capacity. Source-derived estimates are computed
once canonical counts and query rosters exist. Full's 25 stages are preflight,
canonical, content, rosters, capacity, device_probe, WALS, towers, learned tower
index, query encoding, preprocessing, each cutoff's statistics/features for
F1/F2/F3/F4/Validation, one ranker, reload encoding/features/scoring. Use
`--until features_F2` or another actual boundary to stop after verified publication.

Full null semantics are explicit:

| Setting | Full meaning |
|---|---|
| `training_subset: null` | All legal base-history users, no pilot cohort restriction |
| `max_queries_per_fold: null` | Every catalog-matched purchase-active query in each fold and Validation |
| `tt_max_targets: null` | Every unique legal `(user,date,item)` target, not a pilot-sized list |
| `tt_max_steps: null` | No step cap; run the declared ten epochs |
| Catalog/history/cluster row caps: null | No smoke/pilot population caps or activity-minimum filter |

WALS retains 64 dimensions and 15 iterations; Two-Tower retains batch 512,
10 epochs, existing LogQ/multi-positive loss and daily internal Recall@50
selection; K-Means retains K=128; LambdaMART retains at most 1500 rounds and
candidate NDCG early stopping. The existing documented 512-versus-proposal-2048
batch setting remains unchanged. Train-only 1:20 hard/random negative sampling
is intentional protocol behavior, not a full-data cap. Validation is unsampled.

## Recovery and compatibility

After an interruption, keep the same release, profile, device and full run ID;
set `RUN_ACTION = "resume"`, then run the notebook from the top. Equivalent CLI:

```bash
$PYTHON scripts/run_pipeline.py resume --profile full --device cuda \
  --allow-full --run-id "$FULL_ID" --mount-root /content/drive \
  --persistent-root "$PERSIST" --until reload
```

If the previous runtime has stopped but left its lease, set `RECOVER_LOCK = True`
once, or add `--recover-lock`. Never recover a live writer's lease. Restaging and
restore verify file hashes and dependency identities, not just directories.
Code, input, config, schema, cutoffs, requested device, accelerator inventory and
numerical runtime mismatches reject reuse. A pilot cannot become full by renaming
its directory or changing `RUN_PROFILE` on its existing ID.

This release does not import stage artifacts from another run. Inputs are copied
through the existing verified whitelist; within-run upstream reuse goes through
stage manifests and dependencies. New releases/configurations require a new run.

Most incomplete stages restart at the previous committed boundary. Two-Tower
continues from a complete batch with optimizer, scheduler, RNG, epoch, offset,
ordered-target hash and best-model state. Target order is the existing stable
SHA256 order, repeated each epoch; LogQ is derived only from selected Train
unique events. Full keeps the latest complete checkpoint, including best-epoch
weights. A previous checkpoint belonging to this same identity is retired only
after its successor is verified and published. Pilot's retention is unchanged.
Completed stage artifacts and existing runs are never pruned.

## Resource boundaries

Capacity checks are independent: full WALS CSR/factors, ordered targets and
checkpoint working space, four-fold features, complete Validation, group
metadata, and the single actual-sampled-row LambdaMART fit. Full target rows,
query rosters, labels, query vectors, prediction outputs and shard manifests
are streamed or memory-mapped. No full transaction or target list is built in
Python. WALS constructs CSR from bounded Arrow batches.

LightGBM remains one in-memory fit. Its estimate includes Train plus complete
Validation numeric arrays, copies/binning/histograms and overhead; a memory map
is not an out-of-core guarantee. The estimate can reject a runtime even when
feature generation succeeded. Resource failures do not reduce users, targets,
features, candidates or label denominators. Repeated `init_model` fitting is not
used. There is no guarantee that any particular Colab runtime can fit full.

Disk planning reserves native-budget upper bounds, two local output copies,
uncompressed features/provenance, and scratch/checkpoint space. DuckDB retains
2 threads, a 1 GiB limit and 8 GiB spill. The outer adaptive RSS monitor remains
active. Resource estimates and sampled telemetry are evidence, not OS hard caps;
short peaks may be missed. Existing saved resource ceilings remain conservative
on resume. A larger or incompatible runtime may require a new run identity.

## Outputs and verification limits

Local run root:
`/content/e_commerce_recsys/data/runs/colab/<full_run_id>/`.
Verified persistent root:
`/content/drive/MyDrive/projects/e_commerce_recsys/data/runs/colab/<full_run_id>/`.

Inspect `status.json`, `run_summary.json`, `artifacts.json` and
`review_manifest.json`. Logical artifacts include per-stage resource reports,
`capacity/*.json`, `ranker_memory_estimate.json`, `ranker_groups.jsonl`, the
single `models/lambdamart.txt`, `full_results.json`, and
`predictions/reloaded_top12.parquet`. Use `artifacts.json` to resolve them into
immutable stage-attempt paths. It does not replace hash-verified completion.

Full Validation reports native MAP@12, NDCG@12 and Recall@12 over complete truth,
including zero-hit/zero-candidate queries. Candidate-list NDCG remains a separate
optimization diagnostic. The reload report checks feature/score tolerances and
exact Top-12 parity. A small aggregate-only `full_review_<hash>.zip` excludes raw
customer IDs, purchase histories and feature tables. Collect partial diagnostics:

```bash
$PYTHON scripts/run_pipeline.py review --profile full --device cuda \
  --run-id "$FULL_ID" --mount-root /content/drive --persistent-root "$PERSIST"
```

See [implementation report](colab_full_implementation_report.md) for actual local
commands, results and limitations. New-release real Colab full training,
full-data memory/disk/time, CUDA continuation and real full-run Drive recovery
are **not run**. Final Test is disabled in every profile and action.
