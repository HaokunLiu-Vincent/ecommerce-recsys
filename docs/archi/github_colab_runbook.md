# Root Colab Notebook: Pinned GitHub and Explicit Stage Runbook

This guide operates [`colab_training_val.ipynb`](../../colab_training_val.ipynb),
the repository's only active Notebook. It uses one canonical source tree and
one named stage per model/data cell. There is no ZIP-upload bootstrap, pilot
Notebook mode, hidden "run all prerequisites" model cell or automatic final
Test. Source-layout changes require a new run ID.

## Contents

- [1. Prepare and publish reviewed code](#1-prepare-and-publish-reviewed-code)
- [2. Check the immutable Drive inputs](#2-check-the-immutable-drive-inputs)
- [3. Open and configure smoke](#3-open-and-configure-smoke)
- [4. Execute the cells](#4-execute-the-cells)
- [5. Read progress and reports](#5-read-progress-and-reports)
- [6. Start an explicitly authorized full run](#6-start-an-explicitly-authorized-full-run)
- [7. Inspect, retry and recover](#7-inspect-retry-and-recover)
- [8. Final Test approval boundary](#8-final-test-approval-boundary)
- [9. Troubleshooting and unverified behavior](#9-troubleshooting-and-unverified-behavior)

## 1. Prepare and publish reviewed code

The local working tree is the source of truth. Review the
[implementation report](code_architecture_refactor_report.md),
[migration map](refactor_migration_map.md), tests and Git status before your own
commit/push. This implementation task does not publish code for you.

Run the existing quality checks and read-only repository audit:

```bash
uv sync --locked
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked pyright
uv run --locked pytest -q
uv run --locked python scripts/audit_git.py --help
```

Verify that private `data/`, local-only `docs/achive/`, credentials, populated
Notebook outputs, checkpoints and customer-level JSONL/CSV/Parquet are neither
staged nor tracked. Ignore rules do not remove old committed content. Do not
publish the archive, force-add ignored files, rewrite history automatically or
replace the current tree with an old release.

After your reviewed commit is actually published, record its full 40-character
SHA. Open the root Notebook from that same commit on GitHub, or use:

```text
https://colab.research.google.com/github/HaokunLiu-Vincent/ecommerce-recsys/blob/main/colab_training_val.ipynb
```

Replace `main` with the published SHA for an immutable Notebook link. A GitHub
page reflecting an older commit does not contain unpublished local changes.
No code ZIP, release checksum upload or old run folder is an input to this
workflow.

## 2. Check the immutable Drive inputs

The source data root is fixed:

```text
/content/drive/MyDrive/projects/e_commerce_recsys/data/
  original data/
    articles.csv
    customers.csv
    transactions_train.csv
  text_features/
    text_article_ids.parquet
    text_embeddings.npy
    text_features_manifest.json
    text_faiss.index
  image_features/
    image_article_ids.parquet
    image_embedding_failures.parquet
    image_embeddings.npy
    image_features_manifest.json
    image_faiss.index
```

Upload only missing legitimate source artifacts to these existing directories.
Do not regenerate embeddings, change row maps, replace manifests/indexes or
copy old models as input. The staging cell copies exactly the three directories
above into `/content/e_commerce_recsys/data/`; it does not copy all of Drive
`data/` or historical runs. Training and large intermediate files stay local.

The acceptance/preparation stage validates the real catalog/ID/content contract
and reports dimensions and missingness. Equal text/image row counts do not
permit joining vectors by row position. Ordinary missing modalities use masks;
blocking mapping/hash/dimension errors require diagnosis rather than extraction.

## 3. Open and configure smoke

Keep default smoke first on a new code/runtime identity. Replace the SHA example
with the real published commit; do not copy a placeholder as a valid hash.

```python
GITHUB_REPO_URL = "https://github.com/HaokunLiu-Vincent/ecommerce-recsys.git"
GIT_REF = "main"
GIT_COMMIT = "YOUR_PUBLISHED_40_CHARACTER_COMMIT_SHA"
GITHUB_TOKEN_SECRET = None
RUN_PROFILE = "smoke"
RUN_ACTION = "run"
RUN_ID = "colab_smoke_modular_20261103_01"
DEVICE = "cpu"
ALLOW_FULL_RUN = False
RUN_FINAL_TEST = False
ALLOW_FINAL_TEST = False
LOCKED_RUN_SIGNATURE = None
```

Choose a new ID if that run already exists with another identity. IDs contain
letters, digits, underscores or hyphens, not paths. Smoke has explicit bounded
population/compute settings while calling the same six production algorithms.
Its metric is not a full-population Validation benchmark.

For a new run, `GIT_COMMIT=None` may resolve the ref once. Record the printed
resolved commit and set it explicitly before recovery. Existing checkout,
`resume` and `status` require the original explicit SHA. The bootstrap validates
URL/ref/SHA/destination before Git, checks a clean detached checkout and never
uses pull/reset/clean to overwrite source. Different code needs a fresh runtime
and new run identity.

### Private repository access

Opening a Notebook in your browser and cloning from its runtime are separate
permissions. If needed, create a Colab secret containing an appropriately scoped
GitHub token, enable that Notebook's access to it, and set only its **name**:

```python
GITHUB_TOKEN_SECRET = "HM_GITHUB_TOKEN"
```

Never paste the token into configuration, a URL, shell command, `.env`, output
or Git remote. The bootstrap retrieves the named secret only when explicitly
requested, uses an ephemeral askpass process/environment and cleans it up on
success or failure. Missing secret or denied clone access stops safely.

## 4. Execute the cells

Run the Notebook sequentially after reviewing the printed configuration and
resource plan. Once setup/prerequisites exist, a model cell can be run alone:
it invokes only its named stage and validates its prerequisites.

| Order | Stage/cell | What to inspect |
|---|---|---|
| Setup | Git checkout, mount, locked environment and input whitelist | Exact SHA, clean source, actual device/RAM/disk, staged input identity |
| 1 | `prepare` | Immutable split manifest, catalog/content acceptance, canonical counts, all F1–F4/Validation rosters and full truth; Test definitions only |
| 2 | `trend` | Build + validate, quota 60; cutoff counts/Recall, no supervised fit |
| 3 | `repurchase` | Raw-repeat history/decay, quota 30; shortages and Recall |
| 4 | `sibling` | Actual product-code variants, quota 30; seed exclusion and Recall |
| 5 | `itemcf` | Global legal sparse pair/edge work, indexes, quota 50 and Recall |
| 6 | `wals` | Allowed base CSR/fit, fixed factors, current-user folding-in, quota 50 and coverage |
| 7 | `twotower` | Bounded inputs, actual device, epoch/batch/loss, internal selection, catalog encoding, quota 50 and outer coverage |
| 8 | `fusion` | Unique native union, six source records, candidate distribution/Recall, RRF baseline and signed contributions |
| 9 | `features` | All native Train/Validation rows, 49-column schema/order, masks, resource/storage/throughput |
| 10 | `ranker` | Four folds jointly grouped into one fit, native positives, Train-only 1:20 sampling, best iteration and diagnostic |
| 11 | `validation` | Same-roster Trend/RRF/LambdaMART comparison and full-truth MAP/Recall/NDCG@12 |
| 12 | `reload` | Fresh-process model/feature/score/Top-12 checks and frozen run signature |
| Last code cell | `final_test` | **SKIPPED** with defaults; no Test future-label read/scoring |

The six outer target weeks are fixed and disjoint. Preparation may display the
Test date definition [2020-09-16, 2020-09-23), but does not build/read Test future
labels. WALS/Two-Tower base fitting and internal selection use their separate
permitted periods. Future positives are never injected or used to backfill a
channel. Validation keeps zero-candidate and zero-hit queries.

The actual per-cell command has this form:

```bash
python scripts/run_pipeline.py stage \
  --stage trend --stage-action run [COMMON_OPTIONS]
```

Common options are profile, run ID, local data root, persistent run-parent,
mounted root and the complete Git remote/ref/commit trio. Full adds explicit
device/backend and permission. The Notebook prints full argv before execution;
copy that command for diagnosis. It does not use `--until` to secretly execute
all earlier models. Missing prerequisites name the cells/stages to complete.

## 5. Read progress and reports

Structured events come from the actual workers through the CLI to the live
Notebook table and text stream. Depending on the operation, they show rows or
queries, completed/known total, elapsed time, throughput, epoch/batch/loss, RSS,
CUDA allocation/reservation and checkpoint publication state. Percent/ETA is
unknown when the work has no measured denominator. A heartbeat means no new
event, not a stalled job or fabricated completion.

Read aggregate reports at:

```text
/content/e_commerce_recsys/data/runs/colab/<RUN_ID>/reports/<stage>.json
```

Private logs are under `notebook_logs/` in that local run. Verified stage
artifacts/checkpoints persist beneath:

```text
/content/drive/MyDrive/projects/e_commerce_recsys/data/runs/colab/<RUN_ID>/
```

A complete manifest binds outputs, dependency/input/source/config identities
and hashes. Neither a directory nor a log saying "Running" proves completion.
A failed command remains failed and shows its real error/last-work state.
A status command means an inspection ran, not that an incomplete stage finished.

Interpret metrics at their own scope. Candidate-list LightGBM NDCG@12 is an
early-stopping diagnostic, not full-truth MAP@12. A channel's quota Recall is
not a same-K comparison against another quota. Raw GPU memory occupancy does
not establish utilization. Timing intervals can overlap; do not sum nested
CUDA and host timers as independent wall time.

## 6. Start an explicitly authorized full run

Only after inspecting smoke/environment/resources and choosing to start a new
full job in Colab, change:

```python
RUN_PROFILE = "full"
RUN_ACTION = "run"
RUN_ID = "colab_full_modular_20261103_01"
DEVICE = "cuda"
ALLOW_FULL_RUN = True
FULL_PERFORMANCE_BACKEND = "optimized"
RUN_FINAL_TEST = False
ALLOW_FINAL_TEST = False
LOCKED_RUN_SIGNATURE = None
```

Keep the exact published commit and actual source inputs. Explicit CUDA must
exist and be used by the real trainer; no silent CPU fallback is allowed.
Colab does not guarantee any GPU, RAM, disk or runtime duration. Resource checks
may stop. Do not bypass them, silently sample, shorten training or drop features
and still call the result full.

Full's null population/query/target/step limits mean all legal data, not pilot
caps. The formal Train-only negative sampler remains in force. Four folds train
one LambdaMART; outer Validation is complete and unsampled. The full endpoint
in this workflow is completed training, Validation and reload, with final Test
still disabled. Do not launch real full on the local M1 Max.

## 7. Inspect, retry and recover

For a current runtime, use the same explicit commit/run/profile/device/backend
and set `RUN_ACTION="status"` to inspect individual cells. For a failed or
interrupted stage, diagnose its error/resource report, retain the identity and
set `RUN_ACTION="resume"`. Rerun only the desired stage cell. If an earlier
stage never completed or its files fail verification, complete/repair that
specific prerequisite first; the requested model cell does not train it for you.

For a new runtime:

1. Open the same root Notebook revision and set the original explicit Git SHA.
2. Keep original run ID, profile, device/backend, input identity and persistent
   run-parent; set `RUN_ACTION="resume"`.
3. Verify the checkout, mount Drive, install the locked worker environment and
   stage the same three immutable input directories.
4. Run the plan and the desired stage cell. The shared runtime verifies and
   restores compatible persisted prerequisites before continuing that stage.
5. Inspect the aggregate report, publication status and subsequent stages.

Do not change the code/ref/config then force old manifests to match. A new
layout/code/config uses a new run. Do not rename pilot artifacts, reset a dirty
checkout, mutate a live job or execute archived source to reach an old path.

Two-Tower local saves and Drive publications have separate cadences. A runtime
loss can discard progress since the latest complete remote publication. The
previous valid remote checkpoint is retained until a replacement verifies.
Full optimizer/scheduler/RNG/position/selected state is required; weights-only
loading is not equivalent training recovery.

## 8. Final Test approval boundary

The last executable cell defaults to a message only: **SKIPPED**. Preparation
and Validation do not authorize real Test. The structural refactor task does
not execute it.

If the owner later explicitly authorizes final Test after model/config selection
is frozen, the cell requires:

- Completed full training, Validation and reload.
- `RUN_FINAL_TEST=True` and separate `ALLOW_FINAL_TEST=True`.
- `LOCKED_RUN_SIGNATURE` manually copied from the verified completed run.
- Backend identity checks before any Test future-label read.

The approved command adds `--run-final-test --approve-final-test
--locked-signature <recorded_sha256>` to the named `final_test` stage. Missing or
mismatched identities fail. It constructs Test-cutoff past-only candidates and
the same feature schema, scores once without fitting/tuning, evaluates full
unique eligible Test truth and saves a separate report. It cannot retroactively
turn Test into another Validation sweep.

Do not enable Test flags to make a smoke or resource failure pass. Synthetic
Test-gate tests prove interface isolation, not a real H&M Test metric.

## 9. Troubleshooting and unverified behavior

| Symptom | Safe next step |
|---|---|
| Unknown/missing stage prerequisite | Run or restore the named earlier stage; do not hide dependency training inside a model cell |
| Dirty/different Git checkout | Use a fresh runtime at the original SHA, or a new run identity for new source |
| Failed input hash/map/dimension | Diagnose the source artifact; do not overwrite embeddings or invent a repair |
| CUDA explicitly requested but unavailable | Stop and choose a suitable authorized runtime or explicitly version a different device run |
| Resource rejection | Stop; do not reduce the full population and keep its full label |
| Stale/corrupt manifest/checkpoint | Preserve diagnostic evidence and restore the last fully verified publication |
| Missing local report during status | Status is an inspection; resume/restore compatible completed artifacts before expecting local model reports |
| High candidate diagnostic, low end-to-end MAP | Check full ground truth, native retrieval misses and query denominators; do not filter failed queries |

The [implementation report](code_architecture_refactor_report.md) distinguishes
static checks, synthetic real-model execution, measured bounded comparisons and
cloud/full/Test checks not run. A source refactor alone does not prove a speedup,
production readiness or reproduction of the historical full H&M score.
