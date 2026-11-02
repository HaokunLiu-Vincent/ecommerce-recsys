# Codex Task: Colab Training Entry Point and Resumable Scale-Up

**Task type:** Extend the existing implementation; do not rebuild the recommender.  
**Development host:** Apple M1 Max with approximately 7–8 GB of available RAM.  
**Execution target:** A user-operated Google Colab runtime.  
**Default mode:** Smoke. Full training requires an explicit runtime opt-in.  
**Final Test:** Disabled and outside this task's execution scope.

## Contents

- [1. Objective, authority, and execution authorization](#1-objective-authority-and-execution-authorization)
- [2. Repository reconnaissance and protected behavior](#2-repository-reconnaissance-and-protected-behavior)
- [3. Filesystem and artifact contracts](#3-filesystem-and-artifact-contracts)
- [4. Reproducible code release](#4-reproducible-code-release)
- [5. Thin Colab notebook](#5-thin-colab-notebook)
- [6. Dependencies, devices, and process isolation](#6-dependencies-devices-and-process-isolation)
- [7. Smoke, pilot, and full profiles](#7-smoke-pilot-and-full-profiles)
- [8. Scalable data and model execution](#8-scalable-data-and-model-execution)
- [9. Stage orchestration, persistence, and recovery](#9-stage-orchestration-persistence-and-recovery)
- [10. Validation and inference contracts](#10-validation-and-inference-contracts)
- [11. Implementation deliverables](#11-implementation-deliverables)
- [12. Verification and acceptance](#12-verification-and-acceptance)
- [13. Final handoff and stopping conditions](#13-final-handoff-and-stopping-conditions)

## 1. Objective, authority, and execution authorization

Implement a portable, resource-aware Colab workflow on top of the real local
H&M smoke pipeline. A separate notebook must stage a versioned code package and
existing inputs into Colab's local filesystem, invoke the shared implementation,
persist results to Google Drive, and support safe resumption.

Do the implementation and locally available verification, not just a design.
Do not copy model or feature implementations into notebook cells.

Read these repository files before changing code:

1. Root `AGENTS.md` and any applicable scoped agent instructions.
2. `docs/proposal.md`, currently v0.3.2.
3. `docs/archi/architecture.md`.
4. `docs/archi/evaluation_protocol.md`, currently v2.0.
5. `docs/archi/implementation_decisions.md`.
6. `docs/archi/local_smoke_report.md`.
7. Existing code, configs, tests, `pyproject.toml`, and `uv.lock` relevant to the
   task. Read the earlier smoke task if present, but do not apply smoke-only
   sampling rules to full training.

The smoke report identifies `docs/proposal.md` as the active proposal and says
there is no root proposal or Git repository. Inspect the actual checkout rather
than assuming either path or Git exists. Resolve superseded documentation in
favor of the latest explicit user decisions and current architecture. Do not
restore the old overlapping-anchor protocol or archived notebook pipeline.

This task authorizes a cross-phase portability and scale-up implementation. It
does not authorize unlimited training on the development Mac, final Test
execution, cloud deployment, or claims of full-phase completion.

### 1.1 Development-time authorization

On the local Mac, you may inspect inputs, implement code, run bounded synthetic
and unit tests, build the release, and perform an existing-size real smoke
regression when needed. Retain the local smoke's count and resource safeguards.
Use sequential execution and at most two compute threads for local training
checks. Do not run pilot/full training, full feature materialization, or a full
embedding audit on this host. Do not repeatedly scan the full transaction file
merely to test packaging or notebook generation.

Do not modify the existing `data/runs/smoke/real_v1/` evidence. New tests and
regressions need separate output directories.

### 1.2 Colab runtime authorization

Implement notebook-controlled `smoke -> pilot -> full` execution. The notebook
must default to smoke with `ALLOW_FULL_RUN = False` and
`RUN_FINAL_TEST = False`. Full execution is available only after the operator
explicitly selects it, opts in, and passes the applicable gates.

The full run means legal base training, four ranker training folds, complete
global Validation, and fresh-process reload verification. It does not mean
using every transaction as training data for every component.

Do not start, purchase, or provision a Colab/cloud runtime from the development
host. If no actual Colab session is available to you, implement and locally
validate the workflow, then report Colab, CUDA, pilot, and full execution as
`not run`. Do not imply that generated notebook cells were executed remotely.

## 2. Repository reconnaissance and protected behavior

Inspect and reuse the existing interfaces. The smoke report describes:

- `configs/smoke.json`.
- `scripts/local_smoke.py`: preflight, synthetic, real-smoke, and reload.
- `src/hm_recsys/`: adapters, contracts, sampling, resource monitoring, six
  retrievers, trained models, shared features, metrics, and orchestration.
- Existing pytest coverage, Ruff configuration, Pyright, and uv dependencies.
- Sequential fresh processes for PyTorch, FAISS, and LightGBM.

The report is evidence of a previous run, not a guarantee that the checkout is
unchanged. Inspect source signatures and tests; do not invent existing APIs.
Keep the old smoke CLI functional. Refactor shared logic only as necessary and
add regression coverage rather than creating a separate Colab recommender.

Preserve the approved model and data semantics:

| Component | Required behavior |
|---|---|
| Channels | Trend, Repurchase, Product_Code Sibling, ItemCF, WALS, Hybrid Two-Tower |
| Quotas | `60/30/30/50/50/50`; the deduplicated native union has at most 270 candidates per query |
| Fusion | Preserve all native candidates and provenance; RRF is a feature, not a truncating pre-ranker |
| Active pool | At least three purchase lines in the preceding 28 calendar days; an activity proxy, not verified stock |
| WALS | Freeze item factors/maps; use the real fixed-item ALS user solve for folding-in; confidence scaling is applied once |
| Two-Tower | Existing frozen content inputs, trainable projections, shared ID dropout, duplicate-aware multi-positive LogQ loss |
| Ranking | One LightGBM LambdaMART, binary unique-article labels, no injected positives |
| Train sampling | Preserve all native positives; existing documented 1:20 negative cap and hard/random policy; no-pair filtering in Train only |
| Validation | All eligible purchase-active queries and their native candidates, including zero-hit and zero-candidate queries |
| Ground truth | Catalog-matched future purchases; do not intersect truth with active pools, model vocabularies, content coverage, or retrieved candidates |
| Features | Existing shared named schema, currently 49 columns; load the schema rather than hardcode its count |
| Output | Up to 12 unique articles, deterministic tie-breaking; never manufacture scores or pad with duplicates |

Full configuration must restore the actual full architecture settings, including
visual clustering and the base model schedules, rather than copying the reduced
smoke values. Device, worker, batch, and memory settings must be explicit.
Changes to model semantics require a separate documented decision; do not make
such changes merely to get the migration to run.

### 2.1 Dates are immutable contracts

`cutoff_date = T` is inclusive; `prediction_date = a = T + 1 day`.
History is `t_dat < a`, a trailing d-day window is `[a-d, a)`, and labels are
`[a, a+7 days)`.

| Component or partition | Required interval |
|---|---|
| WALS base history | `[2020-02-01, 2020-08-12)` |
| WALS item vocabulary observations | `[2020-06-16, 2020-08-12)` |
| Two-Tower training targets | `[2020-05-19, 2020-08-05)` |
| Two-Tower internal validation targets | `[2020-08-05, 2020-08-12)` |
| Ranker F1 labels | `[2020-08-12, 2020-08-19)` |
| Ranker F2 labels | `[2020-08-19, 2020-08-26)` |
| Ranker F3 labels | `[2020-08-26, 2020-09-02)` |
| Ranker F4 labels | `[2020-09-02, 2020-09-09)` |
| Global Validation labels | `[2020-09-09, 2020-09-16)` |
| Final Test labels, disabled | `[2020-09-16, 2020-09-23)` |

Keep the disjoint-window tests and cutoff gates. Past-only histories can overlap;
label windows cannot. Recompute dynamic rules and ItemCF at each cutoff, update
histories and WALS user folding-in, but never refit the frozen base using ranker
or global Validation purchases.

Reading a source CSV or copying/hashing it may encounter bytes for final Test
dates. That is not permission to materialize final Test labels, build its
purchase-active evaluation roster, evaluate it, or select models using it.
Default training/evaluation extracts must enforce their legal boundaries.

## 3. Filesystem and artifact contracts

### 3.1 User-provided persistent inputs

The exact Drive input root is:

```text
/content/drive/MyDrive/projects/e_commerce_recsys/data/
```

It has the same layout as the local project's data directory:

```text
data/
├── original data/
│   ├── articles.csv
│   ├── customers.csv
│   └── transactions_train.csv
├── text_features/
│   ├── text_article_ids.parquet
│   ├── text_embeddings.npy
│   ├── text_faiss.index
│   └── text_features_manifest.json
└── image_features/
    ├── image_article_ids.parquet
    ├── image_embedding_failures.parquet
    ├── image_embeddings.npy
    ├── image_faiss.index
    └── image_features_manifest.json
```

These are read-only inputs. Do not rename `original data`, regenerate embeddings,
download encoders, overwrite manifests/maps/indexes, or infer that text/image row
orders match. Preserve article IDs as zero-padded strings.

Read separate mappings and manifests, validate real shapes and dtypes, and use
read-only mmap/bounded batches for arrays. Preserve missing-modality masks and
image failure semantics. Full-data consumers must extend the smoke's numeric
validation to every consumed vector, without loading the arrays wholesale or
inventing upstream provenance.

The supplied content FAISS indexes may be copied intact to preserve the input
layout. They are not independent recall channels and are not the learned
Two-Tower index. Do not load/rebuild them without a specific need.

### 3.2 Execution directory and outputs

Use these default paths, configurable in one notebook setup section:

```text
Drive project:       /content/drive/MyDrive/projects/e_commerce_recsys/
Drive releases:      <Drive project>/releases/
Local project:       /content/e_commerce_recsys/
Local data:          <Local project>/data/
Local run:           <Local data>/runs/colab/<run_id>/
Persistent run:      <Drive project>/data/runs/colab/<run_id>/
```

Preserve relative repository paths, not Mac absolute paths. Training reads local
`/content` inputs and writes local artifacts. Drive is the source and persistence
destination, not the working database, feature store, checkpoint scratch space,
or environment installation directory.

Only the notebook/bootstrap configuration knows the Colab absolute defaults.
Shared library code and CLI interfaces accept explicit project/data/run roots
and remain usable locally. Handle spaces in paths correctly.

### 3.3 Idempotent input staging

Implement a whitelist copier for `original data/`, `text_features/`, and
`image_features/`. Do not copy all of `data/`, and never copy old runs, caches,
releases, or checkpoints as new inputs.

Copy files in bounded chunks to temporary local files; verify size and a
streaming content digest, then mark them complete. Hashing during copying is
acceptable and avoids an additional source read. Retain supplied hashes as
separate provenance and verify them when their meaning is established.

Maintain a staging manifest. Reuse an already verified local file only when its
recorded source identity remains valid. Do not treat matching file sizes alone
as cryptographic verification. If reuse relies on size/mtime for unchanged
source detection, state that limitation and offer an explicit reverify option.
Interrupted or mismatched files must be recopied or fail, never silently reused.

Check local free disk before extraction/copying and again before expensive
stages. Include environment, inputs, canonical data, candidates, models, and
DuckDB spill in the estimate. Fail safely if required inputs or space are absent.
Do not delete existing results to make space without explicit authorization.

## 4. Reproducible code release

Implement a release-building command, preferably `scripts/build_release.py`,
using an explicit allowlist. Include the current source, scripts, configs,
tests, `pyproject.toml`, `uv.lock`, required current docs, `AGENTS.md`, and the
new notebook. Include any legitimate build metadata required by pyproject.

Exclude at least:

```text
.venv/                    .git/
__pycache__/              .pytest_cache/
.ruff_cache/              .mypy_cache/
data/                     docs/achive/
.ipynb_checkpoints/        credentials and .env files
model artifacts           user-level outputs and local absolute-path logs
```

Do not initialize Git. If Git metadata exists, record it without changing the
working tree. Otherwise use content hashes and a build ID. Respect existing
user changes; do not reset or reformat unrelated code/notebooks.

Write local release artifacts under `data/releases/`, not inside the inputs.
Produce a ZIP, a SHA256 checksum, and a release manifest with relative paths,
file hashes, dependency-lock identity, and build metadata. Avoid self-referential
hashes. Sort archive members deterministically and do not include executable
Mac environments, cached wheels, or platform binaries.

The user will copy the ZIP and checksum/manifest into Drive's `releases/`.
Also leave `notebooks/01_Colab_Training.ipynb` directly available for opening in
Colab: the notebook must not require unpacking the project before its bootstrap
cells can run. Include it in the ZIP as well for version traceability.

At extraction, validate package integrity and reject absolute paths, `..`
traversal, unsafe links, and attempts to overwrite input/run directories.
Do not overwrite a different existing checkout silently. Reuse a matching
release or require an explicit replacement/new-target choice.

## 5. Thin Colab notebook

Create `notebooks/01_Colab_Training.ipynb`, following AGENTS.md's English,
Google-style Python, notebook, Markdown, and reproducibility requirements.
Preserve existing notebooks; do not use or execute `docs/achive/`.

The notebook should have these sections:

1. **Title, Objective, and Execution Scope**
2. **Run Configuration and Google Drive Mount**
3. **Code Release Verification and Local Staging**
4. **Environment Setup and Resource Preflight**
5. **Input Staging and Validation**
6. **Colab Smoke and Device Verification**
7. **Pilot or Full Training, with Explicit Gates**
8. **Validation and Fresh-Process Reload Checks**
9. **Persistence Verification, Results, and Next Steps**

Use one H1 and a navigable outline. Explain each stage before code. Keep cells
focused; no model training loops, feature SQL, duplicated metric definitions,
or substantial orchestration logic in the notebook. Put reusable work in `src/`
and invoke it through a CLI. Bootstrap code needed before the package exists
must be minimal and limited to mounting, verification, and safe extraction.

Use this configuration intent, adapting the interface to the actual code:

```python
from pathlib import Path

DRIVE_PROJECT_ROOT = Path(
    "/content/drive/MyDrive/projects/e_commerce_recsys"
)
PROJECT_ROOT = Path("/content/e_commerce_recsys")
SOURCE_DATA_ROOT = DRIVE_PROJECT_ROOT / "data"
LOCAL_DATA_ROOT = PROJECT_ROOT / "data"

RELEASE_NAME = "hm_recsys_<build_id>.zip"  # Replace with the built release.
RUN_PROFILE = "smoke"
RUN_ID = "colab_smoke_v1"
ALLOW_FULL_RUN = False
RUN_FINAL_TEST = False
```

Validate unresolved release placeholders before starting. Generate a fresh run
ID or explicitly resume an existing compatible one; do not reset its state on
cell re-execution. Freeze and display the resolved configuration before work.

Use argument lists rather than unquoted shell command strings. Start children
with an explicit interpreter/environment and `cwd`. Capture exit codes, stream
bounded progress summaries, and keep full logs under the run directory. A
failed child must stop its dependent stages, not let the notebook proceed to a
success summary.

The default top-to-bottom run must not unexpectedly start pilot/full training.
A rerun must recognize valid completed stages, not retrain every component.
Only data-free notebook tests can use synthetic outputs; committed notebook
outputs must be empty or genuinely executed, clearly identified evidence.

## 6. Dependencies, devices, and process isolation

Use the project uv environment and lockfile; do not copy the Mac `.venv`.
Do not assume Colab's preinstalled packages match the project. Inspect the
current Python requirement, existing dependency groups, platform markers, and
Linux CPU/CUDA wheel availability before deciding how to extend configuration.

Keep project execution under a single documented project interpreter, e.g.
locked uv commands. The Colab notebook kernel may handle Drive mounting and
presentation, but training/tests/reloads must use the project environment.

Do not blindly upgrade packages or regenerate the lock at notebook runtime.
If compatible Linux/CUDA installation requires a dependency change, make the
smallest explicit development-time change, lock it, document it, and preserve
the existing local CPU path. Do not invent package versions or assume versions
mentioned in the smoke report are portable. Treat platform failures as blockers,
not permission to substitute fake models or change algorithms.

Start with the following execution policy:

| Component | Initial Colab device policy |
|---|---|
| DuckDB, rules, ItemCF | CPU |
| WALS | Existing validated CPU implementation |
| Two-Tower training/encoding | CUDA after an actual device probe; CPU smoke remains supported |
| FAISS | CPU `IndexFlatIP`, preserving existing exact retrieval semantics |
| LambdaMART | CPU |

Verify CUDA in the same project environment and worker process used for training:
record device/driver/runtime details and perform a bounded forward pass,
backward pass, and optimizer update. A visible GPU or `nvidia-smi` output alone
is not sufficient. A profile requesting CUDA must not silently fall back to
CPU; require an explicit alternative configuration.

Preserve sequential fresh-process isolation between PyTorch, FAISS, and
LightGBM. Do not use unsafe duplicate-OpenMP flags, binary patches, or import
order tricks as a substitute for a supported environment. Notebook-owned
processes must terminate cleanly on failure/interruption without killing
unrelated user processes.

Record Python/platform/architecture, resolved package versions, lock/build
hashes, CPU threads, RAM, GPU/VRAM when present, and disk. Report sampled RSS and
CUDA allocated/reserved peaks as measurements with limitations, not hard-cap
proofs. CUDA work must be synchronized appropriately for timing claims.

## 7. Smoke, pilot, and full profiles

Keep the local `configs/smoke.json` contract unchanged. Add explicit Colab
profiles, preferably:

```text
configs/colab_smoke.json
configs/colab_pilot.json
configs/colab_full.json
```

Avoid deeply nested inheritance that hides effective limits. Persist a complete
resolved configuration for every run. Resource checks must exist in the CLI and
training entry points, not only in a notebook Boolean.

### 7.1 Smoke

Reproduce the existing bounded integration design: up to 500 sampled users,
5,000 candidate items, bounded transaction/query rows, short WALS/Two-Tower/
LambdaMART training, six genuine channels, and fresh-process inference. Read the
actual smoke config for all limits rather than reconstructing them from memory.

Run the CPU Colab smoke first to isolate environment portability. Add a separate
bounded CUDA probe and, when GPU is selected for later stages, a tiny genuine
Two-Tower CUDA training/reload check. Record separate run identities/evidence.
Do not require cross-platform retraining to reproduce Mac scores bit-for-bit.

### 7.2 Pilot

Pilot validates scale-sensitive paths without pretending to be a full result.
Use the full protocol's legal catalog and cutoff/aggregate definitions, but
explicitly bound model training work and query counts. Start with F1 and a
bounded global Validation query roster. Define concrete pilot caps in config
before execution; choose them from resource preflight, not favorable metrics.

Where a pilot base-training subset is necessary, define it from legal past
information and record it separately from the catalog and full-history
statistics. Never present sample-derived counts as global counts. Select query
rosters deterministically under the protocol, independently of recall hits;
keep zero-hit queries. Do not shrink truth to the pilot candidate universe.

Measure ItemCF construction, sparse-matrix sizes, embedding batches, candidate
counts, feature throughput, disk/spill, ranker memory, and device behavior.
Clearly label pilot metrics and all training/epoch caps. No tuning sweep.

### 7.3 Full

Full uses the complete legal data and query populations of the current protocol,
not the smoke cohort. Remove smoke-only user activity selection, item ceilings,
target-event limits, and query caps. Preserve the protocol's actual Train-only
negative sampling; full does not mean removing that algorithmic rule.

Use configured base training schedules, full legal vocabularies, frozen
transformations, all four ranker folds, and complete global Validation. Apply
architecture settings rather than smoke settings such as K=8 visual clustering,
16-factor/two-iteration WALS, 50 tower steps, or a 30-round ranker.

Full needs explicit opt-in, verified inputs/environment, a passing Colab smoke,
and compatible pilot/resource evidence. Do not reuse undertrained smoke/pilot
models as full base models. Reuse derived artifacts only when all relevant
semantic, input, training, and code identities match the full specification.

Do not silently reduce queries, candidates, epochs, vocabularies, or features
when memory is tight. Stop with `blocked_resource`, estimates, and evidence.
A deliberately reduced alternative is a separately named pilot/diagnostic, not
a full run. Early stopping explicitly specified by the architecture remains
valid and must be recorded.

Default full execution stops after Validation and reload checks. Do not expose
a generic `all` stage that accidentally includes final Test. Reject Test stage
requests for this task's normal workflow; keep final Test configuration disabled.

## 8. Scalable data and model execution

### 8.1 Data and feature construction

Audit current smoke-only bottlenecks before exposing full mode. Generalize only
what is necessary, preserving algorithms and shared interfaces.

- Use explicit CSV schemas and bounded DuckDB/Arrow/Parquet processing. Do not
  load the entire transaction CSV into pandas.
- Reuse a verified local canonical store where useful to avoid repeated CSV
  parsing; its date bounds and input identity must be recorded.
- Preserve raw repeated purchase lines. Deduplicate only where the current
  interaction or label definition requires it.
- Build query and ground-truth manifests separately from candidate tables.
- Materialize candidates/features one fold and bounded query batch at a time.
  Never concatenate four raw candidate folds plus Validation into a giant
  pandas table just because the smoke implementation did so.
- Use numeric/categorical arrays with deliberate dtypes. Keep identifier joins
  out of model matrices. Do not cast away leading zeros.
- Keep whole query groups contiguous, including across persisted shard
  boundaries or merge steps. Test aggregation across shards explicitly.
- Bound full-catalog score operations by query batches. Do not create an
  all-users-by-all-items dense score matrix.
- Keep ItemCF pair generation and aggregation bounded while preserving its
  approved formula and top-neighbor policy. Do not silently discard high-degree
  users or change similarity definitions to reduce runtime.

### 8.2 Model memory and training semantics

WALS must use the legal sparse matrix and fixed-item user solve. Check sparse
orientation, index widths, factor dimensions, and memory before fitting.

Two-Tower must stream/batch targets and histories. Read source embeddings on CPU
with bounded gathers, then transfer only needed tensors; do not move all source
arrays or transaction histories to GPU. Preserve dropout/masks, date-valid
history, sampler-derived Q, and multi-positive/duplicate treatment.

Choose explicit batch/workers/thread settings after preflight. Account for
in-batch logits/masks growing with `(batch_size, batch_size)`, not just embedding
storage. Do not assume the original batch size fits every GPU. A changed batch
changes in-batch negatives: record it and its consequences. Gradient
accumulation is not equivalent to a larger in-batch negative pool. No silent
OOM retries with different training semantics.

Preserve the approved full visual-clustering fit boundary and K=128. Do not
silently replace the clustering algorithm or fit only a smoke sample in full
mode. Block and report a genuine resource problem.

For LambdaMART, feature sharding is not a claim of out-of-core training. Estimate
actual post-sampling Train rows, complete Validation rows, feature storage,
dataset construction, histograms, categories, and temporary copies. Use the
existing valid LightGBM ingestion path with bounded preprocessing; make any
alternative data-loading path prove group/schema equivalence. Do not treat
sequentially fitting each shard as equivalent to fitting the complete dataset.

Prefer incremental metric accumulation and batched prediction with complete
query coverage. Do not silently sample Validation to make its evaluation fit.

### 8.3 Resource supervision

Retain the process-tree supervisor and add Linux/CUDA-aware checks as needed.
Set explicit CPU threads, DataLoader workers, DuckDB memory/temp-directory/spill
limits, host memory reserve, GPU limits, and disk thresholds. Resolve budgets
from observed resources with a documented reserve and persist them before work.
Do not assume a particular Colab GPU, high-RAM instance, or session lifetime.

The Mac keeps its existing local limits. Colab must use its own resolved resource
profile, not disable all guards or blindly copy the Mac's 5-GiB stop threshold.
DuckDB `memory_limit` does not stand for a whole-process guarantee.

Prevent oversubscription by training stages sequentially. On threshold crossing,
stop owned children, close files, record the partial state, and persist safe
metadata/artifacts when possible. Do not claim a last-second backup is guaranteed
if a runtime is killed abruptly; periodic persistence is required.

## 9. Stage orchestration, persistence, and recovery

Provide a small dependency-aware stage runner, not a new orchestration platform.
Suggested commands/stages may be adapted to the existing CLI, but document the
actual implemented interface and preserve old smoke commands.

The operator needs preflight/plan, run, status, and resume capabilities. A plan
must expose resolved profile, dates, inputs, devices, limits, intended stages,
and output locations before expensive work.

Suggested stage boundaries:

```text
input acceptance / canonical preparation / split manifests
    -> WALS base
    -> Two-Tower base and selected checkpoint
    -> learned encodings / indexes / frozen preprocessing
    -> F1 candidates and features
    -> F2 candidates and features
    -> F3 candidates and features
    -> F4 candidates and features
    -> Validation candidates and features
    -> ranker training
    -> external Validation metrics
    -> fresh-process reconstruction and inference
    -> verified final persistence
```

Use isolated `data/runs/colab/<run_id>/` directories. Protect completed artifacts
from accidental overwrite and prevent conflicting writers for the same run.
Keep local completion and persisted completion as separate states.

Each stage must:

1. Verify dependencies and the configuration/input compatibility signature.
2. Write outputs locally with incomplete/temporary status.
3. Validate outputs, counts, schema, identity, and relevant hashes.
4. Write an output manifest and local completion state.
5. Copy required outputs to the matching persistent run in bounded chunks.
6. Verify the persistent copy and write its completed manifest last.

A directory or marker alone is not proof of completion. Google Drive operations
must not be assumed to form a reliable multi-file transaction. An interrupted
copy, unavailable mount, checksum mismatch, or failed worker must never produce
a successful persistent-stage state.

On a fresh Colab runtime, restage the code and inputs, inspect persistent stage
manifests, restore required derived artifacts locally, and resume at the first
incomplete/incompatible stage. No existing notebook variables are required.

Compatibility must include code/build and dependency identity, input identities,
resolved semantic config, profile, splits/cutoffs, sampling, schema, ID maps,
base-model versions, upstream manifests, and required device/runtime details.
Do not reuse a cache from another cutoff or a smoke base in a full run. Reject
incompatible reuse with a precise reason; do not erase the old run.

Support stage-boundary recovery for every stage. For Two-Tower, additionally
checkpoint at a configured interval/epoch boundary, including model, optimizer,
scheduler and scaler when used, epoch/global step, random states, config,
sampler/data-position information, and selected-checkpoint metadata. Verify
sample order/offset semantics before claiming exact mid-epoch continuation.
If only epoch-boundary restart is reliable, state and test that behavior; do not
pretend weights-only reload is exact continuation.

Persist meaningful intermediate candidate/feature shards after complete shard
boundaries where necessary. Failed shards may be regenerated deterministically;
never double-count them. Other model stages may restart from their last complete
boundary; do not invent unsupported partial-fit behavior.

Persist the minimum resumable bundle: models, checkpoints, learned indexes,
maps/schema, queries/labels as private artifacts, required processed shards,
resolved configs, manifests, metrics, resource evidence, and logs. Do not
re-upload source CSVs/embeddings, virtual environments, or DuckDB spill files.
Use stage/shard-aware copying instead of repeatedly syncing an entire directory.

## 10. Validation and inference contracts

External native macro MAP@12 against complete ground truth is primary.
Candidate NDCG@12 remains a separately named optimization diagnostic; do not
replace the current architecture's early-stopping policy during migration.
Keep Recall@12, full-truth NDCG@12, union recall, channel counts, and relevant
coverage/denominator checks. Do not tune quotas or run hyperparameter sweeps.

Train may remove no-pair groups and sample proxy negatives under the documented
policy. Validation must not. Zero-candidate queries remain in the independent
roster and contribute zero, not disappear from a candidate-table groupby.

After training, launch label-free inference in a new process using only frozen
artifacts, the prediction cutoff, the query roster, and legal history. It must
reconstruct features and candidates rather than simply reread a saved prediction
and declare parity. Label loading is confined to the separate evaluator.

Verify schema/ID maps, candidate IDs/order, features, scores, and unique Top-12
outputs against the same saved run's reference, using explicit established
tolerances. At full scale perform checks in bounded chunks. If a computationally
expensive reconstruction is deliberately limited, use a predetermined sample
and label it as bounded parity verification; do not claim full-row parity.

The goal is same-artifact reload correctness, not bitwise-equivalent retraining
between macOS CPU and Linux CUDA. Record numerical differences honestly.

## 11. Implementation deliverables

Use existing conventions; the names below are preferred new interfaces, not
claims that these files already exist:

| Deliverable | Required purpose |
|---|---|
| `notebooks/01_Colab_Training.ipynb` | Independently openable, thin Colab entry point |
| `scripts/build_release.py` | Allowlisted code release, hashes, and manifest |
| `scripts/run_pipeline.py` or an equivalent extension | Shared profile/stage runner, plan/status/resume, notebook invocation |
| `configs/colab_smoke.json` | Existing-sized bounded integration on Colab |
| `configs/colab_pilot.json` | Explicitly limited scale verification |
| `configs/colab_full.json` | Full protocol data/settings plus resource gates; no final Test |
| Focused additions in `src/hm_recsys/` | Staging, portable paths, resource budgets, scalable execution, persistence |
| Focused tests | Packaging, bootstrap/staging, guards, resume, leakage, group/schema and metric integrity |
| `docs/archi/colab_runbook.md` | Exact operator commands, uploads, notebook steps, recovery, limitations |
| `docs/archi/colab_implementation_report.md` | Actual changes, executed checks, evidence, and unexecuted gates |
| `data/releases/` outputs | Real built ZIP, checksum, release manifest |

Update only necessary project docs. Do not rewrite AGENTS.md, archived reports,
existing notebooks, or the core evaluation protocol to make implementation
easier. If code reveals a genuine unresolved architecture conflict, isolate and
report it instead of silently altering the experiment.

Notebook and repository documentation must be professional English. Follow the
existing Google-style documentation, typing, Ruff, Pyright, pytest, and notebook
standards. Avoid gratuitous abstractions, new platforms, or duplicate configs.
Do not initialize Git, configure cloud credentials, deploy services, launch
Docker/HTTP servers, or add mandatory remote MLflow/DVC infrastructure for this
task. Existing manifests and local artifacts must suffice to execute the path.

## 12. Verification and acceptance

### 12.1 Required local checks

Implement and actually run bounded tests for:

- Existing smoke behavior and shared API compatibility.
- Allowlisted packaging, required files, secret/data/archive exclusions,
  checksum verification, and malicious ZIP path rejection.
- Bootstrap from a clean temporary directory without importing project modules
  before extraction/environment setup.
- Input staging with spaces, independent text/image mappings, read-only sources,
  unchanged-file reuse, interrupted copies, and digest mismatch.
- Missing files, insufficient local space, resource-threshold termination,
  unavailable persistence destination, and failed child exit propagation.
- Resume after an interrupted stage/shard and after deleting the temporary
  local runtime directory; restore from a temporary simulated persistent root.
- Reject changed config/build/input/schema/cutoff and smoke-to-full cache reuse.
- Full opt-in gate and disabled Test; default run does not include pilot/full.
- Full-profile rejection of smoke-only cohort/item/query/training caps.
- Exact non-overlapping windows, cutoff-safe history, and sampler boundaries.
- Chunked vs unchunked synthetic features/metrics; contiguous ranking groups;
  zero-hit/zero-candidate denominators; label-free fresh-process reload.
- Synthetic checkpoint recovery and deterministic boundaries to the extent
  claimed by the implementation.

Use temporary fixture directories to simulate Drive; say explicitly that these
are filesystem tests, not Google Drive service verification. Tests must not
mount a real account, launch large jobs, or require a GPU for the CPU suite.

Validate the notebook JSON/schema, compile/check its code cells appropriately,
and test its meaningful helper paths. A mocked mount or synthetic notebook test
is not a successful real Colab run. Record that distinction in the report.

Run Ruff lint/format checks on changed code and the new notebook, Pyright, and
relevant pytest suites. Run broader regression checks when resource-safe. Retain
and report unrelated existing repository-wide notebook lint failures; do not
suppress or repair them as an unrelated cleanup. Record exact commands, exit
codes, and test counts. No fabricated execution, metrics, or version claims.

Build the actual release, extract it into a fresh temporary directory, verify
its manifest and required entry points, and run the feasible bounded checks
from the extracted package. Do not claim Linux/CUDA dependency compatibility
from a macOS-only extraction test.

### 12.2 Real Colab operator gates

The notebook must produce evidence for these gates when the user runs it:

1. Project environment, inputs, local disk, and resource preflight pass.
2. Colab CPU smoke and same-artifact fresh-process parity pass.
3. Required CUDA probe/tiny training check passes for GPU profiles.
4. Pilot completes with measured resource and persistence/recovery evidence.
5. Full plan fits the resource budget and the operator explicitly opts in.
6. Full legal training plus Validation completes without hidden sampling.
7. Artifacts are persistently verified and reload checks are recorded.

These are runtime gates, not tasks to claim as executed from the Mac. If any
full-scale component remains unsupported, name it explicitly and prevent a
false-success full mode.

## 13. Final handoff and stopping conditions

Start with a brief implementation plan grounded in the inspected checkout, then
implement, test, and package. Do not stop at pseudocode or ask the user to repeat
known paths, architecture decisions, or hardware constraints.

Provide a final report separating:

- Implemented and locally verified.
- Implemented but awaiting actual Colab execution.
- Blocked or not implemented, with concrete reasons.

Include changed files, preserved behavior, any explicit config/dependency
changes, exact executed commands and statuses, local resource evidence,
release/notebook paths, unresolved limitations, and the next operator actions.

The runbook must make the user handoff concrete:

1. Run the actual implemented local release command.
2. Copy the ZIP and checksum/manifest to Drive `releases/`; existing data stays
   in its current Drive location.
3. Open the standalone new notebook in Colab and select the built release.
4. Run the default smoke workflow; inspect evidence before selecting pilot.
5. Start a new full run only after pilot/preflight pass and explicit opt-in.
6. Resume safely after interruption and locate verified persistent artifacts.

Do not invent precise performance forecasts, claim whole-array/source provenance
verification from subset checks, or mark Colab/full/Test/deployment complete
without execution evidence. Do not continue into tuning, final Test, or
unrequested deployment after this task. Deliver the runnable entry point,
release, tests, and honest handoff.
