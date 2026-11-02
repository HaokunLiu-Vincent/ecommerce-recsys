# Codex Task: Deliver the Colab Pilot After Real CPU Smoke Acceptance

**Task type:** Implement, test locally within the existing resource limits, package,
and provide exact Colab pilot operating instructions. Do not merely produce
another prompt, a plan, or a larger smoke configuration.

**Basis:** The existing pilot-completion task, the reviewed release's
`configs/colab_pilot.json`, and the latest supplied real Colab CPU smoke execution
record. Reconcile these with the actual current working tree before editing;
do not overwrite newer user changes with an older release.

## Contents

- [1. Objective and scope](#1-objective-and-scope)
- [2. Read the existing implementation](#2-read-the-existing-implementation)
- [3. Fix the confirmed notebook interface defect](#3-fix-the-confirmed-notebook-interface-defect)
- [4. Implement a shared scalable execution backend](#4-implement-a-shared-scalable-execution-backend)
- [5. Preserve recommendation and evaluation semantics](#5-preserve-recommendation-and-evaluation-semantics)
- [6. Bound local work and separate cloud evidence](#6-bound-local-work-and-separate-cloud-evidence)
- [7. Required tests and acceptance](#7-required-tests-and-acceptance)
- [8. Deliverables and handoff](#8-deliverables-and-handoff)

## 1. Objective and scope

Continue the existing project. The latest submitted execution record supports
completion of the **real Colab CPU smoke milestone**, not merely a local test:

- Linux environment bootstrap and staging of the existing H&M inputs completed.
- The run used 500 users, 5,000 catalog items and 53,642 purchase lines.
- All six recall channels emitted actual candidates; WALS, Two-Tower and
  LambdaMART ran, followed by fresh-process Top-12 inference.
- Fresh-process reload matched 5,487 Validation feature rows and scores, with
  maximum score difference 0. Stage status reported persistence completion.

These are previously supplied execution results, not results you have reproduced.
They do **not** establish CUDA training, a fresh-runtime restore after VM loss,
pilot/full execution, or a measured cloud peak RSS. The supplied resource budget
is not measured memory consumption. Do not copy an old report's blanket
"Colab not run" statement into the current milestone summary without qualification.

The smoke Validation active pool contained only 141 items because its statistics
were computed on the sampled interaction universe. Do not tune the architecture,
relax the active threshold, or inject positives to improve that smoke metric.
The pilot must correct the scope of data processing, not disguise those limits.

Complete the next milestone: an **executable, bounded Colab pilot using a shared
scalable backend**, plus precise instructions for the user to run it. A new
configuration or an unconditional `blocked_implementation` branch is not an
implemented pilot. Remove only blockers whose corresponding operations and tests
actually exist. Report any remaining blockers honestly.

Reuse packaging, staging, isolated workers, persistence, checkpointing, models,
features, metrics and tests. Do not rebuild the framework or add orchestration
platforms. Do not make the pilot depend on completing every full-only feature.
Full execution and final Test remain disabled for this task; hyperparameter
searches and deployment are out of scope.

Implement and test the pilot backend on bounded fixtures locally. Actual pilot
training belongs in Colab and may run only when that environment is genuinely
available and its resource checks pass. Otherwise deliver the implementation,
release and operator instructions, marking new CUDA/real-pilot execution not run.

## 2. Read the existing implementation

Read all applicable `AGENTS.md` files, followed by:

- `docs/proposal.md`
- `docs/archi/architecture.md`
- `docs/archi/evaluation_protocol.md`
- `docs/archi/implementation_decisions.md`
- `docs/archi/local_smoke_report.md`
- `docs/archi/colab_runbook.md`
- `docs/archi/colab_implementation_report.md`
- `docs/archi/codex_colab_migration_prompt.md`
- The latest supplied Colab CPU smoke execution review and run summary, if present.

Historical review files may not be in the repository. Do not invent their contents
or block all implementation solely because an old review attachment is absent.
Use the explicitly supplied status above, and label provenance and verification
limits in the new report.

Inspect the real working tree before editing. The reviewed package was
`hm_recsys_e0d2dd9e00e0daf1.zip`; it is a reference, not permission to overwrite
newer user edits. Preserve existing input files, embeddings, archived notebooks,
old run artifacts, and the original smoke command. Do not initialize Git or
commit/push changes. Follow the project's English documentation and code rules.

Inspect at least:

- `notebooks/01_Colab_Training.ipynb`
- `scripts/probe_device.py` and `scripts/run_pipeline.py`
- `src/hm_recsys/colab/{profiles,runner,worker,state}.py`
- `src/hm_recsys/{data,recall,learned,towers,ranking,pipeline}.py`
- `configs/colab_{smoke,pilot,full}.json`
- Existing smoke, migration, and checkpoint tests.

Produce a brief implementation plan, then implement and test it. Do not stop at
the plan. Current architecture and explicit user decisions take precedence over
archived proposals and notebooks.

## 3. Fix the confirmed notebook interface defect

The latest executed standalone notebook has a malformed optional probe command:
it passes `--output` immediately after the Python interpreter without the script
path, and still passes unsupported `--run-id`. An older packaged notebook may
instead contain only the unsupported-argument variant. Inspect both the working
notebook and the packaged source; fix the entire invocation, not one token.

The current probe interface requires `--device` and `--output`, and does not
accept `--run-id`. The intended invocation is:

```python
if RUN_CUDA_PROBE:
    probe_output = (
        LOCAL_DATA_ROOT / "runs" / "colab" / f"{RUN_ID}_cuda_probe"
    )
    command(
        [
            PYTHON,
            str(PROJECT_ROOT / "scripts" / "probe_device.py"),
            "--device",
            "cuda",
            "--output",
            str(probe_output),
        ],
        PROJECT_ROOT,
        LOG_ROOT / "cuda_probe.log",
    )
```

Keep the following stage/index call intact, while making its profile and run
semantics correct. A prior CPU smoke used `RUN_CUDA_PROBE = False`; it therefore
did not validate this branch. Successfully parsing the command is not GPU proof.

Define safe rerun behavior: use a fresh probe ID or verified compatible evidence.
Do not delete old evidence or infer success from an existing directory. Add a
regression test that captures the actual notebook-generated command and checks
its script path and arguments against the real CLI parser. This test must not
mount Drive, install dependencies or start CUDA. Syntax compilation is insufficient.

Rebuild the standalone notebook, ZIP, checksum and manifest from the same source
snapshot. Never patch an old immutable release in place.

## 4. Implement a shared scalable execution backend

### Pilot workload contract

Start from the current pilot configuration, rather than inventing another
architecture or promising an unmeasured runtime. The reviewed profile specifies:

| Setting | Initial pilot value / rule |
|---|---|
| Learned-model training cohort | At most 5,000 users, stable SHA256 selection using seed 42 among pre-Aug-5 purchasers; no smoke activity threshold |
| Global statistics and ItemCF population | Complete legal-past history; never restricted to those 5,000 users |
| Candidate catalog | Full protocol-legal catalog; no smoke 5,000-item cap |
| Ranking partitions | F1 and Validation only; no need for F2-F4 artifacts in this pilot |
| Query roster | At most 1,000 queries per selected partition; declared deterministic selection, independent of retrieval hits |
| WALS | 64 factors, 3 iterations |
| Two-Tower | 128-dimensional output, batch size 512, at most 20,000 training targets, 1 epoch and 100 optimizer steps |
| Visual clustering | K=128; legal fit population defined by the current protocol |
| LambdaMART | At most 100 rounds; other settings from the current pilot profile |
| Feature query batch | 32 |
| Parallelism | 2 compute threads; 0 loader workers; 1 interop thread |
| Device | CUDA for Two-Tower when explicitly selected; retain CPU implementations for the other components initially |

These are proposed workload limits from the existing configuration, **not a
measured capacity guarantee**. Epoch, target and step caps are maxima; report
actual updates, not the configured upper bound. If the working configuration has
changed, document the difference and its source. Do not silently widen budgets or
alter the loss/candidate semantics to fit a runtime. Changes to batching and
in-batch negatives must be recorded, not described as mathematically equivalent.

Keep distinct manifests for the learned-model training cohort, global statistics
population, candidate vocabulary, per-channel learned coverage, and evaluation
query roster. A full candidate catalog does not imply that every item has a WALS
factor. Report unsupported IDs and apply the documented missing-state behavior;
do not fabricate factors or substitute random scores to claim full coverage.


### 4.1 Data and history access

Replace the pilot's dependency on whole-dataset `Sample.load` and Python
transaction/history objects. Use bounded DuckDB/PyArrow operations and
partitioned canonical data beneath `data/`. Project the needed columns, preserve
string IDs and raw transaction-line semantics, and fetch bounded query batches.

Global legal-past statistics and ItemCF must not be computed only from the pilot
training cohort. Keep the cohort used to limit learned-model training separate
from the population used for dynamic statistics, active pools, and ground truth.
Do not select the learned-model training cohort using Validation outcomes.
Follow the protocol's purchase-active evaluation eligibility rule, then apply a
declared deterministic pilot query limit independent of candidate hits, candidate
scores or model performance. Eligibility based on the benchmark's future-label
rule must never become a retrieval input or a training-cohort selection signal.

Use the pilot configuration above as the starting specification: a stable
past-only base-user subset, full legal candidate universe, bounded target/step
counts, and bounded F1/Validation query rosters. Document these distinctions in
its run manifest. Do not rebrand the existing 5,000-item smoke as a pilot.

### 4.2 Cutoff statistics and ItemCF

Build and reuse cutoff-specific aggregates in bounded, disk-backed operations.
Preserve the current windows, distinct-edge definitions, weighting, top-neighbor
limits, active-pool rules, and deterministic ordering from the architecture.
Avoid a dense item-by-item matrix or an unbounded Python pair dictionary.

Do not add undocumented high-degree truncation or alter the similarity formula
merely to make an implementation fit memory. If an approximation is needed,
identify it explicitly and leave the exact contract gated pending approval.
Verify the bounded implementation against the existing implementation on a
small shared fixture.

### 4.3 WALS and full-vocabulary retrieval

Build the legal sparse training input without first materializing all rows as
Python objects. Preserve confidence weights and true user folding-in against
frozen, unnormalized item factors. Cache item-factor-only work such as `Y.T @ Y`
once per loaded model rather than converting all factors and recomputing it for
every query. Check score equivalence on a fixture.

Replace the smoke-specific 5,000-item index limitation with resource-checked,
profile-aware index construction and batched search. Do not simply delete its
bound while retaining a per-query all-item search followed by a full Python sort.

An exact inner-product backend is acceptable if it is bounded and preserves
semantics. Do not add an approximate index merely because the catalog is larger.
Apply active-pool eligibility correctly before final quota selection. Fixed
small oversampling must not silently omit eligible top candidates. Preserve
article-ID tie-breaking and explicit ID-to-vector alignment.

### 4.4 Two-Tower training, encoding, and device behavior

Connect the profile device to real training and encoding. The current CUDA probe
is separate from `fit_towers`; a `"device": "cuda"` profile does not itself move
models or batches to CUDA. Transfer the model, input tensors, masks and loss
inputs consistently. Record the requested and actual device. Fail clearly when
CUDA was explicitly requested but is unavailable; no silent fallback.

Generate legal training events and bounded histories without retaining all
transactions and all event objects in memory. Preserve the existing
multi-positive LogQ objective, probability/sampling definition, categorical maps,
preprocessing fit boundaries, and internal-validation split.

Implement a real epoch loop and internal-validation checkpoint selection.
Exercise at least two epochs on a tiny synthetic CPU test even when pilot uses
one epoch. Do not use global Validation to fit preprocessing or choose the
retriever's internal checkpoint. Save enough sampler, epoch, optimizer,
scheduler, and RNG state to support the documented continuation guarantees.
CUDA RNG/AMP state must be handled if those execution modes are supported; do
not assert cross-device or cross-platform bitwise training equivalence.

Encode legal item and query vectors in batches, save them with immutable maps,
and build learned indexes from them. The supplied text/image vectors remain
read-only. Their original FAISS indexes are not learned Two-Tower indexes.

### 4.5 Preprocessing, features, and LambdaMART

Fit K=128 visual preprocessing on its documented legal population with bounded
loading. Do not fit it on future data or an unannounced sampled catalog. Preserve
the current algorithm; changing to an approximate clustering algorithm is a
separate methodological decision.

Generate features in bounded query batches and immutable shards. Keep every
query's rows together or reconstruct groups explicitly before fitting. Build
query offsets once instead of scanning a complete customer-ID list for each
query or repeatedly calling `users.count` across the entire table.

Preserve Train-only 1:20 hard/random negative sampling, native positives, and
zero-injection behavior. Validation candidates and ground truth must not inherit
training negative sampling. Retain eligible zero-hit queries in the metric
denominator, including queries with no candidate rows.

Implement resource-aware columnar ranker ingestion. Avoid retaining four folds,
Validation, repeated per-query arrays, and concatenation copies simultaneously.
Estimate actual training rows after the approved sampler, feature matrix size,
LightGBM internal storage, and validation requirements before allocation.

Parquet sharding or a memory-mapped array does not by itself prove out-of-core
LightGBM training. Do not use repeated `init_model` fits over disjoint shards and
call that equivalent to one fit over the intended training population. If a
budget cannot support a valid fit, stop with a resource diagnosis rather than
silently reduce the dataset, feature set, or candidate population.

### 4.6 Profile-aware stages and recovery

The current runner follows a global smoke stage list and writes `is_smoke=True`.
Make stage selection, profile metadata, and model inputs actually profile-aware:

- Smoke keeps its existing end-to-end regression behavior.
- Pilot uses its declared F1/Validation path and recorded bounded training scope.
- Full requires all four folds and complete Validation, but stays gated wherever
  implementation or acceptance evidence is incomplete.

Use truthful states such as `implemented_not_cloud_verified`,
`blocked_implementation`, `blocked_resource`, `passed_synthetic`, and
`passed_real_colab` where appropriate. Do not call a structural test a real run.

Keep completed-stage manifests, compatibility checks, isolated subprocesses,
copy validation, and stage-level resume. Persist results at stage boundaries.
Account for stage working files, checkpoints and committed copies in disk
planning. Reuse existing mechanisms rather than designing another framework.

## 5. Preserve recommendation and evaluation semantics

Retain all six channels and their ordered quotas:

```text
R1  7-Day Trend Popularity         60
R2  Repurchase                    30
R3  Product_Code Sibling Recall   30
R4  Time-Decayed ItemCF           50
R5  WALS                          50
R6  Hybrid Two-Tower              50
```

Use one LambdaMART, frozen base weights, legal dynamic histories, the existing
active-catalog rule and shared training/serving features. The existing rule is at
least 3 purchase lines in the trailing 28 days; compute it from the global legal
history, not from the learned-model training cohort. Apply eligibility before
channel truncation. The native candidate union has at most 270 rows per query
before cross-channel deduplication; do not add an RRF truncation stage.

Preserve the exact non-overlapping split contract, including unused formal folds:

| Partition | Future-label interval, half-open |
|---|---|
| F1 | `[2020-08-12, 2020-08-19)` |
| F2 | `[2020-08-19, 2020-08-26)` |
| F3 | `[2020-08-26, 2020-09-02)` |
| F4 | `[2020-09-02, 2020-09-09)` |
| Validation | `[2020-09-09, 2020-09-16)` |
| Test, disabled | `[2020-09-16, 2020-09-23)` |

The pilot executes only F1 and Validation; it does not move these dates. A cutoff
is the last observable date, and each label starts the next day. Preserve the
separate WALS and Two-Tower fit windows and the Two-Tower internal-validation
window from the current protocol; do not restore an old July training boundary.
All features and histories must be as of their own query cutoff.

Keep ground truth as the complete future set matching the supplied article
catalog. Do not intersect it with the active pool, WALS vocabulary, embedding
availability or recalled items. Preserve zero-candidate and zero-hit queries in
Validation. Legal observations before a later cutoff may update its histories;
that does not authorize gradients or base-factor refitting on future labels.

Do not restore old independent text/image recall, overlapping anchors, oracle
positive injection, future-aware sampling, or random/fixed substitute scores.
Do not claim that active-pool membership measures true stock availability.

The authoritative protocol distinguishes candidate-only diagnostic NDCG from
end-to-end MAP@12. Preserve this distinction. A reduced pilot query roster gives
pilot diagnostics, not full Validation performance. Final Test remains disabled.

Fixed paths:

```text
Drive source: /content/drive/MyDrive/projects/e_commerce_recsys/data/
Local project: /content/e_commerce_recsys/
Local data: /content/e_commerce_recsys/data/
Drive outputs: .../data/runs/colab/<run_id>/
```

Stage only `original data/`, `text_features/`, and `image_features/`. Do not copy
all historical runs, overwrite inputs, extract embeddings again, train on the
mounted Drive directory, or load the original content indexes as learned indexes.

## 6. Bound local work and separate cloud evidence

The development machine remains an M1 Max with approximately 7–8 GB available
memory. Keep the existing conservative local safeguards. Locally, run synthetic
unit/integration tests and only necessary existing-scale smoke regression.
Never start real pilot/full fitting or full candidate generation on the Mac.

Use tiny fixtures to exercise the real scalable code path under a dedicated
test harness. Test-only overrides must not bypass production scale gates for
real data. A standalone synthetic index test with more than 5,000 items is
appropriate; it does not require a correspondingly large training job.

Do not automatically alter dependency versions or regenerate `uv.lock`. Where
an actual implementation requires a dependency change, justify and lock it,
update the environment identity, and retain strict installation failure behavior.

Without actual Colab access, deliver executable code, tests, rebuilt artifacts,
and exact operator commands, but label **this new release's** Drive/Linux/CUDA/
real-pilot execution `not run`. Preserve the earlier real Colab CPU smoke evidence
as historical evidence; do not claim it validates the new backend. Do not promise
background execution or invent cloud measurements.

## 7. Required tests and acceptance

Add focused tests for:

1. Notebook-generated probe arguments matching the real CLI contract, including
   safe rerun behavior and no accidental device computation during parser tests.
2. Bounded history and cutoff statistics matching a reference fixture, including
   date boundaries, duplicate purchase lines, and histories spanning partitions.
3. Full legal candidate maps exceeding 5,000 items, deterministic ties, and
   active filtering that still finds the correct eligible top-K.
4. ItemCF and cached-Gram WALS folding-in matching the current reference scores
   within an explicitly justified numerical tolerance.
5. Pilot stage dispatch using F1/Validation without requiring missing F2–F4
   artifacts, and full dispatch remaining appropriately gated.
6. Real model updates through the scalable backend, multi-epoch state and
   checkpoint continuation on a small CPU fixture; CUDA tests clearly skipped
   rather than represented as passed without a CUDA device.
7. Sharded features, complete groups, the approved negative sampler, retained
   zero-hit Validation queries, metrics and reload parity.
8. Resume after runtime-directory deletion and rejection of incompatible input,
   schema, device/runtime, code, and configuration identities.
9. Resource rejection without silent sampling, final-Test rejection, and local
   real pilot/full rejection.

Run targeted tests, the existing regression suite, Ruff, formatting and Pyright.
Do not modify unrelated archived notebook lint failures. Do not weaken a test
just to obtain a green result. Record exact commands, exits and actual evidence.

The milestone requires real pilot execution logic, not merely a passing plan.
If a blocker remains, identify its concrete code location, missing operation,
and completed work; mark the milestone incomplete without deleting safeguards.

## 8. Deliverables and handoff

Deliver:

- Updated shared execution backend and an unchanged functional smoke interface.
- Fixed thin notebook with smoke as the default and separate pilot opt-in.
- An executable pilot profile, truthful stage status, and bounded test fixtures.
- Updated runbook and implementation report distinguishing code completion from
  actual Colab, CUDA, and real-data verification.
- Rebuilt standalone notebook, ZIP, checksum, and manifest from one consistent
  source snapshot. Validate clean extraction and necessary entry points.

Add `docs/archi/colab_pilot_runbook.md`, or an equally clear pilot section in
the existing runbook, as the user's actual operating instructions. Include:

1. Exact standalone notebook, ZIP, SHA256 and manifest paths; the new release ID;
   and the Drive `releases/` upload destination.
2. A copyable notebook configuration for a new pilot run, using the real implemented
   variable names. Set `RUN_PROFILE = "pilot"`, a fresh pilot `RUN_ID`,
   `RUN_ACTION = "run"`, `ALLOW_FULL_RUN = False`, and `RUN_FINAL_TEST = False`.
   Add an explicit pilot opt-in if appropriate. Do not make pilot require the
   full-training opt-in or reuse an old smoke run ID.
3. Exact cell order and CLI commands for plan/preflight, CUDA probe when requested,
   pilot execution, status inspection and resume. The notebook must actually
   dispatch the pilot stages, not only print a plan or blocked message.
4. Which stages may be restored in a new runtime, which incomplete stages restart,
   how to use the same compatible release/profile/run ID, and how to select a
   new probe output directory safely.
5. Actual completed tests and their exit codes, plus honest labels for cloud/GPU
   checks that have not run. Preserve existing smoke regression evidence and
   explain any intentional numeric differences caused by bounded refactoring.
6. A small downloadable pilot review bundle: resolved config, identities, query/
   vocabulary counts, stage status, resource measurements, run summary, per-channel
   recall and native-union diagnostics, active-pool coverage, MAP@12/NDCG@12,
   reload checks and relevant error summaries. Do not include raw customer IDs,
   purchase histories or full feature tables in a public report.

Keep Notebook defaults on smoke so Run All does not accidentally start pilot.
Provide the explicit pilot configuration separately in its Markdown instructions.
Use a new release and run ID after code changes; old smoke manifests must not be
relabelled or reused as compatible pilot stages without actual identity checks.

The handoff must say whether the **pilot implementation is executable** and,
separately, whether it has run on real Colab. If the implementation remains
incomplete, list concrete code paths and missing operations. Do not claim success
merely because all tests for a blocked CLI pass. Keep full-only blockers separate.
Stop at this milestone: no full run, final Test, deployment or unapproved
architecture changes.
