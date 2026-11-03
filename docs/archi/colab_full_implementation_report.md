# Full Training Adapter Implementation Report

## Contents

- [Scope and evidence](#scope-and-evidence)
- [Implementation](#implementation)
- [Validation](#validation)
- [Compatibility and remaining risks](#compatibility-and-remaining-risks)
- [Delivery](#delivery)

## Scope and evidence

This milestone implements full training, four-fold single-LambdaMART fitting,
complete native Validation and new-process inference. The user reports a
successful real Colab CUDA pilot on the preceding implementation. That is
historical user-supplied evidence, not a new full-data measurement.

The workspace was clean at inspection. Existing environment, models, shared
features/evaluation, input staging and stage persistence were reused. No real
H&M full/pilot job was run on the Mac, no source embedding was changed, and no
Git initialization, commit, push, final Test or deployment was performed.

## Implementation

- Generalized existing pilot dispatch for full across Notebook, CLI, stage
  selection and workers. Full uses 25 stages, F1–F4 plus Validation, and one
  jointly fitted ranker. Explicit authorization and local real-full rejection
  remain independent gates; the implementation blocker was removed only after
  executable operations and tests were added.
- Full null settings mean all legal base users, all Train unique target events,
  all eligible queries and no optimizer-step cap. Full WALS includes legal users
  through the base boundary, not the pilot pre-Aug-5 subset. No active-history
  minimum, 5,000-item cap or implicit sampling is inherited.
- Added bounded Parquet event iteration with row-group-aware offset recovery.
  Stable SHA256 target order, Train-only item probabilities, loss, multi-positive
  masking and ten-epoch internal selection are preserved. Customer category
  fitting and internal daily query reads are batched rather than all-user lists.
- WALS builds the same confidence CSR from Arrow batches after checking exact
  dimensions/nnz. Frozen factors, cached-Gram folding-in and learned indexes
  retain their existing semantics. Original text/image indexes stay read-only.
- Full query rosters and complete labels are Parquet. Query vectors are memmaps;
  feature-manifest groups are streamed JSONL. Features and label attachment use
  bounded complete-query batches. Top-12 Parquet and aggregate metrics are
  written incrementally, including zero-hit/zero-candidate queries.
- Added separate disk/RAM estimates for WALS, targets/towers, four-fold features,
  complete Validation, group metadata and the actual single-ranker fit. A memory
  map is not claimed to remove LightGBM's memory requirement. No repeated
  `init_model`, truncation or hidden approximation substitutes for full fitting.
- Full checkpoint retention keeps the latest verified complete state plus its
  embedded best-model weights. Only a prior checkpoint from the same identity
  is retired after successful publication; pilot retention and completed stage
  artifacts are preserved. Failure records include safe capacity estimates.
- Updated the thin notebook and added a full operator runbook. The build emits
  matching ZIP/checksum/manifest and resolved full instructions outside the
  archive. Dependency declarations and `uv.lock` are unchanged.

The six channels, quotas, disjoint weeks, full ground truth, feature schema,
Train-only native 1:20 hard/random sampling and external MAP@12 remain unchanged.
The existing full batch size 512 and all full compute settings are retained.
Full fixtures change input data size only, plus explicit CPU device selection;
they do not lower factors, iterations, epochs, K=128, ranker limits or remove
null limits. Authenticated fixture bytes are required for the local test hook.

## Validation

Tests use the existing locked Python 3.12 uv environment. Commands are prefixed
with `UV_CACHE_DIR=/tmp/hm-smoke-uv-cache` and use
`/Users/vincent/.local/bin/uv`; no environment/dependency reconstruction occurred.
Evidence is retained under
`data/runs/colab/full_local_validation_20261102_01/`.

| Actual command/check | Exit | Result |
|---|---:|---|
| `uv run --locked ruff check src scripts tests notebooks/01_Colab_Training.ipynb` | 0 | All checks passed |
| `uv run --locked ruff format --check src scripts tests notebooks/01_Colab_Training.ipynb` | 0 | 54 files already formatted |
| `uv run --locked pyright` | 0 | 0 errors, 0 warnings |
| `uv run --locked pytest -q --basetemp=data/runs/colab/full_local_validation_20261102_01/pytest_final --junitxml=data/runs/colab/full_local_validation_20261102_01/pytest_final.xml` | 0 | 40 passed, 1 CUDA test skipped, 111.25 seconds |
| `uv run --locked ruff check .` | 1 | 122 pre-existing historical notebook errors, outside scope |
| `uv run --locked ruff format --check .` | 1 | 8 pre-existing files would be formatted; left untouched |
| `uv run --locked python -c ...` with `nbformat.validate` and `compile` on every code cell | 0 | 9 code cells valid; no notebook execution |
| `uv run --locked python scripts/run_pipeline.py plan --profile full --persistent-root .../unused_persistence` | 0 | All 25 stages; planning only |
| `git diff --check` | 0 | No whitespace errors |

Exact arguments and output logs are in `static_checks.json`. The complete suite
is recorded separately in `pytest_final.log` and `pytest_final.xml`. It covers
legacy smoke/pilot behavior, actual full workers, temporal boundaries, masks,
quotas, full metric denominators, sample/group rules, immutable stage contracts,
interrupted copies/stages, incompatible recovery, unauthorized full rejection,
Notebook-generated CLI commands and a separate 6,001-item exact index.

During development, the cross-row-group offset test exposed a skipped-row bug
in the new reader; it was fixed and retested. A pilot regression exposed a
resource-count field named `users` that conflicted with the review privacy
contract; it is now `user_count`. The initial complete-suite run returned 1
(39 passed, 1 failed, 1 skipped): its zero-candidate test copied a newly added
immutable ranker group audit into a fresh fit. That test now excludes this
output when preparing input; the production no-overwrite guard remains intact.
Earlier failure logs are preserved, not replaced by the final run.

The actual full synthetic fixture has 24 users, 170 catalog items and 4,571 legal
history lines. It executes unchanged full compute settings on CPU, not a pilot
profile under another name:

| Measured operation | Result |
|---|---|
| WALS | 64 factors, 15 iterations, 1,088 nonzero user-item observations |
| Two-Tower | 1,584 unique targets, 10 epochs, 40 updates, 15,840 processed targets |
| Single LambdaMART | 96 groups across F1–F4; 4,938 sampled rows, 480 positive rows |
| Ranker stopping | 52 rounds evaluated; selected model retains 2 rounds and 15 splits |
| Native feature rows | F1: 2,031; F2/F3/F4/Validation: 984 each |
| Complete fixture Validation | All 24 queries retained, including one zero-hit query |
| Fresh-process reload | 984 candidate rows, 24 unique Top-12 lists, score max absolute difference 0 |
| Same-run runtime replacement | Local run deleted after F2; verified stored stages restored and completed |
| Exact checkpoint continuation | Intentional failure after step 1; ten-epoch weights, losses, selection and vectors match uninterrupted CPU run |
| Sampled process-tree RSS | 840,368,128 bytes (about 0.783 GiB), including supervisor |
| Stage timing | 25 stages, 20.41 summed worker seconds; excludes staging/publication/restore overhead |

All six channels emitted candidates in every partition. Their per-fold counts
and full measurements are in `full_fixture_measurements.json`. A pilot regression
also exercises wholly zero-candidate Validation with authentic saved models.
Synthetic accuracy values are retained as test evidence, not an H&M benchmark.
The sampling monitor can miss short RSS peaks; these measurements do not predict
full-data RAM, disk, CUDA use or execution time. Local machine availability is
not used to expand the user's authorized workload.

New-release cloud/CUDA/full-data execution remains **not run**.

## Compatibility and remaining risks

Full must use a new release and new full RUN_ID. No pilot stage is relabelled or
imported across runs. Same-run restore requires exact input/code/config/schema/
cutoff/runtime identity and output hashes. The standalone notebook and packaged
copy must match. Existing source data, prior runs and historical notebooks remain
untouched.

The full backend is executable, but this does not prove real H&M fits Colab.
Global ItemCF, sorting, all-fold features and complete Validation can require
substantial disk/time; one LambdaMART may exceed available RAM. Resource checks
stop rather than shrink the experiment. Sampled RSS can miss brief peaks.
Actual CUDA full training/continuation, full-data resource feasibility, and
fresh Colab-runtime full restore need operator verification. Final Test remains
disabled; no hyperparameter search or deployment is included.

## Delivery

See [full runbook](colab_full_runbook.md) and its resolved companion under
`data/releases/` for exact upload, configuration, status and resume commands.
The release identity is resolved outside packaged documents to avoid hashing a
file that embeds its own archive hash. Verification records include clean
extraction and real entry-point checks. Stop at this implementation milestone.
