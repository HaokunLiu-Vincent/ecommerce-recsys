# Local Smoke Milestone Report

## Result and scope

The first real H&M smoke pipeline passed with run ID `real_v1`. This is a
bounded integration result, not a benchmark or completion of full P1–P5.
Four native ranker folds and global Validation ran; final Test labels were
never materialized, used for selection, or evaluated. No deployment ran.

The active specification is `docs/proposal.md` v0.3.2 and Evaluation Protocol
v2.0. There is no root `proposal.md` or Git repository in this checkout.
No Git repository was initialized. Existing notebooks, archives, AGENTS.md,
source CSVs, supplied embeddings, maps, manifests, and content indexes were
preserved. No archived implementation was imported.

## Implemented files

- `configs/smoke.json`: explicit CPU settings and strict count/memory caps.
- `scripts/local_smoke.py`: preflight, synthetic, real-smoke, and reload CLI.
- `src/hm_recsys/`: contracts, read-only input adapters, sampling, resources,
  six-channel recall, real learned models, shared named features, ranking,
  independent metrics, sequential stage orchestration, and synthetic fixture.
- `tests/conftest.py`, `tests/test_smoke_contracts.py`: bounded offline tests.
- `pyproject.toml`, `uv.lock`: add PyTorch 2.15.0 and psutil 7.2.2 to the
  existing uv development environment; add the pytest `src` import path.
- This report. Runtime artifacts are confined to `data/runs/smoke/`.

The existing Ruff policy was retained. No existing notebook/style error was
suppressed or fixed as an unrelated change.

## Sampling and trained models

The first and only real sample selects 500 users by SHA256(`42:customer_id`)
from users with at least 20 distinct purchase dates in `[2020-02-01,2020-08-05)`.
This eligibility rule was fixed before the first real extraction; no seed or
cohort was retried using recall hits or Validation metrics. It concentrates the
smoke population on historically active customers and is not representative of
all H&M users.

The sample contains **53,642 complete purchase lines**, including repeated
lines, from 2020-02-01 through 2020-09-15. The **5,000-item candidate catalog**
uses pre-Aug-5 sample counts and static product-code sibling fill. Candidate
statistics use the sampled interaction universe. The active-pool rule remains
three purchase lines in 28 days. Future truth is not intersected with this
catalog, active pools, WALS factors, content coverage, or native candidates.

- WALS: 16 factors, **2 actual iterations**, confidence applied once with
  library alpha 1, 18,871 nonzero CSR entries, true fixed-item user solve.
  Its fixed vocabulary intersects the legal June-16–Aug-11 observations with
  the smoke candidate catalog; outside items remain unknown to WALS.
- Two-Tower: 384d supplied text and 512d supplied image inputs, both frozen;
  trainable projections, structured/profile/history branches, shared ID dropout,
  128d output, duplicate-aware multi-positive LogQ loss. Selected 5,000 legal
  targets; **50 optimizer updates / 1,600 processed events** in one bounded
  pass. Text projection maximum parameter change: **0.0193169**. The zero UNK
  row is frozen; padding is independently masked by history weights.
- Visual clustering: K=8, at most 1,000 pre-base images, one initialization and
  at most 10 iterations; frozen before ranker features.
- LambdaMART: **30 actual rounds, 166 splits**, 966 sampled training rows in
  39 valid contiguous groups, including all 46 native positives. No-pair groups
  dropped only from Train: F1=39, F2=40, F3=41, F4=41. No positive injection.
- Every fold and Validation retains its predetermined 50-query population.
  Validation uses all native candidates and retains zero-hit queries.

## Native channel outputs

Counts below are measured channel rows before cross-channel deduplication.
The complete deduplicated union goes to the shared 49-column FeatureBuilder;
RRF does not truncate it.

| Partition | Queries | Union rows | Trend | Repurchase | Sibling | ItemCF | WALS | Two-Tower |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| F1 | 50 | 8249 | 3000 | 239 | 132 | 1805 | 2500 | 2500 |
| F2 | 50 | 8037 | 3000 | 216 | 118 | 1684 | 2500 | 2500 |
| F3 | 50 | 7910 | 3000 | 197 | 123 | 1879 | 2500 | 2500 |
| F4 | 50 | 6387 | 3000 | 111 | 79 | 1045 | 2500 | 2500 |
| Validation | 50 | 5487 | 3000 | 69 | 47 | 671 | 2500 | 2500 |

There are **36,070 feature rows** over the five partitions. Fresh-process
Validation reconstruction adds 5,487 rows, so both materializations together
contain 41,557 rows, below the 100,000-row cap. All six channels produced
nonempty real outputs; an individual query may still have empty channels.

Validation diagnostics: native macro MAP@12 **0.00416667**, Recall@12
**0.01111111**, and NDCG@12 **0.01025544** over all 50 queries. Native union
recall is **0.02106349**; 44 queries have no retrieved positive and remain in
the denominator. These small-sample figures were not used for tuning.

## Fresh-process inference and artifact compatibility

PyTorch encoding, real FAISS IndexFlatIP retrieval/shared features, and
LightGBM scoring execute in sequential fresh processes. This isolation resolves
observed macOS OpenMP runtime conflicts without unsafe duplicate-runtime flags,
package binary patches, algorithm substitutions, or a NumPy retrieval fallback.
The Two-Tower internal daily Recall@50 diagnostic uses bounded exact dot
products; native R6 and saved learned indexes use actual FAISS.

Reload reads a separate history extract ending before 2020-09-09, frozen
weights/maps/schema/content references, and the query roster. It does not read
label files or refit. All 5,487 candidate IDs/order and feature rows matched;
feature tolerance is rtol=1e-5/atol=1e-6. All 50 unique Top-12 lists matched.
Prediction maximum absolute difference was **0.0** with score tolerance
rtol=1e-6/atol=1e-7. No real query needed a short list; unit tests cover short
lists and zero-candidate metric denominators.

Source maps were checked against their declared SHA256 values. Numeric checks
covered all **16,189 consumed text rows** and **16,171 consumed image rows**
(the subset includes history/label metadata outside the candidate catalog).
Original arrays were opened with read-only mmap. Whole-array cryptographic
checksums were not recomputed; declared hashes and unknown upstream provenance
are retained in the acceptance report. Source size/mtime identities remained
unchanged. Neither supplied raw-content FAISS index was loaded.

## Memory and execution evidence

The supervisor sampled its owned process tree every **0.2 seconds**. Measured
peak tree RSS was **0.501831 GiB**; minimum system available memory was
**8.932693 GiB**. No real stage crossed the 4 GiB target, 5 GiB abort threshold,
or 2 GiB available-memory floor. The termination path was tested with a
synthetic threshold crossing, without allocating excessive memory.

DuckDB used two threads, a 1 GB memory limit, and a 2 GiB spill limit inside
this run. The raw CSV was scanned sequentially for selection/count/extraction;
it was not loaded into pandas. Monitoring is sampled, not an OS hard cap, and
may miss between-sample peaks or double-count shared resident pages.

| Stage | Seconds | Sampled peak tree RSS (GiB) |
|---|---:|---:|
| preflight | 0.87 | 0.097 |
| sample | 11.23 | 0.502 |
| wals | 0.65 | 0.176 |
| towers | 5.62 | 0.495 |
| index_towers | 0.44 | 0.117 |
| encode | 1.73 | 0.462 |
| preprocessing | 1.95 | 0.311 |
| features | 3.88 | 0.299 |
| ranker | 1.09 | 0.201 |
| reload_encode | 1.73 | 0.465 |
| reload_features | 1.52 | 0.273 |
| reload | 1.11 | 0.174 |

Total measured stage wall time was **31.83 seconds**, excluding development,
dependency installation, and separate quality checks.

## Commands actually executed and status

The commands below used this exact uv executable/cache prefix:

```bash
UV_CACHE_DIR=/tmp/hm-smoke-uv-cache /Users/vincent/.local/bin/uv
```

| Arguments after the prefix | Exit | Evidence |
|---|---:|---|
| `add --dev torch psutil` | 0 | Updated existing environment and lock; initial sandbox DNS attempt exited 2 |
| `run --locked python scripts/local_smoke.py preflight --run-id preflight_v1` | 1 | Sandbox denied process-tree sysctl; no worker started |
| `run --locked python scripts/local_smoke.py preflight --run-id preflight_v2` | 1 | Duplicate OpenMP initialization before training |
| `run --locked python scripts/local_smoke.py preflight --run-id preflight_v3` | 1 | Runtime import-order attempt also failed; not accepted as a solution |
| `run --locked python scripts/local_smoke.py preflight --run-id preflight_v4` | 0 | Input checks passed after stage isolation |
| `run --locked pytest -q tests/test_smoke_contracts.py -k 'not guarded_integration'` | 0 | Initial 10 contracts passed |
| `run --locked pytest -q tests/test_smoke_contracts.py` | 0 | Initial 11 tests, including model integration, passed |
| `run --locked pytest -q` | 1, then 0 | First 15 passed/1 supervisor exit-accounting assertion failed; corrected implementation, final 16 passed |
| `run --locked pytest -q tests/test_smoke_contracts.py -k 'supervisor or limits or training_rejects'` | 0 | 3 focused guard tests passed |
| `run --locked python scripts/local_smoke.py synthetic --run-id synthetic_v1` | 0 | Persistent complete synthetic pipeline and fresh reload |
| `run --locked python scripts/local_smoke.py reload --run-id synthetic_v1` | 0 | Independent explicit replay; original completed artifacts preserved |
| `run --locked python scripts/local_smoke.py real-smoke --run-id real_v1` | 0 | First/only real sample, three model fits, six channels, fresh reload |
| `run --locked pytest -q -k 'not guarded_integration'` | 0 | Post-real check: 15 passed; already-passed training integration deliberately not repeated |
| `run --locked pyright` | 0 | 0 errors, 0 warnings; rerun after real smoke |
| `run --locked ruff check src scripts tests --output-format concise` | 0 | All current Python source/tests pass |
| `run --locked ruff format --check src scripts tests/conftest.py tests/test_smoke_contracts.py` | 0 | 18 new Python files formatted |
| `run --locked ruff check . --output-format concise` | 1 | 122 pre-existing notebook violations, untouched |
| `run --locked ruff format --check .` | 1 | 6 pre-existing notebooks need formatting, untouched |

Ruff formatting and safe lint fixes were applied only to newly created source,
CLI, and tests during development. Initial type/lint errors in that code were
fixed. The existing text-notebook tests also passed; no notebook was executed.
The current source fingerprints match the code recorded by `real_v1`.

Separate acceptance statuses: synthetic contracts **passed**; sampled real
end-to-end **passed**; fresh-process parity **passed**; new-source quality checks
**passed**. Repository-wide Ruff remains **not clean** because unrelated existing
notebooks are preserved. Exact quality output is retained in `real_v1/*.log`.

## Artifact locations and remaining limits

All real results are under `data/runs/smoke/real_v1/`, including:

- `sample_manifest.json`, `input_acceptance.json`, `split_manifest.json`,
  `config.resolved.json`, `run_manifest.json`, and `feature_schema.json`.
- `models/`, `indexes/`, `queries/`, `labels/`, `features/`, `candidates/`.
- `predictions/reference_top12.json`, `predictions/reloaded_top12.json`,
  reference/reloaded scores, and reloaded features (private customer outputs).
- `smoke_results.json`, `reload_results.json`, `resource_usage.json`,
  `verification_results.json`, `commands.log`, and individual stage logs.

Persistent synthetic evidence is under `data/runs/smoke/synthetic_v1/`;
its explicit replay is under `synthetic_v1/reloads/`. Failed preflight attempts
remain isolated for audit. Tests use temporary fixtures.

Full-data fitting, final Test evaluation, tuning searches, Colab/GPU execution,
serving, Docker, cloud services, and deployment are **not run**. Whole embedding
arrays were not numerically audited, source snapshot provenance was not
independently reproduced, and predictive quality/production readiness is not
established. The current stage isolation is verified on this environment;
other package versions require compatibility checks. Further full-phase work
requires a separate task. The smoke milestone stops here.
