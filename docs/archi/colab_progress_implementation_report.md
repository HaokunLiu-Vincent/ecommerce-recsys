# Colab Live Progress Implementation Report

## Scope

The current Notebook is a smoke/full operator client. Pilot and standalone
benchmark CLIs remain compatible. New release/run IDs protect old artifacts;
no running Colab job, original data or supplied embedding was modified.
The local Mac runs only bounded synthetic fixtures and real small model tests.
Real Colab UX/CUDA/full execution is not claimed.

## Event and execution path

`hm_recsys.progress.emit` writes flushed `HM_PROGRESS ` JSON lines to stdout and
an owned local JSONL. Version 1 fields: UTC timestamp, run/profile/stage/task,
actual completed/nullable total/unit, monotonic elapsed, optional recent rate
and estimated ETA, message, structured aggregate metrics, and status. Periodic
updates are throttled to one second per task; first/final/failed/epoch/iteration
milestones may force emission. No-total operations display indeterminate/ETA
unknown. Failed events have no percentage. Records are limited to 8 KiB.

The resource supervisor tails its real inner worker log in bounded chunks and
forwards flushed events while maintaining its original RSS guard. The Notebook
command helper tails the outer CLI log while that process is alive, invokes the
live UI callback immediately, prints a compact fallback and real argv, and sends
silence heartbeats. A quiet SQL operation never receives a fictional row total.
The UI table is bounded and redraws at most once per second between critical
milestones; fast per-file human text is also throttled while all events remain
in private logs and callbacks. Final unterminated stderr is drained. Failed
leaders and interrupted commands terminate their entire owned process group,
even if the original leader already exited; the last real work events are
retained separately from a bounded traceback tail. The UI table is bounded; complete private logs remain on local disk. Displayed
unstructured text redacts H&M IDs and potential purchase rows. Events contain
aggregates, never customer identities, purchase lines or per-query labels.

Stage completion is emitted only after validation and verified persistence.
Compatible stages are RESTORED, not TRAINED NOW; workers are not rerun. Progress
artifacts `progress/<stage>.jsonl` live in immutable stage deltas. The coordinator
keeps `live_progress.jsonl`; command logs have adjacent `.events.jsonl` files.
Live events are not individually written to mounted Drive. Stage publication
preserves checksums, writer leases, dependency identity and failure propagation.

## Changes by file

| File / function | Previously hidden work | Actual implementation |
|---|---|---|
| `progress.py` emit/scope/Lines/describe | No structured worker observations | Bounded, flushed local event contract, nullable rates/ETA and per-task clocks |
| `colab/notebook.py` command | Outer stdout visible only after exit | Live bounded tail/callback, text fallback, heartbeats, group cancellation, failure tail |
| `resources.py` run_stage | Inner stdout isolated in a separate file | Tail inner log during execution; retain RSS guard and real exit status |
| `colab/runner.py` _execute_profile | Running/verified text only | Identity-bound coordinator context, real stage start and verified/restored status |
| `colab/state.py` persist/execute | Persistence/failure invisible | Actual delta bytes, start/verified completion and safe failure type |
| `colab/files.py` verified_copy/stage_inputs | Large copy/checksum opaque | Actual bytes and 12-file whitelist counters; verification separate; failed copies do not publish success |
| `colab/checkpoints.py` | Save and Drive publication indistinguishable | Local saved, remote publishing/verified/failed and restored milestones |
| `pilot/data.py` canonical/adapt_content/rosters | Blocking SQL and embedding maps | Unknown scan phases; actual retained rows/partitions, mapped vectors, dimensions/missingness and roster counts |
| `pilot/capacity.py` check | Resource decisions hidden | Measured estimate payload and accepted/rejected gates before fitting |
| `compute/prepared.py`, `compute/query_inputs.py` | Raw prepack/cache work hidden | Actual contexts, target/item/profile/history-slot counts and verified cache reuse |
| `learned.py`, `pilot/models.py` WALS | No ALS iteration feedback | Supported callback after actual ALS updates, CSR shape/nnz and latent width |
| `towers.py`, `pilot/towers.py` | Epochs/batches/loss/internal selection hidden | Actual consumed targets, batch/epoch/global updates, loss, host step seconds, allocator samples, catalog/internal query counts and checkpoint/selection status |
| `recall.py`, `pilot/snapshot.py` | Sparse ItemCF work hidden | Actual completed seeds/ranges and neighbors; estimated directed pairs remain explicitly estimates |
| `features.py`, `pilot/models.py` preprocessing | KMeans opaque | Fit start has unknown total; actual n_iter_ and valid image population after fit |
| `pipeline.py`, `pilot/features.py` | Native feature workload hidden | Completed whole queries/native rows/schema/shards; no sampled-row substitution |
| `ranking.py`, `pilot/ranking.py` | Fit and scoring opaque | Dataset rows/groups/positive count, real boosting callback, diagnostic candidate NDCG, actual early-stop iteration and complete Validation/reload row counts |
| `pilot/ranking.py` ExpectedRows | Existing reload compared at most32 reference rows with up to512 query groups | Bounded exact-row consumption across Parquet batches; >32/uneven/EOF regression |
| Notebook | Pilot/benchmark clutter and mechanical comments | Two modes, clean English input/output/command narrative, actual stage table, aggregate results and recovery |
| Runbook, README, release builder | Old client instructions/package allowlist | Current two-mode instructions, protocol navigation and matching source package |

## Counters and limits

WALS iterations, tower batch/epoch, catalog encoding, ItemCF seeds, complete query
rosters, feature rows, boosting and persistence bytes have real denominators.
Configured boosting rounds are an upper bound; early stopping may complete below
100%. Candidate NDCG is labelled diagnostic, internal Recall@50 is distinct from
native end-to-end MAP@12. Opaque SQL scanned-row counts and KMeans live iteration
percentage are unknown. Post-query retained counts are not called scanned rows.
GPU allocated/reserved samples use existing CUDA APIs without synchronization;
host step seconds are not CUDA kernel times. GPU utilization is unavailable
unless measured independently; low VRAM is not interpreted as idleness.

## Validation evidence

Final command statuses, actual model/counter comparisons, live arrival timing,
Notebook structure checks, bounded overhead observations and packaging evidence
are recorded after execution under
`data/runs/benchmark/colab_live_progress_20261103_01/`.
The implementation preserves model objectives, seeds, batch boundaries,
hyperparameters, six quotas, disjoint folds, full truth, Train-only sampling,
feature schema, one LambdaMART and checkpoint contents.

### Executed local checks

The final locked-uv suite exited **0: 59 passed, 2 CUDA skips in 167.31 s**.
It includes a real delayed child inside the real resource supervisor, observed
by the Notebook helper before the child's post-event marker/exit; exact counts
1/3, 2/3, 3/3 survived both process layers. It also covers unknown totals,
100,000-character output truncation, real nonzero exit reasons, unterminated
stderr, cancellation and an exited leader with a still-running descendant.
The failed-leader descendant cannot perform its delayed completion side effect.

The actual shared full backend fixture uses 24 users, 170 items and 4,571 raw
synthetic purchase lines. It made 40 real Two-Tower updates over ten epochs
(15,840 consumed targets), complete internal validation (72 daily queries),
F1–F4 plus 24-query native Validation and fresh-process reload. Persisted worker
counters are checked against their units and artifact populations. Observations:

| Task | Actual final observation |
|---|---|
| WALS | 15/15 iterations; CSR nnz 1,088; latent width 64 |
| Two-Tower | 1,584/1,584 targets in each of ten epochs; global step 40; width 128 |
| Item encoding / internal validation | 170/170 items; 72/72 internal queries |
| F1 ItemCF | 168/168 seed items, without claiming estimated pairs completed |
| Features | 24/24 whole queries in each F1–F4/Validation; schema 49 |
| Ranker | 4,938 sampled Train rows, 96 groups, 480 positives; 984 unsampled Validation rows |
| Boosting | 52 actual iterations against configured ceiling 1,500; early stop selects/retains best iteration 2 |
| Reload | 24/24 queries; Top-12 and score/feature parity passed |

The recorded `before_after_pipeline_parity.json` compares this run with the
preceding deterministic pre-progress full fixture, using the same data/config:
all 120 fold queries/5,967 native candidate rows and feature values match exactly;
the ranker model/group bytes, reference/reloaded Top-12 and full-truth metrics
match. `model_parity.json` additionally checks exact WALS factors, Two-Tower
parameters/item vectors, losses, epoch selection and consumed targets. These
are synthetic production-backend checks, not real H&M quality results.

Independent tiny real-library on/off checks also exited0: two ALS iterations
produce exact user/item factors, and five LambdaMART iterations produce identical
model text and predictions. Their callbacks do actual work; they are not fake
model outputs. Eight already verified upstream stages were restored into a
replacement local filesystem and labelled RESTORED instead of trained again.
The existing interruption/checksum/failed-copy/failed-persistence/resume tests
remain passing. Actual cloud recovery remains unexecuted.

An emitter-only CPU microcheck made 10,000 calls: disabled 0.00289 s versus throttled
0.01298 s; only two events/660 bytes were persisted. This is not a whole-model
or CUDA overhead estimate. First/final epoch and iteration milestones bypass
throttling, so their cost depends on real work duration. No instrumentation
speedup or fixed finish-time claim is made.

| Actual command/check | Exit | Result |
|---|---:|---|
| `uv run --locked pytest -q --basetemp=.../verified_tests --junitxml=.../verified_pytest.xml` | 0 | 59 passed, 2 CUDA skips |
| `uv run --locked ruff check src scripts tests notebooks/01_Colab_Training.ipynb` | 0 | Maintained scope passed |
| `uv run --locked ruff format --check src scripts tests notebooks/01_Colab_Training.ipynb` | 0 | Maintained scope passed |
| `uv run --locked pyright` | 0 | No errors/warnings |
| `uv run --locked ruff check .` | 1 | Existing 122 old/archive Notebook issues retained |
| `uv run --locked ruff format --check .` | 1 | Existing eight old/archive Notebooks retained |
| `nbformat.validate` plus code-cell `compile` | 0 | 19 cells/10 code cells; safe defaults and two-mode CLI contracts |
| Actual WALS and LambdaMART callback parity programs | 0 | Exact factors/model/predictions |
| Full-fixture before/after model and pipeline comparison programs | 0 | Exact saved model/features/candidates/Top-12/metric populations |

Numerical checks use two native threads and the existing 5 GiB local RSS ceiling.
The largest sampled process-tree RSS in the final guarded suite was 956.03 MiB; this is a sampled observation, not an OS hard cap.
No dependency declarations or lockfile changed. Early runs exposed only comment
coverage omissions after import sorting; those were fixed without weakening the
check. The first default-sandbox nested process test lacked macOS process-list
permission; it was rerun with the existing RSS monitor permissions rather than
disabling resource protection. All early logs are preserved. The latest complete
suite is green. Notebook/UI rendering, CUDA and Drive behavior are explicitly
not run in Colab; syntax validation is not cloud execution evidence.

Exact argv/exit records, `counter_evidence.json`, callback event logs, overhead
summary, model/pipeline parity and final packaging verification live in the
evidence directory. Source-only packaging/clean extraction/checksum checks passed, including five
CLI help contracts and 13 focused tests executed from the clean extracted
package. Build and verification records accompany the final source package. The standalone Notebook equals its
packaged copy; the resolved `.live_progress_runbook.md` identifies that release.

## Recovery and remaining validation

Follow [the current live runbook](colab_live_progress_runbook.md). A new runtime
uses the same compatible release/profile/device/run ID and RUN_ACTION=resume.
Only recover a stale lease after confirming the old writer stopped. Failed
stages keep their last real observation and bounded inner traceback, never a
completion marker. Hard termination cannot promise a final checkpoint; progress
since the last verified remote snapshot may be lost.

Local filesystem recovery and synthetic real models do not establish actual
Colab widget rendering, CUDA timings, full-data capacity, Drive/FUSE latency or
new-runtime cloud recovery. Those remain not run for this release. No full job,
final Test, data download, deployment, Git initialization, commit or push ran.
