# Codex Task — Rewrite the Colab Notebook for Observable Smoke / Full Runs

## Objective

Rewrite `notebooks/01_Colab_Training.ipynb` into a **clear, operator-friendly, observable Jupyter notebook** for the existing H&M recommendation pipeline. I need to see what is actually happening **while each stage is running**: current stage, completed/total work, progress percentage when knowable, elapsed time, throughput, meaningful ETA when justified, resource usage, and meaningful stage results. The current notebook mostly hides child-process work behind `command(...)` and only prints `Running (...)` and `Exit`, which is not acceptable.

**Implement the actual progress-data path in the Python CLI / worker modules as needed.** A cosmetic `tqdm` loop around one blocking subprocess call, a fake percentage, or a notebook that only shows results after completion does not satisfy this task.

**The Notebook must expose exactly two run profiles: `smoke` and `full`. Remove the `pilot` profile from the Notebook UI and primary instructions.** Preserve the existing pilot CLI and historical artifacts for backward compatibility unless a change is independently necessary; do not delete or break them. Optional benchmarking is a separate diagnostic CLI / document, **not a third notebook mode**.

## Read and verify first

1. Inspect `AGENTS.md`, current proposal and architecture/evaluation documents, current `docs/archi/` runbooks, and the actual working tree. Respect the most recent user decisions and any uncommitted changes.
2. Read the **uploaded/current** `notebooks/01_Colab_Training.ipynb` and actual source code, especially `src/hm_recsys/colab/notebook.py` (`command`), `src/hm_recsys/colab/runner.py`, `src/hm_recsys/colab/state.py`, the CLI and isolated worker processes, and relevant `src/hm_recsys/pilot/` modules. Verify which messages are currently written only to files.
3. Before making changes, describe briefly: where the subprocess output disappears; which worker loops have known total work; where counters can be measured accurately; and which stages have no meaningful percentage.
4. Do not infer that low GPU VRAM means low utilization or that progress can be estimated from input file size alone.

## A. Rewrite the Notebook: readable end-to-end operation

Organize into clearly named, navigable English Markdown sections with small, readable code cells:

1. **Purpose and execution modes** — explain Smoke vs Full, what is trained, what is not, and the expected outputs.
2. **Configuration** — only `RUN_PROFILE = "smoke" | "full"`, release name, new/resume run ID, `RUN_ACTION`, full authorization, device (Full only), optional CUDA probe, and safe progress display settings. Default: `smoke`, no full permission, final Test disabled.
3. **Mount Drive and verify release** — show release identity and hashes; fail clearly on mismatch.
4. **Prepare Python environment and check resources** — concise status, Python/torch/CUDA versions, available RAM/disk/device; show install progress where available.
5. **Stage and validate input files** — show file-by-file byte progress, completed files/total files, elapsed time, and copy/verification status. Keep inputs read-only and unchanged.
6. **Review plan and capacity** — show the selected stages, expected query/row/target counts if available, missing estimates as `unknown`, and applicable resource gates. For Full, ensure the existing capacity stage is visible **before expensive training**, without bypassing version checks.
7. **Execute the real pipeline with live progress** — a simple operator entry point, real live stage and inner-loop updates, logs accessible during execution, and clear successful/failed/resumed status.
8. **Results and model reload validation** — display actual counts, training steps, fold coverage, native MAP@12/NDCG@12/Recall@12 with correct labels, zero-hit query counts, reload parity and artifact locations.
9. **Recovery instructions** — exact steps for same release/profile/device/RUN_ID with `RUN_ACTION="resume"`; explain stale lease handling and what cannot be resumed.

Use meaningful Markdown to explain WHY each major stage exists and its inputs/outputs, **not** hundreds of boilerplate comments repeating obvious statements. Remove the existing Notebook's mechanically generated comments (for example, `Initialize ... to the explicit value ...`, `Display ... for the current module contract`). Keep English comments/docstrings and preserve the repository coding/documentation standards. Make the Notebook usable with **Restart Kernel + Run All** after configuration, without hidden cell state or duplicated actions.

The Notebook must not copy model training algorithms into cells; it remains a thin, understandable interface to shared `src/` code and CLI. Explain important commands inline before running them so the user can understand what each cell actually does.

## B. Real-time progress: a truthful, structured contract

Introduce a small, reusable structured progress protocol from the **actual isolated worker loops** to the parent runner and Notebook. You may choose JSON Lines or another documented append-only event format. Suggested fields:

```
timestamp, run_id, profile, stage, task,
completed, total (nullable), unit,
elapsed_seconds, message,
metrics (optional, structured), status
```

Rules:

- `completed` counts actual processed entities; `total` is obtained from a grounded query/manifest/counter. If a total is unknown, display `count and elapsed` and an indeterminate indicator, **never invent a percentage**.
- Calculate progress % only from meaningful `completed / total` with a documented unit and denominator. A stage count of 12/25 is not a substitute for 12/25 training batches.
- ETA is optional and shown only when denominator and meaningful recent throughput exist; otherwise `ETA unknown`. Label estimates as estimates.
- Aggregate progress should not double count resumed or verified stages. Mark them explicitly `RESTORED / SKIPPED (verified)` rather than `TRAINED NOW`.
- Emit meaningful milestones and rate-limit fine-grained events (e.g., 1–3 seconds or a batch interval). Avoid printing one message per transaction, candidate or tree node.
- Show live `elapsed`, last event time and a heartbeat when a worker is quiet; `Running (30s)` is acceptable only as fallback, not the sole progress reporting mechanism.
- Write progress to a local file and/or stdout **while running**, then include the final evidence in normal verified stage artifacts; do **not** write live progress to mounted Google Drive on every update.
- Keep source data/user IDs and purchase histories out of progress events and Notebook output. Do not emit hidden labels or future purchases.
- Flush output or use unbuffered subprocesses. If CLI forks more workers, ensure nested events reach the notebook during execution; tailing only the outer CLI log is insufficient if inner logs are isolated.
- Preserve failure propagation. On nonzero exit, show failed stage, last real progress, reason and bounded tail of the relevant **inner** stage log, then re-raise. Do not mark failed stages complete.

Notebook presentation: a small stage-status table or lightweight progress bars (such as `tqdm.auto` / IPython display with text fallback), plus a compact live human-readable log. **No excessive redraws, fake graphics, extra external services or mandatory new visualization packages.** Full detailed logs still go to disk.

## C. Required fine-grained counters by stage

Instrument the real loops where possible; use actual phase names and adapt these requirements to the code you find:

| Stage / task | Live evidence to display |
|---|---|
| Release / input staging | Files `n/N`, bytes copied/total, checksums, elapsed, success/failure |
| Canonical data conversion | Source rows processed, partitions written, current partition, throughput; distinguish scanned from retained rows |
| Catalog / embeddings | Items or embeddings mapped/validated, missing/invalid counts, resulting dimensions and coverage |
| Query rosters / capacity | F1–F4 and Validation query counts, ground-truth eligibility, computed resource estimates and gate result |
| Trend / Repurchase / Sibling statistics | Dates/window and rows or source items processed; counts of output candidates/eligible items |
| ItemCF | Unique user–item edges, directed pair work or seed blocks processed, retained neighbors, elapsed; don't call estimated pairs completed until actually processed |
| WALS | Actual sparse matrix shape/nnz, ALS iteration `i / I`, elapsed and iteration time |
| Two-Tower training | Targets, batches, **epoch `e/E`**, step and batch progress, actual examples processed, examples/sec, time/step, loss when available, checkpoint status; do not reveal training positives |
| GPU information | Actual device and sampled torch CUDA allocated/reserved memory where available; GPU utilization only if measured through a real supported source, else `unavailable` |
| Two-Tower internal validation / item encoding | Catalog items encoded `n/N`, evaluation queries processed, internal Recall@50 clearly distinguished from end-to-end metrics |
| Visual preprocessing | Valid image rows, clustering iteration or fit progress if instrumentable; unknown progress if library offers no truthful counter |
| F1–F4 and Validation feature generation | Query groups `n/N`, native candidate rows created, feature rows/shards written, average rows/query, throughput; distinguish raw candidates from sampled ranker training rows |
| LambdaMART | Actual train rows/groups/positive rows, Validation rows and feature count (currently expected around 49, verify actual schema); boosting iterations `i / configured max`, best iteration/metric if exposed; early stopping may finish before maximum |
| Reload / persistence | Rows rescored, Top-12 parity, feature/score tolerance, files committed and bytes written/verified; stage completion only after verified persistence |

Be precise: WALS latent dimension, Two-Tower representation width, LambdaMART feature count, raw purchase lines, and ranker training rows are **different kinds of quantities**. Present them separately. Report candidate-list NDCG as *diagnostic* and native end-to-end MAP@12 as an evaluation measure, not as interchangeable.

If a dependency (e.g. LightGBM or FAISS) does not provide useful per-iteration callbacks, add the supported callback/hook where feasible; otherwise provide grounded per-stage counts and elapsed time. Do not alter training objectives, numerical settings or stopping rules just to display a progress bar.

## D. Keep exactly two operator modes and all safeguards

- Notebook selection is **only** `smoke` and `full`. Remove Pilot instructions/options, `ALLOW_PILOT_RUN`, `PILOT_DEVICE` and `RUN_PROFILE == "pilot"` branches from the Notebook. Update user-facing docs accordingly. Preserve the old CLI pilot capability if it is needed for historical run recovery; do not corrupt old artifacts.
- Remove `RUN_BENCHMARK`, `BENCHMARK_ONLY`, `BENCHMARK_RUN_ID`, and profiler backend comparisons from the **main training Notebook**. Keep the existing standalone benchmark CLI and performance documentation usable separately; do not delete optimized/ref backend support from shared pipeline if Full uses it. If Full needs a performance backend setting, expose it as a single explicit advanced Full option with a short explanation.
- Defaults are safe: Smoke CPU; `ALLOW_FULL_RUN=False`; `RUN_FINAL_TEST=False`; no automatic pilot/full or long cloud jobs.
- Full requires explicit authorization, fresh RUN_ID for a new experiment, matching release and compatible device, and current resource checks. Avoid silently changing model data limits or dropping zero-hit queries.
- Maintain six retrieval channels with current quotas, WALS, genuine Hybrid Two-Tower, one LambdaMART, four **non-overlapping** ranking folds, native candidates, Train-only negative sampling, unmodified ground truth, point-in-time correctness, and verified stage resume.
- **Do not simplify or replace** the actual backend with toy training, random scores, or dummy progress. Do not run final Test, deploy, download replacement embeddings or overwrite original inputs.
- Preserve stage identity, locks, checksum verification, failure handling, checkpoint restoration and immutable released artifacts. New code needs a new release and run ID; never mutate a running Colab session or existing run in place.

## E. Correctness, UX and performance tests

Implement and execute locally (small deterministic fixtures only):

1. Notebook `nbformat` validation, top-to-bottom code syntax, clear English headings, exactly two selectable profiles; no notebook-only hidden state.
2. A **real child subprocess** emitting multiple delayed progress events: verify that the Notebook/helper sees those events **before the child exits** (not all at the end), with true counters and no unbounded buffering.
3. A nested isolated worker, not only the top-level CLI: verify stage events propagate to the Notebook while the worker runs.
4. No-total tasks, early stopping, interrupted subprocess, failed copying, failed stage, log truncation, and recovered/verified stages; ensure progress never falsely says 100% or success.
5. A small synthetic Smoke and Full path with actual shared model components; verify progress totals match the produced artifacts, query/fold counts and existing group/metric definitions.
6. Compare with the current reference execution: same candidate identities/order, feature values, losses/checkpoint semantics where applicable, Top-12, evaluation denominators and saved model parity under the same deterministic setup. Progress instrumentation must not change algorithms or RNG streams.
7. Verify compatibility of `RUN_ACTION="resume"`, correct stage labeling, and no repeated training of already verified stages.
8. Run Ruff, formatting, Pyright and pytest on changed code. Keep unrelated old notebook lint defects untouched unless this Notebook is being rewritten as requested.

Local machine is M1 Max with only about 7–8 GiB free memory. **Do not start actual H&M full training, global ItemCF rebuild, or an unapproved cloud job.** Complete code, tests, and safe small-fixture demos locally. If no real Colab runtime is available, report Colab real-time UX as **not run**, not verified.

## F. Deliverables and evidence

Deliver actual code changes, not only a new prompt or design:

- Rewritten `notebooks/01_Colab_Training.ipynb` (only Smoke and Full), with clear Markdown and clean, bounded live display.
- Updated shared notebook `command`/runner/worker instrumentation and any relevant model loops; document the progress event schema and fallback.
- Updated user runbook under `docs/archi/` with full startup/resume steps, release upload, where to see live progress, where logs are saved, and what to do after stage failure.
- `docs/archi/colab_progress_implementation_report.md` explaining **each changed file/function**, what was previously invisible, how progress counters are computed, performance/overhead observations, test evidence and remaining limits.
- Updated matching ZIP, SHA256 and manifest, plus a standalone Notebook identical to its packaged copy.

In the final handoff include the exact paths, executed test commands/exit codes, what the Notebook will display during a real Full job, how to resume, and clear distinctions between tested local behavior and **not yet executed** Colab behavior. Do not claim success based solely on notebook syntax checks or simulated print loops.

**Success criterion: During an actual run, the user can see real progress for stage execution, WALS iterations, Two-Tower epochs/batches, candidate/feature row processing and LambdaMART boosting (where measurable), with elapsed time and persisted diagnostics, without sacrificing correctness or recovery.**

Please inspect, implement, test, document, and build the new release directly. Do not stop after planning.
