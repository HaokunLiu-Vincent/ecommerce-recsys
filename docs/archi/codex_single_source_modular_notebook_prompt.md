# Codex Task — Consolidate the H&M Recommender into One Modular Source Tree and One Transparent Colab Notebook

## Mission / expected outcome

**Implement this refactor in the current repository. Do not stop at a proposal or a directory skeleton.** Preserve the proven H&M learning/retrieval/evaluation contract while making the repository easy to read, debug, operate and maintain.

Required outcomes:

1. **One authoritative algorithm implementation** under `src/hm_recsys/`, organized by responsibility rather than the current duplicated `pilot/`, `colab/`, root-level model modules, and old smoke/full paths. `smoke` and `full` must share production algorithms and differ only in selected execution/data/resource configuration.
2. **Exactly one active Colab notebook at the repository root:** `colab_training_val.ipynb`. It must be understandable without reading the entire orchestrator. Each of the six retrieval channels, RRF, and LambdaMART has a clearly labeled **separate executable code cell** that runs its applicable training/build and validation work. A separate explicit, disabled-by-default cell handles **final Test only**.
3. A data-preparation section clearly documents and executes training, validation, and test **split definition**, canonical schemas, catalog/embedding validation, point-in-time feature inputs and GPU-ready batched representations. Preparing test *metadata/split definitions* is allowed; reading test future labels or executing Test scoring is **not** allowed before final Test authorization.
4. Move superseded code, notebooks, old packaging clients, superseded reports and archived experiment material to `docs/achive/` **only after a dependency/privacy audit**; keep that exact directory spelling. The entire directory must be Git-ignored, excluded from tests/lint/type checking, and must never be imported by active code. Preserve bytes and keep personal/transaction-level data out of commits.
5. Add a useful, detailed **module-level docstring at the top of every active Python source and CLI file**. Document its role, major input/output contracts, interactions with other modules, side effects, temporal rules, and costly operations where relevant. All comments/docstrings and repository Markdown are professional English, consistent with `AGENTS.md`. Do **not** add mechanically redundant comments to every trivial statement; favor precise explanation of non-obvious logic and clear variable names.
6. Deliver updated `README.md`, `AGENTS.md`, architectural docs, tests, an exact old-to-new module migration map, and a detailed implementation report under `docs/archi/`.

This is an **engineering-structure refactor**, not permission to change the six-channel strategy, model quality, negative sampling, ranker, temporal split, metrics or training configuration to make tests pass faster.

## 0. Inspect reality before writing code

- First inspect `git status`, tracked files, the current import graph and the actual working tree. There may be newer optimizations than historical release ZIPs and reports. **Use current working code as the source of truth**; do not restore historical ZIP versions or revert recent improvements.
- Read current `AGENTS.md`, proposal (actual path), `docs/archi/evaluation_protocol.md`, `docs/archi/implementation_decisions.md`, architecture document, README, latest Colab runbook, `docs/computing_performance_architecture_audit_report.md` if present, and `colab_full_computing_opt_v1_training_report.md` if available. State explicitly which files are missing; do not invent their contents.
- Inventory **every** file in `src/`, `scripts/`, `tests/`, `configs/`, notebooks and legacy CLIs. For each file classify `ACTIVE_CANONICAL`, `ACTIVE_ENTRYPOINT`, `REFACTOR_MERGE`, `ARCHIVE_ONLY`, `HISTORICAL_REFERENCE`, or `PRIVATE_DATA_DO_NOT_MOVE`. Record imports and call sites.
- Identify duplicate algorithm implementations (e.g. modules with similar jobs under `pilot`, `colab`, and `src/hm_recsys/*.py`). Make an explicit mapping from old API to **one** new module/function. Do not classify a file as dead merely because its name is `pilot`; inspect the real call graph and tests.
- Capture a before-refactor baseline using safe synthetic/integration fixtures: candidate IDs/order, sources, RRF scores, full 49-feature schema/order and selected values, WALS/Two-Tower loss where deterministic, LambdaMART groups, MAP@12/NDCG@12/Recall@12, Top-12 and reload checks. Record the current test commands and which checks actually ran. Never use the historical full MAP score as a synthetic-test assertion.
- Preserve uncommitted user modifications and existing real run artifacts. No `git clean`, `git reset --hard`, unreviewed delete, automatic commit/push or unapproved full job.

## 1. Recommended target structure (adapt minimally where the existing code warrants)

```text
repository/
├── AGENTS.md
├── README.md
├── pyproject.toml
├── uv.lock
├── .gitignore
├── colab_training_val.ipynb            # Only active notebook, at repo root
├── configs/
│   ├── smoke.*
│   └── full.*
├── data/                                # Fixed runtime root, untracked real data
│   ├── original data/                   # Existing inputs: preserve names and bytes
│   ├── text_features/
│   ├── image_features/
│   └── runs/ ...
├── src/hm_recsys/
│   ├── __init__.py
│   ├── core/                            # Contracts, cutoff dates, IDs, config and shared types
│   ├── data/                            # Canonical load, Parquet, split manifests, rosters, labels, histories
│   ├── features/                        # Static content lookup, user/item/cross features, one feature builder
│   ├── retrieval/
│   │   ├── base.py                      # Shared candidate protocol and Top-K contracts
│   │   ├── trend.py                     # R1
│   │   ├── repurchase.py                # R2
│   │   ├── sibling.py                   # R3
│   │   ├── itemcf.py                    # R4
│   │   ├── wals/                        # R5: sparse data, fit, fold-in and inference
│   │   └── two_tower/                   # R6: model, loss, input encoder, trainer, inference
│   ├── fusion/                           # Native union/dedup/provenance; RRF rule/feature
│   ├── ranking/                          # Native feature matrix, train-only sampling, LambdaMART train/predict
│   ├── evaluation/                       # Shared definitions, per-channel recall, internal tower, ranker, comparison
│   └── runtime/                          # Stage orchestration, progress, checkpoints, resources, identities
├── scripts/                             # Thin CLI wrappers ONLY; no duplicated model logic
├── tests/                               # Mirror active modules and Notebook orchestration
└── docs/
    ├── archi/                           # Current authoritative design and implementation reports
    └── achive/                           # Historical, local-only, ignored, never executed/imported
```

This is a **logical package layout**, not a requirement to create empty files or one-off abstractions. Prefer the smallest number of cohesive Python files that cleanly separates the responsibilities. For WALS and Two-Tower, a dedicated subpackage is justified; for the three rule channels, one readable file each is preferable. Avoid creating several copies of common data parsing, feature definitions, negative sampling, or evaluation math.

`data/` stays the only runtime data root, including the existing `original data/`, `text_features/`, and `image_features/` names. Do not regenerate, rename or modify user-provided embeddings, FAISS indexes, mappings or manifests.

### Explicit one-source rule

- `smoke` and `full` call the **same** functions for all six retrievers, fusion, feature generation, ranking and evaluation; no `smoke_model.py`/`pilot_model.py`/`full_model.py` duplicates.
- Files under `src/hm_recsys/pilot/`, `src/hm_recsys/colab/` and old root `recall.py`, `features.py`, `towers.py`, etc., are **migration sources only**. Their needed behavior must be moved or consolidated into the canonical packages, with every import/test/CLI updated. At final delivery, remove parallel active implementations; if an old wrapper remains temporarily, document why and prove it contains **no duplicated algorithms**, and finish removing it before marking the single-source gate complete.
- Exactly one implementation of each evaluation formula: avoid six copy-pasted MAP@12 functions. Use `evaluation/metrics.py` as shared math and targeted retrieval/internal-tower/ranker evaluation adapters as appropriate.
- Avoid cyclic imports. Separate domain contracts from execution orchestration, and keep `runtime/` from being a dependency of core model math.
- No imports or CLI paths may reference `docs/achive/` or archived notebooks.

## 2. Source-file documentation standard

Every active `.py` module (including important scripts and test modules) must begin with a specific, informative **module-level Google-compatible docstring**, before imports, e.g.:

```python
"""Train and score implicit WALS retrieval for time-bounded H&M purchases.

Responsibilities:
    Build confidence-weighted sparse user-item interactions and fit frozen
    item factors. Fold users in with past-only observed purchases at each cutoff.

Inputs:
    Canonical purchase batches, versioned item vocabulary, cutoff config.

Outputs:
    WALS factors and top-K candidate rows with score/rank provenance.

Collaborators:
    hm_recsys.data.history provides cutoff-safe interactions.
    hm_recsys.retrieval.base defines candidate record contracts.
    hm_recsys.runtime.artifacts saves versioned outputs.

Constraints:
    Never consume transactions from a query's label interval. Maintain
    repeat-purchase eligibility and deterministic item-ID ties.
"""
```

Adapt the text **to the actual file**. Document significant input shapes, units, memory usage and side effects when appropriate. Do not use identical boilerplate docstrings everywhere, invented dependencies, or long line-by-line decorative comments. Apply Google-style function/class docstrings and type hints to nontrivial public APIs; explain tricky logic inline. Keep existing algorithm behavior unchanged.

## 3. Root Colab Notebook: human-readable and directly operable

Create exactly `colab_training_val.ipynb` at the repository root. Old active notebooks under `notebooks/` should be moved to local-only `docs/achive/` after checking tests/references and updating links. Do not leave an alternative active training Notebook. The root Notebook should open from GitHub; Colab clones/checks out the **pinned Git commit** into `/content/e_commerce_recsys/`, not into mounted Drive. It installs existing locked dependencies, stages read-only input files from `/content/drive/MyDrive/projects/e_commerce_recsys/data/` into the local `data/` tree, and uses the existing safe stage persistence to Drive. Remove ZIP-upload bootstrap from the main path. No DVC/MLflow/AWS/CI-CD in this task.

Notebook MUST have short English Markdown cells explaining **why** each stage exists, its data grain, input/output paths, expected shape, cache/reload behavior, temporal cutoff and how to interpret results. A user should be able to run cells one by one, inspect intermediate metrics/artifacts, and understand where a failed stage came from. Notebook code should be short and call functions/CLI in `src/` rather than reimplementing models.

### Required notebook flow and executable cells

0. **Overview / architecture and run controls.** Show run ID, pinned Git SHA, profile (`smoke` or `full` only), device, data root, output root, `RUN_ACTION`, `ALLOW_FULL_RUN`, `RUN_FINAL_TEST=False`. Validate and print a readable execution plan. No default automatic Test.
1. **Environment & data acceptance.** Drive mount, checkout verification, lockfile install, input integrity, article_id leading zeros, embedding row-map alignment, resource summary. Show actual counts and missingness.
2. **Data preparation — Train / Validation / Test planning.** Generate and display the immutable split manifest, canonical/cutoff-safe data layer, eligible catalog definitions, full training fold rosters F1–F4, Validation roster/labels, content lookups and bounded GPU-ready ID/tensor/batch representations. Prepare *Test split metadata and availability contract only*; do not read future Test labels or evaluate Test yet. Show per-fold dates, label counts and query counts. CPU feature/SQL preparation is permitted even if the downstream model uses CUDA; do not place the full table on GPU by default.
3. **R1 — 7-Day Trend Popularity (ONE dedicated executable cell):** build cutoff-specific training-fold/Validation trend tables; generate candidates and show per-fold counts, timing and Recall@K. It has **no supervised model fit**; correctly label its action as `build + validate`, not fake training.
4. **R2 — Repurchase (ONE dedicated executable cell):** cutoff-safe history, candidate generation and Train-fold/Validation retrieval metrics.
5. **R3 — Product_Code Sibling (ONE dedicated executable cell):** variant mapping, candidate generation and Train-fold/Validation retrieval metrics.
6. **R4 — ItemCF (ONE dedicated executable cell):** build/reuse legal global cutoff co-purchase index for each fold/Validation, show unique edges, directed-pair work, progress, candidates and retrieval metrics.
7. **R5 — WALS (ONE dedicated executable cell):** fit/freeze WALS using its allowed base period (or load same-identity completed artifact), fold current users in, generate F1–F4/Validation candidates, report matrix shape/nnz/iterations, retrieval results and coverage.
8. **R6 — Hybrid Two-Tower (ONE dedicated executable cell):** fit/freeze the existing multi-positive LogQ Two-Tower (or load same-identity completed artifact), keep its **internal** model-selection window independent of outer Validation, encode the learned item catalog, retrieve F1–F4/Validation candidates, report epochs/steps/loss/internal Recall@50/GPU usage/outer retrieval coverage.
9. **Candidate union and RRF (ONE dedicated executable cell):** merge and deduplicate six channel lists, preserve all source ranks/scores/masks; compute deterministic RRF feature and rule-baseline rankings for training folds and outer Validation. RRF has **no learned weights** in the agreed design and must not be misrepresented as a trainable model or pre-ranking filter. Show native union Recall, per-channel incremental recall, RRF MAP@12 and query denominators.
10. **Shared feature construction (ONE dedicated executable cell):** generate the **same 49-feature schema** for Train F1–F4 and Validation; show rows, queries, feature names/types/missing masks, storage and throughput. Do not expand raw image/text vectors into hundreds of ranking columns. Use identical train/inference feature code.
11. **LambdaMART (ONE dedicated executable cell):** four training folds combined into ONE model; native candidate labels; existing Train-only 1:20 hard/random negative sampling; proper grouped ranking; full unsampled Validation, early-stopping diagnostic kept distinct from end-to-end MAP@12. Display train rows, positives/negatives, group counts, boosting rounds, best iteration, time/RSS and final MAP@12/NDCG@12/Recall@12.
12. **Validation comparison and reload (clearly separated cell):** compare Trend, per-channel Recall, RRF and LambdaMART on **the same eligible outer Validation roster and full truth**, annotate any different K; inspect model/index/feature identities, new-process reload and Top-12 parity. Never call test labels here.
13. **Final Test (ONE standalone gated executable cell at the end):** default is SKIPPED. Require explicit `RUN_FINAL_TEST=True` plus a separate explicit approval flag and frozen model/config/feature/split identities; allow execution only after training and Validation artifacts are complete and configuration locked. Construct test-cutoff past-only features/candidates, read Test labels *only now*, score once without refitting or tuning, evaluate with full unique eligible Test truth and save a separate Test report. A Notebook `Run All` with defaults must NOT read or evaluate test future labels.
14. **Report and next steps:** separate actual measurements from unrun stages; link persisted artifacts and give resume commands. Keep it human-readable.

**Each retrieval/RRF/LambdaMART cell must be independently understandable and invoke only its intended stage(s).** If a prerequisite is missing, print clear names and a safe instruction rather than silently retraining unrelated models; if a stage is already completed, validate identity and show reused/skipped status. The cell must not re-execute all previous stages merely to reach its target. The same stage can be resumed after interruption. User may run a model cell individually to inspect its artifacts.

### Critical temporal protocol (do not invent alternate windows)

Use the exact currently adopted `docs/archi/evaluation_protocol.md` and reconcile with the proposal. The last agreed schedule is:

| Role | Cutoff / target interval |
|---|---|
| WALS base | `[2020-02-01, 2020-08-12)`; vocabulary `[2020-06-16, 2020-08-12)` |
| Two-Tower training targets | `[2020-05-19, 2020-08-05)` |
| Two-Tower internal selection | `[2020-08-05, 2020-08-12)` |
| F1 | cutoff 2020-08-11; labels `[2020-08-12, 2020-08-19)` |
| F2 | cutoff 2020-08-18; labels `[2020-08-19, 2020-08-26)` |
| F3 | cutoff 2020-08-25; labels `[2020-08-26, 2020-09-02)` |
| F4 | cutoff 2020-09-01; labels `[2020-09-02, 2020-09-09)` |
| Outer Validation | cutoff 2020-09-08; labels `[2020-09-09, 2020-09-16)` |
| Final Test | cutoff 2020-09-15; labels `[2020-09-16, 2020-09-23)` |

These six 7-day outer label windows are pairwise disjoint. For each query, behavioral features and retrieval use `t_dat <= cutoff`, equivalent to `< prediction_date`. Do not shrink truth by the active pool or retrieval vocabulary; keep zero-hit queries in Validation/Test denominators. Repeat purchases remain eligible; raw duplicate purchase lines are preserved in transactions, with labels unique by article. No future-positive injection. Report native macro MAP@12 as the primary metric. If the current authoritative protocol differs, **stop and report the precise conflict before semantic changes**; do not silently reinterpret dates.

## 4. Transparent execution, performance and recovery

- Keep genuine live worker → CLI → notebook progress. For each cell show stage name/status, processed rows or queries / actual total if known, duration, throughput, RSS, GPU allocation/reservation when available, errors and persistence/checkpoint state. Show 'unknown' for unknown totals and never fabricate ETAs or percent complete.
- Measure expensive phases separately: global ItemCF pair generation; WALS fitting/folding-in; Two-Tower data prep, host→device, forward/backward, internal evaluation, checkpoint I/O; candidate fusion; 49-column feature building; LambdaMART fit/validation.
- **Preserve existing compute improvements** in the current source (static content caches, bounded readers, learned-vector lookups, CPU/GPU optimizations); don't replace them with older ZIP code or a simpler but slower implementation. A structural refactor is not proof of performance improvement.
- Preserve stage manifest/hash identity, checkpoint/resume, actual device and Python library environment checks. A path/name move must not silently resume an incompatible old checkpoint. Prefer a **new run identity for the new code layout**. Old runs remain immutable/read-only; cross-format reuse requires verified schema/artifact/algorithm compatibility.
- Staged data and training run on Colab `/content/e_commerce_recsys/`; only authorized checkpoint/artifacts persist to Drive. Avoid dozens of noisy Notebook outputs or dumped user identifiers.
- Smoke runs are bounded and share the same algorithm path. Full may be initiated only with explicit permission/resource gates. No full run on local M1 Max (~7–8 GiB available).

## 5. Archive, Git safety and deletion policy

- Use `docs/achive/` **exactly** (not `docs/archive/`), and add `docs/achive/` to `.gitignore` and Ruff/pytest/Pyright traversal exclusions. This local directory is not maintained or shipped in GitHub checkouts. Existing archived reports can be left in place; don't reformat or rewrite them.
- **Never move actual customer histories, raw transactions, model outputs or source embeddings into a Git-tracked docs tree.** They remain beneath ignored `data/` or a private local-only archive. Do not use `git add -f` to publish archived/private content.
- An ignored file already tracked by Git remains tracked. Inspect `git ls-files` and the index; when untracking archive paths, use a **non-destructive, explicit index-only operation**, record every affected path, and ensure worktree copies remain intact. If sensitive content exists in committed history, report that before any push; `.gitignore` cannot undo history.
- Use safe, non-destructive moves; preserve a migration manifest (old path, new path or archive path, checksum, reason). Only archive files proven inactive or superseded; do not archive required code just to make the active tree look smaller. Do not leave imports/tests/document links dangling.
- Archive old root or nested notebooks, old ZIP release scripts and legacy smoke/pilot implementations only when the replacement is working and tested. Do not delete unrelated user work. Only the root `colab_training_val.ipynb` is active; tests and README must enforce this.
- Do not initialize a new remote, commit, push, force-push or rewrite history automatically. GitHub code publication remains the user's reviewed action.

## 6. Tests and acceptance gates

Run as many of these as possible safely on the local M1 Max; label cloud/full checks NOT RUN when not executed:

1. **Module consolidation:** AST/import-based inventory proves a unique canonical implementation of each retriever, feature builder, loss and evaluator, and no active import from `pilot`, legacy wrappers or `docs/achive`. No active duplicate training notebook. No outdated script paths.
2. **Contracts:** preserve six quota values `60/30/30/50/50/50`, candidate IDs/ranks/source masks and deterministic ties, RRF rules, `49` feature columns/order/values, label and group semantics, same Train-only negative sampler, stable Top-12, complete truth and native MAP@12 denominator.
3. **Split/no-leakage tests:** exact F1–F4, Validation and Test intervals; no overlapping labels; boundary test fixtures prove no transaction at/after the query prediction date enters features. Test label files and Test scoring are not touched with default Notebook Run All.
4. **Model-path tests:** for each rule recall: build + outer Validation candidates; ItemCF global history correctness; WALS train/fold-in; Two-Tower train/in-batch multi-positive LogQ + internal selection; union/RRF; single LambdaMART group training; per-stage cache/resume and missing-prerequisite behavior.
5. **Notebook testing:** root filename, no other active notebook, executable cell count and order, actual CLI argument parsing and stage binding, clean kernel Run All with Test disabled on a bounded synthetic fixture, independent stage-cell execution and progress streaming, failure visibility, restart/resume behavior.
6. **Parity:** identical eligible queries, candidates/order/ties, values, loss/gradients where deterministic, ranked scores within documented numeric tolerances, final Top-12 and metric values across equivalent reference and refactored bounded fixtures. Do not require cross-platform CUDA bitwise equality; explain differences. Cache/warm-start changes cannot alter semantics.
7. **Security:** ensure `data/`, `docs/achive/`, secrets and customer-level runs are ignored AND not staged/tracked. Run a repository-content scan that catches private JSONL/Parquet/CSV and populated notebook output cells. Do not claim public-Git safety based on `.gitignore` alone.
8. **Style/tools:** `ruff check`, `ruff format --check`, `pyright`, `pytest -q`, `nbformat.validate`; also validate Markdown links, Mermaid if changed, shell/CLI help, and notebook unit/cell tests. Do not run tests on `docs/achive/`; do not weaken useful tests to make migration pass.
9. **Performance regression:** on exactly matched small data, measure candidate/feature row throughput, model-step throughput and resource usage where possible. Document any clear overhead; do not promise a full-data speedup without a matched measured full run.

The full historical Validation result (MAP@12 ≈ 0.023638 over 72,019 queries) is **historical evidence, not a required reproduction on a local fixture**. Report precisely whether it was merely referenced, independently audited previously, or re-run. Final Test remains not run unless user separately authorizes it.

## 7. Documentation deliverables

- `README.md`: high-level architecture, dependency installation, local smoke, GitHub push, opening the root notebook from GitHub, Colab paths, input schema/artifact inventory, per-cell action/outputs, six channels, RRF, ranker, split table, progress, persistence, resume, explicit final Test gate and limitations. Include a **complete active source file map** explaining each file's purpose and primary callers.
- `AGENTS.md`: update all active path rules and tool exclusions; remove references to old active notebooks, duplicated model paths, `data/raw`/`data/embeddings` conventions that conflict with actual input directories, or automatic DVC/MLflow requirements. Preserve Google Python/Notebook/Markdown quality requirements. State only the root notebook is active and archives are ignored/unmaintained.
- `docs/archi/architecture.md`: actual active import/call graph and stage/data flow (Mermaid if helpful, validated). `docs/archi/evaluation_protocol.md` and `implementation_decisions.md`: update references only; do not quietly change the evaluation protocol.
- `docs/archi/refactor_migration_map.md`: old→new mapping and hashes; active/archive rationale; import updates; any deliberately retained compatibility wrappers (must be removed to meet final single-source gate).
- **`docs/archi/code_architecture_refactor_report.md`:** itemized old problem → exact file/function changes → tests → observed/not-run performance → known limitations. Include separate sections for repository privacy, Notebook Test isolation and archive changes.
- Root Notebook includes explanatory Markdown sections and an explicit 'not run / skipped / success' result summary.

## 8. Execution order and stop conditions

Implement incrementally and verify at each step:

**A. Baseline and privacy audit → B. Shared contracts/data → C. Six canonical retrieval channels → D. Fusion/features/ranking/evaluation → E. Shared runtime/CLI → F. Single root Notebook → G. Archive only proven-unused files → H. README/AGENTS/docs → I. Full bounded test suite/report.**

Keep users informed about materially risky moves and blockers. If the current repo has already-diverged code, missing source files, unknown data/private artifacts, unsafe Git history or an unresolvable semantic discrepancy, make best-effort safe progress and identify the exact blocker; do not silently delete or guess.

**Do not run actual full H&M training or final Test; do not push GitHub.** The final report must distinguish implemented source refactor, bounded local verification, real Colab full compatibility (not run unless separately authorized), and final Test (not run). Stop after a verified, reviewable commit-ready working tree.

## 9. Final response expected from Codex

Summarize: current git status; exact changed/moved/archived files; updated canonical source tree; per-cell Notebook plan and Test gate; input/output contracts; baseline/parity tests; executed commands and exit statuses; privacy and ignored tracked-file checks; performance tests; what is not verified; manual local-smoke → commit/push → Colab checkout instructions. Do not claim a new model-quality uplift from restructuring alone.
