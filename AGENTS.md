# AGENTS.md — H&M Recommendation System

Version: 0.3.2 · Updated: 2026-11-02

These instructions apply to the repository unless a more specific, applicable
instruction explicitly overrides them. Use the uppercase filename `AGENTS.md`
at the repository root for Codex discovery. A lowercase download copy is not a
second source of authority; do not maintain two divergent instruction files.

## Contents

- [1. Project purpose and scope](#1-project-purpose-and-scope)
- [2. Read first and preserve decision provenance](#2-read-first-and-preserve-decision-provenance)
- [3. Repository layout and path rules](#3-repository-layout-and-path-rules)
- [4. Execution workflow and evidence](#4-execution-workflow-and-evidence)
- [5. Non-negotiable ML and data contracts](#5-non-negotiable-ml-and-data-contracts)
- [6. Python Code Style & Documentation Standards](#6-python-code-style--documentation-standards)
- [7. Applying the standards in this repository](#7-applying-the-standards-in-this-repository)
- [8. Required verification](#8-required-verification)
- [9. Resource, privacy, and reproducibility rules](#9-resource-privacy-and-reproducibility-rules)
- [10. Completion report and change discipline](#10-completion-report-and-change-discipline)
- [11. Jupyter Notebook & Markdown Documentation Standards](#11-jupyter-notebook--markdown-documentation-standards)
  - [11.1 General Principles](#111-general-principles)
  - [11.2 Jupyter Notebook Standards (.ipynb)](#112-jupyter-notebook-standards-ipynb)
  - [11.3 Markdown Document Standards (.md)](#113-markdown-document-standards-md)
  - [11.4 Machine Learning Specific Requirements](#114-machine-learning-specific-requirements)
  - [11.5 Quality Assurance](#115-quality-assurance)

## 1. Project purpose and scope

Build a reproducible H&M next-seven-day purchase recommender from observed
purchase history and static article content. The system has exactly these six
logical recall channels and one learned ranker:

| Channel | Default quota | Update policy |
|---|---:|---|
| 7-Day Trend Popularity | 60 | Recompute at each cutoff |
| Time-Decayed Repurchase | 30 | Recompute at each cutoff |
| Product_Code Sibling Recall | 30 | Recompute at each cutoff |
| Time-Decayed ItemCF | 50 | Rebuild at each cutoff |
| Time-Decayed WALS | 50 | Freeze item factors; fold in current user history |
| Multi-modal Two-Tower | 50 | Freeze weights; encode current user history |

Merge and deduplicate the native candidates, construct shared features, apply
one LightGBM LambdaMART model, and return unique Top-12 articles. The initial
pre-deduplication quota sum is 270. RRF is a feature or rule baseline, not a
second trained ranker. Text and image signals enter Two-Tower and ranking
features; they are not separate recall channels.

Do not add pre-ranking, Logistic Regression as an additional learned ranker,
LightGCN, DeepFM, DIN, a search service, a feature-store platform, or cloud
infrastructure without an explicit scope change. Separate logical channels do
not require separate deployed services.

The task is purchase prediction, not CTR/CVR modeling, inventory prediction,
returns analysis, or causal sales-lift measurement. Never fabricate missing
impressions, orders, sizes, inventory, sessions, or return events.

## 2. Read first and preserve decision provenance

Before editing, inspect the actual repository and read the relevant portions of:

1. `proposal.md`: current scope, channel algorithms, inputs, and defaults.
2. `docs/archi/evaluation_protocol.md`: exact temporal and metric contract.
3. `docs/archi/implementation_decisions.md`: decisions, especially D06/D07/D14.
4. `docs/archi/architecture.md`: data flow and module boundaries.
5. `docs/phase0-data download & EDA report.md`: historical measured data facts.

The latest explicit user decisions and approved new architecture control core
requirements. Keep current proposal/architecture/protocol/instructions aligned;
this file's old path examples must never restore superseded architecture.
Report and reconcile conflicts explicitly, recording semantic changes in the
current implementation decision log.

The user reports prepared text/image artifacts under `data/`. Treat them as
existing source inputs, not a pending extraction task. The older Phase 0
statement that embeddings were not yet generated does not supersede that user
update. Do not claim integrity, coverage, dimensions, or provenance have been
verified until their real artifacts are inspected.

`docs/hm_mvp_full_reports_colab_v1.md`,
`docs/achive/hm_mvp_full_reports_colab_v1.zip`, and
`docs/achive/mvp_notebooks_colab/` are limited historical references. Preserve
the literal `achive` spelling. Read only relevant sections; do not automatically
unzip, run, import, or port the archived pipeline. Individually reused helpers
must be reviewed and tested against current paths, dates, interfaces, sampling,
and metric definitions; record provenance and adaptation.

Do not resurrect the old 2020-07-28 cutoff, overlapping fifteen-anchor
protocol, fixed C0, injected positives, NDCG@20 primary selection, or standalone
text/image recall. Old metrics are not evidence for the new architecture.
Never invent uninspected report/notebook content or rewrite archived results.

## 3. Repository layout and path rules

- `data/` is the fixed runtime data root. Do not create parallel `dataset/`,
  `datasets/`, root-level `models/`, `outputs/`, or `artifacts/` data trees.
- Preserve source CSVs in `data/original data/`: `articles.csv`, `customers.csv`,
  and `transactions_train.csv`. The space in the directory name is intentional.
- Preserve `data/text_features/`: `text_article_ids.parquet`,
  `text_embeddings.npy`, `text_faiss.index`, and `text_features_manifest.json`.
- Preserve `data/image_features/`: `image_article_ids.parquet`,
  `image_embedding_failures.parquet`, `image_embeddings.npy`,
  `image_faiss.index`, and `image_features_manifest.json`.
- Do not move these inputs to `data/raw/` or `data/embeddings/`, duplicate them,
  overwrite original manifests/indexes, rerun extraction, or require an
  unprovided image directory. Source regeneration/replacement is a separate
  explicitly authorized task, never an automatic loader fallback.
- Add derived tables under `data/canonical/`; cutoff pools, candidates, labels,
  features, and optional adapter caches under `data/processed/`; fitted models
  under `data/models/`; new ItemCF/learned-factor indexes under `data/indexes/`;
  acceptance/split/schema/lineage reports under `data/manifests/`; metrics under
  `data/metrics/`; local MLflow artifacts under `data/runs/`; scratch under
  `data/tmp/`. Create directories only when needed.
- Preserve `docs/achive/` and both user-listed reports at the `docs/` root.
  Add current design files under `docs/archi/`. Do not force a migration into
  `docs/references/`, rename the archive, or overwrite historical notebooks.
  Existing archive files are references, not a destination for new runtime data.
- Keep reusable Python under `src/hm_recsys/`, SQL under `sql/`, configurations
  under `configs/`, thin entry points under `scripts/`, tests under `tests/`,
  and new exploratory notebooks under `notebooks/`.
  `src/hm_recsys/embeddings/` loads/validates/adapts existing vectors; it is not
  a default source-embedding extraction pipeline.
- Resolve `<repository>/data/` independently of the process working directory.
  Mount external data there. Use `pathlib.Path`; quote spaces in shell paths.
  Never fall back to a personal path, `/content/data`, or another data tree.
- Keep all supplied inputs immutable. Use explicit UTF-8 text encoding, atomic
  publication of derived artifacts, and manifests linking inputs to outputs.
  Tests use synthetic fixtures and `tmp_path`, never mutate real source files.

## 4. Execution workflow and evidence

Implement the assigned phase, not an imagined complete platform. Inspect the
existing environment, package configuration, and tests before choosing APIs or
creating new abstractions. Reuse existing project conventions when compatible
with these instructions.

State the intended change and its acceptance checks. Keep changes scoped and
prefer small, testable functions and explicit data contracts. Update associated
schema, configuration, documentation, and tests together when behavior changes.

A task to write or review documentation does not authorize dataset downloads,
large embedding jobs, full training, external uploads, cloud deployment, paid
resources, or destructive filesystem operations. Run those only when explicitly
included in the requested work and within approved resource constraints.

Do not overwrite unrelated user changes. Do not claim a script, benchmark,
training run, deployment, or test succeeded unless it was actually executed and
its result inspected. Distinguish planned work, synthetic checks, sampled
smoke tests, and full-data results. Never invent run IDs or fill tables with
plausible-looking metrics.

If a required input or unresolved contract blocks one stage, report the precise
blocker and the completed work. Do not silently substitute a different model,
dataset, negative sampler, time interval, or evaluation population.

## 5. Non-negotiable ML and data contracts

### 5.1 Date semantics and model updates

`cutoff_date = T` is inclusive. `prediction_date = a = T + 1 day`.

- Observable transactions satisfy `t_dat <= T`, equivalently `t_dat < a`.
- A trailing `d`-day window is `[a - d days, a)`.
- Labels are `[a, a + 7 days)`.
- A Two-Tower positive dated `t` uses history strictly before that date. Do not
  invent intraday ordering or use other same-day purchases as prior history.

The base boundary is inclusive `2020-08-11`. WALS fits transactions from
`2020-02-01` through that day; its item vocabulary is observed purchases from
`2020-06-16` through that day. Two-Tower training targets run `2020-05-19` through
`2020-08-04`, with internal validation `2020-08-05` through `2020-08-11`.

Ranker cutoffs are `2020-08-11`, `2020-08-18`, `2020-08-25`, and `2020-09-01`.
Global validation uses cutoff `2020-09-08`; final test uses `2020-09-15`.

D06 is a hard gate: the six seven-day ranker/Validation/Test target windows
must be pairwise disjoint and exactly match the current date table. They cover
2020-08-12 through 2020-09-22 once. Two-Tower train/internal-validation targets
are also disjoint. Persist a split manifest; assert dates, interval lengths,
empty intersections, and cutoff/start alignment before execution. Reject the
old three-day anchor stride and random row splitting. Past-only histories may
overlap; target assignments may not. Earlier completed weeks may enter later
observed history without becoming duplicate target windows.

Do not refit base item factors or Two-Tower weights on ranking, global
validation, or test labels. Dynamic past-only ItemCF, statistics, user history,
and WALS user folding-in are intentional and must be versioned by cutoff.

### 5.2 Keys, interactions, and active items

Preserve `article_id` as a zero-padded ten-character string and `customer_id` as
a string. Use the actual `product_code` column, not a guessed ID prefix.
Preserve raw purchase lines, including exact repeats. Distinguish purchase
lines, distinct user-item edges, distinct purchase dates, and unique labels.

The active pool is articles in the catalog with at least three purchase lines
in the trailing 28 days. It is a purchase-activity proxy, not verified stock.
Apply it to candidates before each channel's Top-K, never to the main future
positive denominator. Do not discard ground truth because an image, factor,
channel result, or content field is missing.

Do not globally filter purchased articles. The implicit-library recommendation
path must disable already-liked filtering where applicable. Missing histories
or factors may yield an empty channel; global popularity is the explicit
fallback. Keep native shortages and fallback provenance visible.

### 5.3 Candidate and feature semantics

Use `(customer_id, cutoff_date)` as query identity and
`(customer_id, cutoff_date, article_id)` as candidate identity. Deduplicate
articles, retain all channel provenance, and resolve ties deterministically.
Every channel's rank is one-based and score is channel-local.

Generate native candidates without consulting future labels. Do not inject
missing positives at any stage. Do not expand budgets silently or turn RRF
into an undocumented pre-ranking filter.

Use one feature implementation for training and inference. Freeze and validate
column names, order, dtypes, category vocabularies, missing-value rules, and
content/model versions. Missing recall scores/ranks are `NaN` with explicit
source masks, not arbitrary mixtures of zero and sentinel values. Raw IDs,
fold keys, cutoff identifiers, labels, and future-derived fields are not ranker
input features.

The source's advertised 38 features contain 39 listed names. The current schema
adds six channel masks and four content-availability masks, for a proposed
49-column default. Load the feature list from its versioned schema; do not
hard-code any of these counts in algorithms.

Respect static metadata/image snapshot limitations. Fit K-Means or any learned
preprocessor only on its documented pre-base population, then freeze it.
Image filename coverage is not decoding or embedding coverage. Record encoder
checkpoint, revision, dimensions, normalization, preprocessing, and failures.
Do not assume all CLIP/ResNet checkpoints return 512-dimensional vectors.

### 5.4 Recall-specific safeguards

- ItemCF uses the documented asymmetric formula. Do not reinterpret its
  `alpha=0.75` as stronger popularity suppression or silently replace its math.
  Build sparse neighborhoods; preserve the documented edge/timestamp policy.
- WALS uses one confidence transform. The current proposal stores full
  confidence in CSR and uses library `alpha=1.0` to avoid double scaling.
  Verify the installed library's behavior with a tiny controlled test.
- Recalculate the WALS user vector with fixed item factors and the same
  confidence convention. Weighted pooling plus normalization is not ALS
  folding-in. Do not normalize WALS factors or change the retrieval metric
  without a separately versioned experiment.
- Keep WALS item vocabulary fixed; new users can be folded in only when they
  have representable history. Unknown items do not acquire random factors.
- Two-Tower item representations must remain independent of the requesting
  user. The supplied frozen content vectors feed trainable projections; ranker
  training never backpropagates into the recall models or source vectors.
- Handle duplicate in-batch items and known multi-positive user/date labels.
  Estimate LogQ from training sampling only; do not use global future counts.
  Keep padding, zero-vector, unknown-ID, and training-only ID-dropout behavior
  explicit and tested.
- `IndexFlatIP` is exact inner-product search. Do not call it an approximate
  index or change it to cosine without the prescribed normalization contract.

### 5.5 Ranking, sampling, and evaluation

LambdaMART training consumes `X [N,F]`, binary `y [N]`, and contiguous query
sizes `group [Q]`. Assert `sum(group) == N`, identical row alignment, unique
candidate keys, and consistent grouping after every filter or sample.
Prediction consumes the same feature schema and returns ranking scores, not
probabilities; outer orchestration retains query boundaries.

Training may drop groups without both positive and proxy-negative candidates.
Retain every native positive and sample up to twenty negatives per positive,
using the versioned hard/random allocation and seed. Validation and test must
use full native candidates with no negative sampling. Do not remove zero-hit
or zero-candidate queries from evaluation.

Main offline queries are customer/cutoff pairs with at least one future
catalog-matched purchase. Ground truth is the unique future article set,
including active-pool exclusions and recall misses. Macro MAP@12 uses
`min(12, number_of_full_ground_truth_items)` as the AP denominator. Preserve an
independent query manifest so inner joins cannot erase failed queries.

LightGBM's candidate-list NDCG is a diagnostic, not the pipeline's complete
MAP@12. `eval_at=12` does not turn LambdaRank into a MAP objective. Log the
actual early-stopping criterion separately from the primary model-selection
metric. Test is evaluated once after configuration is locked; never select
quotas, features, or checkpoints on test scores.

### 5.6 Existing content artifact contract

Read the real manifests and schemas before configuring the multimodal models.
Use verified `D_text` and `D_image` for trainable projections; do not hardcode
384/512 or replace a supplied encoder to match an old example. Keep original
vectors frozen. Inspect actual text construction rather than assuming a single
source field.

Validate array shape/dtype, finite/nonzero usable values, row counts, canonical
article IDs, duplicate IDs, catalog joins, mapping semantics, and image failure
records. Text/image maps are independent; join by article ID, not row position.
Never sort only one side of an ID/vector pair. Record complete versus sampled
checks honestly, including unknown provenance and upstream fitted transforms.

Keep validation reports and derived adapters separate from source manifests.
Blocking mapping/dimension/integrity errors stop affected consumers; ordinary
missing modalities use masks and never delete future ground truth. Do not
fabricate repair values or launch extraction automatically.

Existing content FAISS files are optional source artifacts, not R7/R8 and not
substitutes for WALS/Two-Tower learned indexes. Before any use, validate type,
metric, normalization, dimension, vector count, ID mapping, and available
fingerprints. The default vector loader does not require those indexes.
Never overwrite a content index with a learned-factor index of the same shape.

## 6. Python Code Style & Documentation Standards

Follow the official Google Python Style Guide:

https://google.github.io/styleguide/pyguide.html

### 6.1 General Principles

- Write clean, readable, maintainable, production-oriented Python.
- Use English for all identifiers, comments, docstrings, and error messages.
- Follow Google-style docstrings and PEP 257.
- Use type hints for public functions, methods, and important internal
  interfaces.
- Prefer descriptive variable and function names over excessive comments.
- Avoid unnecessary abstraction and boilerplate.

### 6.2 Docstrings

Use Google-style docstrings with triple double quotes.

Document public APIs, nontrivial functions, classes, and methods.

Include the following sections when applicable:

- Args
- Returns
- Raises
- Yields
- Attributes

Document input/output semantics, shapes, data types, units, assumptions, and
side effects when relevant.

Do not generate verbose docstrings for trivial private helpers.

### 6.3 Inline Comments

- Explain WHY a decision is made, not WHAT an obvious statement does.
- Document non-obvious algorithms, assumptions, constraints, and performance
  optimizations.
- Avoid redundant line-by-line comments.
- Keep comments accurate when modifying code.
- Never add decorative comments merely to increase documentation coverage.

### 6.4 Machine Learning Documentation

For ML, data engineering, and recommendation systems, explicitly document
important assumptions involving:

- Input and output tensor shapes.
- Feature definitions and preprocessing.
- Label construction.
- Positive and negative sampling.
- Temporal split boundaries.
- Data leakage prevention.
- Candidate generation and filtering.
- Loss functions and optimization objectives.
- Evaluation metrics and denominators.
- Random seeds and reproducibility.
- Training versus inference behavior.
- Computational complexity where relevant.

Pay particular attention to point-in-time correctness and training-serving
consistency.

For evaluation code, clearly specify the evaluation population, candidate
universe, ground truth, ranking cutoff, and treatment of missing interactions.

### 6.5 Engineering Quality

- Use Ruff for linting and formatting.
- Use Pyright for static type checking.
- Use pytest for automated tests.
- Follow existing project conventions where they do not conflict with these
  requirements.
- Do not suppress warnings without justification.
- Do not modify algorithmic behavior solely to satisfy formatting rules.

### 6.6 Documentation Quality

Documentation should help another engineer understand and maintain the system
without repeatedly reading implementation details.

Prioritize clarity, correctness, and conciseness.

Do not over-comment straightforward code.

When implementing complex logic, explain the underlying assumptions and design
tradeoffs.

**The objective is professional, production-oriented code with concise,
technically meaningful documentation—not maximum comment density.**

## 7. Applying the standards in this repository

The six subsections above are the user's required standard. The following are
project-specific enforcement defaults, not a claim that Google mandates Ruff
or Pyright. The user's requested tools take precedence over alternative tools
mentioned in the style guide.

- Use four-space indentation and a target line length of 80 characters, subject
  to the guide's justified exceptions. Let Ruff format Python code.
- Use `snake_case` for functions/variables/modules, `PascalCase` for classes,
  and `UPPER_SNAKE_CASE` for constants. Keep imports explicit and side effects
  out of module import time.
- Prefer keyword-only arguments for easy-to-confuse temporal/configuration
  parameters. A parameter named `cutoff_date` must have the documented inclusive
  semantics; do not mix it with a prediction date.
- Do not repeat type hints mechanically in docstrings. Add semantics that types
  cannot express: matrix axes, sortedness, coordinate mappings, null handling,
  interval closure, units, mutation, and storage side effects.
- Use professional English for all authored project documentation, Markdown
  headings, notebook narrative, comments, docstrings, and emitted log/error
  messages. Preserve quoted source material and immutable reference files as
  references; do not silently translate or rewrite archived sources.
- Use explicit exceptions for invalid external data, schema mismatches, and
  incompatible model artifacts. Do not use `assert` as the only production
  validation, catch broad exceptions and continue silently, or log secrets and
  raw customer histories.
- Wrap poorly typed third-party boundaries in narrow typed adapters when useful.
  Do not spread `Any`, unqualified `type: ignore`, `noqa`, or warnings filters to
  hide issues. A necessary suppression must be local, targeted, and justified.
- Keep algorithm explanations near non-obvious code; keep long derivations,
  experimental alternatives, and architecture rationale in `docs/archi/`.
  Do not paste the proposal into every function docstring.

When bootstrapping `pyproject.toml`, use the following as a proposed baseline;
merge with the existing configuration rather than replacing unrelated settings:

```toml
[tool.ruff]
line-length = 80
src = ["src"]
extend-exclude = ["data", "docs/achive"]

[tool.ruff.lint]
select = ["E4", "E7", "E9", "F", "I", "B", "UP", "D"]

[tool.ruff.lint.pydocstyle]
convention = "google"

[tool.ruff.format]
quote-style = "double"
indent-style = "space"
docstring-code-format = true

[tool.pyright]
include = ["src", "scripts", "tests"]
exclude = ["data", "docs/achive", ".venv"]
typeCheckingMode = "standard"

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "--strict-markers"
markers = [
    "integration: exercises multiple pipeline components",
    "slow: requires explicit resource authorization",
]
```

Use the repository's supported Python version and pinned tool versions; do not
invent version pins from memory. Type-check important public and internal
interfaces, and strengthen narrow modules where practical. Do not disable
checks across source packages just to obtain a green build. Automated style
checks do not replace review of comment correctness or ML semantics.

## 8. Required verification

Run appropriate focused tests during development and the repository-wide checks
when the environment supports them:

```bash
ruff format --check .
ruff check .
pyright
pytest -q
```

Use the project's environment runner if one already exists. A missing command
or dependency is a reported verification limitation, not a passed check.
Formatting may be applied with `ruff format .` before running the check; inspect
lint fixes and never apply unsafe fixes without reviewing behavior.

At minimum, relevant implementation phases need tests covering:

1. Exact trailing-window boundaries, seven-day labels, same-day exclusion, and
   future-row perturbations that do not change an earlier query's features.
2. Leading zeros, catalog joins, retained raw duplicates, deduplicated labels,
   and deterministic date/article ties.
3. Independent channel limits, active-pool filtering, repeat purchases, sibling
   self-exclusion, overlap provenance, empty history, and fallback behavior.
4. WALS CSR orientation/confidence scaling, fixed item mapping, folding-in,
   unknown items, and save/load score parity.
5. Two-Tower history masks, duplicate/multi-positive negatives, training-only
   dropout, frozen source vectors, expected shapes, zero-vector safety, and index maps.
6. Cutoff-safe features, NaN/mask semantics, frozen preprocessing, categorical
   handling, and training-serving feature parity.
7. Group contiguity, train-only negative sampling, reproducible query-local
   seeds, no injected positives, and preservation of all evaluation queries.
8. Hand-computed MAP@12 and NDCG examples, full-ground-truth denominators,
   zero hits, zero candidates, short lists, duplicates, and misses caused by
   active-pool or vocabulary restrictions.
9. Fresh-process artifact reload, immutable versions, missing/corrupt file
   handling, and a small end-to-end integration fixture.

Also test pairwise-disjoint target windows and rejection of overlapping
legacy anchors; literal source paths with spaces; separate text/image row maps;
manifest/array dimension conflicts; image failure masks; and non-use of raw
content indexes as learned indexes. Test fixtures must not trigger extraction.

Keep tests offline and bounded by default. Large-data, accelerator, or external
service tests must be separately marked and explicitly invoked. Do not mistake
a synthetic pipeline run for an H&M benchmark.

## 9. Resource, privacy, and reproducibility rules

Use DuckDB and Arrow/Parquet batches for large tables. Use CSR/sparse neighbor
representations for CF. Never materialize a full dense user-item matrix or
unbounded all-user candidate DataFrame. Estimate pair enumeration, batch sizes,
and index memory before expensive jobs.

Record seeds for Python, NumPy, PyTorch, samplers, clustering, and model
initialization where applicable. Record library versions, device, thread counts,
config hashes, data fingerprints, and model/index/schema mappings. Do not promise
bitwise determinism when the runtime cannot provide it.

Runtime caches, checkpoints, metrics, and manifests remain under `data/`.
MLflow must use explicitly configured local paths under `data/runs/` rather than
silently creating a root-level `mlruns/`. DVC can use its normal metadata/cache
mechanism; that does not authorize additional dataset roots. Never commit raw
customer data, secrets, credentials, downloaded weights, or large artifacts to
Git. Reference artifacts through DVC and immutable manifest identities.

## 10. Completion report and change discipline

For each implementation task, report the changed files, behavior introduced or
changed, commands actually executed and their outcomes, and any unverified
assumptions or blockers. Include actual artifact paths and run IDs only when
they exist. State whether tests were synthetic, sampled, or full-data.

When dates, candidate pools, quotas, sampling, model objectives, feature schemas,
or metric denominators change, update the applicable proposal/protocol/decision
record and invalidate incompatible cached artifacts. Do not edit archived
sources or silently compare runs from different protocol versions.

Definition of done: the assigned behavior is implemented and tested, relevant
interfaces and assumptions are documented, the requested quality checks are
reported honestly, and no unrelated model or infrastructure scope was added.

## 11. Jupyter Notebook & Markdown Documentation Standards

### 11.1 General Principles

Follow these references where applicable:

- Google Python Style Guide
- Google Markdown Style Guide
- Jupyter Notebook Best Practices
- PEP 8 and PEP 257

All documentation, Markdown headings, comments, and docstrings must be written
in professional English.

Prioritize correctness, readability, reproducibility, and maintainability.

### 11.2 Jupyter Notebook Standards (.ipynb)

#### Structure

Organize notebooks in a logical sequence:

1. Title and Objective
2. Environment and Configuration
3. Data Loading and Validation
4. Methodology / EDA
5. Experiments
6. Results and Interpretation
7. Conclusions and Next Steps

Use Markdown headings to create a navigable outline.

#### Code Cells

- Follow Google Python Style Guide.
- Group imports near the beginning.
- Define configuration, paths, random seeds, and parameters explicitly.
- Keep cells focused on one logical operation.
- Use meaningful variable names and type hints where useful.
- Avoid excessively long code cells.
- Avoid hardcoded absolute paths.
- Never hardcode credentials or secrets.
- Avoid unnecessary package installation inside notebooks.
- Move reusable models, data pipelines, metrics, and utilities into Python
  modules under `src/`.
- Use notebooks primarily for exploration, experimentation, visualization, and
  analysis.

#### Markdown Cells

- Explain the purpose of each major section before executing its code.
- Describe WHY an analysis is performed, not merely WHAT the code executes.
- Document important assumptions and methodological decisions.
- Present mathematical definitions where necessary.
- Explain results immediately after relevant tables or figures.
- Distinguish observations from interpretations.
- Avoid redundant narration and decorative headings.

#### Experiments and Reproducibility

- Notebooks must execute sequentially from top to bottom.
- Avoid hidden state and out-of-order cell dependencies.
- Use explicit random seeds.
- Record dataset versions, code versions, configurations, and important package
  versions.
- Clearly document train/validation/test boundaries.
- Prevent temporal leakage and accidental test-set contamination.
- Never fabricate experimental results.
- Clearly mark experiments that have not been executed.
- Never claim successful execution without verification.
- Validate with Restart Kernel + Run All when feasible.
- Remove irrelevant debugging outputs and excessively large cell outputs before
  committing.
- Preserve meaningful experimental figures and results.

#### Visualizations

- Use descriptive plot titles.
- Label axes with units where appropriate.
- Include legends when needed.
- Use consistent visual styles.
- Avoid unnecessary visual complexity.
- Explain what each important figure demonstrates.
- Save important figures as reproducible artifacts when appropriate.

### 11.3 Markdown Document Standards (.md)

#### Structure and Formatting

- Use one H1 heading per document.
- Use hierarchical H2, H3, and H4 headings without skipping levels.
- Include a linked Table of Contents for long technical reports.
- Use descriptive section titles.
- Keep paragraphs concise and logically organized.
- Prefer tables for structured comparisons.
- Use ordered lists for sequential procedures.
- Use fenced code blocks with explicit language identifiers.
- Use relative links for repository files.
- Avoid unnecessary HTML.

#### Mathematics

- Use `$...$` for inline LaTeX.
- Use `$$...$$` for display equations.
- Place display math delimiters on separate lines.
- Leave blank lines before and after display equations.
- Define symbols and assumptions.
- Maintain consistent notation throughout the document.
- Use supported KaTeX-compatible commands where practical.
- Verify that formulas render correctly in VS Code Markdown Preview.
- Avoid unsupported custom LaTeX macros unless explicitly configured.

#### Technical Diagrams

- Use Mermaid for flowcharts, architecture diagrams, and sequence diagrams.
- Keep diagrams readable and technically accurate.
- Explain important relationships in accompanying text.
- Validate Mermaid syntax before delivery.

#### Research and Engineering Documentation

- Reference original papers and official documentation.
- Explain architecture choices and tradeoffs.
- State experimental assumptions explicitly.
- Include hardware, dataset, and evaluation settings when reporting benchmarks.
- Distinguish measured results from estimates.
- Do not invent citations or performance metrics.
- Clearly document limitations and unresolved issues.

### 11.4 Machine Learning Specific Requirements

For recommendation system notebooks and reports, explicitly document:

- Dataset and interaction definitions
- Temporal split and feature cutoff
- Candidate universe
- Ground truth and label construction
- Negative sampling strategy
- Model inputs and outputs
- Loss functions
- Evaluation metrics and aggregation
- Warm-start and cold-start handling
- Training versus inference behavior
- Reproducibility settings

Ensure all documented assumptions match the actual implementation.

### 11.5 Quality Assurance

Before delivering notebooks or Markdown files:

- Check heading hierarchy.
- Verify formulas and Markdown rendering.
- Verify internal links and code references.
- Validate Mermaid diagrams.
- Run Ruff and relevant tests.
- Check notebook execution order.
- Check that results correspond to executed code.
- Remove obsolete comments and misleading outputs.
- Preserve the existing document structure unless restructuring is requested.

For this repository, apply these requirements together with Sections 3, 4, 8,
and 9. Reusable notebook code belongs under `src/hm_recsys/`; resolve runtime
inputs and outputs through the fixed repository `data/` root. A documentation
task does not authorize an expensive dataset or training run. If notebook
execution, preview rendering, diagram validation, or required tools cannot be
verified, report the unverified checks and reasons; textual review alone is not
proof of execution or successful rendering.

**The goal is clean, self-explanatory, reproducible engineering documentation—not
maximum documentation length.**
