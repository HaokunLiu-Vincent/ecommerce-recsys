# H&M Next-Seven-Day Recommendation System

A purchase-history recommender with six native retrieval channels, one shared
point-in-time feature implementation and one LightGBM LambdaMART returning
unique Top-12 articles. The only active Colab client is
[`colab_training_val.ipynb`](colab_training_val.ipynb) at the repository root.
Every retrieval channel, union/RRF, feature build and ranker has its own
executable stage cell. Model and evaluation logic stays in `src/hm_recsys/`.

This is purchase prediction, not CTR/CVR, stock verification, returns analysis
or causal lift measurement. There are no fabricated impressions, sessions,
orders, sizes or inventory. Static content/customer snapshots and unobserved
purchase negatives remain explicit limitations.

## Contents

- [Architecture and scope](#architecture-and-scope)
- [Development environment](#development-environment)
- [Inputs, outputs and ownership](#inputs-outputs-and-ownership)
- [Time and evaluation contracts](#time-and-evaluation-contracts)
- [Six retrievers, RRF and the ranker](#six-retrievers-rrf-and-the-ranker)
- [Root Notebook operation](#root-notebook-operation)
- [CLI and local smoke](#cli-and-local-smoke)
- [Resources, progress and recovery](#resources-progress-and-recovery)
- [Final Test isolation](#final-test-isolation)
- [Active Python file map](#active-python-file-map)
- [Verification and publication](#verification-and-publication)
- [Design and implementation evidence](#design-and-implementation-evidence)

## Architecture and scope

The canonical packages are organized by responsibility:

```text
colab_training_val.ipynb       # Only active Notebook
src/hm_recsys/
  core/                       # IDs, configuration, temporal contracts, progress
  data/                       # Canonical/history/target/query readers
  retrieval/                  # Six algorithms and persisted channel candidates
    wals/                     # Confidence CSR, fit, fold-in
    two_tower/                # Model/loss, inputs, training and frozen inference
  fusion/                     # Native deduplication, provenance, RRF
  features/                   # Content, preprocessing and one feature builder
  ranking/                    # Train sampling/groups, LambdaMART and reload
  evaluation/                 # Shared macro metric definitions
  runtime/                    # Stages, identities, resources, checkpoints, CLI
scripts/                      # Thin executable clients
configs/                      # Explicit smoke/full and compute settings
tests/                        # Offline bounded tests
data/                       # Immutable inputs and private runtime artifacts
docs/archi/                 # Current architecture, runbook and reports
docs/achive/                # Ignored local historical material; never imported
```

Smoke and full share production algorithms. They differ through explicit
population, compute and resource configuration. No active `pilot/`, `colab/`,
legacy model wrapper or second metric implementation is an alternative source
of truth. A source-layout change gets a new run identity rather than rewriting
old checkpoint hashes. See the [migration map](docs/archi/refactor_migration_map.md)
for preserved old paths and exact move/checksum evidence.

The [architecture document](docs/archi/architecture.md) describes actual call
boundaries. No deployed service, DVC/MLflow platform, HTTP server or AWS setup is
required by this repository.

## Development environment

Reuse the existing Python 3.12 `uv` project and `uv.lock`. This is a non-package
source checkout; thin scripts resolve `src/` independently of the shell's current
working directory. Do not recreate dependency management or add unneeded packages.

```bash
export PATH="$HOME/.local/bin:$PATH"
uv python install 3.12
uv sync --locked
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked pyright
uv run --locked pytest -q
```

DuckDB is the base dependency. The locked development group supplies PyArrow,
NumPy/SciPy, PyTorch, implicit, FAISS, LightGBM, nbformat and quality tools.
Dependency declaration changes require a consistent lock update. Actual version
and device information is recorded per run, rather than inferred from a GPU name.

On the local M1 Max, keep checks small and native threads bounded:

```bash
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 uv run --locked pytest -q
uv run --locked python scripts/local_smoke.py --help
uv run --locked python scripts/run_pipeline.py --help
```

Do not start real full training, full candidate construction or unrestricted
prepacking on a machine with only 7–8 GiB available. Synthetic fixtures test
real algorithms on small inputs; their scores are not H&M benchmark results.
The Notebook prepares the existing locked worker environment separately from
Colab's kernel and does not replace its CUDA-enabled kernel PyTorch install.

## Inputs, outputs and ownership

`data/` is the only runtime data root. These supplied inputs are immutable:

| Directory | Required artifacts and meaning |
|---|---|
| `data/original data/` | `articles.csv`, `customers.csv`, `transactions_train.csv`; literal space preserved |
| `data/text_features/` | `text_article_ids.parquet`, `text_embeddings.npy`, `text_features_manifest.json`, original `text_faiss.index` |
| `data/image_features/` | `image_article_ids.parquet`, `image_embedding_failures.parquet`, `image_embeddings.npy`, `image_features_manifest.json`, original `image_faiss.index` |

Text/image maps are independent and joined by ten-character article ID, not row
position. Validate real dimensions, row counts, duplicates, catalog joins,
finite/nonzero usable values and failures. Ordinary missing modalities use
masks. Blocking mapping/dimension/integrity errors stop consumers; they never
trigger automatic extraction, encoder downloads or source-array repair.
Original content indexes are not learned WALS/Two-Tower indexes.

New local runs live under `data/runs/<kind>/<run_id>/`. Their canonical tables,
channel candidates, native union, labels, 49-feature shards, model/index files,
aggregate reports, scratch, manifests and checkpoints stay within that root.
Existing completed runs and source files are not moved or overwritten.

| Colab path | Ownership |
|---|---|
| `/content/e_commerce_recsys/` | Pinned code checkout and worker environment |
| `/content/e_commerce_recsys/data/` | Local staged inputs, training and large intermediate work |
| `/content/drive/MyDrive/projects/e_commerce_recsys/data/` | Existing Drive inputs; only `original data/`, `text_features/`, `image_features/` are staged |
| Drive `data/runs/colab/<RUN_ID>/` | Verified stage deltas and restart checkpoints |

Do not copy the entire Drive `data/` or old runs as training inputs. Do not
train directly on the mounted Drive. Customer-level rows, checkpoints and
populated Notebook outputs are private, even when a report contains only
aggregate metrics. Ignoring a directory does not remove it from Git history.

## Time and evaluation contracts

The authoritative [Evaluation Protocol v2.0](docs/archi/evaluation_protocol.md)
and [proposal v0.3.2](docs/proposal.md) define:

| Role | Observable cutoff / target dates | Labels, end exclusive |
|---|---|---|
| WALS base | [2020-02-01, 2020-08-12); item vocabulary [2020-06-16, 2020-08-12) | Implicit confidence |
| Two-Tower training | Targets [2020-05-19, 2020-08-05) | History strictly before each target day |
| Two-Tower internal selection | [2020-08-05, 2020-08-12) | Daily internal Recall@50 |
| F1 | 2020-08-11 | [2020-08-12, 2020-08-19) |
| F2 | 2020-08-18 | [2020-08-19, 2020-08-26) |
| F3 | 2020-08-25 | [2020-08-26, 2020-09-02) |
| F4 | 2020-09-01 | [2020-09-02, 2020-09-09) |
| Validation | 2020-09-08 | [2020-09-09, 2020-09-16) |
| Final Test, gated | 2020-09-15 | [2020-09-16, 2020-09-23) |

`cutoff=T` is inclusive; prediction starts `a=T+1`. History is `t_dat<a`, a
trailing d-day window is `[a-d,a)`, and labels are `[a,a+7)`. All six outer
label weeks are disjoint. Earlier completed weeks may enter later history;
frozen base factors, tower weights and preprocessing do not refit on them.

Transactions retain raw repeats. Labels are unique future catalog articles.
The active pool is catalog items with at least three purchase lines in the
trailing 28 days; it is not an inventory measure. Apply it before channel Top-K,
never to full truth. Keep purchases missed by active pools, factors or retrieval
in denominators. Query identity is customer/cutoff; candidate identity adds
article. Repeated purchases remain eligible recommendations.

The primary metric is native end-to-end macro MAP@12. Each query's AP divides
its precision-weighted hit sum by `min(12, full_truth_size)`. Macro Recall@12 and
NDCG@12 also use full truth, retaining zero-hit/zero-candidate queries. Do not
substitute candidate-only NDCG, a retrieved-only truth set or micro recall.
No future-positive injection is permitted at any stage.

## Six retrievers, RRF and the ranker

| Channel | Quota | Canonical algorithm and update |
|---|---:|---|
| Trend | 60 | Seven-day time-decayed global/age-bucket popularity; explicit global backfill |
| Repurchase | 30 | Sixty-day raw counts and existing time decay; repeats retained |
| Product-code sibling | 30 | Actual `product_code` variants from recent seeds; seed self-exclusion |
| ItemCF | 50 | Global legal 28-day sparse asymmetric similarity, last-edge-date decay, frozen neighborhood/seed limits |
| WALS | 50 | Fit allowed confidence CSR once; freeze unnormalized item factors; solve current users by genuine folding-in |
| Hybrid Two-Tower | 50 | Train existing multi-positive LogQ model with frozen source content and trainable projections; select internally, freeze and encode current histories |

Native fusion deduplicates all six lists and preserves scores, one-based ranks,
source masks and fallback metadata. RRF is `sum(1/(60+source_rank))`; it has no
learned weights and adds no pre-ranking truncation. The quota sum is 270.
Removing a channel for a union-Recall diagnostic does not redistribute quotas.

Two-Tower caches model-independent IDs, raw content, profiles, historical IDs
and weights. Learned item/user vectors and projection outputs are not frozen as
training caches. Static item inputs can live on CUDA when measured memory
allows; host lookup remains explicit otherwise. Targets/history stay bounded,
full multi-positive masking and Q are preserved, and internal validation runs
on its separate dates. No AMP/TF32/compile or batch/optimizer changes are implied
by moving modules.

One `FeatureBuilder` emits the same 49 ordered float32 columns for training and
inference: recall scores/ranks/masks, user/item/cross history, content similarity
and availability, and fixed categorical preprocessing. Raw embeddings are not
expanded into ranker columns. Missing source values are NaN with explicit masks.

One LambdaMART consumes contiguous query groups across F1–F4. Only Train may
drop groups without positive/negative pairs and sample up to twenty negatives
per native positive, preserving the existing half-hard/query-local-random rule.
Full native Validation/Test are unsampled. The LightGBM candidate-list NDCG@12
early-stopping diagnostic is separate from external full-truth MAP@12.

## Root Notebook operation

Open
[the root Notebook in Colab](https://colab.research.google.com/github/HaokunLiu-Vincent/ecommerce-recsys/blob/main/colab_training_val.ipynb)
after the reviewed changes are published by the repository owner. This task
does not commit or push them. GitHub may not yet contain the working-tree version.
The client itself clones/checks out a verified pinned commit locally; no ZIP
upload or remote Python execution is needed.

Defaults:

```python
RUN_PROFILE = "smoke"
RUN_ACTION = "run"
RUN_ID = "colab_smoke_modular_20261103_01"
DEVICE = "cpu"
ALLOW_FULL_RUN = False
RUN_FINAL_TEST = False
ALLOW_FINAL_TEST = False
LOCKED_RUN_SIGNATURE = None
```

| Executable cell | Action and aggregate outputs |
|---|---|
| Git/environment/input staging | Validate HTTPS URL/ref/SHA/destination, mount Drive, install locked worker environment and copy only three inputs |
| `prepare` | Canonical acceptance, split/catalog/content inputs, F1–F4/Validation rosters/full truth and resource plan; Test metadata only |
| `trend` / `repurchase` / `sibling` | One separate rule build + retrieve + validate cell each |
| `itemcf` | Its own sparse global index build/reuse, retrieval and metrics |
| `wals` | Its own fit/freeze/fold-in/retrieve/validate stage |
| `twotower` | Its own data prep/training/internal selection/frozen indexing/retrieval stage |
| `fusion` | Persist native union; show overlap, Recall, signed contributions and RRF baseline |
| `features` | Persist shared 49-feature Train/Validation rows, schema and bounded resource/throughput diagnostics |
| `ranker` | One jointly grouped four-fold LambdaMART fit; full Validation predictions and diagnostic |
| `validation` | Same-roster rule/channel/ranker comparison with correct full-truth metrics |
| `reload` | New-process feature/score/Top-12 consistency and frozen identities |
| Last code cell: `final_test` | Skipped by default; separately approved locked Test inference only |

Each cell prints actual argv and live worker progress. A missing prerequisite
produces a named error; a channel cell does not implicitly train all earlier
models. Matching completed stages are reused only after manifest/file checks.
Reports live at `data/runs/colab/<RUN_ID>/reports/NAME.json` and private logs at
that run's `notebook_logs/`.

See the [step-by-step runbook](docs/archi/github_colab_runbook.md) for private
GitHub secrets, exact smoke/full settings and recovery. A status inspection is
not proof that an unexecuted stage completed.

## CLI and local smoke

The same stage interface backs every Notebook cell:

```bash
uv run --locked python scripts/run_pipeline.py stage --help
uv run --locked python scripts/run_pipeline.py stage \
  --stage trend --stage-action status \
  --profile smoke --run-id YOUR_EXISTING_RUN_ID \
  --data-root /path/to/project/data \
  --persistent-root /path/to/verified/run-parent
```

For Git-aware Colab execution also supply the complete original
`--git-remote`, `--git-ref`, `--git-commit` trio and the mounted Drive root.
`--stage-action run` builds only that stage; `resume` restores/verifies compatible
stage inputs; `status` inspects. The names are `prepare`, `trend`, `repurchase`,
`sibling`, `itemcf`, `wals`, `twotower`, `fusion`, `features`, `ranker`,
`validation`, `reload`, `final_test`.

`local_smoke.py` is a thin client of the same canonical pipeline. Default
behavior runs input preflight without training. An explicit bounded synthetic
end-to-end run is:

```bash
uv run --locked python scripts/local_smoke.py --synthetic --run \
  --run-id local_modular_smoke_20261103_01 --until reload
```

Omitting `--synthetic` uses the real input root and requires a deliberate
bounded real-smoke decision. `--until` selects a whole-run endpoint only in
this local convenience client; independent Notebook cells use the named-stage
interface above. Reusing the same identity verifies completed stages/reload.
Normal pytest fixtures also exercise real bounded WALS/Two-Tower/LambdaMART;
random-score substitutes do not constitute an end-to-end smoke.

No local full run or real final Test is authorized by this README. Full's null
caps mean all legal data, not a hidden query/user/target sample. Do not bypass
host/resource guards by setting internal fixture flags on real data.

## Resources, progress and recovery

Large tables use DuckDB, Arrow/Parquet batches, sparse matrices and bounded
context arrays. Measure separate WALS, prepared targets, four-fold features,
complete Validation and single-LightGBM capacity. Memmaps bound array residency;
they do not make LightGBM arbitrarily out-of-core. Insufficient resources stop
without relabeling a reduced run as full.

Workers provide actual rows/queries, epochs/batches/loss, timings, rates, RSS,
CUDA allocated/reserved memory and checkpoint state. Unknown denominators/ETAs
stay unknown. GPU allocation is not GPU utilization. Per-operation CUDA events
measure asynchronous device intervals; their nested timings are not simply
added to host wall time.

Stage completion requires verified outputs, dependency/source/input/config
identity and atomic manifest publication. A failed stage remains incomplete.
Two-Tower restart checkpoints include weights, optimizer/scheduler/RNG,
training position and selected state; the static input bank is separate.
Local save cadence and remote Drive publication cadence differ. Lower remote
frequency can lose more progress on runtime loss; the previous valid remote
checkpoint remains usable until a new publication verifies successfully.

For a new runtime, pin the original explicit commit, reuse the exact run ID,
profile/device/backend/input identity, restage the three source directories and
set `RUN_ACTION="resume"`. Run the needed cell; missing earlier prerequisites
are named rather than automatically trained. Use a new run ID for source/layout
or configuration changes. Do not mutate running-job directories, rewrite old
manifests or rename pilot outputs to full.

## Final Test isolation

Preparation defines Test dates and availability only. With default settings,
no Test future labels are constructed/read and no Test scorer is called.
The final executable cell requires all of:

1. A completed full training, Validation and reload identity.
2. `RUN_FINAL_TEST=True` and separate `ALLOW_FINAL_TEST=True`.
3. An explicit recorded `LOCKED_RUN_SIGNATURE` matching source, inputs, config,
   models, feature schema and splits.
4. Backend checks before any Test future-label read or scoring.

Approved Test is a separate once-only report with past-only cutoff features and
full eligible unique truth, without refitting, negative sampling or tuning.
A synthetic authorization/recovery test is not real H&M Test evaluation. Full
and real Test remain unrun unless an actual separately authorized report says
otherwise.

## Active Python file map

The following inventory covers every active Python source, script and test
module. Package initializers define boundaries/exports, not alternate models.
“Callers” identifies primary static importers or entry points; dynamic worker
selection is also described in the architecture/runbook. The migration report
records the final before/after paths and checksums.

<!-- BEGIN ACTIVE PYTHON INVENTORY -->

### CLI scripts

| File | Purpose and main interfaces | Primary callers |
|---|---|---|
| [scripts/audit_git.py](scripts/audit_git.py) | Thin read-only Git working-tree/index/history privacy-audit entrypoint. Interfaces: `main`. | Named runtime/CLI dispatch or direct tests |
| [scripts/benchmark_compute.py](scripts/benchmark_compute.py) | Thin entrypoint for explicitly bounded compute/parity benchmarks. | Named runtime/CLI dispatch or direct tests |
| [scripts/local_smoke.py](scripts/local_smoke.py) | Thin guarded local smoke client of the canonical production algorithms. | Named runtime/CLI dispatch or direct tests |
| [scripts/probe_device.py](scripts/probe_device.py) | Thin parser for an isolated real CPU/CUDA/library probe and its output directory. Interfaces: `build_parser`, `main`. | Named runtime/CLI dispatch or direct tests |
| [scripts/run_pipeline.py](scripts/run_pipeline.py) | Thin plan/stage/staging/status/recovery CLI used by every Notebook stage cell. | Named runtime/CLI dispatch or direct tests |

### Package root

| File | Purpose and main interfaces | Primary callers |
|---|---|---|
| [src/hm_recsys/__init__.py](src/hm_recsys/__init__.py) | Package boundary and documented exports; no alternate training or metric implementation. | Package imports |

### core

| File | Purpose and main interfaces | Primary callers |
|---|---|---|
| [src/hm_recsys/core/__init__.py](src/hm_recsys/core/__init__.py) | Package boundary and documented exports; no alternate training or metric implementation. | Package imports |
| [src/hm_recsys/core/contracts.py](src/hm_recsys/core/contracts.py) | Validated configuration, canonical IDs/JSON hashes, stable seeds and exact split manifests. Interfaces: `write_json`, `read_json`, `digest`, `stable_key`, `article_id`, `split_manifest`; additional helpers in file. | [scripts/probe_device.py](scripts/probe_device.py); [src/hm_recsys/core/synthetic.py](src/hm_recsys/core/synthetic.py); 61 other static importers |
| [src/hm_recsys/core/progress.py](src/hm_recsys/core/progress.py) | Append/forward real aggregate worker events; bounded line framing and truthful count/rate rendering. Interfaces: `scope`, `emit`, `Lines`, `describe`. | [src/hm_recsys/data/history.py](src/hm_recsys/data/history.py); [src/hm_recsys/features/content.py](src/hm_recsys/features/content.py); 21 other static importers |
| [src/hm_recsys/core/synthetic.py](src/hm_recsys/core/synthetic.py) | Deterministic tiny raw-data/content fixtures; never a production-data substitute. Interfaces: `create_fixture`. | [src/hm_recsys/runtime/testing.py](src/hm_recsys/runtime/testing.py); [tests/test_canonical_data_features.py](tests/test_canonical_data_features.py); 4 other static importers |

### data

| File | Purpose and main interfaces | Primary callers |
|---|---|---|
| [src/hm_recsys/data/__init__.py](src/hm_recsys/data/__init__.py) | Package boundary and documented exports; no alternate training or metric implementation. | Package imports |
| [src/hm_recsys/data/history.py](src/hm_recsys/data/history.py) | Bounded canonical purchase conversion, cutoff-safe SQL history and independent rosters/full labels; gated Test roster adapter. Interfaces: `allocation`, `connect`, `transaction_view`, `canonical`, `History`, `rosters`; additional helpers in file. | [src/hm_recsys/data/streaming.py](src/hm_recsys/data/streaming.py); [src/hm_recsys/features/benchmark.py](src/hm_recsys/features/benchmark.py); 23 other static importers |
| [src/hm_recsys/data/sample.py](src/hm_recsys/data/sample.py) | Typed raw purchase/sample records, CSV contract inspection and deterministic age buckets. Interfaces: `Transaction`, `Sample`, `inspect_csv`, `age_bucket`. | [src/hm_recsys/core/synthetic.py](src/hm_recsys/core/synthetic.py); [src/hm_recsys/data/history.py](src/hm_recsys/data/history.py); 8 other static importers |
| [src/hm_recsys/data/streaming.py](src/hm_recsys/data/streaming.py) | Bounded SQL export, deterministic target iteration and partition/query/shard readers. Interfaces: `export_query`, `rows`, `TargetStream`, `Partition`, `Shards`, `load_manifest`. | [src/hm_recsys/data/history.py](src/hm_recsys/data/history.py); [src/hm_recsys/features/benchmark.py](src/hm_recsys/features/benchmark.py); 18 other static importers |

### evaluation

| File | Purpose and main interfaces | Primary callers |
|---|---|---|
| [src/hm_recsys/evaluation/__init__.py](src/hm_recsys/evaluation/__init__.py) | Package boundary and documented exports; no alternate training or metric implementation. | Package imports |
| [src/hm_recsys/evaluation/metrics.py](src/hm_recsys/evaluation/metrics.py) | Single shared full-truth macro Recall/MAP/NDCG definitions and streaming metric accumulation. Interfaces: `recall_at_k`, `metrics`, `union_recall`, `MetricAccumulator`. | [src/hm_recsys/features/benchmark.py](src/hm_recsys/features/benchmark.py); [src/hm_recsys/fusion/pipeline.py](src/hm_recsys/fusion/pipeline.py); 5 other static importers |

### features

| File | Purpose and main interfaces | Primary callers |
|---|---|---|
| [src/hm_recsys/features/__init__.py](src/hm_recsys/features/__init__.py) | Package boundary and documented exports; no alternate training or metric implementation. | Package imports |
| [src/hm_recsys/features/benchmark.py](src/hm_recsys/features/benchmark.py) | Matched bounded feature/reference timing and value/order diagnostics. Interfaces: `compare`. | Named runtime/CLI dispatch or direct tests |
| [src/hm_recsys/features/builder.py](src/hm_recsys/features/builder.py) | One ordered point-in-time 49-feature implementation and frozen visual-cluster lookup. Interfaces: `FeatureBuilder`, `frozen_clusters`. | [src/hm_recsys/features/benchmark.py](src/hm_recsys/features/benchmark.py); [src/hm_recsys/features/pipeline.py](src/hm_recsys/features/pipeline.py); 8 other static importers |
| [src/hm_recsys/features/content.py](src/hm_recsys/features/content.py) | Validate immutable modality arrays/maps/failures and build separate ID-aligned masked content adapters. Interfaces: `ContentSource`, `ContentBank`, `adapt_content`, `MappedContent`. | [src/hm_recsys/features/benchmark.py](src/hm_recsys/features/benchmark.py); [src/hm_recsys/features/builder.py](src/hm_recsys/features/builder.py); 16 other static importers |
| [src/hm_recsys/features/pipeline.py](src/hm_recsys/features/pipeline.py) | Construct train/inference features from persisted native unions without rerunning retrieval. Interfaces: `generate`. | [src/hm_recsys/runtime/stages.py](src/hm_recsys/runtime/stages.py); [tests/test_modular_backend.py](tests/test_modular_backend.py) |
| [src/hm_recsys/features/preprocessing.py](src/hm_recsys/features/preprocessing.py) | Fit/persist pre-base categorical/visual preprocessing once with its schema and fit identity. Interfaces: `preprocessing`. | [src/hm_recsys/runtime/stages.py](src/hm_recsys/runtime/stages.py); [tests/test_canonical_data_features.py](tests/test_canonical_data_features.py) |
| [src/hm_recsys/features/storage.py](src/hm_recsys/features/storage.py) | Write contiguous-query float32 Parquet feature shards and schema-aligned tables. Interfaces: `FeatureShards`, `feature_table`. | [src/hm_recsys/features/pipeline.py](src/hm_recsys/features/pipeline.py); [tests/test_computing_refactor.py](tests/test_computing_refactor.py) |

### fusion

| File | Purpose and main interfaces | Primary callers |
|---|---|---|
| [src/hm_recsys/fusion/__init__.py](src/hm_recsys/fusion/__init__.py) | Package boundary and documented exports; no alternate training or metric implementation. | Package imports |
| [src/hm_recsys/fusion/native.py](src/hm_recsys/fusion/native.py) | Native Candidate/provenance representation, exact RRF rule and deterministic article-key merge. Interfaces: `Candidate`, `merge`. | [src/hm_recsys/features/builder.py](src/hm_recsys/features/builder.py); [src/hm_recsys/fusion/pipeline.py](src/hm_recsys/fusion/pipeline.py); 6 other static importers |
| [src/hm_recsys/fusion/pipeline.py](src/hm_recsys/fusion/pipeline.py) | Fuse saved six-channel candidates and evaluate native union/RRF/complementarity on full truth. Interfaces: `fuse`. | [src/hm_recsys/runtime/stages.py](src/hm_recsys/runtime/stages.py); [tests/test_modular_backend.py](tests/test_modular_backend.py); 1 other static importers |
| [src/hm_recsys/fusion/storage.py](src/hm_recsys/fusion/storage.py) | Bounded persisted candidate readers preserving source rank/score/fallback metadata. Interfaces: `CandidateReader`, `load_candidates`. | [src/hm_recsys/features/benchmark.py](src/hm_recsys/features/benchmark.py); [src/hm_recsys/features/pipeline.py](src/hm_recsys/features/pipeline.py); 3 other static importers |

### ranking

| File | Purpose and main interfaces | Primary callers |
|---|---|---|
| [src/hm_recsys/ranking/__init__.py](src/hm_recsys/ranking/__init__.py) | Package boundary and documented exports; no alternate training or metric implementation. | Package imports |
| [src/hm_recsys/ranking/model.py](src/hm_recsys/ranking/model.py) | Train-only query-local hard/random sampling, grouped matrices, boosting progress and stable Top-12 ordering. Interfaces: `BoostProgress`, `select_training_rows`, `matrix`, `ordered_predictions`. | [src/hm_recsys/features/benchmark.py](src/hm_recsys/features/benchmark.py); [src/hm_recsys/ranking/pipeline.py](src/hm_recsys/ranking/pipeline.py); 1 other static importers |
| [src/hm_recsys/ranking/pipeline.py](src/hm_recsys/ranking/pipeline.py) | One jointly grouped LambdaMART fit, complete partition scoring and serialized-model/reload parity. Interfaces: `read_shard`, `feature_manifest`, `train`, `ExpectedRows`, `reload`, `predict_partition`. | [src/hm_recsys/features/benchmark.py](src/hm_recsys/features/benchmark.py); [src/hm_recsys/runtime/stages.py](src/hm_recsys/runtime/stages.py); 2 other static importers |

### retrieval

| File | Purpose and main interfaces | Primary callers |
|---|---|---|
| [src/hm_recsys/retrieval/__init__.py](src/hm_recsys/retrieval/__init__.py) | Package boundary and documented exports; no alternate training or metric implementation. | [src/hm_recsys/retrieval/pipeline.py](src/hm_recsys/retrieval/pipeline.py); [src/hm_recsys/retrieval/snapshot.py](src/hm_recsys/retrieval/snapshot.py); 1 other static importers |
| [src/hm_recsys/retrieval/base.py](src/hm_recsys/retrieval/base.py) | Channel Hit type, one-based ranks and deterministic score/article-ID Top-K contract. Interfaces: `Hit`, `top`. | [src/hm_recsys/fusion/native.py](src/hm_recsys/fusion/native.py); [src/hm_recsys/fusion/storage.py](src/hm_recsys/fusion/storage.py); 7 other static importers |
| [src/hm_recsys/retrieval/itemcf.py](src/hm_recsys/retrieval/itemcf.py) | Global legal sparse asymmetric co-purchase neighborhoods and cutoff-safe query retrieval. Interfaces: `build`, `retrieve`. | [src/hm_recsys/retrieval/pipeline.py](src/hm_recsys/retrieval/pipeline.py); [src/hm_recsys/retrieval/snapshot.py](src/hm_recsys/retrieval/snapshot.py); 2 other static importers |
| [src/hm_recsys/retrieval/learned.py](src/hm_recsys/retrieval/learned.py) | Batched frozen WALS/Two-Tower retrieval adapters; delegates model/encoding math to dedicated packages. Interfaces: `TowerRetriever`, `learned_batch`, `wals_batch`, `tower_batch`. | [src/hm_recsys/retrieval/pipeline.py](src/hm_recsys/retrieval/pipeline.py) |
| [src/hm_recsys/retrieval/pipeline.py](src/hm_recsys/retrieval/pipeline.py) | Execute one channel across required partitions; persist source candidates and full-roster retrieval metrics. Interfaces: `run_channel`. | [src/hm_recsys/runtime/stages.py](src/hm_recsys/runtime/stages.py); [tests/test_modular_backend.py](tests/test_modular_backend.py); 1 other static importers |
| [src/hm_recsys/retrieval/repurchase.py](src/hm_recsys/retrieval/repurchase.py) | Canonical raw-repeat/count/time-decay repurchase scoring and active retrieval. Interfaces: `reference_scores`, `scores`, `retrieve`. | [src/hm_recsys/retrieval/pipeline.py](src/hm_recsys/retrieval/pipeline.py); [src/hm_recsys/retrieval/snapshot.py](src/hm_recsys/retrieval/snapshot.py) |
| [src/hm_recsys/retrieval/sibling.py](src/hm_recsys/retrieval/sibling.py) | Actual product-code variant scoring with recent seeds and seed self-exclusion. Interfaces: `reference_scores`, `scores`, `retrieve`. | [src/hm_recsys/retrieval/pipeline.py](src/hm_recsys/retrieval/pipeline.py); [src/hm_recsys/retrieval/snapshot.py](src/hm_recsys/retrieval/snapshot.py) |
| [src/hm_recsys/retrieval/snapshot.py](src/hm_recsys/retrieval/snapshot.py) | Cutoff statistics/active-pool snapshots and bounded user-history adapters shared by channels/features. Interfaces: `build_statistics`, `build_snapshot`, `Snapshot`, `GlobalSnapshot`. | [src/hm_recsys/features/benchmark.py](src/hm_recsys/features/benchmark.py); [src/hm_recsys/features/builder.py](src/hm_recsys/features/builder.py); 8 other static importers |
| [src/hm_recsys/retrieval/trend.py](src/hm_recsys/retrieval/trend.py) | Canonical seven-day global/age-bucket popularity and quota-limited active retrieval. Interfaces: `popularity`, `retrieve`. | [src/hm_recsys/retrieval/pipeline.py](src/hm_recsys/retrieval/pipeline.py); [src/hm_recsys/retrieval/snapshot.py](src/hm_recsys/retrieval/snapshot.py) |
| [src/hm_recsys/retrieval/vectors.py](src/hm_recsys/retrieval/vectors.py) | Bounded exact inner-product block search, active-row filtering and deterministic ties. Interfaces: `block_topk`, `ActiveBlocks`, `BlockIndex`. | [src/hm_recsys/retrieval/learned.py](src/hm_recsys/retrieval/learned.py); [src/hm_recsys/retrieval/pipeline.py](src/hm_recsys/retrieval/pipeline.py); 7 other static importers |

### retrieval / two_tower

| File | Purpose and main interfaces | Primary callers |
|---|---|---|
| [src/hm_recsys/retrieval/two_tower/__init__.py](src/hm_recsys/retrieval/two_tower/__init__.py) | Package boundary and documented exports; no alternate training or metric implementation. | Package imports |
| [src/hm_recsys/retrieval/two_tower/execution.py](src/hm_recsys/retrieval/two_tower/execution.py) | Measured host/CUDA timing, bounded prefetch and static raw item-input placement. Interfaces: `Measurements`, `prefetch`, `ItemBank`. | [src/hm_recsys/retrieval/two_tower/inputs.py](src/hm_recsys/retrieval/two_tower/inputs.py); [src/hm_recsys/retrieval/two_tower/query_inputs.py](src/hm_recsys/retrieval/two_tower/query_inputs.py); 3 other static importers |
| [src/hm_recsys/retrieval/two_tower/inference.py](src/hm_recsys/retrieval/two_tower/inference.py) | Batch frozen WALS/tower query encoding for explicit partition rosters. Interfaces: `encode`. | [src/hm_recsys/runtime/stages.py](src/hm_recsys/runtime/stages.py) |
| [src/hm_recsys/retrieval/two_tower/inputs.py](src/hm_recsys/retrieval/two_tower/inputs.py) | Model-independent profile/category/content/history tensor assembly with strict target-day exclusion. Interfaces: `event_history`, `TowerEncoder`, `SqlEncoder`. | [src/hm_recsys/retrieval/two_tower/inference.py](src/hm_recsys/retrieval/two_tower/inference.py); [src/hm_recsys/retrieval/two_tower/training.py](src/hm_recsys/retrieval/two_tower/training.py); 7 other static importers |
| [src/hm_recsys/retrieval/two_tower/model.py](src/hm_recsys/retrieval/two_tower/model.py) | Trainable multi-modal item/user towers and unchanged duplicate-aware multi-positive LogQ loss. Interfaces: `TwoTower`, `positive_mask`, `logq_loss`. | [src/hm_recsys/retrieval/two_tower/inference.py](src/hm_recsys/retrieval/two_tower/inference.py); [src/hm_recsys/retrieval/two_tower/inputs.py](src/hm_recsys/retrieval/two_tower/inputs.py); 7 other static importers |
| [src/hm_recsys/retrieval/two_tower/prepared.py](src/hm_recsys/retrieval/two_tower/prepared.py) | Identity-checked bounded context/target/mmap preparation and cache verification. Interfaces: `identity`, `verify`, `daily_rows`, `prepare`, `Prepared`. | [src/hm_recsys/retrieval/two_tower/query_inputs.py](src/hm_recsys/retrieval/two_tower/query_inputs.py); [src/hm_recsys/retrieval/two_tower/training.py](src/hm_recsys/retrieval/two_tower/training.py); 4 other static importers |
| [src/hm_recsys/retrieval/two_tower/query_inputs.py](src/hm_recsys/retrieval/two_tower/query_inputs.py) | Reusable bounded cutoff/user input arrays for frozen query encoding. Interfaces: `QueryInputs`. | [src/hm_recsys/retrieval/two_tower/inference.py](src/hm_recsys/retrieval/two_tower/inference.py); [tests/test_computing_refactor.py](tests/test_computing_refactor.py) |
| [src/hm_recsys/retrieval/two_tower/training.py](src/hm_recsys/retrieval/two_tower/training.py) | Actual-device optimization, complete internal selection, full-state checkpoint cadence and frozen catalog encoding. Interfaces: `device_for`, `move`, `item_vectors`, `internal_recall`, `cached_internal_recall`, `fit`. | [src/hm_recsys/retrieval/two_tower/inference.py](src/hm_recsys/retrieval/two_tower/inference.py); [src/hm_recsys/runtime/benchmark.py](src/hm_recsys/runtime/benchmark.py); 2 other static importers |

### retrieval / wals

| File | Purpose and main interfaces | Primary callers |
|---|---|---|
| [src/hm_recsys/retrieval/wals/__init__.py](src/hm_recsys/retrieval/wals/__init__.py) | Package boundary and documented exports; no alternate training or metric implementation. | Package imports |
| [src/hm_recsys/retrieval/wals/model.py](src/hm_recsys/retrieval/wals/model.py) | One confidence transform, WALS model state and genuine fixed-item user folding-in. Interfaces: `confidence`, `fold_user`, `fold_users_batch`, `WALS`. | [src/hm_recsys/retrieval/learned.py](src/hm_recsys/retrieval/learned.py); [src/hm_recsys/retrieval/pipeline.py](src/hm_recsys/retrieval/pipeline.py); 3 other static importers |
| [src/hm_recsys/retrieval/wals/training.py](src/hm_recsys/retrieval/wals/training.py) | Legal base-window sparse fitting, frozen factor/maps persistence and resource checks. Interfaces: `fit_wals`. | [src/hm_recsys/runtime/stages.py](src/hm_recsys/runtime/stages.py) |

### runtime

| File | Purpose and main interfaces | Primary callers |
|---|---|---|
| [src/hm_recsys/runtime/__init__.py](src/hm_recsys/runtime/__init__.py) | Package boundary and documented exports; no alternate training or metric implementation. | [tests/test_compute.py](tests/test_compute.py); [tests/test_git_safety.py](tests/test_git_safety.py); 1 other static importers |
| [src/hm_recsys/runtime/acceptance.py](src/hm_recsys/runtime/acceptance.py) | Input/content acceptance and shared contract/reference utilities; production stages use canonical channel/fusion/feature pipelines. Interfaces: `catalog_ids`, `preflight`. | [src/hm_recsys/runtime/local_cli.py](src/hm_recsys/runtime/local_cli.py); [src/hm_recsys/runtime/stages.py](src/hm_recsys/runtime/stages.py); 1 other static importers |
| [src/hm_recsys/runtime/benchmark.py](src/hm_recsys/runtime/benchmark.py) | Controlled same-input/initial-state reference versus optimized parity and bounded performance reports. Interfaces: `reference_batches`, `optimized_batches`, `update`, `new_model`, `parity`, `input_view`; additional helpers in file. | [scripts/benchmark_compute.py](scripts/benchmark_compute.py); [tests/test_compute.py](tests/test_compute.py) |
| [src/hm_recsys/runtime/capacity.py](src/hm_recsys/runtime/capacity.py) | Explicit model/feature working-set and disk estimates; reject resource excess without sampling full. Interfaces: `check`, `feature_capacity`. | [src/hm_recsys/data/streaming.py](src/hm_recsys/data/streaming.py); [src/hm_recsys/ranking/pipeline.py](src/hm_recsys/ranking/pipeline.py); 6 other static importers |
| [src/hm_recsys/runtime/checkpoints.py](src/hm_recsys/runtime/checkpoints.py) | Complete-state atomic local saves, verified remote publication and last-valid recovery. Interfaces: `save_checkpoint`, `read_previous`, `load_checkpoint`. | [src/hm_recsys/retrieval/two_tower/training.py](src/hm_recsys/retrieval/two_tower/training.py); [src/hm_recsys/runtime/benchmark.py](src/hm_recsys/runtime/benchmark.py); 3 other static importers |
| [src/hm_recsys/runtime/cli.py](src/hm_recsys/runtime/cli.py) | Validate actual plan/staging/stage/status/resume arguments and dispatch the canonical runtime. Interfaces: `main`. | [scripts/run_pipeline.py](scripts/run_pipeline.py) |
| [src/hm_recsys/runtime/device.py](src/hm_recsys/runtime/device.py) | Bounded real native-library/device probe; explicit CUDA failures remain visible. Interfaces: `probe`. | [src/hm_recsys/runtime/stages.py](src/hm_recsys/runtime/stages.py); [tests/test_modular_backend.py](tests/test_modular_backend.py) |
| [src/hm_recsys/runtime/files.py](src/hm_recsys/runtime/files.py) | Safe relative destinations, atomic JSON, input-whitelist staging and verified resumable copies/digests. Interfaces: `canonical_hash`, `atomic_json`, `relative_path`, `safe_destination`, `require_space`, `file_identity`; additional helpers in file. | [scripts/probe_device.py](scripts/probe_device.py); [src/hm_recsys/data/history.py](src/hm_recsys/data/history.py); 28 other static importers |
| [src/hm_recsys/runtime/git_checkout.py](src/hm_recsys/runtime/git_checkout.py) | Validate GitHub URL/ref/SHA, safe clean pinned checkout and ephemeral private-token askpass. Interfaces: `repository_url`, `validate_ref`, `validate_commit`, `git_environment`, `inspect_checkout`, `checkout`; additional helpers in file. | [src/hm_recsys/runtime/cli.py](src/hm_recsys/runtime/cli.py); [src/hm_recsys/runtime/runner.py](src/hm_recsys/runtime/runner.py); 1 other static importers |
| [src/hm_recsys/runtime/git_safety.py](src/hm_recsys/runtime/git_safety.py) | Read-only filesystem/index/history privacy audit; detect private paths, content and populated outputs. Interfaces: `path_indicators`, `content_indicators`, `audit_repository`. | [scripts/audit_git.py](scripts/audit_git.py); [tests/test_git_safety.py](tests/test_git_safety.py) |
| [src/hm_recsys/runtime/local_cli.py](src/hm_recsys/runtime/local_cli.py) | Guarded local preflight and explicit bounded synthetic/real smoke through the same production stage graph. Interfaces: `main`. | [scripts/local_smoke.py](scripts/local_smoke.py) |
| [src/hm_recsys/runtime/modular.py](src/hm_recsys/runtime/modular.py) | Independent named-stage prerequisites, reports, identity-checked reuse and locked final-Test authorization. Interfaces: `prerequisites`, `resolved_profile`, `safe_report`, `stage_summary`, `Session`, `run_all`; additional helpers in file. | [src/hm_recsys/runtime/cli.py](src/hm_recsys/runtime/cli.py); [src/hm_recsys/runtime/local_cli.py](src/hm_recsys/runtime/local_cli.py); 7 other static importers |
| [src/hm_recsys/runtime/notebook.py](src/hm_recsys/runtime/notebook.py) | No-shell subprocess execution with genuine live event/text streaming, privacy redaction and failure cleanup. Interfaces: `safe_text`, `command`. | [tests/test_progress.py](tests/test_progress.py) |
| [src/hm_recsys/runtime/profiles.py](src/hm_recsys/runtime/profiles.py) | Resolve explicit smoke/full configuration, legal stage plans, device/runtime identities and execution gates. Interfaces: `stages_for`, `load_profile`, `is_colab_runtime`, `execution_gate`, `runtime_identity`, `resource_plan`; additional helpers in file. | [src/hm_recsys/runtime/benchmark.py](src/hm_recsys/runtime/benchmark.py); [src/hm_recsys/runtime/cli.py](src/hm_recsys/runtime/cli.py); 13 other static importers |
| [src/hm_recsys/runtime/resources.py](src/hm_recsys/runtime/resources.py) | Bounded child process trees, RSS/available-memory guards, owned-process cleanup and event forwarding. Interfaces: `constrained_env`, `memory_violation`, `tree_rss`, `stop_owned`, `run_stage`. | [scripts/probe_device.py](scripts/probe_device.py); [src/hm_recsys/runtime/benchmark.py](src/hm_recsys/runtime/benchmark.py); 4 other static importers |
| [src/hm_recsys/runtime/review.py](src/hm_recsys/runtime/review.py) | Produce verified aggregate review material from completed run artifacts without retraining. Interfaces: `publish_review`. | Named runtime/CLI dispatch or direct tests |
| [src/hm_recsys/runtime/runner.py](src/hm_recsys/runtime/runner.py) | Resource-supervised isolated worker execution, dependency/source identities and verified stage publication. Interfaces: `code_identity`, `execution_identity`, `execute`. | [src/hm_recsys/runtime/benchmark.py](src/hm_recsys/runtime/benchmark.py); [src/hm_recsys/runtime/modular.py](src/hm_recsys/runtime/modular.py); 5 other static importers |
| [src/hm_recsys/runtime/settings.py](src/hm_recsys/runtime/settings.py) | Recorded reference/optimized settings and separate local/remote checkpoint cadence. Interfaces: `apply`, `Cadence`. | [src/hm_recsys/retrieval/two_tower/training.py](src/hm_recsys/retrieval/two_tower/training.py); [src/hm_recsys/runtime/modular.py](src/hm_recsys/runtime/modular.py); 3 other static importers |
| [src/hm_recsys/runtime/stages.py](src/hm_recsys/runtime/stages.py) | Stage implementation dispatch/validation around canonical data/model/feature modules. Interfaces: `cutoff_for`, `run_stage`, `validate`. | [src/hm_recsys/runtime/modular.py](src/hm_recsys/runtime/modular.py); [src/hm_recsys/runtime/worker.py](src/hm_recsys/runtime/worker.py) |
| [src/hm_recsys/runtime/state.py](src/hm_recsys/runtime/state.py) | Single-writer leases and checksum-bound stage completion/dependency/restore state. Interfaces: `writer_lease`, `StageStore`. | [src/hm_recsys/runtime/benchmark.py](src/hm_recsys/runtime/benchmark.py); [src/hm_recsys/runtime/modular.py](src/hm_recsys/runtime/modular.py); 3 other static importers |
| [src/hm_recsys/runtime/testing.py](src/hm_recsys/runtime/testing.py) | Byte-authenticated tiny generated fixtures and bounded real-model stage integration helpers. Interfaces: `create_source`, `assert_fixture`, `fixture_profile`, `execute_fixture`, `create_full_source`, `execute_full_fixture`. | [src/hm_recsys/runtime/benchmark.py](src/hm_recsys/runtime/benchmark.py); [src/hm_recsys/runtime/local_cli.py](src/hm_recsys/runtime/local_cli.py); 9 other static importers |
| [src/hm_recsys/runtime/worker.py](src/hm_recsys/runtime/worker.py) | Fresh-process numerical stage entrypoint with validated runtime inputs. Interfaces: `main`. | Named runtime/CLI dispatch or direct tests |

### Bounded test modules

| File | Purpose and main interfaces | Primary callers |
|---|---|---|
| [tests/conftest.py](tests/conftest.py) | Bound numerical libraries before test modules import their dependencies. | pytest |
| [tests/test_canonical_data_features.py](tests/test_canonical_data_features.py) | Verify shared canonical data, feature and metric contracts on tiny sources. Interfaces: `test_smoke_sql_keeps_truth_separate_from_candidate_cap`, `test_smoke_row_budget_rejects_complete_cohort`, `test_test_truth_has_an_explicit_gate_and_exact_window`, `test_one_columnar_builder_is_cutoff_safe_and_keeps_schema`; additional helpers in file. | pytest |
| [tests/test_colab_checkpoints.py](tests/test_colab_checkpoints.py) | Validate canonical complete-state checkpoint contents using an actual tower update. Interfaces: `trained_state`, `test_complete_model_optimizer_scheduler_and_rng_restore`. | pytest |
| [tests/test_colab_migration.py](tests/test_colab_migration.py) | Verify input staging, manifest recovery and bounded canonical execution. Interfaces: `test_input_whitelist_reuse_corruption_and_source_integrity`, `test_copy_interruption_checksum_missing_and_space`, `test_stage_failure_resume_fresh_runtime_and_incompatible_identity`, `test_local_completion_survives_failed_persistence`; additional helpers in file. | pytest |
| [tests/test_compute.py](tests/test_compute.py) | Verify bounded canonical inputs, gradients, cache invalidation and publication. Interfaces: `test_cadence_failure_clock_and_overlay`, `test_prepared_inputs_updates_invalidation_and_prefetch`, `test_remote_publication_failure_preserves_last_valid`, `test_cli_contracts_and_guards`; additional helpers in file. | pytest |
| [tests/test_computing_refactor.py](tests/test_computing_refactor.py) | Audit-driven optimized execution with unchanged temporal/model contracts. Interfaces: `test_query_inputs_full_roster_and_corrupt_cache`, `test_batched_wals_and_exact_ties`, `test_itemcf_seed_layout_and_decay_parity`, `test_incremental_shards_and_stat_receipt_invalidation`; additional helpers in file. | pytest |
| [tests/test_documentation.py](tests/test_documentation.py) | Validate useful module documentation and the README's complete active map. Interfaces: `active_python_files`, `test_active_modules_begin_with_specific_docstrings`, `test_readme_maps_every_active_code_and_configuration_file`. | pytest |
| [tests/test_final_test_guard.py](tests/test_final_test_guard.py) | Verify the optional final-stage guard using generated synthetic labels only. Interfaces: `test_synthetic_final_stage_lock_and_no_refitting`. | pytest |
| [tests/test_full_backend.py](tests/test_full_backend.py) | Exercise the canonical full stage graph on generated, bounded inputs only. Interfaces: `test_full_none_population_targets_and_truth`, `test_full_authorization_and_resource_rejection`, `test_full_actual_training_restore_and_complete_validation`, `test_parquet_resume_crosses_row_groups`; additional helpers in file. | pytest |
| [tests/test_full_checkpoints.py](tests/test_full_checkpoints.py) | Verify complete-batch continuation in the canonical production tower trainer. Interfaces: `test_full_checkpoint_exact_continuation`. | pytest |
| [tests/test_git_checkout.py](tests/test_git_checkout.py) | Offline Git bootstrap, immutable resume and credential-boundary tests. Interfaces: `git`, `local_remote`, `test_local_clone_new_commit_and_exact_resume`, `test_existing_checkout_refusals`; additional helpers in file. | pytest |
| [tests/test_git_safety.py](tests/test_git_safety.py) | Audit publication safely using commits in isolated temporary repositories. Interfaces: `test_clean_repo_and_unique_cli_reports`, `test_tracked_ignore_staged_secret_and_deleted_history`, `test_large_blob_is_metadata_only`, `test_invalid_repository_threshold_and_cli_output`; additional helpers in file. | pytest |
| [tests/test_git_workflow.py](tests/test_git_workflow.py) | Verify pinned Git identities around the real canonical smoke stage graph. Interfaces: `test_git_identity_failed_stage_resume_and_incompatibility`, `test_git_smoke_pipeline_matches_local_reference`. | pytest |
| [tests/test_github_notebook.py](tests/test_github_notebook.py) | Exercise the root Notebook's safe Git bootstrap against owned offline remotes. Interfaces: `git_project`, `test_actual_notebook_bootstrap_local_remote`, `test_notebook_bootstrap_validation_precedes_git`, `test_notebook_private_secret_askpass_is_ephemeral`; additional helpers in file. | pytest |
| [tests/test_modular_backend.py](tests/test_modular_backend.py) | Validate shared SQL retrieval, genuine smoke stages and portable recovery. Interfaces: `canonical_fixture`, `test_global_snapshot_and_bounded_histories`, `test_large_exact_index_and_cached_wals`, `test_modular_staged_fixture_restore_and_empty_validation`; additional helpers in file. | pytest |
| [tests/test_modular_notebook.py](tests/test_modular_notebook.py) | Validate the root Colab client's explicit stage and final-Test boundaries. Interfaces: `test_root_notebook_structure_and_safe_defaults`, `test_exactly_one_active_notebook`, `test_each_stage_has_one_ordered_dedicated_cell`, `test_stage_cell_executes_only_requested_cli_stage`; additional helpers in file. | pytest |
| [tests/test_modular_retrieval.py](tests/test_modular_retrieval.py) | Check canonical rule stages, native fusion and immutable upstream statistics. Interfaces: `test_native_union_preserves_complete_provenance`, `test_rule_cells_and_itemcf_do_not_mutate_shared_statistics`, `test_columnar_fusion_retains_zero_candidate_queries`. | pytest |
| [tests/test_modular_runtime.py](tests/test_modular_runtime.py) | Stage isolation, dependency identities and default future-label safety. Interfaces: `test_sibling_cells_have_no_hidden_training_dependencies`, `test_independent_stage_publication_and_restore`, `test_default_test_denial_precedes_any_input_read`, `test_only_two_active_profiles_and_null_full_limits`. | pytest |
| [tests/test_pilot_checkpoints.py](tests/test_pilot_checkpoints.py) | Retain checkpoint rejection coverage after removal of the parallel pilot trainer. Interfaces: `test_canonical_checkpoint_rejects_pointer_and_payload_corruption`. | pytest |
| [tests/test_progress.py](tests/test_progress.py) | Real nested-process latency, failure, unknown totals and model parity evidence. Interfaces: `test_nested_worker_events_arrive_before_exit`, `test_unknown_totals_failure_and_bounded_lines`, `test_reader_interrupt_propagates`, `test_reload_expected_rows_cross_batch_boundary`; additional helpers in file. | pytest |
| [tests/test_smoke_contracts.py](tests/test_smoke_contracts.py) | Exercise canonical temporal, numerical, resource and reload contracts. Interfaces: `fixture_data`, `test_ids_splits_and_age_boundaries`, `test_content_mapping_readonly_and_missing`, `test_content_invalid_consumed_vector`; additional helpers in file. | pytest |

### Configuration files

| File | Purpose |
|---|---|
| [configs/colab_full.json](configs/colab_full.json) | Uncapped formal full profile with explicit permission/resource gates. |
| [configs/colab_smoke.json](configs/colab_smoke.json) | Bounded smoke profile for the same canonical algorithms and four folds. |
| [configs/compute_optimized.json](configs/compute_optimized.json) | Recorded input/cache/device/checkpoint performance settings, without changing the learning objective. |
| [configs/smoke.json](configs/smoke.json) | Shared bounded Config defaults used by guarded local fixtures/preflight. |

<!-- END ACTIVE PYTHON INVENTORY -->

## Verification and publication

Required checks are Ruff lint/format, Pyright, pytest, `nbformat.validate` and
explicit compilation of all Notebook code cells. Relevant bounded tests cover
exact date boundaries and future perturbations, ID maps/repeats, six quotas,
WALS confidence/folding-in, multi-positive LogQ, candidate/source uniqueness,
49-column parity, grouped sampling, full-truth metrics, model reload, stage
failure/resume and default-Test isolation.

The [refactor report](docs/archi/code_architecture_refactor_report.md) records
commands, exit statuses, matched before/after fixture results and unrun cloud
checks. Local test success does not establish full-data GPU execution, memory
feasibility, production serving or a structural speedup. Historical full-run
scores are evidence of that historical run, not synthetic-test targets.

Before publication, inspect `git status`, tracked/indexed files and history.
`data/`, `docs/achive/`, credentials, private JSONL/CSV/Parquet, embeddings,
checkpoints and populated Notebook outputs must not be staged. Run the existing
read-only Git audit and resolve findings; do not rely on ignore rules alone.
A contaminated history needs a separate deliberate remediation decision.
No automatic commit, push, history rewrite or forced archive add is performed.

`docs/achive/` is local-only historical material, ignored and unmaintained.
It is excluded from active tests/lint/type traversal and cannot be a runtime
import, code bootstrap or required documentation dependency. Moving old source
there preserves bytes after dependency/privacy review; it does not move real
customer inputs or active models out of their data ownership.

## Design and implementation evidence

- [AGENTS.md](AGENTS.md): engineering, privacy, Notebook and resource rules.
- [Proposal](docs/proposal.md): approved algorithms, inputs and defaults.
- [Evaluation protocol](docs/archi/evaluation_protocol.md): exact dates and metrics.
- [Implementation decisions](docs/archi/implementation_decisions.md): decision provenance.
- [Architecture](docs/archi/architecture.md): canonical packages and stage/data flow.
- [Colab runbook](docs/archi/github_colab_runbook.md): pinned checkout and per-cell operation.
- [Migration map](docs/archi/refactor_migration_map.md): file moves, rationale and hashes.
- [Implementation report](docs/archi/code_architecture_refactor_report.md): actual verification and limitations.
