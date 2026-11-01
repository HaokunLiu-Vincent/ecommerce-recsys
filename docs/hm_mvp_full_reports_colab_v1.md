# H&M Personalized Fashion RecSys — Colab MVP v1.0 Full Run Report

**Project:** `e_commerce_recsys`  
**Report date:** 2026-11-01  
**Status:** First full, end-to-end offline Colab run completed; model performance remains a research baseline  
**Protocol:** Evaluation Protocol v1.0  
**Artifact version:** `hm-mvp-1.0.0`  
**Execution mode:** `full`  
**Report basis:** Archived Colab full-run manifests, metrics and performance logs (`hm_mvp_full_reports.zip`), the Phase 0 EDA report and the execution logs shared during this run.

> **Executive conclusion:** The five-notebook, six-retriever pipeline completed full-scale retrieval, candidate fusion, learning-to-rank, independent validation and a frozen native Test evaluation. Test macro **NDCG@20 = 0.036211**, **MAP@12 = 0.023663**, and final candidate-pool **Recall@500 = 0.183866**. LambdaMART beats RRF on the defined benchmark, but absolute purchase-relevance coverage is weak. The run establishes an engineering baseline, **not** a deployment-quality recommendation claim.

## 1. Scope and goals of this run

This was an **end-to-end feasibility and debugging run**, with the immediate goals of checking reproducibility, temporal contracts, six retrieval lanes, ranker training and evaluation, artifacts, memory usage, and full-scale execution characteristics. It was **not** a systematic model-optimization study.

The first full run used these five Google Colab notebooks:

| Order | Notebook | Main responsibility | Full-run status |
|---|---|---|---|
| 00 | `00_Data_Preparation.ipynb` | Canonical Parquet, fixed eligible catalog, anchor/label protocol | Completed |
| 01 | `01_CPU_Retrievers.ipynb` | Popularity, ItemCF, implicit ALS, CLIP image retrieval | Completed; 60 CPU candidate stages separately verified in Colab |
| 02 | `02_GPU_Retrievers.ipynb` | Two-Tower training/retrieval and text encoding/retrieval | Completed; 30 candidate anchor outputs in performance log |
| 03 | `03_Fusion_Ranking.ipynb` | SQL RRF, ranking features, LR/LambdaMART, Validation selection | Completed; LambdaMART selected |
| 04 | `04_Offline_Evaluation.ipynb` | Independent macro metrics, Validation confirmation, frozen Test | Completed; Test report published |

Persistence root: `/content/drive/MyDrive/projects/e_commerce_recsys/data/` (with model/ranking outputs under `retrievers/`, `candidates/` and `ranking_v2/`). Raw images were not re-encoded during these retrieval runs; the prior CLIP image artifacts were reused.

## 2. Dataset and established evaluation contract

### 2.1 Data available from Phase 0

| Item | Observed data |
|---|---:|
| Purchase lines | 31,788,324 |
| Customer catalog records | 1,371,980 |
| Article catalog records | 105,542 |
| Distinct customer–article interactions | 27,306,439 |
| Transaction dates | 2018-09-20 through 2020-09-22 |
| Extracted CLIP image embeddings | 105,100 × 512, FP32; no failures in the extraction run |
| Fixed **C0** eligible article catalog in v1.0 | **98,085** |

C0 is the intersection of articles found in `articles.csv` and articles purchased **before 2020-07-28**. This is intentionally a **warm-item evaluation**, not a historical inventory-availability model. In the image retriever, **97,649** C0 articles have available image vectors (per its model manifest). The text retriever encoded all **98,085** C0 articles using `sentence-transformers/all-MiniLM-L6-v2` with the fields `prod_name`, `product_type_name`, `product_group_name`, `colour_group_name`, and `detail_desc`.

### 2.2 Chronology

| Period | Date interval | Definition |
|---|---|---|
| Retrieval model fit | `t_dat < 2020-07-28` | Frozen model weights, graph statistics and base indexes |
| Ranking training | 2020-07-28–2020-08-15 | Five 7-day-label anchors |
| Validation | 2020-08-16–2020-09-03 | Five anchors; native end-to-end NDCG@20 for selection |
| Test | 2020-09-04–2020-09-22 | Five anchors; frozen selection, native candidates only |

Within each 19-day block, anchors use offsets **0, 3, 6, 9, 12**. Future labels are **unique eligible purchased `article_id` values** in `[anchor_date, anchor_date + 7 days)`; historical query features use strictly `t_dat < anchor_date`. Five windows in a block overlap; the same future purchase can be relevant to more than one query. Retrievers are **not refitted** after the first cutoff, although the observable user history can advance across anchors.

| Protocol query count | Ranking train | Validation | Test |
|---|---:|---:|---:|
| Future-purchase-active user–anchor queries before C0 filtering | 393,443 | 373,663 | 358,945 |
| Retained eligible user–anchor queries | **385,312** | **319,112** | **260,755** |
| Excluded after C0 filtering | 8,131 | 54,551 | 98,190 |
| Query exclusion rate | 2.07% | 14.60% | 27.36% |

Totals: **965,179** retained eligible user–anchor queries across all 15 anchors. These are **queries**, not necessarily distinct customers. Eligible evaluation queries are conditioned on **at least one eligible purchase in the future horizon**. Therefore these offline metrics are not an all-customer conversion measure or an estimate of online business uplift.

## 3. Retrieval implementation and full-run output

Six implemented lanes:

| ID | Retrieval lane | Key signal / implementation |
|---|---|---|
| R1 | Popularity / Trending | Pre-cutoff item frequency/recency ranking and fallback |
| R2 | ItemCF | Sparse item co-purchase neighborhood |
| R3 | Implicit ALS | Frozen 64-factor implicit-feedback user/item representations; 15 iterations, recorded confidence weighting |
| R4 | Two-Tower | Trained user/item towers with 64-dimensional item vectors; frozen model and FAISS item index |
| R5 | Text | Frozen pretrained 384-dimensional MiniLM item embeddings and FAISS |
| R6 | Image | Previously extracted 512-dimensional CLIP vectors and FAISS, mapped to C0 |

Each lane emitted **up to 100** distinct eligible articles per query. Source ranks and raw scores were retained for deterministic RRF, with `k0=60`; the merged native pool was capped at **M=500**. Previously purchased articles were not globally excluded. Missing-history users received a documented fallback.

The aggregated six-lane candidate count was **578,468,232 rows**, including **385,546,808** CPU-lane and **192,921,424** GPU-notebook-lane rows. This is the raw per-channel candidate total, **not** the number of distinct user–article pairs.

| Channel | Persisted candidate rows across 15 anchors |
|---|---:|
| popularity | 96,517,900 |
| itemcf | 95,993,964 |
| als | 96,517,900 |
| two_tower | 96,517,900 |
| text | 96,403,524 |
| image | 96,517,044 |

### 3.1 Per-lane macro Recall

| Lane | Validation Recall@100 | Test Recall@100 | Test signed incremental Recall@500 |
|---|---:|---:|---:|
| popularity | 6.196% | 6.008% | +1.764 pp |
| itemcf | 6.168% | 6.042% | +1.443 pp |
| als | 7.324% | 8.106% | +2.688 pp |
| two_tower | 6.944% | 6.644% | +1.336 pp |
| text | 3.267% | 3.641% | -0.121 pp |
| image | 2.830% | 3.288% | +0.137 pp |

The signed incremental number means **Recall(all six, M=500) − Recall(all except this lane, M=500)**, with RRF and Top-500 **recomputed** after lane removal. It is not a measure of unique hits that can be added across lanes. In this experiment, **ALS** has the strongest single-channel Recall@100 and the largest signed incremental contribution. **Text** has a slightly *negative* incremental contribution under this specific RRF/Top-500 policy; that does **not** prove the text encoder has no information or that removing it always helps.

### 3.2 Fusion coverage

| Metric | Validation | Test |
|---|---:|---:|
| Macro Union Recall@500 | 17.985% | 18.387% |
| Mean pre-budget unique candidates per query | 498.03 | 494.86 |
| Mean post-budget native candidates per query | 467.09 | 463.84 |

At Test, **77,189 / 260,755** eligible queries have at least one retrieved relevant article; **183,566 (70.40%)** have **none**. This is the strongest observed explanation for why final Top-20 performance remains low. A ranker cannot recover articles absent from its native candidate set.

## 4. Ranking setup, Validation and frozen Test results

### 4.1 Ranking training

- Training sample: **19,451** deterministically selected user–anchor queries out of 385,312 eligible Ranking Training queries (~5.05%).
- Training candidate rows: **9,193,142**.
- Positive training rows: **65,529**.
- Missing positive articles injected into the training pool: **54,517**; original native positive rows: **11,012**, by subtraction.
- **83.20% of positive training rows came from injection.**
- Models: pointwise **Logistic Regression** baseline and **LightGBM LambdaMART** (`lambdarank`); no DeepFM, LightGCN or separate pre-ranker.
- Training candidate features (schema `ranking-1.0`): RRF; six lane-specific present/rank/score triples; user purchase count and recency; item purchase count; user–item purchase count. These are computed as of each anchor.

**Important methodological risk:** Injected positives are, by construction, missing native-retrieval provenance; all six source-presence flags are zero, with missing RRF/rank/score signals. Even when an explicit `was_injected` column is excluded from model inputs, that pattern may serve as an **injection identity proxy**. Since injected rows in training are positive, this can create a shortcut and training/serving distribution mismatch. This is a **risk inferred from the construction and feature schema, not a proven root cause** of the model's performance. The accepted v1.0 protocol explicitly permits train-only injection.

### 4.2 Native end-to-end ranking (macro over eligible queries)

| Pipeline | Validation NDCG@20 | Test NDCG@20 | Test Recall@20 | Test MAP@12 |
|---|---:|---:|---:|---:|
| Popularity only | 0.010914 | 0.010661 | 0.020755 | 0.004766 |
| RRF only | 0.026701 | 0.027592 | 0.051146 | 0.015601 |
| Logistic regression | 0.005698 | 0.005557 | 0.010695 | 0.002668 |
| LambdaMART | 0.034543 | 0.036211 | 0.058985 | 0.023663 |

The primary v1.0 model-selection metric was **native end-to-end macro NDCG@20**. **LambdaMART** was selected on Validation and frozen before Test.

- Validation LambdaMART NDCG@20: **0.034543** vs RRF **0.026701** (**+29.37% relative**).
- Test LambdaMART NDCG@20: **0.036211** vs RRF **0.027592** (**+31.24% relative**).
- Test LambdaMART MAP@12: **0.023663**.
- Test LambdaMART Recall@20: **0.058985**.

The Validation-to-Test scores are broadly similar under the **same v1.0 protocol**, but this is only one historical held-out period. It does not establish general online performance.

### 4.3 Distinct evaluation views

| View | Interpretation | Example outcome |
|---|---|---|
| **Native end-to-end** | Ranks actually retrieved articles; missed positives count as zero. **Primary model-selection view.** | Test LambdaMART NDCG@20 **0.036211** |
| **Conditional native** | NDCG against only the *retrieved relevant items*, excluding queries with zero retrieved positives. **Not end-to-end.** | Test LambdaMART conditional NDCG@20 **0.154361**, from 77,189 queries |
| **Injected Validation diagnostic** | Adds missing positives with oracle knowledge; **never an end-to-end result or selection criterion**. | Validation diagnostic LambdaMART NDCG@20 **0.873086** |

The injected diagnostic's large value is not evidence of high-quality serving recommendations, particularly given the injection-proxy risk noted above. **Test used native candidates only**; its evaluation manifest states `no_test_injection=true` and `no_test_refit=true`.

## 5. Engineering performance and resource observations

### 5.1 Full retrieval jobs

| Measured component | Full-run duration | Main observation |
|---|---:|---|
| CPU Retriever notebook total | **145.82 min** (2.43 h) | Includes extensive validation of older candidate shards |
| CPU Image candidate compute | **84.86 min** | Exact CPU FAISS / image retrieval is the dominant fresh-compute cost |
| CPU ALS candidate compute | **27.64 min** | Frozen factors and cache still incur retrieval overhead |
| CPU ItemCF fresh-compute (14 anchors) | **2.65 min** | Sparse precomputed neighbors enable much faster queries |
| Verification of pre-existing completed CPU candidate stages | **26.78 min** | Legacy multi-file manifests incur many small Drive reads/checks |
| GPU Retriever notebook total | **42.35 min** | Two-Tower + Text retrieval, including text encoding |
| Text FAISS exact search | **33.08 min** | Dominant cost of GPU notebook; search executed on CPU FAISS |
| Two-Tower FAISS exact search | **3.70 min** | Much faster than 384D text exact search in this implementation |
| Text item encoding | **52.31 s** | 98,085 item embeddings built successfully |

**Recorded memory:** CPU Retriever peak process RSS **3.94 GiB**; GPU notebook peak process RSS **4.86 GiB**, with recorded PyTorch peak allocated GPU memory **0.54 GiB**. These figures are process-level/log-specific, not total system memory or a universal minimum memory requirement.

The optimized candidate writers usually produced **one larger Parquet shard per anchor**, whereas legacy Popularity candidate outputs retained **3,777** old small shards. A separate user-run CPU data integrity pass checked 60/60 candidate stages, hashing the full file set, and took **25.21 minutes**. The small-file problem had a measurable cost.

### 5.2 Fusion, ranking and evaluation observations

From the Colab execution logs shared for Notebook 03 and 04:

- G4 High-RAM ranking run: **16** configured threads, **32GB** DuckDB limit, ranking feature generation handling roughly **28–31 million native rows per Validation anchor**, with process peak RSS near **35.04 GiB**.
- Ranker training: **9,193,142** rows / **19,451** sampled queries; the recorded model-fit stage took approximately **31.64 s** (not the entire feature-building workflow).
- Independent Full Validation on a smaller runtime: **8** threads, ~**12.30GB** DuckDB limit, approximately **43 min** across the five independent evaluation Anchor timers; peak RSS about **13.20 GiB**.
- Final Native Test: five Test anchors completed; **zero** injected rows in the logged Test feature outputs; native ranking features, predictions and independent evaluation were published. Measured component times for Test feature construction, model scoring and evaluation were much larger than simple CSV serialization.
- A LightGBM warning persisted: `X does not have valid feature names, but LGBMRanker was fitted with feature names`. This is a **feature-schema consistency warning** worth removing; it does not on its own demonstrate incorrect rankings.

**Operational lesson:** Increasing thread counts alone was insufficient. Removing redundant per-query history scans, replacing Python row dictionaries with Arrow/SQL operations, using wider batches and fewer shards, and reusing frozen model work were the highest-value engineering changes in this iteration. Exact CPU FAISS and independent evaluation remain notable bottlenecks.

## 6. Artifact integrity, repeatability and limits of verification

The archived Full Run report includes a frozen selection manifest and `ranking_v2` artifact lineage; the independent Test evaluation manifest records the selected LambdaMART pipeline, and states native-only/no refit. **All five** files listed by each of the **Validation and Final Test** `_COMPLETE.json` manifests were independently checked against their archived byte lengths and SHA-256 digests during preparation of this report; **all checks passed**.

A separate Colab verification reported **60/60 CPU candidate stages passed** with schema, size, SHA-256 and row-count checks. The archive contains many lightweight candidate `_COMPLETE.json` files but **does not contain the large model files, indexes or full candidate Parquet bodies**; therefore this report does not claim an independent end-to-end recomputation of retrieval or ranking metrics from source transactions. Actual Colab runs and packaged manifest statements are distinguished accordingly.

Important artifact identifiers:

| Contract | Identifier |
|---|---|
| Protocol version | `1.0` |
| Artifact version | `hm-mvp-1.0.0` |
| Mode | `full` |
| Catalog ID | `2551c83a6042b94960e47e78e58c3b59e33a1e6013ffabf613e50db6102f8e7b` |
| Ranking feature schema | `ranking-1.0` |
| Candidate budget / RRF constant | `M=500`, `k0=60` |
| Frozen selected pipeline | `lambdamart` |
| Full ranking outputs | `data/ranking_v2/` |

## 7. Main findings, risks and their evidentiary status

| Finding | Status | Evidence / qualification |
|---|---|---|
| The notebook-first pipeline can complete a full historical pass | **Observed** | Five completed notebook stages, full metrics and published Test report |
| LambdaMART improves NDCG over the chosen RRF baseline on v1.0 Test | **Observed** | Native Test NDCG@20 0.036211 vs 0.027592 |
| Retrieval coverage limits end-to-end performance | **Observed** | Test Union Recall@500 18.387%; 183,566 zero-positive-retrieval queries |
| ALS has the strongest observed incremental retrieval contribution | **Observed under RRF/Top-500** | Test signed incremental Recall +2.688 percentage points |
| Text retrieval may displace better candidates under current fusion | **Observed under this fusion** | Signed leave-one-out contribution -0.121 percentage points |
| Ground-truth injection may create a label-proxy shortcut | **Risk / hypothesis** | 83.2% of training positives injected, with absent native-source features |
| Frozen retrieval catalog and model fit may impair fashion freshness | **Hypothesis** | Pre-2020-07-28 frozen fit, later windows and increasing C0 exclusions |
| A single 7-day purchase horizon may be too restrictive for broader business goals | **Research question** | Current evaluation only measures next-seven-day purchases |
| Re-encoding or fine-tuning content models will improve purchasing predictions | **Unproven** | Need matched controlled comparisons; image/text similarity is not purchase propensity |

## 8. Retrospective and recommended follow-up work

### 8.1 Engineering improvements, prioritized

1. **Progress observability:** each notebook/stage should print current anchor, completed/total queries, rows, compute/write/publish seconds, recent throughput, peak RAM/VRAM and an ETA **derived from measured comparable batches**. Persist a small execution summary.
2. **Reduce I/O overhead:** consolidate old tiny Parquet shards in a new verified artifact generation rather than repeatedly hashing thousands of files; preserve immutable original artifacts.
3. **Optimize exact nearest-neighbor search:** profile CPU FAISS settings and, if practical, benchmark batched GPU exact inner-product search with score/tie-order parity checks. Image and Text lanes are the largest recorded compute bottlenecks.
4. **Improve notebook readability:** smaller sections, clear input/output contracts and named data shapes; avoid dense monolithic cells or unnecessary abstraction.
5. **Strengthen ranker training diagnostics:** report native-vs-injected positives, effective training/query counts, missing-source feature patterns, and native-only ranking metrics. Protect against proxies for injection status.
6. **Resource-aware evaluation:** independently profile DuckDB memory/spill and wall-clock time of macro metrics and leave-one-out fusion; do not assume a small RAM runtime will match G4 performance.

### 8.2 Model research: proposed, **not executed in v1.0**

- **Revisit the purchase horizon.** The prior protocol answers a precise 7-day forecasting task; it is not necessarily the best horizon for longer-cycle fashion purchases. Consider a **single fixed 30-day Research Validation** as a new, explicitly versioned evaluation design. Changing the label horizon alters the task and its denominators; scores must not be compared as if they came from the same benchmark.
- **Investigate one-anchor purchase evaluation.** Define the user population using prior history and report the observed future purchase-active fraction to avoid conflating purchase incidence with recommendation quality.
- **Audit native vs injected ranker training.** Compare a controlled native-only or mitigated-injection training design against the current compromise **using development-only data**, not repeated tuning on the already observed Test.
- **Audit time freshness and the C0 exclusions.** Before changing temporal or catalog policies, measure how many future positives and queries are removed and how much this limits apparent recall.
- **Rebuild content representations as a separate study.** Compare frozen pretrained Image/Text embeddings against domain-/purchase-aware fine-tuned versions with identical item coverage, data splits and retrieval budgets. Version new embeddings; do not overwrite CLIP 512D or MiniLM 384D baseline artifacts.
- **Retain simple purchase baselines.** Historical recency/popularity and repeat-purchase strategies should be evaluated alongside learned models rather than presumed inferior.

**Research discipline:** The previously held-out v1.0 Test has now been evaluated and inspected. Subsequent experimentation must not treat the same Test as a pristine unbiased final holdout. A new monthly task should have a separately documented Validation protocol and, when feasible, newly acquired or independently held-out future data for eventual final testing.

## 9. Open limitations and interpretation

- The 7-day, stride-3 overlapping labels create correlated user–anchor observations; macro averaging alone does not make queries independent.
- This benchmark excludes all future-positive articles outside C0 and then excludes queries with no remaining eligible positive; **Test excludes 98,190 of 358,945 future-active queries (27.36%)**, versus 14.60% in Validation.
- Frozen pre-cutoff collaborative representations may be stale by late Test. Static metadata/image files are not guaranteed to be point-in-time authentic.
- Unpurchased native candidates are proxy zero labels; true impressions, stock availability and non-purchase intentions are unknown.
- Injection-based training and Validation diagnostics are oracle-assisted; only Native Validation/Test scores describe the full candidate-to-rank funnel.
- The observed low ranking metrics do not uniquely identify the cause. Horizon length, catalog constraints, temporal freshness, representation mismatch and injection artifacts are competing explanations needing separate tests.
- No production serving latency, online personalization quality, revenue lift, or cross-domain transfer is established by this run.

## 10. Evidence index

Files used from `hm_mvp_full_reports.zip` (relative to the Drive `data/` root):

- `protocol/mvp_v1/full/protocol_manifest.json` — fixed dates, 15 anchors, C0 and cohort diagnostics.
- `ranking_v2/evaluation/validation_report/mvp_v1/full/validation_ranking_metrics.csv` — Full Validation ranking and diagnostic views.
- `ranking_v2/evaluation/validation_report/mvp_v1/full/retrieval_metrics.csv` and `fusion_metrics.csv` — Full Validation retrieval/fusion metrics.
- `ranking_v2/evaluation/final_test_report/mvp_v1/full/final_test_metrics.csv` — Full native Test metrics.
- `ranking_v2/evaluation/final_test_report/mvp_v1/full/retrieval_metrics.csv` and `fusion_metrics.csv` — Test lane and signed ablation metrics.
- `ranking_v2/rankers/models/mvp_v1/full/training_summary.json` and `feature_schema.json` — ranker training budget/labels and features.
- `ranking_v2/rankers/selection/mvp_v1/full/selection.json` — frozen Validation-based model selection and retrieval contracts.
- `ranking_v2/evaluation/*/mvp_v1/full/_COMPLETE.json` — report integrity manifests.
- `retrievers/cpu_performance/.../performance.json`, `retrievers/gpu_performance/.../performance.json` — measured retrieval runtime/throughput.
- Relevant full-mode model `_COMPLETE.json` files — Two-Tower/Text/Image/ALS configs and persisted artifact identities.

**Bottom line:** This run is a valid **MVP v1.0 full-scale engineering and offline benchmark baseline**. It shows a real LambdaMART uplift over RRF on the defined seven-day warm-catalog task, but low end-to-end relevance coverage and several methodological constraints. Preserve the run unmodified, then evolve engineering observability and the next research protocol deliberately rather than attributing low scores to any single untested cause.
