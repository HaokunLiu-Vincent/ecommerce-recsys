# Full Colab Computing Optimization v1: Training Audit and Technical Report

**Run:** `colab_full_computing_opt_v1`  
**Audit scope:** `docs/achive/colab_full_computing_opt_v1/`  
**Architecture:** proposal v0.3.2, Evaluation Protocol v2.0  
**Method:** immutable-artifact inspection, complete saved-Validation verification,
read-only SQL, bounded numerical checks, and historical-source review. No
training pipeline, notebook, configuration, model or input was modified.

## Contents

1. [Executive summary](#1-executive-summary)
2. [Experiment identity and configuration](#2-experiment-identity-and-configuration)
3. [Artifact inventory and integrity](#3-artifact-inventory-and-integrity)
4. [Pipeline execution status](#4-pipeline-execution-status)
5. [Dataset and temporal protocol verification](#5-dataset-and-temporal-protocol-verification)
6. [Retrieval performance](#6-retrieval-performance)
7. [Candidate fusion analysis](#7-candidate-fusion-analysis)
8. [Ranking training results](#8-ranking-training-results)
9. [Offline evaluation results](#9-offline-evaluation-results)
10. [Computational performance and resources](#10-computational-performance-and-resources)
11. [Correctness audit and risks](#11-correctness-audit-and-risks)
12. [Engineering and maintainability](#12-engineering-and-maintainability)
13. [Limitations and missing evidence](#13-limitations-and-missing-evidence)
14. [Prioritized recommendations](#14-prioritized-recommendations)
15. [Final assessment](#15-final-assessment)
16. [Appendix: evidence and reproduction](#16-appendix-evidence-and-reproduction)

Evidence labels used throughout:

- **Observed:** read directly from an artifact, without assuming its claim is true.
- **Independently verified:** recomputed or checked against retained underlying data.
- **Derived:** arithmetic or a new audit diagnostic using identified artifacts.
- **Estimated:** a resource/planning estimate, rather than a measurement.
- **Unverified:** evidence is absent, incomplete or outside this audit's bounds.

Links such as [run summary][R] and [artifact map][A] identify actual files. A
logical name such as `models/wals.json` means the corresponding entry in [A];
its immutable physical path includes a stage attempt identifier. Section 16
lists the exact paths used for the main checks. Decimal metric values lie in
[0,1]; they are recommendation metrics, not purchase or click probabilities.

## 1. Executive summary

The evidence supports a completed, real-data-sized **full training and complete
Validation workload**, including genuine WALS, multi-modal Two-Tower and one
LambdaMART, followed by saved reload verification. It does **not** establish a
complete, directly restorable archival export or production readiness.

| Main result | Evidence classification and qualification |
|---|---|
| Native Validation MAP@12 **0.023638417433** | Independently verified over all 72,019 saved queries and 227,910 unique ground-truth pairs; agrees with reported output within floating-point rounding. [CQ][LG][PR][R] |
| Native Recall@12 **0.047971186689**, NDCG@12 **0.034834373313** | Independently verified with full ground truth and zero-hit queries retained. [LG][PR] |
| Stored-RRF-feature baseline MAP@12 **0.017888745342** | Derived audit baseline on the identical native candidate pool, using stored float32 RRF and article-ID ties. LambdaMART improves MAP by **0.005749672092**, or **32.14% relative**, on this Validation set. This is not a significance or Test claim. [FM][LG] |
| Native union Recall **0.166374395434** | Independently verified from all reconstructed Validation candidates, with no future-positive injection. [FM][LG] |
| **26 successful worker-stage records**, **18,644.056 s** total | Observed records and derived deduplicated worker time: **5 h 10 m 44 s**. This excludes incompletely timed setup, staging and publication; it is not proven end-to-end elapsed time. [V] (`resources.stages`) |
| Largest sampled process-tree RSS **3.992 GiB**; tower CUDA allocation **484.881 MiB** | Observed sampled/recorded measurements, not system RAM usage or GPU utilization. [V][TP] |
| **21 present stage manifests**, **4,702 declared outputs** pass integrity checks | Independently verified manifest self-hashes, output sizes/SHA256 and common run identity. Five original candidate/feature stage directories are missing from this copy. [A][S] |
| **23,232 logical artifact-map entries unavailable locally** | Independently counted. The export cannot be treated as complete merely because `status.complete=true`. [A][S] |

The main measured costs are feature-stage work, **53.36%** of deduplicated
worker time, and Two-Tower, **26.37%**. Within Two-Tower, internal validation
consumes **3,199.038 s**, much more than catalog encoding. Low VRAM occupancy
alone does not establish poor GPU utilization. There is no matched reference
full run here, so no optimization speedup is established. [V][TP]

No leakage was demonstrated by the checks performed. All five saved label
populations exactly match the retained canonical transactions; all selected
Two-Tower target dates and Q frequencies pass checks. Four past-only features
were independently replayed for 32 Validation queries. These results support
point-in-time correctness, but do not certify every original feature value or
raw-source lineage. [SP][CQ][LG][FM][TI]

A separate publication risk requires attention: this directory contains
customer-level histories, labels and group records. Some JSONL records are
currently untracked **and not ignored**. The aggregate review JSON's privacy
statement applies to that diagnostic object, not to the entire directory.
Do not publish the archive wholesale. No publication occurred in this audit.

## 2. Experiment identity and configuration

### 2.1 Authority and the legacy checklist

The requested `docs/evaluation_protocol.md` does not exist. The newer explicitly
approved [protocol v2.0](archi/evaluation_protocol.md),
[proposal v0.3.2](proposal.md), root [AGENTS.md](../AGENTS.md) and
[implementation decisions](archi/implementation_decisions.md) are authoritative.
They are also byte-identical to their copies in the matching historical source
package, so this is not a retrospective substitution of a newer metric contract.
The historical package's architecture document differs from the current one.

The task's LR, standalone text/image retrieval, positive injection and Top-500
SQL-RRF checklist describes the superseded architecture. This run actually uses
**Trend, Repurchase, Sibling, ItemCF, WALS and Two-Tower**, quotas
**60/30/30/50/50/50**, one LambdaMART, native candidates and Top-12 output.
Those legacy components are marked **not executed by design**, not failed or
silently assumed present. [I] (`profile`)[P]

### 2.2 Source identity and runtime

| Property | Observed value / independent check |
|---|---|
| Run ID | `colab_full_computing_opt_v1`, explicitly present in preflight `colab_stage.json` local/persistent checkpoint paths |
| Run signature | `1dc6eff1b26f6162deeca6ff7d8374a33092bd37b50c6d22413994cce9beabf3` |
| Identity format | Version 1, source-file hashes; no Git commit provenance |
| Matching historical source | `data/releases/hm_recsys_a736135841d66476.zip`, 339,684 bytes, 89 members |
| Source-package SHA256 | `eff8d6b6c38d18138d6b15c574951c4f4361e413137d38cc84fdf8aa61c9ad69`; matches its sidecar |
| Source-file agreement | All **58** archived code/config/lock hashes match package member bytes; no extraction or execution was needed |
| Current workspace comparison | 47 of those 58 files differ; 11 match. Current model source was not substituted for the historical source |
| Run profile | `full`; `is_smoke=false`, `is_benchmark=false`, `synthetic_fixture=false` |
| Host / Python | Colab, Linux x86_64, Python 3.12.16 |
| Accelerator | NVIDIA A100-SXM4-80GB; driver 580.82.07; reported capacity 81,920 MiB |
| Torch | Runtime identity reports distribution version 2.15.0; checkpoint pointer records actual build `2.15.0+cu132` |
| Lock SHA256 | `cfdae9f71dfaad81ea20f6f4d2eb798564f452ad5420802ea5ce339113108a7e` |
| Final Test | Explicitly disabled |

Sources: [I], preflight `colab_stage.json`, [V] (`code_identity`, `runtime`),
[tower metadata][TI] and the [matching source package][SRC]. Package matching
binds the available historical implementation to recorded hashes; it is not
independent attestation of every instruction executed in the original runtime.

Recorded numerical libraries are DuckDB 1.5.6, PyArrow 26.0.0, NumPy 2.5.4,
SciPy 1.18.1, scikit-learn 1.9.1, implicit 0.7.3, FAISS CPU 1.15.1, LightGBM
4.7.0 and psutil 7.2.2. The package's 18-cell Notebook is the older release-based
client; the current 21-cell GitHub client is not evidence of how this run was
started. Both have safe smoke defaults. Actual full execution is established
by the resolved profile and outputs, not by a Notebook's default configuration.
[I][P][SRC]

### 2.3 Effective execution configuration

| Component | Recorded configuration |
|---|---|
| Population/query limits | `training_subset`, user/catalog/transaction/query/target/step/cluster-row caps all null; no smoke cohort or 5,000-item limit |
| Folds | F1, F2, F3, F4 and complete Validation |
| Native recall | Six fixed channel quotas; no extra RRF truncation |
| WALS | 64 factors, 15 iterations, regularization 0.05, full-confidence alpha 20, library alpha 1.0; unnormalized factors |
| Two-Tower | Batch 512, 10 epochs, output 128, FP32 without AMP; trainable 384→32 text and 512→32 image projections |
| Two-Tower optimizer | Matching source: AdamW, learning rate 0.001, weight decay 0.0001; cosine scheduler; existing multi-positive LogQ loss, temperature 0.07 |
| Sampling/order | Fixed SHA256 event order, repeated per epoch; Q from selected training events; raw repeated purchase history retained |
| Feature preparation | 32-query batches; 49-column shared schema; frozen preprocessing |
| Ranker sampling | Train only; retain native positives; up to 20 negatives/positive, half hard by RRF and remainder query-local seeded random |
| LambdaMART | Up to 1,500 rounds; learning rate 0.03; 63 leaves; depth 6; minimum leaf 100; L1=1, L2=5; column/row subsample 0.8; seed 42; deterministic, two threads |
| Optimization backend | `optimized`, raw item placement `auto`→actual device; prefetch 2 batches |
| Checkpoints | Effective local cadence 100 steps or 60 s; remote cadence 300 s, plus explicit boundary publications |

Sources: [I] (`profile`), [TI], [WS], [M] and matching ZIP members
`pilot/towers.py`, `towers.py`, `pilot/ranking.py`, `ranking.py`.

Batch 512 differs from the proposal's 2,048 example/default and changes the
in-batch negative population. The profile explicitly records this difference;
it is not equivalent to gradient accumulation at 2,048. The old
`checkpoint_every_steps=10` field remains alongside the effective performance
cadence; actual save counts below avoid inferring behavior from that legacy
field alone. [I] (`profile.batch_change_note`, `performance`)

## 3. Artifact inventory and integrity

### 3.1 Physical inventory

**Independently counted:** **4,740 files**, **4,309,416,991 bytes**
(**4.013 GiB**). This is the size of the supplied copy, not peak training disk
usage. Formats: `no extension`: 9; `.duckdb`: 6; `.faiss`: 2; `.json`: 103; `.jsonl`: 2; `.log`: 42; `.npy`: 2,273; `.parquet`: 2,299; `.pt`: 3; `.txt`: 1.

| Physical stage directory | Files | Bytes | MiB |
|---|---:|---:|---:|
| `Root and auxiliary metadata` | 12 | 5,386,730 | 5.137 |
| `canonical` | 33 | 1,144,518,512 | 1,091.498 |
| `capacity` | 7 | 5,336 | 0.005 |
| `content` | 10 | 380,384,795 | 362.763 |
| `device_probe` | 6 | 605,113 | 0.577 |
| `encode` | 7 | 190,447,286 | 181.625 |
| `index_towers` | 7 | 109,983,124 | 104.888 |
| `preflight` | 12 | 39,565 | 0.038 |
| `prepare_towers` | 25 | 869,132,737 | 828.870 |
| `preprocessing` | 8 | 2,480,009 | 2.365 |
| `ranker` | 2,265 | 112,208,962 | 107.011 |
| `reload` | 7 | 9,597,680 | 9.153 |
| `reload_encode` | 8 | 36,973,293 | 35.260 |
| `reload_features` | 2,261 | 704,678,527 | 672.034 |
| `rosters` | 19 | 30,408,425 | 29.000 |
| `stats_F1` | 6 | 60,315,478 | 57.521 |
| `stats_F2` | 6 | 58,743,996 | 56.023 |
| `stats_F3` | 6 | 57,696,801 | 55.024 |
| `stats_F4` | 6 | 56,649,607 | 54.025 |
| `stats_Validation` | 6 | 59,010,349 | 56.277 |
| `towers` | 12 | 180,104,319 | 171.761 |
| `wals` | 11 | 240,046,347 | 228.926 |

The inventory includes nine `.DS_Store` files. They are packaging clutter,
not model evidence. Root and auxiliary metadata include the final summaries,
identity, artifact map and the independent checkpoint pointer.

| Artifact family / names | Actual format and scope |
|---|---|
| `identity.json`, `status.json`, `artifacts.json`, `run_summary.json`, `full_review.json`, `resources.resolved.json`, `review_manifest.json` | JSON identities, summaries, resource policy and export references |
| Canonical `history.duckdb`, monthly `.../data_0.parquet`, `cohort.parquet` | Six DuckDB files exist overall: canonical history plus five cutoff snapshots; canonical has 25 monthly transaction Parquet files and its cohort table |
| `queries/{F1,F2,F3,F4,Validation}.{json,parquet}` and corresponding `labels/*.parquet` | Five complete independent query/label populations |
| `prepared/tower/{contexts,ordered,internal-2020-08-05,...,internal-2020-08-11}.parquet` | Raw input/context caches and ordered training events |
| `prepared/tower/{histories,weights,profiles,target_context,target_item,positive_offsets,positive_items,item_codes}.npy` | Memory-mappable model-independent inputs; no learned vectors cached as training inputs |
| `models/wals.json`, `models/wals_user_factors.npy`, `models/wals_users.parquet`, `indexes/wals.{json,npy,faiss}` | Frozen WALS factors and mappings |
| `models/twotower.{json,pt}`, `models/twotower_items.{json,npy}`, `indexes/twotower.{json,npy,faiss}` | Learned tower model, catalog vectors and separate learned index |
| `checkpoints/step-00071480-e10-selected.pt` inside the tower stage | 97,157,103-byte complete-state checkpoint; exported model weights are 23,808,305 bytes |
| `models/lambdamart.txt` | 1,448,797-byte text model; 197 trees |
| Ranker `predictions/reference-00000.npy` through `reference-02250.npy` | 2,251 reference-score shards |
| Reload `predictions/reloaded_features/part-00000.parquet` through `part-02250.parquet` | 2,251 reconstructed Validation feature shards; manifests/group metadata retained |
| `predictions/reference_top12.parquet`, `predictions/reloaded_top12.parquet` | Complete saved Top-12 lists, independently compared |
| `resource_usage.json`, `tower_performance.json`, `ranker_performance.json`, `feature_performance/Validation_True.json`, `capacity/*.json` | Timings, sampled resources, estimates and diagnostics |
| 42 `.log` files | Stage commands and outcomes; several are cumulative snapshots rather than distinct runs |

Use [A] for full logical names and physical attempt paths. The table groups
sequential shard names instead of printing thousands of near-identical rows.
It does not imply that entries merely listed in [A] exist in this copy.

### 3.2 Manifest and byte integrity

The following checks were independently executed, read-only:

| Check | Result |
|---|---|
| Hash `identity.json` using the historical canonical JSON function | Exactly equals the run signature in [S] |
| Present completion manifests | 21; all self-hashes valid and all bind the same run signature |
| Declared outputs of present manifests | **4,702**, all physically present |
| Output size and streamed SHA256 | **4,702 passed**, zero mismatches; **4,303,206,822 bytes** hashed in 4 MiB buffers |
| Dependencies whose manifests are present | 210 edges, all hashes agree |
| Dependencies whose manifests are absent | 30 references to the five missing feature stages; not independently resolvable |
| Conflicting declarations for the same physical output | Zero |
| Root summary versus ranker `full_results.json` | Shared result fields agree; status strings have different stage/coordinator scope |

Sources: `stages/*/complete.json`, [I], [S], [R], [FR]. These checks establish
integrity relative to the supplied manifests, not authenticity against an
external signed release or raw data source.

### 3.3 Missing, duplicated and non-authoritative artifacts

[A] contains **27,892** logical entries: **4,660 resolve locally**, while
**23,232 do not**. The absent directories are exactly
`stages/features_F1`, `features_F2`, `features_F3`, `features_F4` and
`features_Validation`. Their original candidate/feature shards, completion
manifests, logs and separate performance reports are not in this copy.

| Missing logical family | Missing entries |
|---|---:|
| Original candidates F1 / F2 / F3 / F4 / Validation | 2,223 / 2,253 / 2,509 / 2,371 / 2,252 |
| Original features F1 / F2 / F3 / F4 / Validation | 2,224 / 2,254 / 2,510 / 2,372 / 2,253 |
| `candidate_results.json` | 1 |
| Original `feature_performance/*_False.json` | 5 |
| Original feature-stage logs | 5 |

Family counts include their manifests, not only Parquet shards. Missing byte
volume cannot be recovered from this logical-name map alone. `status.complete`
and successful cumulative resource records do not make those files locally
verified. The pattern supports an incomplete export; it is not proof that the
original training job failed.

Among the verified stage outputs, two exact duplicate-content pairs account
for **55,937,414 redundant bytes**: learned tower vectors appear as both
`models/twotower_items.npy` and `indexes/twotower.npy` (54,037,632 bytes each),
and the associated catalog mapping repeats (1,899,782 bytes each). Distinct
model/index contracts may justify these copies; deletion is not recommended
without dependency review. Cumulative command/resource snapshots also recur
under different stage paths, but are not independent observations.

The root `checkpoints/towers/latest.json` points to a `.pt` absent from that
same directory. The matching file **is present** in the tower-stage checkpoint
folder; both pointers agree on step, hash and size. This is a portable-recovery
layout gap, not evidence that original runtime recovery failed. The review ZIP
named by [RV] is also absent, although its extracted `full_review.json` is
present; the absent ZIP's claimed checksum cannot be independently checked.

Local mtimes span 2019-12-31 to 2026-11-03. The oldest belongs to the extracted
review JSON, while other files have copied run/export times. They cannot be
used as a reliable start-to-finish training clock or, by themselves, as proof
of stale model contents.

## 4. Pipeline execution status

Status refers to the available evidence, not a rerun of training. A verified
serialized output does not independently prove every optimizer operation.

| Requested stage | Audit status | Basis and limits |
|---|---|---|
| Data loading/preprocessing | **Completed and verified**, conditional on retained canonical data | Present checksummed data; complete counts/date bounds; all five saved label/query populations rebuilt against canonical data |
| Frozen content adaptation/preprocessing | **Completed and verified** for retained artifacts | Shapes/masks, schema and manifests; bounded numeric checks; upstream extraction remains unverified |
| WALS training/loading | **Completed and verified** for model/index artifacts | Factors, dimensions, ID maps, source contract and hashes; 15 iterations are recorded rather than independently rerun |
| Two-Tower training/loading | **Completed and verified** for artifacts and training-record consistency | CUDA metadata, 71,480 finite losses, 10 epoch records, target/Q checks, model/checkpoint hashes; no GPU rerun |
| Per-channel retrieval | **Completed and verified** for reconstructed Validation; **completed but not independently verified** for original training outputs | Complete Validation masks/recalls reproduced; original candidate tables missing |
| Native merge/deduplication | **Completed and verified** for reconstructed Validation set integrity | All candidate keys unique, active, sourced and within budget; exact six-rank RRF reconstruction remains unverified |
| Original F1–F4/Validation feature generation | **Completed but not independently verified** as original outputs | Successful run records and consistent downstream counts; five physical stage directories absent |
| Reload Validation feature construction | **Completed and verified** in scope | All 11,150,979 keys checked; first shard's schema/order and four independently replayed temporal features checked, not every numeric feature |
| LR baseline | **Not executed**, by design | Superseded architecture; no missing required LR model |
| LambdaMART | **Completed and verified** for serialized fit and recorded groups | One model, 197 trees, consistent sampling/group totals and early-stopping curve; original sampled rows unavailable |
| Validation inference | **Completed and verified** for saved outputs and bounded model replay | Complete saved tables equal; 4,726 local loaded-model predictions equal reference scores exactly |
| Test inference/evaluation | **Not executed**, by design | Explicit false flags; no Test labels constructed or evaluated in this audit |
| Offline Validation evaluation | **Completed and verified** | Complete metrics independently recomputed with full ground truth and zero-hit queries |
| Final model serialization | **Completed and verified** for present outputs | Model/index/checkpoint bytes match stage manifests |
| Portable archive/export completion | **Partially completed** | Missing five stage directories, broken root checkpoint-file layout and absent review ZIP prevent complete recovery certification |

Sources: [S], [V] (`resources.stages`), [A], [R], [FR], [RF] and the checks
specified in Sections 3, 5–9. No retained successful record is classified as
failed merely because an export is incomplete; no incomplete export is called
fully verified merely because a success flag exists.

## 5. Dataset and temporal protocol verification

### 5.1 Population and source inputs

| Quantity | Observed / independently checked value |
|---|---:|
| Catalog articles | 105,542, unique |
| Catalog customers | 1,371,980 |
| Retained canonical purchase lines | **31,548,013**, across 25 monthly Parquet files |
| Canonical date range | 2018-09-20 through 2020-09-15 |
| Full base-history users | 1,330,267; no selected pilot cohort |
| Catalog-unmatched history rows | Reported 0 |
| Full ranker training queries | 299,204 across F1–F4 |
| Validation queries / unique label pairs | 72,019 / 227,910 |
| Total five-partition queries | 371,223 |
| Reported total native feature rows | 56,156,407 |

Sources: [R] (`sample`, `partitions`), canonical `population_manifest.json`,
canonical transaction Parquet files, [CQ][LG]. Complete query/label counts and
canonical dates were independently checked; raw CSV conversion fidelity was
not, because original CSV bytes are not included in this archive.

All full population/target/query caps are null. The smaller WALS fitted user
and item populations reflect legal training windows and representability,
not a hidden 5,000-user or 5,000-item smoke limit. [I][WS]

### 5.2 Exact temporal windows

| Role | Inclusive cutoff / training range | Label/evaluation range, end exclusive |
|---|---|---|
| WALS fit | 2020-02-01 through 2020-08-11 | Not a future-label fit |
| WALS item vocabulary | 2020-06-16 through 2020-08-11 | Frozen before F1 |
| Two-Tower targets | 2020-05-19 through 2020-08-04 | History strictly before each target date |
| Two-Tower internal validation | Daily 2020-08-05 through 2020-08-11 | Separate from gradient-training target dates |
| F1 | 2020-08-11 | [2020-08-12, 2020-08-19) |
| F2 | 2020-08-18 | [2020-08-19, 2020-08-26) |
| F3 | 2020-08-25 | [2020-08-26, 2020-09-02) |
| F4 | 2020-09-01 | [2020-09-02, 2020-09-09) |
| Validation | 2020-09-08 | [2020-09-09, 2020-09-16) |
| Final Test, not run | 2020-09-15 | [2020-09-16, 2020-09-23) |

[SP], [I] (`splits`), [WS] and matching historical source agree. All six declared
weekly label intervals are independently verified as seven days, pairwise
disjoint, cutoff+1 aligned, and covering exactly 42 dates. The Test interval
was checked as a manifest definition only, not evaluated.

Saved labels and rosters were independently rebuilt from retained canonical
transactions with catalog matching and unique user/article labels:

| Partition | Unique queries | Unique ground-truth pairs | Missing / extra labels or queries |
|---|---:|---:|---:|
| F1 | 71,094 | 237,160 | 0 / 0 |
| F2 | 72,035 | 230,825 | 0 / 0 |
| F3 | 80,253 | 255,172 | 0 / 0 |
| F4 | 75,822 | 238,074 | 0 / 0 |
| Validation | 72,019 | 227,910 | 0 / 0 |

This validates label construction **relative to the saved canonical data**.
It does not prove that the CSV-to-canonical import retained every original
line or that source metadata was historically available in 2020.

### 5.3 Point-in-time and label separation

The matching historical source establishes the following intended execution:

- `pilot/data.py` canonical construction excludes dates on/after 2020-09-16;
  roster construction uses the exact future week and catalog, without an
  active-pool, WALS-vocabulary or recall-result intersection.
- `pilot/streaming.py::TargetStream` selects distinct legal user/date/item
  training events, applies an optional limit only when non-null, and derives
  Q from that selected training population.
- `compute/prepared.py` keeps same-day positives separate and only advances
  prior-day raw purchase lines into the latest-20 history cache. Repeated
  history lines remain observations; same-day intraday order is not invented.
- `pilot/features.py` retrieves and constructs features before reading truth.
  Frozen base models remain fixed while cutoff statistics/history change.
- `ranking.py::select_training_rows` is applied only to training folds. It
  never injects missing future positives. Validation uses full native rows.

These are source-level verifications from [SRC], not assumptions based on
current, different source files.

Additional independent checks were executed:

1. All **3,659,422** selected Two-Tower targets point to contexts dated inside
   [2020-05-19, 2020-08-05); none points into internal validation or Test.
   Context IDs are contiguous, and empirical target-item frequencies equal
   stored Q exactly, over **40,080** observed training items.
2. All **85,441** internal contexts match their daily file dates. Daily counts
   are 15,669 / 15,676 / 13,023 / 10,448 / 7,434 / 10,799 / 12,392.
3. For the first 32 Validation queries and 4,726 reload rows,
   `user_tx_count_30d`, `user_tx_count_90d`, `user_avg_price_90d` and
   `sales_channel_preference` were recomputed from canonical past transactions.
   Counts match exactly; maximum price and channel-preference differences are
   respectively **8.875e-10** and **1.987e-8**, within recorded tolerances.

Sources: `prepared/tower/{manifest.json,contexts.parquet,target_context.npy,
target_item.npy,internal-*.parquet}`, [N], canonical history and [FM].
Exhaustive replay of all historical slots and all 49 original feature columns
was not performed. No claim of universal leakage freedom follows from these
bounded feature checks.

### 5.4 Active pool, IDs and content availability

The source contract is catalog items with at least three raw purchase lines in
the trailing 28 days, applied before each channel's Top-K. Independently:

- Validation snapshot contains **20,362** unique active articles.
- All **20,356** distinct reconstructed candidate articles belong to that pool.
- **17,070** Validation ground-truth pairs lie outside the active pool and remain
  in the denominator. Macro active-pool coverage recomputes to
  **0.933331888707**, matching the report.
- Prediction IDs are valid ten-digit strings; reference/reloaded tables have
  no duplicates inside Top-12 and no missing or extra roster queries.

Sources: `snapshots/Validation.duckdb` table `active`, [FM], [LG][CQ][PR].

Content adapters contain float32 text `[105542,384]` and image `[105542,512]`
arrays. Text availability is 105,542; image availability is 105,100, with **442
missing**. Independent 64-row numeric samples per adapter contain no nonfinite
values or zero vectors for present modalities. These are sampled numerical
checks, not a new extraction audit. [C]

The copied source manifest identifies MiniLM `all-MiniLM-L6-v2` revision
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, 14-field text construction, and CLIP
`openai/clip-vit-base-patch32` revision
`3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`. Source extraction and pretrained-model
provenance were not reproduced. Loader reports explicitly say
`numeric_validation_scope="consumed rows only"` and source-array checksum
status `not_verified`; input identity hashes and stage-output integrity are
separate evidence. Raw CSV/embedding originals are not in this copy. [C][IA][I]

Both learned indexes were independently loaded read-only as **IndexFlatIP**:
WALS dimension 64 / 36,774 vectors; tower dimension 128 / 105,542 vectors;
metric type inner product. Sixty-four deterministic reconstructions per index
match the corresponding NPY rows exactly. WALS factors are not normalized.
Source text/image FAISS files were recorded as **not loaded** by the default
content consumer; they are not substitutes for these learned indexes. [WS][C]

## 6. Retrieval performance

All main comparisons use the same **72,019** Validation queries and **227,910**
full catalog-matched future pairs. Empty channels contribute zero rather than
removing queries. The following quota recalls, row counts and source coverage
were independently recomputed from complete reconstructed Validation features.
[FM][LG][CQ]

| Logical channel | Quota | Recall@12, reported | Recall at quota, verified | Candidate rows | Nonempty queries | Distinct exposed articles |
|---|---:|---:|---:|---:|---:|---:|
| Trend | 60 | 0.023196742 | 0.080553538 | 4,321,140 | 72,019 | 131 |
| Repurchase | 30 | 0.033993313 | 0.034915118 | 336,418 | 45,621 | 18,970 |
| Product-code sibling | 30 | 0.019424070 | 0.022615506 | 451,600 | 32,030 | 13,481 |
| ItemCF | 50 | 0.025179379 | 0.045078034 | 1,611,573 | 32,379 | 20,344 |
| WALS | 50 | 0.019293042 | 0.038150482 | 3,067,600 | 61,352 | 3,097 |
| Multi-modal Two-Tower | 50 | 0.012237014 | 0.028572106 | 2,808,150 | 56,163 | 9,919 |

The Trend, ItemCF, WALS and Two-Tower @12 ranks are also independently checked.
Repurchase/Sibling @12 remain reported because their complete original source
ranks are not retained in the feature schema. Distinct exposed articles count
candidate exposure, not model representability or final Top-12 exposure.

Trend is the strongest individual channel at its larger quota, whereas
Repurchase is strongest at @12. This is not a same-K claim for the configured
quota column. WALS can represent 36,774 fitted items but exposes only 3,097
articles here; Two-Tower exposes 9,919 despite an index covering the full catalog.
This indicates concentrated candidate exposure, not necessarily incorrect IDs
or failed indexing. The differing empty-channel counts can arise from legal
history/representation/active-pool constraints; the retained bundle does not
assign every shortage to a specific reason. [WS][V] (`coverage`)[FM]

Reported temporal recall at each quota:

| Channel | F1 | F2 | F3 | F4 | Validation |
|---|---:|---:|---:|---:|---:|
| Trend@60 | 0.053115 | 0.080877 | 0.088425 | 0.089789 | 0.080554 |
| Repurchase@30 | 0.034553 | 0.031053 | 0.027790 | 0.032485 | 0.034915 |
| Sibling@30 | 0.021453 | 0.021193 | 0.018473 | 0.020219 | 0.022616 |
| ItemCF@50 | 0.043524 | 0.039366 | 0.036565 | 0.042717 | 0.045078 |
| WALS@50 | 0.070747 | 0.058445 | 0.043875 | 0.039487 | 0.038150 |
| Two-Tower@50 | 0.047857 | 0.040172 | 0.029908 | 0.028834 | 0.028572 |

Source: [R] (`partitions.*.channel_recall`). The training-fold entries cannot be
independently recreated from their missing original candidates. Frozen-model
recall declines away from the base cutoff while dynamic Trend remains useful.
Temporal drift is a plausible interpretation, not a demonstrated causal
explanation; query populations differ by week.

The run records @12 and each channel's configured quota, not a common
Recall@10/@20/@100 suite. Missing metrics are not filled in with invented
values or with different-population internal Recall@50. Independent text/image
channels do not exist in this architecture. No prediction failure is inferred
solely from an empty legal channel; complete native union coverage is nonzero
for every Validation query.

## 7. Candidate fusion analysis

### 7.1 Actual implementation and contract

The historical implementation is **not SQL-based RRF fusion**. DuckDB provides
history/statistics; `recall.py::Snapshot.retrieve` merges Python channel hits
by article ID and retains score, one-based rank and fallback metadata.
`Candidate.rrf` computes:

$$
\operatorname{RRF}(i,q)=\sum_{c\in S(i,q)}\frac{1}{60+r_c(i,q)}.
$$

Here $S$ is the set of sources returning the article and $r_c$ is that source's
one-based rank. The full native union is returned in deterministic article-ID
order, without an extra RRF candidate truncation. The ranker later sorts by
model score, then article ID. The maximum sum of channel quotas is **270**,
not 500. This differs from the legacy task checklist but follows v2.0.
Source: [SRC] members `recall.py` (Candidate RRF and retrieval validation),
`pilot/features.py` and `pilot/ranking.py`.

Exact reconstruction of all six original rank/score tuples is **unverified**
because original candidate files are missing and Repurchase/Sibling source
ranks are not fully represented in the retained 49 features. Source inspection
verifies the formula; the stored RRF feature is float32. The derived RRF
baseline below explicitly uses that stored precision, not reconstructed double
precision ranks.

### 7.2 Counts and deduplication

| Partition | Queries | Native candidate/feature rows | Mean/query | Union Recall | Zero-retrieved-positive queries |
|---|---:|---:|---:|---:|---:|
| F1 | 71,094 | 10,593,977 | 149.014 | 0.155797714 | 49,013 |
| F2 | 72,035 | 10,800,496 | 149.934 | 0.171264906 | 47,816 |
| F3 | 80,253 | 12,059,609 | 150.270 | 0.163910193 | 53,968 |
| F4 | 75,822 | 11,551,346 | 152.348 | 0.169463146 | 50,857 |
| Validation | 72,019 | 11,150,979 | 154.834 | 0.166374395 | 48,609 |

Sources: [R] (`partitions`); the Validation row is independently verified from
[FM] and [LG]. Across all partitions, **56,156,407** native rows are reported.
Original training rows are reports, not independently loaded tables.

Complete reconstructed Validation checks establish:

- **11,150,979** unique query/article rows in **2,251** shards.
- Candidates/query: minimum **60**, median **155**, mean **154.833849**,
  p95 **216**, maximum **252**; no query exceeds 270.
- Zero duplicate candidate rows, zero invalid/null source masks and zero rows
  without a source; all candidate articles are active.
- Source rows sum to **12,596,481**; union deduplication removes **1,445,502**
  overlapping appearances. This counts overlapping hits, not necessarily the
  number of articles shared by exactly two channels.
- Union retrieves **33,420** of 227,910 unique ground-truth pairs. Its macro
  Recall is **0.166374395434**; the pair-weighted micro ratio is a different
  metric and must not replace it.

No Recall@500 is reported: the native set is bounded by 270 and was not
expanded for this audit.

### 7.3 Independent leave-one-channel-out complementarity

For each channel, the audit recomputed the signed difference

$$
\Delta R_c=\operatorname{MacroRecall}(C)
          -\operatorname{MacroRecall}(C\setminus c),
$$

using all queries and full truth. Removing a source retains an article if any
other source still supplies it. There is no quota redistribution, refitting,
positive injection or learned reranking. These are **new derived audit
results**, not pre-existing experiment metrics. [FM][LG][CQ]

| Removed channel | Signed union-Recall contribution | Ground-truth pairs supplied exclusively by that channel |
|---|---:|---:|
| Trend | +0.066598205711 | 14,298 |
| Repurchase | +0.011754109497 | 1,796 |
| Sibling | +0.002979944656 | 603 |
| ItemCF | +0.010305850418 | 2,080 |
| WALS | +0.013525407938 | 2,904 |
| Two-Tower | +0.007986287880 | 1,849 |

All six contribute complementary relevant candidates. These contributions are
not additive: shared positives remain in the union when one channel is removed.
Under this pure set-removal definition the differences are nonnegative; no
absolute-value transformation was applied. They are not end-to-end Top-12
ranking ablations and do not show that removing a channel would improve or
harm a retrained ranker by the same amount.

## 8. Ranking training results

### 8.1 Training population and proxy-negative sampling

| Quantity | Observed / independently checked value |
|---|---:|
| Eligible F1–F4 queries | 299,204 |
| Retained training groups | 97,550 |
| Dropped no-preference-pair groups | 201,654 |
| Retained groups by F1/F2/F3/F4 | 22,081 / 24,219 / 26,285 / 24,965 |
| Training rows | 2,965,468 |
| Native positives | 141,591 |
| Proxy negatives, derived | 2,823,877 |
| Aggregate negatives/positive | 19.9439 |
| Positive fraction | Approximately 4.775% |
| Features | 49 |

[RS], `ranker_groups.jsonl`, [FR]. All group sizes sum to the row total;
retained plus dropped groups equal all training queries. Negative count is
below 20 times positive count. Exact query-level retention of every native
positive and hard/random allocation cannot be independently rechecked without
the missing original training features/candidates. The matching source keeps
all native positives and uses the documented query-local seed and allocation.

Dropping unlearnable training groups is permitted. Removing zero-hit queries
from Validation is not; the independent evaluation confirms they remain.
The negative-sampling label is a proxy absence of purchase, not an observed
negative impression or causal non-interest.

### 8.2 Fitted model and early stopping

The saved [model][M] has **197 trees**, 49 feature names in exact schema order,
`max_feature_idx=48`, and objective `lambdarank`. Its split-count importance
sums to **11,938**, matching [FR]. The recorded diagnostic curve has **247
rounds**, peaks at iteration **197** with candidate NDCG@12
**0.7247144839026839**, then continues for the configured 50-round patience.
The retained model agrees with that best iteration. [FR][M][FS]

Matching source calls `model.fit` once across the jointly grouped four-fold
training set. It does not repeatedly fit shards with `init_model` and claim
an equivalent single fit. Memory maps bound array storage but do not make
LightGBM arbitrary-scale out-of-core training.

Leading split counts are visual cluster 2,148; product type 955; RRF 779;
7-versus-28-day sales velocity 473; and ItemCF score 436. These are **split
counts**, not gain importance or causal explanations. Correlated feature
families and categorical split opportunities prevent a causal interpretation.
[FR] (ranker feature importance)[M]

### 8.3 Independent bounded serialization replay

The audit loaded the serialized text Booster in a new local Python process
and predicted the first retained reload shard: **4,726 rows, 49 float32
features, 32 queries**. Maximum absolute difference from
`predictions/reference-00000.npy` was **0.0**, and stable score/article-ID
ordering reproduced every saved Top-12 list in that shard. No training or
parameter update occurred. A loader warning said its model-file parameters
supersede supplied constructor parameters; it did not invalidate predictions.
[M][N][SC]

This is stronger than checking output existence, but does not independently
re-execute the full 11-million-row inference or the CUDA tower. The complete
saved reference/reload Top-12 tables were nevertheless compared and are
exactly equal. The run's all-row feature/score parity claim remains supported
by its own reload report plus these independent, narrower checks. [RF][PT][PR]

## 9. Offline evaluation results

### 9.1 Complete independent recomputation

For each eligible query $q$, let $G_q$ be its complete unique future catalog
purchase set and $P_q$ its saved ordered unique predictions. The primary AP is

$$
\operatorname{AP@12}(q)=
\frac{\sum_{j=1}^{\min(12,|P_q|)}
 \operatorname{Precision@}j(q)\,\mathbf{1}[P_q[j]\in G_q]}
 {\min(12,|G_q|)}.
$$

MAP is the arithmetic mean over **all** 72,019 eligible queries. Recall divides
by $|G_q|$; NDCG uses the ideal list with $\min(12,|G_q|)$ positives. Queries
with zero hits or candidates receive zero. The audit did not intersect truth
with the active pool, model vocabulary or candidate set. [LG][CQ]

| Metric | Run report | Independent saved-output computation |
|---|---:|---:|
| MAP@12 | 0.023638417433196977 | 0.023638417433197 |
| Recall@12 | 0.047971186689203990 | 0.047971186689203756 |
| NDCG@12 | 0.034834373312758626 | 0.034834373312758474 |
| Queries | 72,019 | 72,019 |
| Zero-hit queries | 64,714 | 64,714 |
| Zero-candidate queries | 0 | 0 |
| Empty-truth queries excluded | 0 | 0 |

Sources: [R] (`validation`), [CQ][LG][PR]. Maximum metric disagreement is
**2.36e-16**. There are **1,289** queries with more than 12 positives, so the
AP denominator is consequential; using retrieved-only truth would be wrong.
Both saved prediction files contain exactly 12 unique valid articles for every
query, and no queries outside the roster.

The final zero-hit rate is **89.8568%**. Native retrieval already has **48,609
zero-hit queries (67.4947%)**, which no ranker can repair without changing the
candidate set. Ranker and retrieval quality must therefore be discussed
separately. [FM][LG][PR]

### 9.2 Matched native baselines derived during this audit

The audit streamed one feature shard at a time, retaining only its query IDs,
article IDs and relevant ranking values. It did not materialize all Validation
candidates in Python. RRF sorts the **stored float32 `rrf_fused_score` feature**, descending,
then article ID; Trend sorts stored popularity rank. Both use the same query
roster, full truth and native eligibility as the saved model. These are audit
computations, not previously logged baseline experiments. [FM][CQ][LG]

| Ranking | MAP@12 | Recall@12 | NDCG@12 | Zero-hit queries |
|---|---:|---:|---:|---:|
| Trend channel | 0.007506074540 | 0.023196741815 | 0.014076623175 | 67,791 |
| Stored-RRF-feature rule | 0.017888745342 | 0.046731850878 | 0.029438107382 | 65,254 |
| Saved LambdaMART | **0.023638417433** | **0.047971186689** | **0.034834373313** | **64,714** |
| Logistic Regression | Not executed | Not executed | Not executed | Not applicable |

LambdaMART minus stored-RRF rule:

| Metric | Absolute difference | Relative difference |
|---|---:|---:|
| MAP@12 | +0.005749672092 | +32.14% |
| Recall@12 | +0.001239335811 | +2.65% |
| NDCG@12 | +0.005396265930 | +18.33% |

The much larger MAP than Recall improvement is consistent with better hit
ordering rather than a large increase in the number of relevant recommended
items. This is an interpretation of these metrics, not a causal decomposition.
RRF float32 ties may differ from hypothetical ordering by the missing original
six double-precision source ranks. The comparison is therefore explicitly to
the **available stored-RRF-feature baseline**.

No confidence intervals, paired significance tests, independent Test results
or online outcomes are available. Global Validation is also used for the
reported early-stopping choice. These comparisons establish observed
Validation performance, not an unbiased estimate of future deployment uplift.

### 9.3 Diagnostic metrics that must remain separate

Candidate-list NDCG@12 **0.724714** is not native end-to-end NDCG@12 **0.034834**.
The candidate relevance universe differs, and library handling of no-relevance
groups must not be assumed to match the protocol's full-truth, zero-hit-as-zero
metric. As a useful
sanity bound, if all 48,609 no-native-positive groups received zero and every
other group scored one, the macro result could be at most **0.325053**; thus
0.724714 cannot be interpreted as that end-to-end metric. The exact library
zero-relevance-group convention was not independently replayed here. [FR][FM]

Conditional NDCG@20, native @20 results, a uniform @10/@20/@100 retrieval suite
and injected diagnostic scores were not recorded as this run's official
metrics. They are not invented or substituted for @12. Positive injection is
prohibited by this protocol, so absence of injected/LR outputs is expected.
No final Test evaluation or test-driven selection is evidenced or performed.

## 10. Computational performance and resources

### 10.1 Timing scope and worker-stage table

Individual `resource_usage.json` files are cumulative. Deduplicating the **26**
distinct records yields **18,644.056 s**; naively summing every copy yields
143,545.897 s, an incorrect **7.70×** overcount. All distinct records have exit
code zero, no termination reason and `target_exceeded=false`. [V] (`resources`)

| Worker stage | Seconds | Sampled peak tree RSS, GiB |
|---|---:|---:|
| `preflight` | 1.811 | 0.215 |
| `canonical` | 64.794 | 1.275 |
| `content` | 3.825 | 0.823 |
| `rosters` | 2.818 | 0.516 |
| `capacity` | 0.403 | 0.094 |
| `device_probe` | 6.238 | 1.524 |
| `wals` | 88.581 | 1.459 |
| `prepare_towers` | 129.456 | 3.156 |
| `towers` | 4,916.374 | 3.819 |
| `index_towers` | 0.806 | 0.815 |
| `encode` | 668.608 | 1.734 |
| `preprocessing` | 45.896 | 1.114 |
| `stats_F1` | 28.990 | 0.518 |
| `features_F1` | 1,940.298 | 1.214 |
| `stats_F2` | 28.788 | 0.600 |
| `features_F2` | 1,885.814 | 1.305 |
| `stats_F3` | 27.796 | 0.587 |
| `features_F3` | 2,159.494 | 1.317 |
| `stats_F4` | 27.580 | 0.588 |
| `features_F4` | 2,019.733 | 1.326 |
| `stats_Validation` | 29.798 | 0.572 |
| `features_Validation` | 1,943.797 | 1.326 |
| `ranker` | 673.301 | 3.992 |
| `reload_encode` | 129.234 | 1.617 |
| `reload_features` | 1,649.751 | 1.312 |
| `reload` | 170.072 | 0.402 |

Source: [V] (`resources.stages`), corroborated by the final cumulative
`stages/reload/.../files/resource_usage.json`. This table includes successful
records for the five physically missing feature stages; their timing is
observed in cumulative metadata, not reconstructed from absent stage logs.

| Aggregate work | Seconds | Share of unique worker time |
|---|---:|---:|
| F1–F4 plus original Validation features | 9,949.136 | 53.36% |
| Two-Tower stage | 4,916.374 | 26.37% |
| Reload encoding/features/scoring | 1,949.056 | 10.45% |
| Initial query encoding | 668.608 | 3.59% |
| Five cutoff statistics snapshots | 142.953 | 0.77% |

Feature-stage duration includes retrieval, native merge, features, audit and
writing; it is not a pure feature-arithmetic timer. Canonical conversion is
64.794 s, not total Drive data-loading time. Setup, source staging, coordinator
work, stage-level publication and unarchived attempts are not fully timed.
No exact end-to-end elapsed time or restart count is established.

### 10.2 Two-Tower compute, transfer and validation

| Recorded timing | Seconds | Scope |
|---|---:|---|
| Target read/wait | 6.600 | Cached consumption; excludes the 129.456 s preparation stage |
| Host-to-device, host elapsed | 207.039 | Host timing overlaps device timing |
| Host-to-device, CUDA events | 200.323 | Device interval; not additive with host timing |
| User forward, CUDA events | 118.007 | Training operation interval |
| Item forward, CUDA events | 75.215 | Training operation interval |
| Loss, CUDA events | 60.978 | Existing multi-positive LogQ computation |
| Backward, CUDA events | 285.870 | Training operation interval |
| Optimizer, CUDA events | 72.046 | Training operation interval |
| CUDA batch intervals | 913.321 | Outer measured training intervals, not total tower time |
| Internal validation including catalog | **3,199.038** | 10 complete passes |
| Internal retrieval | **3,126.379** | Nested inside validation; do not add again |
| Internal catalog encoding | 3.906 | Included in validation |
| Final catalog encoding | 0.395 | One final encoding |

Source: [TP] (`host_seconds`, `cuda_event_seconds`, `calls`). Asynchronous host
return times are not presented as GPU kernel timings. Named intervals can
nest or overlap and do not form a disjoint wall-time decomposition.

Internal validation covers **85,441 daily queries per epoch**, ten times,
with **26,730 retrieval calls**. In the matching historical source,
`cached_internal_recall` moves user vectors to CPU NumPy and calls
`pilot/vectors.py::block_topk`: at most 32 queries, 4,096-item blocks,
per-query lexicographic sorting and Python top-K merging. This specific
internal path is not GPU FAISS search, even though learned FAISS artifacts
are serialized elsewhere. The code path and measured retrieval span identify
a concrete investigation target. They do not prove a particular replacement
will be faster or numerically equivalent. [SRC] (`pilot/towers.py:203–244`,
`pilot/vectors.py:12–56`)[TP]

Actual static raw input placement is CUDA, **384,172,880 bytes (366.376 MiB)**,
with **299,008 bytes** pinned staging and `checkpoint_included=false`.
Observed peak CUDA allocation is **508,434,432 bytes (484.881 MiB)** and
reservation **526,385,152 bytes (502 MiB)**. GPU allocation is not utilization;
no utilization time series or CPU utilization series was found. Upstream
encoder measurements embedded in content manifests are separate historical
jobs and are not this training run's VRAM measurements. [TI][TP][C]

Derived throughput is **7,443 examples/s** over the whole tower worker versus
**40,067 examples/s** when dividing examples only by accumulated CUDA batch
intervals. These have different denominators and are not a speedup comparison.
Reported host step p50/p95 are **12.208/14.092 ms**; CUDA step p50/p95 are
**12.196/14.053 ms**. They describe only the **last 4,096 completed batches**.
[TP]

Training records are internally consistent: **3,659,422** unique training
targets, **7,148 batches/epoch**, **71,480** steps and **36,594,220** examples.
Mean recorded batch loss falls from approximately **6.1342** in epoch 1 to
**5.5219** in epoch 10. Internal Recall@50 rises from **0.0474104** to
**0.0648719**, selecting epoch 10. Declining loss alone is not evidence of
better global recommendation quality; internal populations/dates differ from
Validation, where tower Recall@50 is 0.0285721. [TI]

### 10.3 Checkpoint and feature I/O

Local checkpoint serialization/write costs **301.775 s**, and hashing costs
**197.955 s**, with **721 calls** each. Recorded remote copy/hash, file verify
and pointer publication total only **8.606 s**, with 11 calls each. Combined
recorded checkpoint work is **508.336 s**, about **10.34%** of the tower stage.
This is not all stage-level Drive publication overhead. [TP]

The full-state checkpoint is 92.656 MiB, versus 22.705 MiB exported weights,
a ratio of 4.08. Full-state retention is important: faster weights-only saving
would not be an equivalent restart strategy. No interrupted Colab runtime
recovery was independently executed during this audit.

Only reload Validation feature timing is retained separately: [FP].

| Named reload work | Seconds |
|---|---:|
| Feature builder | 361.142 |
| Two-Tower retrieval | 287.738 |
| WALS retrieval | 178.739 |
| Trend | 141.955 |
| ItemCF | 9.838 |
| Repurchase | 2.877 |
| Sibling | 2.017 |
| Total | 1,649.147 |

Named buckets explain 984.307 s; **664.840 s (40.31%)** remains unattributed.
It must not be labelled exclusively I/O, RRF or serialization without further
instrumentation. Original feature stages run at approximately **5,460–5,737
candidate rows/s**, derived from their recorded rows and stage times. Their
missing detailed telemetry prevents a matched per-fold breakdown. [R][V][FP]

There are 2,251 small reference-score NPY files and 2,251 reload-feature Parquet
files. Metadata and transfer overhead are plausible concerns, but no measured
file-consolidation speedup or whole-run I/O bandwidth is available.

### 10.4 Ranker time and memory planning

| Ranker operation | Seconds | Share of recorded ranker total |
|---|---:|---:|
| Train-only negative sampling | 196.796 | 29.35% |
| Dataset construction | 64.151 | 9.57% |
| LightGBM fit | 256.714 | 38.29% |
| Validation scoring | 152.744 | 22.78% |
| Recorded total | 670.405 | 100% |

[RP]. Outer worker duration is 673.301 s, a different scope. Only 256.714 s is
actual LightGBM fitting; ranker work is not the dominant full-job cost.

The run uses 49 float32 columns. Numeric feature payloads are approximately
**581,231,728 bytes** for sampled training and **2,185,591,884 bytes** for full
Validation, before labels, binning, copies and model overhead. The ranker
estimate is explicitly **10× numeric payload plus 512 MiB**, not an out-of-core
guarantee. Original five-partition candidate upper bound is **100,230,210**
rows (270×371,223), versus 56,156,407 reported actual rows. The disk upper bound
is **206,073,311,760 bytes (191.921 GiB)**, not measured disk consumption.
[RM][V] (`capacity`)

### 10.5 Resource estimates versus measurements

| Estimate | Estimated memory, GiB | Related measured worker-tree peak, GiB |
|---|---:|---:|
| WALS | 1.422 | 1.459 |
| Prepared contexts | 0.125 | 3.156 |
| Two-Tower | 0.766 | 3.819 |
| Training feature working set | 0.500 | 1.214–1.326 |
| Complete Validation working set | 0.500 | 1.326 |
| Single LambdaMART | 26.806 | 3.992 |

Sources: [V] (`capacity` and `resources.stages`). These estimates have different
scopes and describe working-set assumptions, not consistently complete-process
predictions. WALS is close; preparation/tower estimates understate complete tree
RSS; the ranker estimate is conservative. The run does not establish that the
estimator is universally calibrated.

Policy initially observes **164.103 GiB** available, sets target **89.272 GiB**,
RSS abort **111.590 GiB**, available-memory floor **32.821 GiB**, two threads
and **0.2 s** monitoring. The largest observed tree sample is 3.992 GiB. These
are sampled process-tree RSS values including the supervisor; they neither
measure total system RAM use nor enforce an OS hard limit. Saved memmaps,
filesystem page cache and peaks between samples require caution. [RES][V]

The prepared-history DuckDB profile records **387,383,296 bytes (369.438 MiB)**
peak temporary-directory use, **1.441 GiB** peak buffer memory and **3.223 s**
latency for that query. This proves some spill occurred; it is not whole-job
peak scratch usage. Actual maximum temporary disk, total disk, staging transfer
and stage-publication time remain unmeasured. [HD]

No inference that full training is safe on the 7–8 GiB-available M1 Max follows
from these Colab samples. This audit used bounded read-only analysis and did
not launch local full training.

## 11. Correctness audit and risks

| Severity | Finding and evidence | Potential impact | Recommended verification/remediation |
|---|---|---|---|
| **High; Critical if published** | Customer-level archive resides beneath `docs/`; `ranker_groups.jsonl` contains `user`, and reload `shards.jsonl` contains query groups. Targeted Git checks show these untracked paths are not ignored | Accidental disclosure when adding documentation directories | Keep archive private; authorize explicit ignore/export remediation separately; publish only an aggregate report; review history before any push |
| **High** | Five original candidate/feature stages and 23,232 logical entries absent despite success flags [A][S] | Cannot independently audit original training rows/provenance or treat this copy as restorable | Recover the original matching stage manifests/files from the original run; validate identity, dependencies and bytes |
| **High** | Full original rank/score provenance unavailable for all six channels | Exact RRF and original sampling verification incomplete | Retain all source hit fields in an audit-accessible representation; do not infer them from masks alone |
| **High** | Root checkpoint pointer's referenced file is absent at that root, though retained inside the tower stage | Direct interrupted-stage continuation from this copied layout can fail | Restore validated state at its expected location and test real runtime recovery; no identity rewriting |
| **Medium** | Candidate NDCG diagnostic differs radically from full-ground-truth native NDCG [FR][PR] | Misleading quality claim or selection interpretation | Report both with explicit population/denominator and verify zero-relevance behavior; retain native MAP as headline |
| **Medium** | Original CSV/source-array lineage not independently reconstructed; loader says source checksums not verified [IA][C] | Cannot exclude upstream conversion/snapshot/provenance problems | Retain original input fingerprints/staging receipt and perform authorized source-level reconciliation |
| **Medium** | Full feature serving parity is a reported check; independent numeric replay covers four columns/32 queries | Unseen feature/temporal bugs remain possible | Expand cutoff-safe replay across folds, missing modalities and cold histories after recovering original features |
| **Medium** | Resource estimates under/over-shoot complete worker RSS by substantial, differently scoped amounts [V] | Bad acceptance decisions on different runtimes | Separate payload/working-set/process-tree estimates and calibrate observed peaks with headroom |
| **Medium** | No matched reference full run or utilization/time/disk series [TP][V] | Unsupported speedup, GPU-utilization or capacity claims | Same data/order/steps/validation/checkpoint work, with cold/warm scope separated and actual utilization recorded |
| **Medium** | Validation is used for early stopping; no uncertainty estimates or locked Test result | Optimistic generalization or significance claims | Keep claims Validation-specific; reserve one explicitly authorized locked Test evaluation for later |
| **Low** | Repeated feature-name/deprecated-eval warnings in `ranker.log`; OpenBLAS thread warning in `wals.log` | Log noise, API maintenance or possible avoidable overhead | Fix genuine API/threading issues in a separate implementation task, then measure; warnings alone do not prove wrong predictions |
| **Low** | Nine Finder files, redundant vector/map copies, tiny-shard proliferation | Packaging confusion and plausible I/O overhead | Exclude Finder metadata from future exports; review dependency-aware consolidation with measured parity |

No actual GitHub data publication, detected future-positive injection,
incorrect metric denominator, duplicate reconstructed candidate or ID/index
mismatch was found. The risks above distinguish demonstrated gaps from
unverified possibilities. Their remediation is a future task; no pipeline or
artifact change was made here.

## 12. Engineering and maintainability

The archive has useful integrity features: immutable stage attempts,
content hashes, independent query rosters, complete saved predictions,
source/config/lock identities, shared schemas, explicit device settings and
separate numerical-worker stages. Source matching permits historical review
without executing old code. These strengths are supported by the independent
checks, not merely by naming conventions.

The main reproducibility weakness is **export completeness**. The aggregate
review correctly describes its JSON as containing no customer-level data, but
that statement must not be generalized to the parent archive. An export
validator should distinguish an aggregate review bundle from a complete
resumable run and explicitly enumerate missing stage/dependency objects.

This run's version-1 identity must not be impersonated as a current GitHub
version-2 run. Preserve its exact historical code/package and original data,
profile/schema/runtime identities for recovery. Renaming directories, changing
RUN_ID or editing `identity.json` is not a valid migration.

For performance work, inspect whether the current implementation has already
changed the historical 32-query CPU block-search path before planning a fix.
Candidate fusion and much residual feature time are not separately timed here.
A new profiler alone would not establish a speedup, and an algorithm-changing
shortcut would not be a comparable optimization.

## 13. Limitations and missing evidence

| Missing / unverified evidence | What remains unresolved |
|---|---|
| Original five feature-stage directories, original candidate manifests/tables, `candidate_results.json` and detailed original feature timing | Full original provenance, sampled-row labels/selection, all-column train/serve parity, original fusion timing |
| Original CSV and source-embedding bytes plus staging receipt in this archive | End-to-end raw-input fidelity, original numeric/source checksums, extraction/provenance |
| Interrupted-runtime recovery execution and complete root checkpoint layout | Real Colab reset/Drive recovery, rather than same-run process reload |
| All-rows independent feature replay and full model rescoring | Universal numeric/PIT correctness beyond complete saved-output arithmetic and bounded replay |
| Whole-run setup/staging/publication clock; maximum disk/scratch; CPU/GPU utilization series | Exact elapsed time, I/O bottlenecks, actual accelerator utilization and capacity |
| Matched reference full run | Any compute speedup attributable to this optimization |
| Original full-precision six-channel ranks for RRF | Exact original RRF rule ordering beyond available float32 feature baseline |
| Confidence intervals / paired uncertainty analysis / final locked Test / serving tests | Statistical significance, generalization, latency, reliability and production readiness |

The report's computed baselines and leave-one-out results are new **analysis**
of immutable outputs, not previously logged training runs. Read-only hashes
establish consistency with local manifests, not an external attestation of
original data provenance. No CUDA training/inference, model refitting, full
pipeline, dataset download, cloud service, final Test or Git publication was
performed by this audit.

## 14. Prioritized recommendations

1. **Protect customer-level artifacts before publication.** Keep this directory
   private, explicitly review JSON/JSONL exposure and use an authorized export
   policy. Preserve current bytes; do not treat the aggregate privacy flag as
   certification of the whole archive.
2. **Repair audit/recovery completeness.** Recover the five missing original
   stage directories and referenced checkpoint placement from the same original
   run, then verify all dependency and file hashes. Do not synthesize replacements
   under the old identity or mix different code/config manifests.
3. **Keep native end-to-end evaluation authoritative.** Publish this complete
   Validation recomputation and matched baseline caveats; retain zero-hit and
   inactive-item misses. Do not headline candidate NDCG or injected diagnostics.
4. **Audit feature and sampling correctness across folds.** Extend the four-column
   replay to the complete schema and verify every sampled positive/negative
   contract once original artifacts are available.
5. **Prioritize measured compute costs.** Investigate internal CPU block search
   and repeated active-item gathers/sorts during per-epoch validation; preserve
   every query, epoch, loss and checkpoint-selection rule. Separately attribute
   the 40.31% residual reload-feature time before optimizing presumed RRF/I/O.
6. **Review local checkpoint frequency and hashing costs.** Preserve full state,
   atomic verification and the last valid remote snapshot. Any lower publication
   cadence must state the larger possible loss of unpersisted progress.
7. **Calibrate resource and I/O telemetry.** Compare equally scoped memory
   estimates and measurements, record whole-run elapsed/disk/utilization data,
   and measure any file-consolidation or BLAS-threading change on equal work.
8. **Measure ranking generalization only after locking configuration.** The
   observed MAP improvement merits preservation, not an immediate Test-driven
   tuning loop. Any final Test or deployment work requires separate authorization.

These are prioritized future engineering actions. None was implemented as part
of this report-only task.

## 15. Final assessment

The strongest established result is a **complete, independently verified saved
Validation benchmark**: 72,019 queries, full ground truth, valid unique Top-12,
exact agreement between saved reference/reloaded predictions, and a bounded
serialized-ranker replay with zero score difference. LambdaMART exceeds the
available stored-RRF-feature baseline in Validation MAP@12, while retrieval
misses remain the main structural ceiling.

The completed workload is supported by consistent full-mode configuration,
source hashes, model artifacts, optimizer records, canonical labels and
successful worker records. The **local archive is incomplete**, and global
leakage freedom, full interrupted-runtime recovery, a compute speedup and
production readiness are not established.

This is a useful full-training/Validation milestone with actionable evidence.
It is not a final Test result, a deployment approval, or permission to publish
customer-level artifacts.

## 16. Appendix: evidence and reproduction

### 16.1 Exact stage locations

Paths are relative to `docs/achive/colab_full_computing_opt_v1/`.

| Stage alias | Immutable `files/` base |
|---|---|
| Preflight | `stages/preflight/fba96bdfecf14b7789910e781e77ee8b/files/` |
| Canonical | `stages/canonical/50f8a148610d41de88925e90cac13af8/files/` |
| Content | `stages/content/c4cf70d5a1034e0aa738f815fd88fb39/files/` |
| Rosters | `stages/rosters/51464d6a336842ec94b19879532376e9/files/` |
| WALS | `stages/wals/5e73b803db204e67a211483062345244/files/` |
| Prepared towers | `stages/prepare_towers/c44ff2842f354a76876180b1588f23c9/files/` |
| Towers | `stages/towers/4cd2f016bf1e466c9c8647ce4535d02c/files/` |
| Learned index | `stages/index_towers/ab378dbdcb8a4f9da9b5f83e91743e51/files/` |
| Preprocessing | `stages/preprocessing/34d7e2823eda48378c40dba99b496984/files/` |
| Ranker | `stages/ranker/f8ceec7bf13a4cdf8a3321e000df7cfe/files/` |
| Validation snapshot | `stages/stats_Validation/c2be7176bb194b4cb1d43e96b1b28a3a/files/` |
| Reload encoding | `stages/reload_encode/a97221901cb1488abd50912f2823b6da/files/` |
| Reload features | `stages/reload_features/7aede4498d6b47e3955210f2fed834b5/files/` |
| Reload scoring | `stages/reload/028c5cc2f9724502837be28118c5a638/files/` |

### 16.2 Checks actually executed

All analysis used the existing local Python environment with bytecode writing
disabled (`.venv/bin/python -B`), without installing packages. DuckDB analyses
used an in-memory connection, two threads, 512 MB managed-memory limit and
`max_temp_directory_size='0B'`; saved databases were attached read-only.
Parquet projections and one-shard-at-a-time loops avoided full Python candidate
tables. The largest observed process RSS among these SQL audits was approximately
629 MB, including allocations outside DuckDB's managed limit.

| Executed check | Outcome |
|---|---|
| Recursive filename/format/size inventory and logical-map resolution | 4,740 physical files; 23,232 unresolved logical names |
| Streamed size/SHA256 checks on all declared present stage outputs | 4,702 passed; no output mutation |
| Historical JSON identity/manifest hash and dependency checks | Identity and 21 manifests pass; 210 present dependency links agree; 30 reference missing stages |
| In-memory ZIP member hashes against all archived code identities | 58/58 match; no extraction/import/execution |
| Read-only canonical counts and complete five-fold label/roster joins | Exact counts, no extras/missing/duplicate pairs |
| Batched target/context/Q checks | All 3,659,422 training references legal; all 85,441 internal contexts legal; exact empirical Q |
| Full Validation metric, query/list equality and denominator checks | All 72,019 queries; complete saved reference/reloaded agreement |
| Full reconstructed candidate count, duplicate, masks and active-pool checks | 11,150,979 valid native rows; all within active set and quota sum |
| Complete quota recall, union and leave-one-out calculations | Reported retrieval metrics reproduce; new signed LOO table derived |
| Streamed Trend and stored-RRF-feature baselines | Complete matched-population table in Section 9 |
| First-shard temporal/schema/order checks and loaded Booster prediction | Four temporal features and 4,726 scores pass; max score difference 0 |
| Learned FAISS/NPY dimensions, mappings and bounded reconstructions | Two IndexFlatIP artifacts; 64 exact reconstruction matches each |
| Read-only Git ignore/tracking checks for customer-level JSONL | Both targeted paths untracked and not ignored |

The local analysis times are not substituted for Colab stage measurements.
Two read-only DuckDB setup attempts used unsupported parameterized/view forms
and exited nonzero before evaluation; corrected in-memory read-only attachments
then succeeded. No failed check was relabelled as a passing result. No repository
pytest/model-training suite was run: this was a report-only task, with bounded
artifact computations rather than a new training experiment.

For identity reproduction, use the historical function's serialization exactly:

```python
import hashlib
import json

signature = hashlib.sha256(
    json.dumps(identity, sort_keys=True, allow_nan=False).encode("utf-8")
).hexdigest()
```

Do not change JSON separators when reproducing these hashes. For a completion
manifest, hash all fields except `manifest_sha256`; then verify each output's
bytes and SHA256 at `stages/<stage>/<attempt>/files/<logical-name>`.

For metric reproduction, resolve [CQ], [LG] and [PR], explode the ordered
prediction lists with one-based positions, join full unique ground truth,
compute cumulative hit precision and discount, then **left-join every query
from the independent roster** before averaging. For RRF/Trend/LOO, use the
same full truth and reconstructed feature shards; preserve their stored
precision, channel masks and deterministic article-ID ties. These procedures
require no model retraining and no final Test access.

### 16.3 Supplemental historical-source references

The matching source package is [SRC], outside the supplied run directory.
Its bytes were inspected in memory only. Relevant ZIP member locations:

| Member / region | Contract checked |
|---|---|
| `src/hm_recsys/colab/files.py::canonical_hash` | Exact identity serialization |
| `src/hm_recsys/pilot/data.py`, canonical construction and roster lines 432–453 | Legal history and full catalog-matched weekly truth |
| `src/hm_recsys/pilot/streaming.py`, lines 62–78 | Unique legal targets, optional limits and fixed order |
| `src/hm_recsys/compute/prepared.py`, around lines 187–204 | Prior-day history advancement and complete same-day positive sets |
| `src/hm_recsys/recall.py`, Candidate RRF and retrieval around lines 273–289 | Smoothing 60, full native merge, quotas/source provenance |
| `src/hm_recsys/learned.py`, lines 35–50 | Genuine fixed-item WALS normal-equation folding-in |
| `src/hm_recsys/pilot/features.py`, around lines 148–190 | Candidate/features built before truth attachment |
| `src/hm_recsys/ranking.py`, lines 17–31 | Native-positive retention and train-only negative sampling |
| `src/hm_recsys/pilot/ranking.py`, lines 125–228 and 252–275 | Memory estimate, one fit, complete Validation metrics |
| `src/hm_recsys/evaluation.py`, around lines 37–40 | Full-ground-truth denominators |
| `src/hm_recsys/pilot/towers.py`, lines 203–244 | CPU blocked internal retrieval within epoch validation |
| `src/hm_recsys/pilot/vectors.py`, lines 12–56 | Exact active-first block search and stable ties |

Line numbers refer to these ZIP members, not the different current workspace.

[I]: achive/colab_full_computing_opt_v1/identity.json
[S]: achive/colab_full_computing_opt_v1/status.json
[A]: achive/colab_full_computing_opt_v1/artifacts.json
[R]: achive/colab_full_computing_opt_v1/run_summary.json
[V]: achive/colab_full_computing_opt_v1/full_review.json
[RES]: achive/colab_full_computing_opt_v1/resources.resolved.json
[RV]: achive/colab_full_computing_opt_v1/review_manifest.json
[P]: achive/colab_full_computing_opt_v1/stages/preflight/fba96bdfecf14b7789910e781e77ee8b/files/profile.resolved.json
[SP]: achive/colab_full_computing_opt_v1/stages/preflight/fba96bdfecf14b7789910e781e77ee8b/files/split_manifest.json
[IA]: achive/colab_full_computing_opt_v1/stages/preflight/fba96bdfecf14b7789910e781e77ee8b/files/input_acceptance.json
[C]: achive/colab_full_computing_opt_v1/stages/content/c4cf70d5a1034e0aa738f815fd88fb39/files/content/acceptance.json
[WS]: achive/colab_full_computing_opt_v1/stages/wals/5e73b803db204e67a211483062345244/files/models/wals.json
[TI]: achive/colab_full_computing_opt_v1/stages/towers/4cd2f016bf1e466c9c8647ce4535d02c/files/models/twotower.json
[TP]: achive/colab_full_computing_opt_v1/stages/towers/4cd2f016bf1e466c9c8647ce4535d02c/files/tower_performance.json
[FS]: achive/colab_full_computing_opt_v1/stages/preprocessing/34d7e2823eda48378c40dba99b496984/files/feature_schema.json
[FR]: achive/colab_full_computing_opt_v1/stages/ranker/f8ceec7bf13a4cdf8a3321e000df7cfe/files/full_results.json
[RS]: achive/colab_full_computing_opt_v1/stages/ranker/f8ceec7bf13a4cdf8a3321e000df7cfe/files/ranker_sampling.json
[RP]: achive/colab_full_computing_opt_v1/stages/ranker/f8ceec7bf13a4cdf8a3321e000df7cfe/files/ranker_performance.json
[RM]: achive/colab_full_computing_opt_v1/stages/ranker/f8ceec7bf13a4cdf8a3321e000df7cfe/files/ranker_memory_estimate.json
[M]: achive/colab_full_computing_opt_v1/stages/ranker/f8ceec7bf13a4cdf8a3321e000df7cfe/files/models/lambdamart.txt
[PT]: achive/colab_full_computing_opt_v1/stages/ranker/f8ceec7bf13a4cdf8a3321e000df7cfe/files/predictions/reference_top12.parquet
[SC]: achive/colab_full_computing_opt_v1/stages/ranker/f8ceec7bf13a4cdf8a3321e000df7cfe/files/predictions/reference-00000.npy
[FM]: achive/colab_full_computing_opt_v1/stages/reload_features/7aede4498d6b47e3955210f2fed834b5/files/predictions/reloaded_features/manifest.json
[FP]: achive/colab_full_computing_opt_v1/stages/reload_features/7aede4498d6b47e3955210f2fed834b5/files/feature_performance/Validation_True.json
[N]: achive/colab_full_computing_opt_v1/stages/reload_features/7aede4498d6b47e3955210f2fed834b5/files/predictions/reloaded_features/part-00000.parquet
[CQ]: achive/colab_full_computing_opt_v1/stages/rosters/51464d6a336842ec94b19879532376e9/files/queries/Validation.parquet
[LG]: achive/colab_full_computing_opt_v1/stages/rosters/51464d6a336842ec94b19879532376e9/files/labels/Validation.parquet
[PR]: achive/colab_full_computing_opt_v1/stages/reload/028c5cc2f9724502837be28118c5a638/files/predictions/reloaded_top12.parquet
[RF]: achive/colab_full_computing_opt_v1/stages/reload/028c5cc2f9724502837be28118c5a638/files/reload_results.json
[HD]: achive/colab_full_computing_opt_v1/stages/prepare_towers/c44ff2842f354a76876180b1588f23c9/files/prepared/tower/history_query_profile.json
[SRC]: ../data/releases/hm_recsys_a736135841d66476.zip
