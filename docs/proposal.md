# H&M Fashion Recommendation System — Project Proposal

**Version:** v0.3.2 · six-channel cascade  
**Updated:** 2026-11-02  
**Evaluation contract:** v2.0; existing dates retained, non-overlap now an explicit hard gate  
**Status:** Design and implementation contract. Text/image artifacts are reported by the user as already prepared; their contents have not been inspected in this documentation revision. No new training or evaluation is claimed.  
**Task:** Predict unique articles purchased in the next seven days; return Top-12  
**Only learned ranker:** LightGBM LambdaMART (`objective="lambdarank"`)  
**Primary metric:** Native end-to-end macro MAP@12  
**Fixed data root:** `<repository>/data/`

## Contents

- [0. Sources, precedence, and revision scope](#0-sources-precedence-and-revision-scope)
- [1. System architecture and design principles](#1-system-architecture-and-design-principles)
- [2. Global chronological split](#2-global-chronological-split)
- [3. Retrieval layer: six channels](#3-retrieval-layer-six-channels)
- [4. Ranking layer: LambdaMART only](#4-ranking-layer-lambdamart-only)
- [5. Evaluation, repository layout, and implementation](#5-evaluation-repository-layout-and-implementation)
- [6. Risks, experiments, and completion criteria](#6-risks-experiments-and-completion-criteria)

## 0. Sources, precedence, and revision scope

The current design follows the user's revised six-channel architecture: the
`60/30/30/50/50/50` quotas, frozen WALS/Two-Tower base, dynamic statistical
retrieval and ItemCF, four non-overlapping ranker training weeks, and one
LambdaMART ranker. This revision applies the user's confirmed input and
reference directories without redesigning those algorithms or dates.

The latest decisions are:

- **Existing content artifacts:** Read the prepared files in
  `data/text_features/` and `data/image_features/`. Do not schedule text/image
  extraction, encoder replacement, or data migration as a prerequisite.
- **D06:** The new chronological schedule is mandatory. Overlapping target
  windows, including the old fifteen-anchor schedule, are prohibited.
- **D07:** Preserve the supplied `data/` and `docs/` layout. Historical reports
  and archived notebooks are limited references, never the implementation
  specification when they conflict with the new architecture.

### 0.1 Authority and conflict handling

Apply these rules in order:

1. The latest explicit user decisions and the approved new architecture govern
   core requirements.
2. This proposal and the synchronized documents in `docs/archi/` express that
   architecture. [AGENTS.md](AGENTS.md) governs engineering, Python, notebook,
   and documentation quality; it must not restore an obsolete architecture.
3. [Implementation decisions](docs/archi/implementation_decisions.md) make
   clarifications and unresolved evidence explicit. They cannot silently
   override a newer user decision.
4. Historical reports, ZIP contents, notebooks, and older proposals/protocols
   are reference-only. Resolve conflicts in favor of the new architecture and
   update the affected current documents together.

Do not use the old `2020-07-28` base cutoff, fixed `C0`, overlapping anchors,
positive injection, NDCG@20 primary selection, or standalone text/image recall.
Do not mix outputs from different protocols as directly comparable results.

### 0.2 Evidence and limited reference use

The [Phase 0 report](<docs/phase0-data download & EDA report.md>) supplies the
previously measured tabular facts. Its historical statement that embedding
creation was outstanding does **not** override the user's later confirmation
that the embedding artifacts have been prepared and downloaded.

The listed [historical MVP report](docs/hm_mvp_full_reports_colab_v1.md),
`docs/achive/hm_mvp_full_reports_colab_v1.zip`, and
`docs/achive/mvp_notebooks_colab/` may be consulted for relevant implementation
ideas or prior observations. Their contents were not supplied for inspection in
this revision; do not invent their results or assume their code matches the
current design.

Consult only the portions relevant to the assigned task. Do not automatically
unzip, execute, port, or import the archived pipeline. Any reused helper must
be reviewed for current date semantics, paths, interfaces, sampling, metrics,
and training-serving consistency, then tested under the new contract. Record
its provenance and changes. Historical scores are historical evidence only,
not a baseline measured under this protocol.

## 1. System architecture and design principles

### 1.1 Data-driven scope and current readiness

This is a purchase-history prediction project, not CTR/CVR estimation,
inventory forecasting, returns analysis, or causal sales-lift measurement.
Repurchase, sibling variants, recent popularity, and content matching are
hypotheses to evaluate, not established business benefits.

Phase 0 recorded 31,788,324 purchase lines, 1,371,980 catalog customers, and
105,542 catalog articles, with transactions from 2018-09-20 to 2020-09-22.
It also measured 105,100 local image filename mappings. Filename presence is a
historical audit result, not proof of embedding quality or complete coverage.
Preserve raw repeated purchase lines; deduplicate articles within each label
set. See the [Phase 0 report](<docs/phase0-data download & EDA report.md>).

**Current input status:** The user reports that text and image embeddings,
article mappings, manifests, existing FAISS indexes, and an image failure log
are already in `data/`. Treat them as supplied inputs, not unimplemented
extraction deliverables. Actual dimensions, row alignment, normalization,
encoder identities, and successful coverage must be read and checked locally
before their consumers run. This documentation task has not performed that
artifact audit.

### 1.2 Six retrieval channels

| ID | Channel | Main signal | Default Top-K |
|---|---|---|---:|
| R1 | 7-Day Trend Popularity | Global and age-bucket recent purchases | 60 |
| R2 | Time-Decayed Repurchase | Recent repeat-purchase evidence for the same article | 30 |
| R3 | Product_Code Sibling Recall | Other article variants of a purchased product code | 30 |
| R4 | Time-Decayed ItemCF | Recent local collaborative relationships | 50 |
| R5 | Time-Decayed WALS | Global low-rank implicit-feedback preferences | 50 |
| R6 | Multi-modal Two-Tower | Purchase-supervised history, attributes, text, and image | 50 |

The pre-deduplication bound is **270**. Actual unique candidate counts must be
measured; the source estimates of 100–180 candidates and 22,000–28,000 active
articles are not runtime facts. Send the complete native deduplicated union to
LambdaMART. RRF is a feature or non-learned baseline, not an additional
pre-ranking stage or trained ranker.

The data flow is six-channel retrieval, union and deduplication with provenance,
shared feature construction, LambdaMART scoring, and stable Top-12 selection.
The module view is in [architecture.md](docs/archi/architecture.md).

No standalone Text/Image recall, pre-ranker, second learned ranker, LightGCN,
DeepFM, or DIN is in scope. Existing `text_faiss.index` and `image_faiss.index`
files do not authorize extra channels.

### 1.3 Technology and execution boundaries

Use DuckDB/Parquet for bounded-memory ETL; SciPy/implicit for sparse CF;
PyTorch for Two-Tower; FAISS for vector search; LightGBM for the only ranker;
DVC for artifact versioning; and MLflow for experiment lineage.

The multimodal branches consume the supplied frozen content vectors. The
original encoders are not rerun or fine-tuned in the default pipeline. A
replacement or recovery extraction job requires a separate explicit request
and must not overwrite the supplied artifacts.

The learned Two-Tower index uses its own 128-dimensional item representations;
WALS uses its own item factors. Neither is interchangeable with a supplied
raw-content FAISS index. The current learned-index design uses `IndexFlatIP`
for exact inner-product search. Do not infer the type, metric, or normalization
of the supplied indexes from their filenames.

First deliver a local, reproducible training and batch-inference pipeline.
HTTP serving, containers, and cloud execution are later scope, not completed
facts. Documentation changes do not authorize paid resources or large jobs.

## 2. Global chronological split

### 2.1 Date semantics

`cutoff_date = T` is the last observable calendar date, inclusive.
`prediction_date = a = T + 1 day`.

```text
Observed history:       t_dat <= T, equivalently t_dat < a
Trailing d-day window:  [a - d days, a)
Seven-day labels:       [a, a + 7 days)
```

For example, the trailing seven days at cutoff 2020-08-11 are 2020-08-05 through
2020-08-11, not an eight-date interval. There is no observed intraday ordering.
Use one window implementation in training and inference; persist both dates.

### 2.2 Mandatory four-stage schedule

| Stage | Input interval or cutoff | Target interval / role |
|---|---|---|
| WALS base training | `[2020-02-01, 2020-08-12)` | Fit user/item factors |
| WALS item vocabulary | Purchases in `[2020-06-16, 2020-08-12)` | Freeze item-column map |
| Two-Tower training targets | `[2020-05-19, 2020-08-05)` | Prior history to same-date positive |
| Two-Tower internal validation | `[2020-08-05, 2020-08-12)` | Daily Recall@50 checkpoint selection |
| Ranker train fold 1 | `T=2020-08-11` | `[2020-08-12, 2020-08-19)` |
| Ranker train fold 2 | `T=2020-08-18` | `[2020-08-19, 2020-08-26)` |
| Ranker train fold 3 | `T=2020-08-25` | `[2020-08-26, 2020-09-02)` |
| Ranker train fold 4 | `T=2020-09-01` | `[2020-09-02, 2020-09-09)` |
| Global validation | `T=2020-09-08` | `[2020-09-09, 2020-09-16)` |
| Final test | `T=2020-09-15` | `[2020-09-16, 2020-09-23)` |

Use these exact dates rather than recomputing dates from approximate week
counts in the original source.

### 2.3 D06: non-overlap is a hard gate

The six seven-day target intervals—four ranker folds, global validation, and
final test—must be **pairwise disjoint**. Each contains exactly seven calendar
dates, and together they cover 2020-08-12 through 2020-09-22 exactly once.
Two-Tower training target dates and its internal validation target dates must
also be disjoint. No target date may be reassigned across these supervised
splits. Random row splitting and overlapping weekly anchors are prohibited.

Persist the schedule in a versioned split manifest and fail before candidate
or feature generation if any date is wrong, any target interval overlaps, any
interval is not seven days, or a cutoff is not the day before its target start.
Unit tests must reject the old three-day anchor stride and boundary off-by-one
errors; this rule is not just a notebook comment.

**Non-overlap concerns split/target assignments, not historical lookback reuse.**
Past-only 7/28/60/90-day histories may overlap across cutoffs. A purchase from a
completed earlier week may become observed history for a later query, but must
not be relabeled in a second overlapping target week. The base models can
share historical inputs as specified. Users can recur across weeks; do not
claim statistical independence.

### 2.4 Frozen parameters and advancing history

Freeze WALS item factors/maps, Two-Tower weights and encodings, and fitted
visual clustering before the first ranker query. WALS may use purchases
through 2020-08-11; Two-Tower gradient targets end on 2020-08-04, with internal
validation through 2020-08-11.

At each later cutoff, rebuild R1–R4, the active pool, current histories, WALS
user folding-in, and historical ranking statistics using only `t_dat <= T`.
Updating observed history or rebuilding an active-subset index is not refitting
base item factors or Two-Tower weights. Never use current/future target dates
in candidate generation or features. Cutoff-specific caches must not be
silently reused for a different date.

## 3. Retrieval layer: six channels

### 3.1 Shared candidate pool and output contract

Define the active candidate pool at cutoff $T$, with $a=T+1$ day:

$$
A(T)=\{i\in C_{catalog}:\operatorname{purchase\_lines}_{[a-28,a)}(i)\ge3\}.
$$

This is a purchase-activity proxy, not verified stock. It can exclude new,
low-frequency, and subsequently purchased articles. Do not shrink the main
ground truth to this pool; measure its coverage loss.

Apply the current pool before each channel's Top-K. WALS additionally requires
a frozen item factor; do not impose its vocabulary on other channels.
Two-Tower maps unknown IDs to frozen `<UNK>` behavior and uses available
content; it must not add randomly initialized, untrained ID rows.

Each channel returns:

```text
query_id, customer_id, cutoff_date, article_id,
channel, raw_score, source_rank, retriever_version
```

A query identifies `(customer_id, cutoff_date)`. Ranks start at one; articles
are unique per channel; empty history may yield no candidates; ties use
`article_id` ascending. Scores are channel-local, not directly additive.
Preserve all six presence flags, raw scores, and source ranks after union.

### 3.2 R1 — 7-Day Trend Popularity

**Build input:** Purchases `(customer_id, article_id, t_dat)` in `[a-7,a)` and
static customer age. Compute a global list and age-bucket lists:

$$
S_{pop}(i,b)=\sum_{(u,i,t):bucket(u)=b}\exp(-(T-t)/3).
$$

Age buckets are `<25`, `[25,35)`, `[35,50]`, and `>50`. Missing/invalid age is
`UNKNOWN`; customer attributes follow the static-snapshot assumption.

**Build output:** `cutoff_date, age_bucket, article_id, pop_score, pop_rank`,
including the global list. There are no labels or gradient updates.

**Inference:** Read the user's bucket and current pool; return up to 60
`(article_id, pop_score, pop_rank)`. Unknown age uses the global list. A short
bucket list may backfill from the global list within the same 60-item quota;
preserve fallback provenance rather than inventing bucket counts. This is the
explicit fallback for users without usable history.

### 3.3 R2 — Time-Decayed Repurchase

**Build input:** The user's purchase lines in `[a-60,a)`.

$$
S_{rep}(u,i)=\sum_{t\in H_{60}(u,i)}\exp(-(T-t)/14).
$$

**Build output:** User/article aggregates containing the decay score, purchase
line count, and last purchase date.

**Inference:** Read user aggregates and the current pool; return up to 30
`(article_id, repurchase_score, repurchase_count, days_since_last_buy)`.

Do not globally exclude already-purchased articles. Counts are purchase lines,
not distinct orders. Same-day repeats and cross-day repurchases are different
interpretations. Empty history returns no candidates; do not invent returns,
exposure, or negative feedback.

### 3.4 R3 — Product_Code Sibling Recall

**Preparation:** Build the actual `product_code -> article_id` mapping from
`articles.csv`; do not replace it with a guessed ID-prefix rule.

**Inference input:** Seeds in `[a-30,a)`, each seed's last purchase date,
candidate seven-day sales, and the active pool. For seed $i$, admit $j\ne i$
with the same product code:

$$
S_{sib}(u,j)=\max_i\left[\exp(-(T-t_{u,i})/14)\log(1+sales_{7d}(j))\right].
$$

**Output:** Up to 30 `(article_id, sibling_score, sibling_source_days)`. When
multiple seeds nominate one candidate, use the maximum score and deterministic
tie handling. A sibling may itself have been purchased; overlap with R2 is
allowed and retains both provenance flags.

This is an article-variant relationship, not evidence of size recommendations,
returns, or within-order complementarity.

### 3.5 R4 — Time-Decayed ItemCF

**Build input:** Purchases in `[a-28,a)`, rebuilt at each cutoff. Use distinct
user/article edges; $t_{ui}$ is that pair's last purchase date in the window.
Preserve raw purchase lines for count-based features.

Retain the source similarity formula:

$$
w_{ij}=\frac{\sum_{u\in U(i)\cap U(j)}\frac{\exp(-|t_{ui}-t_{uj}|/7)}{\log(1+|I_u|)}}{\sqrt{|U(i)|\,|U(j)|^{\alpha}}},\qquad\alpha=0.75.
$$

Here $I_u$ is the distinct item set in the window. This formula is generally
directional. Relative to $\alpha=1$, $0.75$ weakens the candidate-popularity
denominator; do not describe it as stronger suppression or silently substitute
a different formula.

**Build output:** Sparse top-100 current-pool neighbors per seed and cutoff.
Never allocate a dense all-item similarity matrix. Measure high-degree user
pair-enumeration costs; any pruning must be configured and logged.

**Inference input:** Up to 15 recent distinct articles within the 28-day
window, with last purchase dates. Sum `exp(-(T-t_ui)/10) * w_ij` across seeds.
Return up to 50 `(article_id, itemcf_score, itemcf_rank)`.

### 3.6 R5 — Time-Decayed WALS

#### Training input, objective, and output

Use `[2020-02-01, 2020-08-12)` transactions. Columns are articles purchased in
`[2020-06-16, 2020-08-12)`; rows are users with at least one interaction with
those columns. Measure actual matrix sizes rather than reusing estimates.

With $T_{base}=2020\text{-}08\text{-}11$:

$$
r_{ui}=\sum_t\exp(-(T_{base}-t)/30),\qquad p_{ui}=\mathbf{1}[r_{ui}>0].
$$

$$
c_{ui}=1+\alpha_w\log(1+r_{ui}/\epsilon),\qquad\alpha_w=20,\quad\epsilon=1.
$$

The user-by-item CSR matrix stores full confidence for observed entries;
unlisted entries mean $p=0,c=1$. Minimize:

$$
\sum_{u,i}c_{ui}(p_{ui}-\mathbf{u}_u^T\mathbf{v}_i)^2+
\lambda\left(\sum_u\|\mathbf{u}_u\|_2^2+\sum_i\|\mathbf{v}_i\|_2^2\right).
$$

Initial settings remain `factors=64`, `regularization=0.05`, `iterations=15`,
`use_cg=True`, and seed 42. Since full confidence already includes
`confidence_alpha=20`, use library `alpha=1.0`; do not scale twice. These are
initial configurations, not measured optimal hyperparameters.

**Output:** `user_factors [U,64]`, `item_factors [I,64]`, bidirectional ID maps,
training intervals, and confidence configuration.

#### Inference with fixed item factors

Use the user's `[2020-02-01,a)` history on the frozen item vocabulary and the
same confidence function, now decayed relative to $T$. Recalculate the user
factor with the validated fixed-item ALS solve (`recalculate_user` or an
equivalent regularized least-squares implementation).

Search `A(T) intersect I_WALS` by the resulting user/item inner product; return
up to 50 `(article_id, wals_score, wals_rank)`. A learned-factor active-subset
`IndexFlatIP` is separate from both supplied content indexes. Do not normalize
WALS factors into a different cosine objective. Disable already-liked filtering.
No representable history means an empty WALS result and rule-based fallback.

The source's normalized mixture of an old user vector and pooled item vectors
is an unselected heuristic, not an equivalent ALS folding-in operation. Items
outside the frozen vocabulary do not acquire factors through user folding-in.

### 3.7 R6 — Multi-modal Two-Tower

#### Samples and tower inputs

Training target events run from 2020-05-19 through 2020-08-04. A positive dated
$t$ uses only history `<t`; same-day purchases cannot enter each other's
history. The existing default history lower bound is 2020-05-19, retaining the
latest 20 purchase lines with deterministic same-date truncation, not invented
intraday chronology. Document sampling/deduplication separately from the
preserved raw data.

Internal validation targets run 2020-08-05 through 2020-08-11, with daily
past-only queries and Recall@50 for checkpoint selection. Do not backpropagate
these labels or describe this diagnostic as external seven-day MAP@12.

| Branch | Input | Processing / output |
|---|---|---|
| User profile | `age_bucket` 8d; `club_member_status` 4d; `fashion_news_frequency` 4d | Explicit missing categories |
| User history | Latest 20 article, product-type, and colour embeddings | Masked time-decayed mean pooling; decay constant 14 days |
| User MLP | Profile and history aggregate | `[256,128]`, LayerNorm, L2 normalization; `[B,128]` |
| Item IDs/attributes | Article 64d; product code 32d; product type 16d; graphical appearance 8d; colour 8d; section 8d | Concatenated structured representation |
| Item text | Verified row from `data/text_features/text_embeddings.npy` | `Linear(D_text,32)` using the inspected source dimension |
| Item image | Verified row from `data/image_features/image_embeddings.npy` | `Linear(D_image,32)` using the inspected source dimension |
| Item MLP | IDs, attributes, projected text/image, availability masks | `[256,128]`, LayerNorm, L2 normalization; `[B,128]` |

`D_text` and `D_image` come from the supplied arrays and manifests. The source's
384/512-dimensional encoder examples are **not** authority to replace the
prepared files. Do not assume text was generated from `detail_desc` alone;
record the actual text construction and encoder identity when available.
A dimension/manifest conflict is a loader error, not permission to silently
re-extract content or change encoders.

Keep source vectors frozen; train the projection layers and towers. Text and
image use separate article-to-row maps, joined by canonical `article_id`, not
assumed equal row positions. Section 5.2 defines the artifact acceptance gate.

Build article/product-code vocabularies only from legal training targets and
history observed through 2020-08-04. Later IDs use `<UNK>` without randomly
adding untrained rows. Static customer/catalog content remains an explicit
snapshot assumption.

Retain shared ID dropout `p=0.25`: during training, zero the article and
product-code ID branches together while retaining attributes/content. Real
unknown IDs have their own `<UNK>` semantics; inference disables dropout.
This mechanism does not itself prove that content improves retrieval.

#### Loss, training outputs, and inference

Retain in-batch negatives and LogQ-corrected InfoNCE:

$$
L=-\frac{1}{B}\sum_k\log\frac{
\exp(\mathbf{e}_{u_k}^{T}\mathbf{e}_{i_k^+}/\tau-\log Q(i_k^+))}{
\sum_j\exp(\mathbf{e}_{u_k}^{T}\mathbf{e}_{i_j^+}/\tau-\log Q(i_j^+))},
\qquad\tau=0.07.
$$

Here $B$ is batch size, $\mathbf{e}$ denotes tower outputs, and $Q$ is the
training sampler's item distribution. Resolve duplicate candidates,
multi-positive user/date targets, and padding before implementation. Never
turn known same-query positives into negatives. Persist the sampler, smoothing,
and probability definition; do not estimate $Q$ from validation/test data.
Do not add the training correction directly to inference scores.

Initial settings remain AdamW `lr=1e-3`, `weight_decay=1e-4`, cosine scheduling,
batch size 2048, and at most 10 epochs. Record resource-driven changes; changing
batch size also changes the negative sampling context.

**Training outputs:** Tower weights, frozen maps, references/hashes for the
supplied feature manifests, normalization rules, learned item vectors
`[I,128]`, and their own search ID mapping.

**Inference:** Encode the current profile and latest 20 prior purchases into
`[B,128]`. Use the fixed Item Tower or a compatible cache for `A(T)`, and search
its learned `IndexFlatIP` to return up to 50
`(article_id, twotower_score, twotower_rank)`. Building this index is not
retraining the towers. Item vectors must not depend on the requesting user.

Neither `image_faiss.index` nor `text_faiss.index` can substitute for this
learned index, even if its dimension happens to match. The MVP sends users
without effective history to R1. Missing content uses masks and does not remove
the article from other channels or ground truth.

## 4. Ranking layer: LambdaMART only

### 4.1 Samples, labels, and groups

Generate native candidates and cutoff-safe features before consulting labels.
Do not inject missed positives in Train, Validation, Test, or disguised backfill.

```text
query_key = (cutoff_date, customer_id)
row_key = (cutoff_date, customer_id, article_id)
y = 1 if article_id is in the query's unique future-purchase set else 0
```

Use binary relevance `{0,1}`; repeated purchases in a target week do not become
`y=2`. Zero means unobserved purchase, not an exposed negative impression.

**Training:** Concatenate the four disjoint folds. Keep groups with at least
one native positive and one proxy negative. Preserve all native positives;
select `min(available_negatives, 20 * positives)` negatives. The configured
initial hard share is 0.5: take `floor(0.5 * selected_negatives)` by native RRF,
then sample the remainder without replacement using a stable query-local seed.
RRF is computed without future labels; do not score hardness from future
purchase counts or final test outcomes. The hard share is an explicit previous
implementation clarification, not an experimentally optimal value.

**Validation/Test:** Keep every purchase-active query, without negative sampling
or positive-hit filtering. A zero-candidate query remains in the separate query
manifest and contributes zero to end-to-end metrics.

Verify unique row keys, contiguous groups, and `sum(group_sizes)==len(X)` after
transformations. Sort stably by `(cutoff_date, customer_id, article_id)`.
Customer/article IDs are join/audit keys, not ranker features.

### 4.2 Feature schema and content reuse

The source lists 39 core features, despite calling them 38. Keep the listed
names; add six source masks and four content masks for a proposed **49-column**
default. Read `F` from the versioned schema, never a hardcoded dimension.

| Family | Names and count |
|---|---|
| Recall and consensus · 15 | `pop_score`, `pop_rank`, `repurchase_score`, `repurchase_count`, `days_since_last_buy`, `sibling_score`, `sibling_source_days`, `itemcf_score`, `itemcf_rank`, `wals_score`, `wals_rank`, `twotower_score`, `twotower_rank`, `recall_channel_hit_count`, `rrf_fused_score` |
| Content cross · 4 | `user_item_img_cosine_sim`, `user_item_last1_img_sim`, `user_item_text_cosine_sim`, `item_visual_cluster_id` |
| Item trend · 7 | `item_sales_3d`, `item_sales_7d`, `item_sales_14d`, `item_sales_28d`, `sales_velocity_3d_vs_7d`, `sales_velocity_7d_vs_28d`, `item_repurchase_ratio_28d` |
| User profile · 5 | `user_age`, `sales_channel_preference`, `user_tx_count_30d`, `user_tx_count_90d`, `user_avg_price_90d` |
| User-item affinity · 8 | `price_diff_from_user_avg`, `price_ratio_to_user_avg`, `user_garment_group_affinity`, `user_section_affinity`, `user_colour_affinity`, `item_rank_in_user_age_bucket_7d`, `product_type_no`, `index_group_no` |
| Availability · 10 | `hit_trend`, `hit_repurchase`, `hit_sibling`, `hit_itemcf`, `hit_wals`, `hit_twotower`, `has_item_image`, `has_item_text`, `has_user_image_history`, `has_user_text_history` |

Use the following definitions consistently:

- `rrf_fused_score = sum(1 / (60 + rank_c))` over actual channel hits. It does
  not truncate the union. Missing source scores/ranks are `NaN`, with mask 0;
  never fabricate a rank or mix arbitrary zero/sentinel conventions.
- Content profiles use valid supplied vectors among the latest five purchase
  lines: normalize each vector, average, then normalize the mean. A missing or
  zero-norm profile is unavailable, not silently normalized. `last1` refers to
  the deterministically selected latest history row, not the latest row that
  happens to have an image. A missing image leaves that feature unavailable.
- Use the provided vector arrays for these similarities. No raw image or
  encoder download is required. Missing/failure records produce explicit masks,
  not random replacement vectors or fabricated coverage.
- Fit K-Means `K=128` on valid image vectors of articles observed by the base
  cutoff, then freeze it; later items are transformed only. The input vectors
  are already supplied, but this downstream clustering is a separate learned
  artifact. `item_visual_cluster_id` is categorical, with missing-image handling.
- Velocity features are `(sales_3d/3)/(sales_7d/7+0.1)` and
  `(sales_7d/7)/(sales_28d/28+0.1)`.
- `item_repurchase_ratio_28d` means users buying the item on at least two
  distinct dates divided by distinct users buying it in the window. It is not
  an order ratio; a zero denominator is missing.
- User average price uses 90 days; item average price uses 28 days. Compute
  price differences/ratios only from those past values, preserving dataset
  units rather than assuming a currency.
- `sales_channel_preference` is the user's 90-day fraction of purchase lines
  with `sales_channel_id=2`, without inferring unverified business labels.
- Attribute affinities use matching 90-day purchase-line fractions. The age
  bucket rank uses seven-day bucket sales, with explicit missing behavior.

One `FeatureBuilder` combines cutoff-aware SQL and vector operations. Training
and inference share column order, dtype, categorical maps, availability rules,
and content versions. Do not omit costly inference columns while keeping an
incompatible trained schema.

### 4.3 Training inputs and outputs

**Inputs:** `X_train [N,F]`, numerical float32 and explicitly encoded categorical
columns; `y_train [N]`, integer binary relevance; and `group_sizes [Q]` for
contiguous query rows. Validation uses untrimmed native candidates and aligned
groups.

Keep the existing initial settings: `objective=lambdarank`,
`boosting_type=gbdt`, `n_estimators=1500`, `learning_rate=0.03`, `num_leaves=63`,
`max_depth=6`, `min_child_samples=100`, `subsample=0.8`, `subsample_freq=1`,
`colsample_bytree=0.8`, `reg_alpha=1.0`, `reg_lambda=5.0`, and seed 42.
Configure thread counts explicitly; these are not measured optimal values.

Pass `eval_at=(12,)`, `eval_group`, and categorical columns explicitly.
Candidate NDCG@12 and 50-round early stopping remain optimization diagnostics.
NDCG@12 is not MAP@12; `eval_at` does not by itself redefine the loss truncation.
Select the final configuration using external native MAP@12 with complete
ground truth. Any future end-to-end MAP early-stopping callback must preserve
the independent full query/label manifests.

**Outputs:** LightGBM model, best iteration, feature schema and category maps,
sampling configuration, base/content versions, MLflow run identity, and data
lineage. Ranking scores are not purchase probabilities.

### 4.4 Inference

**Input:** Native candidate features `X_inference [N,F]`, with no labels and no
`group` argument to `predict()`. Preserve outer query/article mappings.

**Output:** One continuous score per row. Within each query, sort by
`score descending, article_id ascending` and return up to 12 unique articles.
Short offline lists are valid; missing positions earn no credit. Never pad
with duplicate articles. A later competition-format backfill must be explicitly
specified and included in offline evaluation before use.

Two-Tower, WALS, and LambdaMART train separately. The ranker does not backpropagate
into the towers or original encoders. Content similarity is a feature, not
another learned ranker.

## 5. Evaluation, repository layout, and implementation

### 5.1 Evaluation contract

See [Evaluation Protocol v2.0](docs/archi/evaluation_protocol.md). For query $q$,
$G_q$ is the unique future seven-day purchase set matched to the article
catalog, not filtered to the active pool, WALS vocabulary, image coverage, or
retrieved candidates. Let $rel_q(j)$ indicate that the rank-$j$ recommendation
is in $G_q$; $Precision@j$ is the relevant fraction through that position.

$$
AP@12(q)=\frac{\sum_{j=1}^{12}Precision@j(q)\,rel_q(j)}{\min(12,|G_q|)},
\qquad MAP@12=\operatorname{mean}_q AP@12(q).
$$

Unfilled positions earn no credit. Average equally over purchase-active
customer/cutoff queries with at least one catalog-matched future purchase.
Count excluded zero-purchase queries when constructing a broader population.
This is not an all-traffic or causal online metric.

Report native MAP@12 as primary; Recall@12, full-ground-truth NDCG@12,
per-channel Recall@K, union Recall, active-pool coverage, actual candidate
counts, and necessary exclusions as supporting evidence. Conditional ranking
metrics have a different denominator and must be labeled separately.

Use the same query, label, pool, and date contracts across comparisons. Default
leave-one-channel-out ablations remove that quota without redistribution;
redistributed budgets are separate configurations. Tune only on global
Validation, lock the configuration, then evaluate Test once.

### 5.2 Existing data layout and embedding acceptance

The following is the **user-reported initial input layout**, not a newly
created target structure. Preserve the literal directory names, including the
space in `original data`.

```text
data/
├── image_features/
│   ├── image_article_ids.parquet
│   ├── image_embedding_failures.parquet
│   ├── image_embeddings.npy
│   ├── image_faiss.index
│   └── image_features_manifest.json
├── original data/
│   ├── articles.csv
│   ├── customers.csv
│   └── transactions_train.csv
└── text_features/
    ├── text_article_ids.parquet
    ├── text_embeddings.npy
    ├── text_faiss.index
    └── text_features_manifest.json
```

Do not rename or move these files to `data/raw/` or `data/embeddings/`, create
parallel copies, regenerate them by default, or require a raw image directory
that was not supplied. Read source files in place and treat them as immutable.
Resolve paths from the repository root using `pathlib.Path`; quote paths with
spaces in shell commands. Do not depend on the notebook's working directory.

#### Artifact roles

| Existing input | Role in the new pipeline |
|---|---|
| Three CSVs in `data/original data/` | Source tables for typed canonicalization |
| `text_embeddings.npy` and `text_article_ids.parquet` | Frozen text vectors and their verified article/row mapping |
| `image_embeddings.npy` and `image_article_ids.parquet` | Frozen image vectors and their verified article/row mapping |
| Two `*_features_manifest.json` files | Source provenance/configuration, interpreted from actual contents |
| `image_embedding_failures.parquet` | Failure/coverage audit input; schema and meaning must be inspected |
| `text_faiss.index`, `image_faiss.index` | Preserve as existing raw-content indexes; not default recall channels or learned tower/factor indexes |

#### Required acceptance checks before consumption

Inspect the real files rather than inferring their schema from names. Record
results under `data/manifests/` without editing the original manifests.

1. Confirm paths and parse both manifests. Record available encoder/checkpoint
   identity, source text fields, preprocessing, dtype, dimension, normalization,
   creation metadata, fingerprints, and any learned postprocessing. Mark absent
   provenance as unknown; do not invent values or quietly select a new encoder.
2. Open arrays with bounded-memory reads, using memory mapping where suitable.
   Verify a two-dimensional `[N,D]` numeric shape, actual dtype, finite values,
   and nonzero usable vectors. Distinguish a sample check from a complete check.
3. Validate ID uniqueness, row counts, catalog joins, and explicit row-index
   semantics. If positional alignment is the source convention, verify and
   record it before relying on it. Never independently sort an ID table without
   applying the identical permutation to its vectors. Preserve ten-character
   article IDs through a documented, tested canonicalization.
4. Join text/image by article ID, not array row number. Audit the image failure
   log against successful mappings; report conflicts and missing modalities.
   Failure records need not be assumed disjoint from later success records
   without knowing their schema. Do not silently delete ground truth or claim
   complete coverage from the existence of an array.
5. Before any existing FAISS index is used, validate its dimension, row/ID
   mapping, `ntotal`, metric, index type, normalization, and source-vector
   fingerprint where available. It is not a dependency for using valid vector
   arrays in the default architecture. A failed optional index check does not
   authorize an extra recall route or automatic source-file replacement.
6. Record the static-content assumption and check any learned transformation's
   fit population. Preparation after the historical prediction date is not by
   itself evidence of behavioral leakage, but a transform trained on future
   interactions cannot be declared cutoff-safe without review. Unknown
   provenance remains an explicit limitation, not a fabricated pass.

A blocking mapping, dimension, or numeric-integrity conflict stops the affected
consumer and produces a report. Ordinary missing-content cases use documented
masks. Do not auto-repair sources or launch embedding regeneration. Model
configuration uses verified `D_text`/`D_image`; maintain original arrays while
any normalized/adapted cache is a separate versioned derived artifact.

### 5.3 Existing documentation and additive output directories

Preserve this user-reported initial documentation layout, including the exact
spelling `achive` and spaces in notebook names:

```text
docs/
├── achive/
│   ├── hm_mvp_full_reports_colab_v1.zip
│   └── mvp_notebooks_colab/
│       ├── 00_Data_Preparation.ipynb
│       ├── 01_CPU_Retrievers.ipynb
│       ├── 01_CPU_Retrievers_v2.ipynb
│       ├── 02_GPU_Retrievers v2.ipynb
│       ├── 02_GPU_Retrievers.ipynb
│       ├── 03_Fusion_Ranking v2.ipynb
│       ├── 03_Fusion_Ranking.ipynb
│       ├── 04_Offline_Evaluation v2.ipynb
│       └── 04_Offline_Evaluation.ipynb
├── hm_mvp_full_reports_colab_v1.md
└── phase0-data download & EDA report.md
```

Add current architecture documents under `docs/archi/`. Do not rename `achive`
to `archive`, require a migration into `docs/references/`, overwrite historical
notebooks, or modify archived outputs to make them appear compatible.
The archived ZIP/notebooks remain reference exceptions inside `docs/`; they
are not a place to write new runtime datasets or model outputs.

Create derived directories only as needed:

```text
repository/
├── AGENTS.md
├── proposal.md
├── README.md                         # Repository setup and verified status
├── pyproject.toml                    # Environment and quality tooling
├── configs/                          # Versioned source configuration
├── data/                             # Existing input directories stay in place
│   ├── canonical/                    # Derived typed Parquet tables
│   ├── processed/                    # Cutoff pools, candidates, labels, features
│   ├── models/                       # WALS, towers, K-Means, LambdaMART
│   ├── indexes/                      # New ItemCF / learned-factor search indexes
│   ├── manifests/                    # Acceptance, splits, IDs, schemas, lineage
│   ├── metrics/                      # Machine-readable evaluation results
│   ├── runs/                         # Local experiment artifacts and logs
│   └── tmp/                          # Bounded scratch space
├── docs/
│   └── archi/                        # Added; existing references remain untouched
│       ├── architecture.md
│       ├── evaluation_protocol.md
│       └── implementation_decisions.md
├── src/hm_recsys/
│   ├── data/                         # Source readers and canonicalization
│   ├── embeddings/                   # Existing-vector loaders, validation, adapters
│   ├── recall/                       # Six logical channel implementations
│   ├── features/                     # Shared training/inference features
│   ├── ranker/                       # LambdaMART only
│   ├── evaluation/                   # Full-query metric computation
│   ├── inference/                    # End-to-end orchestration
│   └── tracking/                     # Manifests and experiment lineage
├── sql/                              # Parameterized past-only queries
├── scripts/                          # Thin CLI entry points
├── tests/                            # Synthetic unit/integration/leakage tests
└── notebooks/                        # New-architecture exploration and analysis
```

This is an additive output/code plan, not proof that those modules exist.
`src/hm_recsys/embeddings/` is a loader/adapter boundary, not an instruction to
extract vectors again. All new machine-readable results and learned artifacts
stay under `data/`, not in `docs/` or root-level `outputs/`/`artifacts/`.
Mount data at the fixed root across environments; never fall back to a personal
absolute path or `/content/data`. Tests use `tmp_path`, not mutable real inputs.

### 5.4 Pipeline and lineage

The default pipeline is:

1. Inspect the repository and validate the supplied CSV/content artifact
   contracts; canonicalize source tables without modifying them.
2. Validate and freeze the non-overlapping split/query manifests.
3. Bind existing text/image vectors by verified article mappings; fit only
   permitted downstream transformations and frozen base retrieval models.
4. For each cutoff, build the active pool, dynamic R1–R4, WALS folding-in,
   Two-Tower queries, and the native candidate union.
5. Build shared features; attach labels separately; sample negatives in ranker
   Train only; fit LambdaMART; select using global Validation.
6. Lock configuration and immutable artifact identities; run final Test once.

There is **no default download, image-decode, or content-embedding extraction
stage**. Creating learned 128d Two-Tower item vectors, K-Means assignments, or
WALS factors is downstream model work, not repetition of source preprocessing.

CLI entry points take an explicit cutoff/split and call reusable `src/` code.
Generating past-only test features does not authorize fitting on test labels.
Load immutable run/release identities rather than a changing model alias at
final evaluation time.

DVC tracks large artifacts; Git does not contain customer histories, vectors,
weights, or candidate matrices. MLflow/manifests record code revision, seeds,
package versions, data and content fingerprints, dimensions, ID-map hashes,
source manifests, model versions, and feature schema. Keep source and derived
artifact lineage distinct.

Use bounded DuckDB/Arrow/Parquet batches. Do not create a dense user-item matrix,
unbounded all-customer pandas table, or per-request full-catalog image encoding.
Report resource limits rather than silently shrinking Validation/Test queries.

### 5.5 Implementation phases and acceptance gates

| Phase | Deliverable | Exit criterion |
|---|---|---|
| P0 | Input inventory and evidence reconciliation | Record existing artifacts as user-reported; distinguish supplied files, checked integrity, and historical Phase 0 facts |
| P1 | Canonical tables; embedding readers/acceptance report; split/query/pool manifests | Source mappings and dimensions verified; exact disjoint dates and leakage/metric tests pass |
| P2 | R1–R4 and unified retrieval interface | Correct cutoffs, reloadable artifacts, actual counts/Recall outputs |
| P3 | WALS, Two-Tower, and frozen visual clustering using existing content | Fixed mappings, safe sampler, internal validation, learned-index lineage, reload parity; no default re-extraction |
| P4 | Shared feature schema and native-only LambdaMART training | No injection; Train-only sampling; full-query evaluation |
| P5 | Batch inference and optional local serving | Fresh-process reload, feature parity, missing-state and fallback tests |
| P6 | Locked configuration and final Test report | No test tuning; traceable runs; measured facts separated from plans |

Scope each Codex task to its assigned phase. Synthetic tests are not H&M
benchmarks, and code presence is not verified quality, cost, or deployment.

## 6. Risks, experiments, and completion criteria

Prioritize active-pool coverage, incremental Repurchase/Sibling contribution,
ItemCF/WALS complementarity, and Two-Tower content ablations. Precomputed
content availability does not establish predictive benefit; poor raw-content
similarity does not establish that supervised multimodal features are useless.

The data do not establish historical inventory, sizes, orders, exposure,
returns, or point-in-time customer attributes. Content provenance and static
snapshot assumptions must remain explicit. The active pool limits cold-item
coverage even if a content branch can encode those items.

Completion requires reproducible input/split manifests, six actual retrieval
outputs, a native-only ranker, full-ground-truth Validation/Test metrics,
necessary ablations, reload/feature-parity tests, and honestly reported Ruff,
Pyright, and pytest results. Follow the English Python, notebook, and Markdown
standards in [AGENTS.md](AGENTS.md).

**This revision updates documentation only. It does not claim to have opened
the user's local CSV/NPY/Parquet/FAISS/JSON artifacts, validated their contents,
executed archived notebooks, regenerated embeddings, trained models, or run
H&M evaluation.**
