# Phase 0 — Data Download & Exploratory Data Analysis Report

**Project:** H&M Personalized Fashion Recommendation System  
**Dataset:** Kaggle — H&M Personalized Fashion Recommendations  
**Report date:** 2026-10-29  
**Status:** Core tabular EDA complete; local image filename coverage measured; independent official image-inventory and image-decode validation outstanding.  
**Scope:** Data acquisition, quality, descriptive analysis, and feasibility assessment. **No model training or frozen evaluation split was performed.**

## 1. Executive summary

The Phase 0 analysis confirms that H&M provides a substantially connected purchase-history dataset suitable for further investigation of personalized next-seven-day article recommendations. The audited files contain **31,788,324 purchase lines**, **1,371,980 catalog customers**, and **105,542 catalog articles**. The transaction table covers **2018-09-20 through 2020-09-22**, contains **27,306,439 distinct customer–article pairs**, and joins completely to both customer and article catalogs. Article and customer primary keys are unique, and article descriptive fields are largely populated.

The subsequent local image audit found **105,100 valid, uniquely mapped JPG image filenames**, covering **99.5812% of catalog articles** and **99.5701% of recorded purchase lines**. These are **file-presence** measurements conditional on the user's confirmation of a complete local Kaggle download. They do **not** certify a complete official Kaggle image inventory, successful decoding of every file, or generated embeddings.

**Phase 0 decision:** Proceed to formalize the canonical data layer and chronological offline evaluation protocol. Do not yet freeze model lanes, candidate budgets, a feature-store platform, or a production deployment architecture.

## 2. Sources and reproducibility

- Source competition: <https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data>
- Executed notebook: `01_Data_Dowload&EDA.ipynb` (existing spelling retained).
- Additional executed image coverage cells and pasted category-segment outputs provided after the notebook run.
- Local Colab working directory: `/content/data/`.
- Google Drive backup destination: `/content/drive/MyDrive/projects/e_commerce_recsys/data/original data/`.
- Tabular EDA aggregates written to `/content/eda_outputs/` as CSV, with a JSON gate summary. Additional image audit outputs include `image_inventory.parquet`, `image_coverage_summary.csv`, image coverage segmentation CSVs, and `image_audit_summary.json`.

The KaggleHub run reported a **28.7 GB archive download**, followed by extraction. The notebook records moving the data into `/content/data/`, successfully verifying four CSV copies with matching SHA-256 checksums, and copying and verifying an `images.tar` archive to Google Drive. These observations reflect the logged Colab session, not an independent re-download.

**Download inventory:** `articles.csv`, `customers.csv`, `transactions_train.csv`, `sample_submission.csv`, and `images/` (archived as `images.tar` for backup). DuckDB queried CSV-backed views rather than loading the full transaction table into pandas. Identifier columns were read as strings to preserve leading zeros.

## 3. Key dataset metrics

| Metric | Observed value | Interpretation |
|---|---:|---|
| Article catalog rows / unique IDs | 105,542 / 105,542 | Catalog SKU-level items |
| Customer catalog rows / unique IDs | 1,371,980 / 1,371,980 | Unique customer records |
| Purchase lines | 31,788,324 | Line records, **not** unique user-item edges |
| Customers appearing in transactions | 1,362,281 | Purchasers observed in transaction history |
| Articles appearing in transactions | 104,547 | 995 catalog articles have no observed purchase |
| Unique customer–article pairs | 27,306,439 | Distinct interaction edges |
| Unique customer–article–day keys | 28,575,395 | Distinct day-level purchase keys |
| Exact duplicate transaction rows | 2,974,905 | Preserve: may reflect multiple units/purchases |
| Transaction date range | 2018-09-20 to 2020-09-22 | No malformed or missing dates found |
| Customer and article transaction join coverage | 100% / 100% | No unmatched transaction IDs |

**Important grain distinction:** A purchase line, a distinct customer–article pair, and a distinct customer–article–day key are different units. Without a quantity or order ID, identical lines cannot safely be classified as erroneous duplicates. The EDA performed **no automatic transaction deduplication**.

## 4. Article catalog quality

- **105,542 rows**, all with unique, nonblank, ten-digit string `article_id` values; **zero** duplicate IDs, conflicting IDs, or exact duplicate article rows.
- **25 fields** inspected. `detail_desc` is missing for **416 articles (0.394%)**; the remaining inspected article columns recorded zero null/blank values under the notebook's checks. These checks do not prove all category labels are semantically valid (`Unknown` is still a value).
- Product taxonomy is rich: **131 product types**, **19 product groups**, **56 sections**, and **21 garment groups**. Examples of leading product types: Trousers (11,169), Dress (10,362), Sweater (9,302), T-shirt (7,904).
- `product_code` groups multiple `article_id` variants. The distribution includes **24,066 product codes with one article** and **11,921 with two articles**, with groups extending to larger variant counts. `product_code` should not be substituted for the article-level recommendation key.

**Implications:** Article metadata supports item grouping and investigation of text-based retrieval. The article identifier remains the initial prediction unit; style-level concentration/diversity should be measured later, not assumed to be a problem already solved.

## 5. Customer catalog quality and privacy

- **1,371,980 rows / unique customers**; no null, blank, duplicate, or conflicting primary keys.
- Major missingness rates: `Active` **66.151%**, `FN` **65.238%**, `fashion_news_frequency` **1.167%**, `age` **1.156%**, and `club_member_status` **0.442%**.
- Age band **20–29** is the largest observed band (**528,358** customers). The notebook also checks age group distributions and potentially suspicious values; no claim is made that all ages or attributes have validated business semantics.
- Most frequent recorded membership status: `ACTIVE` (**1,272,491** customers). The most frequent fashion-news category is `NONE` (**877,711**).
- **352,899** distinct anonymized postal-code values; only cardinality was reported. No customer IDs or postal-code values are included in this report.

**Interpretation:** Missing `Active` or `FN` must not automatically be interpreted as inactive or unsubscribed. Use null indicators or document validated meanings before modeling. Customer-level raw exports should remain private and out of public repositories.

## 6. Transaction behavior, pricing, and integrity

### 6.1 Interaction concentration

| Distribution of purchase-line counts | Per purchasing customer | Per purchased article |
|---|---:|---:|
| Median (p50) | 9 | 65 |
| p90 | 60 | 797 |
| p99 | 187 | 3,220 |
| Maximum | 1,895 | 50,287 |

These are **purchase-line** counts rather than distinct neighboring users/items. They nevertheless show substantially repeated behavior and heavy-tailed demand, which merits full-graph collaborative-retrieval experiments.

- **3,515,413** customer–article pairs have repeated purchase lines, with **4,481,885** lines beyond the first occurrence per repeated pair.
- **2,717,844** customer–article–day keys have repeated lines, with **3,212,929** excess lines beyond the first per repeated day-level key.
- **914,308** purchasers made purchases on multiple distinct dates (**67.116%** of purchasing customers).
- **676,044** purchasers have at least ten observed purchase lines over the full period; this is a descriptive whole-period statistic, **not** an admissible warm-user cohort for a historical cutoff.

### 6.2 Prices and sales channels

The raw `price` values have no confirmed currency in this analysis. No missing, invalid numeric, nonfinite, negative, or zero prices were detected. Observed median price is **0.025364**, p99 is **0.096495**, and maximum is **0.591525** in dataset units. `sales_channel_id=2` appears in **22,379,862** lines and `sales_channel_id=1` in **9,408,462**; business-channel meanings were not inferred from the numeric codes.

### 6.3 Temporal distribution and integrity

Monthly and weekly purchase counts were saved as aggregates. The dataset has no malformed transaction dates and no transaction lines with customer IDs absent from `customers.csv` or article IDs absent from `articles.csv` (**100% join coverage** for both). This validates referential integrity for the supplied snapshots; it does **not** establish whether any article was available for sale at a particular past date.

## 7. Diagnostic next-seven-day feasibility (not a final split)

A diagnostic window from **2020-09-16 through 2020-09-22** gives:

| Diagnostic metric | Result |
|---|---:|
| Customers with a purchase in the seven-day window | 68,984 |
| Those with a purchase before the window | 63,412 |
| Those with at least 10 prior purchase lines | 54,242 |
| Purchase lines within the window | 240,311 |
| Prior-history median among returning purchasers | 40 lines |
| Prior-history p90 among returning purchasers | 148 lines |

**63,412 / 68,984 ≈ 91.9%** of purchasers in this diagnostic window have some prior history. This indicates a practical warm-history evaluation population exists. However, the diagnostic uses the final observed seven days and is **not** a frozen validation/test design, a prediction result, or an estimate of achievable MAP@12.

The competition's target is future seven-day purchases and its primary metric is **MAP@12**. Formal split dates, active-query population, historical item eligibility, and whether repeated purchases count once or multiple times in the label representation remain open decisions.

## 8. Image inventory and coverage audit

### 8.1 Local filename inventory

| Inventory check | Measured result |
|---|---:|
| Total local image files | 105,100 |
| Valid JPG files with ten-digit article ID stems | 105,100 |
| Invalid extension or filename stems | 0 |
| Distinct image article IDs | 105,100 |
| Article IDs mapped to multiple image files | 0 |
| Duplicate image mappings | 0 |
| Image IDs absent from article catalog | 0 |

### 8.2 Coverage based on the confirmed local download

| Coverage metric | Numerator / denominator | Percentage |
|---|---|---:|
| Catalog articles with a matching local image filename | 105,100 / 105,542 | **99.5812%** |
| Catalog articles without a matching image filename | 442 / 105,542 | **0.4188%** |
| Purchase lines for articles with image filenames | 31,651,678 / 31,788,324 | **99.5701%** |
| Purchase lines for articles without image filenames | 136,646 / 31,788,324 | **0.4299%** |

By transaction-observed status, **104,106 / 104,547 (99.578%)** historically purchased articles have an image filename; among **995** never-purchased catalog articles, **994 (99.899%)** do.

### 8.3 Coverage by major product group

| Product group | Articles | Missing images | Filename coverage |
|---|---:|---:|---:|
| Garment Upper body | 42,741 | 61 | 99.857% |
| Garment Lower body | 19,812 | 42 | 99.788% |
| Garment Full body | 13,292 | 16 | 99.880% |
| Accessories | 11,158 | 153 | 98.629% |
| Shoes | 5,283 | 127 | 97.596% |
| Underwear | 5,490 | 28 | 99.490% |

Accessories and Shoes account for **280 of 442** missing image mappings. Coverage was also grouped by `product_type_name`, `section_name`, and transaction-observed status; the corresponding complete outputs are in `/content/eda_outputs/`.

### 8.4 Audit evidence and limits

The *original notebook* recorded `image_coverage_status=NOT_MEASURED` because no complete archive was available for an independent comparison and the local completeness switch was not enabled. **After the original notebook run**, the user confirmed the extracted local dataset was complete and ran additional image-mapping cells, obtaining `MEASURED_CONFIRMED_LOCAL_DOWNLOAD` and the values above. The later audit therefore supersedes **only the earlier image-coverage status and related measurements**.

This status represents a **user-attested complete local download**, not independently verified equality to a paginated official Kaggle image filename inventory. Image **filename presence is not evidence of image decodability**; bounded decode sampling and actual CLIP embedding creation remain unperformed. The dataset's provided article metadata and images are snapshots, not validated point-in-time historical inventories.

## 9. Modeling and data-management implications (provisional)

1. **Keep full historical interaction structure:** Avoid random user sampling that severs user–item graph connectivity. Use chronological pre-cutoff history for training, while defining warm/cold evaluation cohorts separately.
2. **Treat purchase as implicit feedback:** A missing purchase is unobserved, not an explicit negative; keep transaction-line frequency separate from unique user–article edges.
3. **Start with article-level targets:** Preserve ten-digit string `article_id`, and retain `product_code` for optional style-level diagnostics or controls.
4. **Use temporal correctness as a first-class contract:** Fit interaction-based features, popularity, and retrieval models only on information available before a given prediction cutoff. Document the static-catalog-metadata assumption.
5. **Consider text and visual retrieval viable candidates:** Article text is well populated and local image filename coverage is high. Neither observation establishes offline retrieval uplift, image decode success, embedding quality, or latency/cost feasibility.
6. **Prefer reproducible minimal infrastructure:** Next create typed, versioned canonical Parquet datasets and a frozen evaluation protocol before selecting a heavyweight feature-store platform or cloud serving architecture.

## 10. Phase 0 completion checklist and Phase 1 handoff

| Gate / deliverable | Status |
|---|---|
| Download and local extraction | Complete in logged Colab run |
| Google Drive CSV and image archive backup | Copy and checksum verification logged |
| Article IDs, schema, missingness, taxonomy | Complete |
| Customer IDs, missingness, privacy-safe aggregates | Complete |
| Transaction dates, repeat behavior, pricing, referential integrity | Complete |
| Diagnostic next-seven-day feasibility | Complete; not a formal split |
| Full local image filename enumeration and article mapping | Complete |
| Local image filename presence and transaction-weighted coverage | Measured with user-attested download completeness |
| Independent official Kaggle image inventory comparison | Not completed |
| Image decode quality / embedding coverage | Not completed |
| Frozen chronological train/validation/test | Not completed |
| Models, ranking, performance benchmark | Not started |

**Phase 1 priorities:** (a) define immutable canonical table schemas and versioned Parquet transforms; (b) freeze a leak-resistant chronological query/label/evaluation protocol with MAP@12; (c) define catalog eligibility and cold/warm cohorts; (d) validate data and feature snapshots across cutoffs; (e) separately smoke-test a bounded image decode sample before committing to visual feature extraction.

## 11. Known limitations

- No independent audit against the complete official Kaggle image filename list.
- No full-image decode, image-quality, or embedding audit.
- No reliable impression, cart, inventory, or point-in-time catalog availability logs.
- No established real-world price currency or full customer attribute business semantics.
- No historical train/test split; the final-seven-day calculation was diagnostic only.
- No trained models, Recall@K, NDCG@K, MAP@12, or online effect claims.

**Overall finding:** The audited H&M dataset has clean join keys, meaningful repeat purchasing, extensive article metadata, and excellent local image filename presence. It passes the initial **tabular data readiness and evaluation-feasibility review**. Remaining image verification and temporal evaluation design should be explicitly addressed as engineering gates, not silently treated as complete.
