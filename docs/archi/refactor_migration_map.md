# Single-Source Refactor: File and API Migration Map

**Run record:** `single_source_20261103_01`  
**Scope:** the actual working tree before this refactor, including uncommitted
user changes; this is not a map restored from an old release or Git HEAD.  
**Algorithm contract:** proposal v0.3.2 / Evaluation Protocol v2.0 unchanged.

## Contents

- [1. Evidence and classification](#1-evidence-and-classification)
- [2. Exhaustive prior-file map](#2-exhaustive-prior-file-map)
- [3. Split and consolidated APIs](#3-split-and-consolidated-apis)
- [4. Import and caller provenance](#4-import-and-caller-provenance)
- [5. Archive byte preservation and Git index](#5-archive-byte-preservation-and-git-index)
- [6. Mapped active destination fingerprints](#6-mapped-active-destination-fingerprints)
- [7. Documentation snapshots](#7-documentation-snapshots)
- [8. Privacy and acceptance boundaries](#8-privacy-and-acceptance-boundaries)

## 1. Evidence and classification

The local evidence root is
`data/runs/refactor/single_source_20261103_01/`. It is an ignored runtime record,
not a public checkout dependency. The map was built from:

| Evidence | Purpose |
|---|---|
| `before_inventory.json` | All prior source/script/test/config/Notebook paths, original sizes/SHA256 and static project imports |
| `before_source/<old_path>` | Exact pre-edit copies, including the user's uncommitted source changes |
| `initial_path_mapping.json` | Mechanical first-pass module moves and original hashes; corrected below where the final split/archive differs |
| `before_docs/` | Exact README/AGENTS/architecture/runbook copies before their edits; also current test copies before test-specific migration |
| `archive_untracking.json` | Explicit index-only removal record for ten pre-existing archive paths, with worktree presence/size/hash |
| Current source AST and archive bytes | Final modules/functions/callers and preserved archive destinations |

**Independently checked while writing this map:** all **83** inventory entries
match their `before_source` SHA256; all **47** initial mapping hashes agree with
that inventory. None was replaced with a later Git/archive version. The
inventory comprises **51 source files, six scripts, 17 tests, five configs and
four Notebooks**. Every one appears once in Section 2.

| Classification | Meaning | Prior-file count |
|---|---|---:|
| `ACTIVE_CANONICAL` | Retained maintained source/test/config path; imports/docs/contracts may be updated. | 20 |
| `ACTIVE_ENTRYPOINT` | Retained thin CLI; model/math implementations live in canonical packages. | 5 |
| `REFACTOR_MERGE` | Needed behavior moved/split/consolidated into named canonical destinations; no old-namespace module wrapper remains. | 46 |
| `ARCHIVE_ONLY` | Superseded interface/client/namespace; original evidence preserved locally, not imported. | 9 |
| `HISTORICAL_REFERENCE` | Historical exploratory/extraction Notebook; not the active training/input-generation interface. | 3 |

`PRIVATE_DATA_DO_NOT_MOVE` applies outside this source inventory to the existing
`data/original data/`, `data/text_features/`, `data/image_features/` and completed
run/model/checkpoint trees. They are not refactor source files or archive
migration inputs. Existing private full-run material under `docs/achive/`
remains local-only and was not moved into an active/public tree. This map does
not claim an exhaustive new byte audit of those large runtime inputs.

The initial `runtime/legacy_*`, ZIP bootstrap and release destinations were
intermediate migration steps, not final active compatibility shims. Their
functional replacements are `runtime.local_cli`, `runtime.modular` and the
pinned-Git client. The final table supersedes the initial destination column
while preserving its original hashes.

## 2. Exhaustive prior-file map

Each SHA256 is the **pre-refactor file's** hash, not the current destination's
hash. Section 6 lists current mapped active hashes. Multiple destinations mean
an explicit split; they do not mean parallel profile-specific implementations.
Archive paths are local code-formatted locations, intentionally not public
runtime dependencies. A new root Notebook replaces the old training client;
its file role is documented in the README rather than misrepresented as a
byte-preserving rename.

### 2.1 Python source

| Prior path | Classification | Final canonical or original archive destination(s) | Original bytes | Original SHA256 |
|---|---|---|---:|---|
| `src/hm_recsys/__init__.py` | `ACTIVE_CANONICAL` | `src/hm_recsys/__init__.py` | 63 | `401b63bcbf561cad770d2d7a387dbf0068c786022649a2f70d35b0b2c228165c` |
| `src/hm_recsys/cli.py` | `REFACTOR_MERGE` | `src/hm_recsys/runtime/local_cli.py` | 16,372 | `940c47da7f5575f9c5c4564b7686daf71282c907a5cdbf2fdce4b5ced4cfaaf6` |
| `src/hm_recsys/colab/__init__.py` | `ARCHIVE_ONLY` | `docs/achive/single_source_refactor_20261103_01/source_before/src/hm_recsys/colab/__init__.py` | 73 | `2c88d6443ad637d07b0f65a43271c26c10934262217eb02007be94f3e9e88af3` |
| `src/hm_recsys/colab/bootstrap.py` | `ARCHIVE_ONLY` | `docs/achive/single_source_refactor_20261103_01/source_before/src/hm_recsys/colab/bootstrap.py` | 13,825 | `7a5a9231f2af7d2b23762c0adc9b9ba7f30aebb0115d10c5ef8550031fcce709` |
| `src/hm_recsys/colab/checkpoints.py` | `REFACTOR_MERGE` | `src/hm_recsys/runtime/checkpoints.py` | 17,499 | `df3f5fed84b929fb095191a464466484913d7ff41267cbdbab4540b2aab1e58b` |
| `src/hm_recsys/colab/cli.py` | `REFACTOR_MERGE` | `src/hm_recsys/runtime/cli.py` | 17,913 | `d7c07681dcafbeff2adc79a5f7c8d62b7afad9e014f8f0ba3a6093f28fc57e63` |
| `src/hm_recsys/colab/device.py` | `REFACTOR_MERGE` | `src/hm_recsys/runtime/device.py` | 10,329 | `07a8c58a79927b09c6cd2b200980e5a58ad8aec95d3e00554fbfc8b4088ac7de` |
| `src/hm_recsys/colab/files.py` | `REFACTOR_MERGE` | `src/hm_recsys/runtime/files.py` | 35,809 | `7abc3ed16aae4ba2b8a235568cfc515d7ecd497f76d9e9a00ad593aa532dd58f` |
| `src/hm_recsys/colab/git_checkout.py` | `REFACTOR_MERGE` | `src/hm_recsys/runtime/git_checkout.py` | 25,267 | `c5ca7a95f5010e392d2b4913716768a30c28ecf9275e9de9e0d1ef34e5d47a3f` |
| `src/hm_recsys/colab/git_safety.py` | `REFACTOR_MERGE` | `src/hm_recsys/runtime/git_safety.py` | 24,940 | `bf205456cbc143a3799ddc2f5c7a75b6eb0c58c50a33bbfc3998f1478f791b6b` |
| `src/hm_recsys/colab/notebook.py` | `REFACTOR_MERGE` | `src/hm_recsys/runtime/notebook.py` | 13,708 | `04f522cd0c4215b2740b2e678733dd711631491a56bfe92d3315db4f87974a2e` |
| `src/hm_recsys/colab/profiles.py` | `REFACTOR_MERGE` | `src/hm_recsys/runtime/profiles.py` | 22,401 | `207949fcc6328f05e7a2d150f5adbcacd052fcc86df7726de0af109552a6b164` |
| `src/hm_recsys/colab/release.py` | `ARCHIVE_ONLY` | `docs/achive/single_source_refactor_20261103_01/source_before/src/hm_recsys/colab/release.py` | 15,915 | `adf77d7cd234324f025a916031823f20bdbafdbc429ff1341e752aa2ddc1d98c` |
| `src/hm_recsys/colab/review.py` | `REFACTOR_MERGE` | `src/hm_recsys/runtime/review.py` | 8,969 | `380a4382113c7e836828d47319b145d0245403b819d61fc48bd838c499a7a483` |
| `src/hm_recsys/colab/runner.py` | `REFACTOR_MERGE` | `src/hm_recsys/runtime/runner.py`; `src/hm_recsys/runtime/modular.py` | 37,691 | `3552b558a939da73b5124550641c1526dbd6809e094b32a59092448d33e206b2` |
| `src/hm_recsys/colab/state.py` | `REFACTOR_MERGE` | `src/hm_recsys/runtime/state.py` | 35,114 | `7d412fa64081063713e84247a43aa6b021f8728068ff93365c7bf928d97ef16f` |
| `src/hm_recsys/colab/worker.py` | `REFACTOR_MERGE` | `src/hm_recsys/runtime/worker.py` | 8,930 | `1ff82d8fd1b8aa4938a00c3f751a71e3e266579ce57d4aed15793c546b6e1ded` |
| `src/hm_recsys/compute/__init__.py` | `ARCHIVE_ONLY` | `docs/achive/single_source_refactor_20261103_01/source_before/src/hm_recsys/compute/__init__.py` | 85 | `25f09e02403f85b30aff13ac3ec3e1d42d57d63cf5cd629b0019f9692bd27c42` |
| `src/hm_recsys/compute/benchmark.py` | `REFACTOR_MERGE` | `src/hm_recsys/runtime/benchmark.py` | 71,398 | `4d0d0f1a8567ed2403ab9ab62510c11af89066be7773c8234260783bc422d530` |
| `src/hm_recsys/compute/feature_benchmark.py` | `REFACTOR_MERGE` | `src/hm_recsys/features/benchmark.py` | 17,670 | `65583a5487d64baa94362e4516258b5cbb23f2cfd262f52d640341fbfe75fb10` |
| `src/hm_recsys/compute/prepared.py` | `REFACTOR_MERGE` | `src/hm_recsys/retrieval/two_tower/prepared.py` | 38,798 | `3bf833095db89def83c1056e8410cc5bb26b16e9ef5f544242b99037a316b6e8` |
| `src/hm_recsys/compute/query_inputs.py` | `REFACTOR_MERGE` | `src/hm_recsys/retrieval/two_tower/query_inputs.py` | 18,754 | `e4eb8127056eb4fdaf937316a400e5b68ea64377e59d21307b41a436c1405645` |
| `src/hm_recsys/compute/runtime.py` | `REFACTOR_MERGE` | `src/hm_recsys/retrieval/two_tower/execution.py` | 24,919 | `5fa365393e93f1352948e27052d59e76bbbe41ab8e9609ca4e92d8058502506f` |
| `src/hm_recsys/compute/settings.py` | `REFACTOR_MERGE` | `src/hm_recsys/runtime/settings.py` | 5,962 | `89d76ea6ddd9b584ecd74f152081684d09ba9d2f5e022fc43ec43a074befd619` |
| `src/hm_recsys/compute/shards.py` | `REFACTOR_MERGE` | `src/hm_recsys/features/storage.py` | 7,435 | `5938bbf920a196eff7dafdb366d6bb40ca6b3d75c0c8e01dd9cacf99248f6287` |
| `src/hm_recsys/content.py` | `REFACTOR_MERGE` | `src/hm_recsys/features/content.py` | 17,781 | `49a843aef0b6237aefdb8762ba39298ab302b16bf7fe2b456b6e4edb7dbff534` |
| `src/hm_recsys/contracts.py` | `REFACTOR_MERGE` | `src/hm_recsys/core/contracts.py` | 18,131 | `48b12f96ea6fa1034bce1f80026ea0c0c5818919a09af44e46a774f6f65f8a8d` |
| `src/hm_recsys/data.py` | `REFACTOR_MERGE` | `src/hm_recsys/data/sample.py`; `src/hm_recsys/data/history.py`; `src/hm_recsys/data/streaming.py` | 29,124 | `62ff21fb65080cc8c3539a6221ab2ba167d89c5db92b7c6b1309304ad08b454d` |
| `src/hm_recsys/evaluation.py` | `REFACTOR_MERGE` | `src/hm_recsys/evaluation/metrics.py` | 5,832 | `0a588c017c5029ce241cdb192cde97b47ae7335bb2da1629a8c73a26e538cf3f` |
| `src/hm_recsys/features.py` | `REFACTOR_MERGE` | `src/hm_recsys/features/builder.py`; `src/hm_recsys/features/preprocessing.py` | 60,168 | `f71402ef96f13a8dbe1d44e3689537a3405c27470dcef49928ee85fdaab62011` |
| `src/hm_recsys/learned.py` | `REFACTOR_MERGE` | `src/hm_recsys/retrieval/wals/model.py`; `src/hm_recsys/retrieval/learned.py`; `src/hm_recsys/retrieval/vectors.py` | 32,590 | `00122022ccb8cb05f12a1cdf46ecf4e74dc4dcef11f62aafcfa3fa272b02b3e0` |
| `src/hm_recsys/pilot/__init__.py` | `ARCHIVE_ONLY` | `docs/achive/single_source_refactor_20261103_01/source_before/src/hm_recsys/pilot/__init__.py` | 80 | `de89fe514d2dac2c0517de7839c31c54e16d4b5855266ee3d176f4a70a966185` |
| `src/hm_recsys/pilot/capacity.py` | `REFACTOR_MERGE` | `src/hm_recsys/runtime/capacity.py` | 6,751 | `90078bd5276cbfdf521bde6c7967b2fe362b228e53012bc70346f7a2f35f9c66` |
| `src/hm_recsys/pilot/data.py` | `REFACTOR_MERGE` | `src/hm_recsys/data/history.py` | 51,770 | `d7039f7b580f197d0ce5da30963fe7968656db4149e66f6f8f8c1d8fd82a9037` |
| `src/hm_recsys/pilot/features.py` | `REFACTOR_MERGE` | `src/hm_recsys/features/pipeline.py`; `src/hm_recsys/retrieval/learned.py`; `src/hm_recsys/retrieval/pipeline.py`; `src/hm_recsys/fusion/pipeline.py` | 35,301 | `0df6c9aac7bb51d9117da0b89e0f99e7bfd39419fac2a12362bf3b7b33296861` |
| `src/hm_recsys/pilot/models.py` | `REFACTOR_MERGE` | `src/hm_recsys/retrieval/wals/training.py`; `src/hm_recsys/features/preprocessing.py` | 23,056 | `e8ab6b8f1975c014c77d02d4dedf96be195b8432f2081f4ccd00e468eda85103` |
| `src/hm_recsys/pilot/ranking.py` | `REFACTOR_MERGE` | `src/hm_recsys/ranking/pipeline.py` | 53,213 | `d9293de4234026525b9c919059df18cfa1a4d3da841b908428055e682e62299b` |
| `src/hm_recsys/pilot/snapshot.py` | `REFACTOR_MERGE` | `src/hm_recsys/retrieval/snapshot.py`; `src/hm_recsys/retrieval/itemcf.py` | 26,051 | `acbfb93c21e65b7100595f9b03e285b5a0f696d01f456f6ee5093786bbef4d92` |
| `src/hm_recsys/pilot/streaming.py` | `REFACTOR_MERGE` | `src/hm_recsys/data/streaming.py` | 20,994 | `2e28b1979c81dc13d4a085f3f313adb1276c352cd90f3871b597616f1a84b540` |
| `src/hm_recsys/pilot/testing.py` | `REFACTOR_MERGE` | `src/hm_recsys/runtime/testing.py` | 17,668 | `3a793e387b3b1a0142d4ad46274a2cf9fbe809463bca33dda14871113f437768` |
| `src/hm_recsys/pilot/towers.py` | `REFACTOR_MERGE` | `src/hm_recsys/retrieval/two_tower/training.py`; `src/hm_recsys/retrieval/two_tower/inputs.py`; `src/hm_recsys/retrieval/two_tower/inference.py` | 87,632 | `906aef5709391765c039ae14f2229743fe84375b78b2bb8766680c073e366792` |
| `src/hm_recsys/pilot/vectors.py` | `REFACTOR_MERGE` | `src/hm_recsys/retrieval/vectors.py` | 17,373 | `b9f4afd35227fcf63e12d24971443bd79ed49831f3f9c0fc29fa502b6f443392` |
| `src/hm_recsys/pilot/worker.py` | `REFACTOR_MERGE` | `src/hm_recsys/runtime/stages.py` | 21,986 | `745d79f62938a0dab809c38c3f2b2eacd348abe86c7a20a3f63c64d7db59a657` |
| `src/hm_recsys/pipeline.py` | `REFACTOR_MERGE` | `src/hm_recsys/runtime/acceptance.py`; `src/hm_recsys/runtime/modular.py`; `src/hm_recsys/features/pipeline.py` | 40,237 | `7fbdc75e7e967cc85a19a812dea2cdcb58dcc28d2edec86048439569b02ab71d` |
| `src/hm_recsys/progress.py` | `REFACTOR_MERGE` | `src/hm_recsys/core/progress.py` | 10,115 | `02f26b886879cc24a74ab0f15992f0e1d9ecf2e52ea85b00403fcc68410bf373` |
| `src/hm_recsys/ranking.py` | `REFACTOR_MERGE` | `src/hm_recsys/ranking/model.py` | 24,794 | `68b1978a1e6a71565f6e763c3971b3cae939123e4528c446e78f24ba51733f87` |
| `src/hm_recsys/recall.py` | `REFACTOR_MERGE` | `src/hm_recsys/retrieval/base.py`; `src/hm_recsys/fusion/native.py`; `src/hm_recsys/retrieval/snapshot.py`; `src/hm_recsys/retrieval/trend.py`; `src/hm_recsys/retrieval/repurchase.py`; `src/hm_recsys/retrieval/sibling.py`; `src/hm_recsys/retrieval/itemcf.py` | 36,184 | `933d0311e0b0197f5de79aef1c76e35b4629f2a0a169b3e8f84e3b744f80a856` |
| `src/hm_recsys/resources.py` | `REFACTOR_MERGE` | `src/hm_recsys/runtime/resources.py` | 19,903 | `6daeff645d6d9fb6f008702a1852fce4c806e1bd2bd501e65630ee2daa8c7ddd` |
| `src/hm_recsys/synthetic.py` | `REFACTOR_MERGE` | `src/hm_recsys/core/synthetic.py` | 12,641 | `5d45cc94daaf2b8dac0746fa860f5d797fc57c96399582bd6e52ff00746d8512` |
| `src/hm_recsys/towers.py` | `REFACTOR_MERGE` | `src/hm_recsys/retrieval/two_tower/model.py`; `src/hm_recsys/retrieval/two_tower/inputs.py`; `src/hm_recsys/data/streaming.py`; `src/hm_recsys/retrieval/two_tower/training.py` | 54,157 | `327944cd9ea08c3eb5be25f8985401700c638286843577f9a197e9b61d65fd97` |
| `src/hm_recsys/worker.py` | `REFACTOR_MERGE` | `src/hm_recsys/runtime/worker.py` | 11,640 | `3b8450eff5b32026a3fe12ed019eb37ac7f9a2d0b781f280b69e081ae6ac068b` |

### 2.2 Thin entry points

| Prior path | Classification | Final canonical or original archive destination(s) | Original bytes | Original SHA256 |
|---|---|---|---:|---|
| `scripts/audit_git.py` | `ACTIVE_ENTRYPOINT` | `scripts/audit_git.py` | 4,403 | `7934e59109425d0e8c38db6fc3113bdc8d65ca2896d22543a2acb4a89ff12fd3` |
| `scripts/benchmark_compute.py` | `ACTIVE_ENTRYPOINT` | `scripts/benchmark_compute.py` | 983 | `56f7554b96e0e6896907985f2b06703dab27b0fa550f98f59b62f01034168f28` |
| `scripts/build_release.py` | `ARCHIVE_ONLY` | `docs/achive/single_source_refactor_20261103_01/source_before/scripts/build_release.py` | 5,701 | `9e845dec385f5d2f8b05c4df9de63b0405b28907993ff9c5a7e691554cb742e4` |
| `scripts/local_smoke.py` | `ACTIVE_ENTRYPOINT` | `scripts/local_smoke.py` | 1,176 | `acfa1b966b178e2da52f6f6557c907bc438954fc34b62010ebb5a1b8e8b63e4d` |
| `scripts/probe_device.py` | `ACTIVE_ENTRYPOINT` | `scripts/probe_device.py` | 4,578 | `56197948adbe418450cb14bab8722fa0b10929387fc766811ec192240a67b5f3` |
| `scripts/run_pipeline.py` | `ACTIVE_ENTRYPOINT` | `scripts/run_pipeline.py` | 1,228 | `3d9ad305b6d2446ad09a06558bf4df6f17329c260fa0f29345f17039cb68425c` |

### 2.3 Configurations

| Prior path | Classification | Final canonical or original archive destination(s) | Original bytes | Original SHA256 |
|---|---|---|---:|---|
| `configs/colab_full.json` | `ACTIVE_CANONICAL` | `configs/colab_full.json` | 1,374 | `2b61f7f3070f7148c02a7d69dc1644613e7a7c430641599ae49fd93840a737f0` |
| `configs/colab_pilot.json` | `ARCHIVE_ONLY` | `docs/achive/single_source_refactor_20261103_01/superseded/configs/colab_pilot.json` | 1,497 | `2e4342ad7f5e7f0bb8f605d0dd43dea7d8a0943b37773dafda66911109e66a5b` |
| `configs/colab_smoke.json` | `ACTIVE_CANONICAL` | `configs/colab_smoke.json` | 1,325 | `a6bd2a37d6fcf1e9f9c608558ad0f10fd5f42bfa650c8d2dfd73fe989fc2bc17` |
| `configs/compute_optimized.json` | `ACTIVE_CANONICAL` | `configs/compute_optimized.json` | 185 | `2bd394ee0803e189e2a92a778615cd47c01859b445afa47357fa6f8c0b17e473` |
| `configs/smoke.json` | `ACTIVE_CANONICAL` | `configs/smoke.json` | 800 | `52f805b9c5de6f23595a5af883bb5ac90fb88d3572eb331f1f5da05d7d62fd8a` |

### 2.4 Superseded Notebooks

| Prior path | Classification | Final canonical or original archive destination(s) | Original bytes | Original SHA256 |
|---|---|---|---:|---|
| `notebooks/01_Colab_Training.ipynb` | `ARCHIVE_ONLY` | `docs/achive/single_source_refactor_20261103_01/notebooks/01_Colab_Training.ipynb` | 36,654 | `826b43c4158a5c2a08534508124fe80b287a8a5323c8646c1202f9571414927f` |
| `notebooks/01_Data_Dowload&EDA.ipynb` | `HISTORICAL_REFERENCE` | `docs/achive/single_source_refactor_20261103_01/notebooks/01_Data_Dowload&EDA.ipynb` | 433,446 | `f66b72c2bde557dceee636bc481aa465ec14503d6ca4ec6409a2e9766cc6bbb6` |
| `notebooks/02_image_embeddings_extraction.ipynb` | `HISTORICAL_REFERENCE` | `docs/achive/single_source_refactor_20261103_01/notebooks/02_image_embeddings_extraction.ipynb` | 113,235 | `129d67acb0dadfdb718f4aa8e4f68865d8679a55e62faecf7d94bcb6f148be8a` |
| `notebooks/03_text_embeddings_extraction.ipynb` | `HISTORICAL_REFERENCE` | `docs/achive/single_source_refactor_20261103_01/notebooks/03_text_embeddings_extraction.ipynb` | 42,328 | `64292fae10a97cda5fb3f1616dc51b5eba818a33e4dacd91eba2e86fc3ac2b7b` |

### 2.5 Test migration

| Prior path | Classification | Final canonical or original archive destination(s) | Original bytes | Original SHA256 |
|---|---|---|---:|---|
| `tests/conftest.py` | `ACTIVE_CANONICAL` | `tests/conftest.py` | 1,057 | `4c844748842e3577d00c50a5c445fb0ca8d32bef3990c892cb1af14112238cf9` |
| `tests/test_colab_checkpoints.py` | `ACTIVE_CANONICAL` | `tests/test_colab_checkpoints.py` | 10,845 | `362031346fe1fde01cc74fe0631daf6aa554c1973e024048d7069f49d8717472` |
| `tests/test_colab_migration.py` | `ACTIVE_CANONICAL` | `tests/test_colab_migration.py` | 40,631 | `c52745619c237c2fb656f29a702409a72082c2076002af1dec8366acbad40298` |
| `tests/test_compute.py` | `ACTIVE_CANONICAL` | `tests/test_compute.py` | 40,791 | `1f01fb54db28129cf7ccad3001040fbf3b104432876ee45932664e04ea960617` |
| `tests/test_computing_refactor.py` | `ACTIVE_CANONICAL` | `tests/test_computing_refactor.py` | 25,529 | `4001bb115c56e957423a0c06b48b146d885e8569ee271e5c886325be365fb037` |
| `tests/test_documentation.py` | `ACTIVE_CANONICAL` | `tests/test_documentation.py` | 4,215 | `76bc8d7f7961a7cd2c5787596ed88e438b8976b5ffeed38c9957d53fa0bbf254` |
| `tests/test_full_backend.py` | `ACTIVE_CANONICAL` | `tests/test_full_backend.py` | 25,335 | `dada86c6bd2a44bce9f7d6a5e6d40f0ba38f7fe6d410b2e6745c83cad99bd76c` |
| `tests/test_full_checkpoints.py` | `ACTIVE_CANONICAL` | `tests/test_full_checkpoints.py` | 10,917 | `dd00b4acd9ee876f63ee9963f9c6ebf08a2ceb7872172db4cabdd2fd5572a4b9` |
| `tests/test_git_checkout.py` | `ACTIVE_CANONICAL` | `tests/test_git_checkout.py` | 17,969 | `8d4fd14d0df6059b5f1ba3426b9fc161649b0fbb51ffea03dc62c83883f56aae` |
| `tests/test_git_safety.py` | `ACTIVE_CANONICAL` | `tests/test_git_safety.py` | 13,597 | `69dd91bca438f5054ab0271d73b9320884dd656e504d0c268f44500f0bbec8dc` |
| `tests/test_git_workflow.py` | `ACTIVE_CANONICAL` | `tests/test_git_workflow.py` | 14,809 | `8929b80a74d7c16f60db83e19f97ccbca587424d9ce9f32dbab3477b3f310d75` |
| `tests/test_github_notebook.py` | `ACTIVE_CANONICAL` | `tests/test_github_notebook.py` | 16,267 | `dfb84c2248354c8f334f8d6f7de7ae81aa993a3c4f85c6d3ed8e61ccd1824e54` |
| `tests/test_pilot_backend.py` | `REFACTOR_MERGE` | `tests/test_modular_backend.py` | 46,855 | `bb10076faddd4adbd104338718ab4d39c274a39ee39e58441971cc1ddca0999d` |
| `tests/test_pilot_checkpoints.py` | `ACTIVE_CANONICAL` | `tests/test_pilot_checkpoints.py` | 10,612 | `b6149da0ec14b3d5ef9200b70a9aec33f725e23c079346e2549437d05f407032` |
| `tests/test_progress.py` | `ACTIVE_CANONICAL` | `tests/test_progress.py` | 12,990 | `639ee54ec170a6b8d079a7b496aaba530c6ad0d2ee3149df3c5761b4801923c0` |
| `tests/test_smoke_contracts.py` | `ACTIVE_CANONICAL` | `tests/test_smoke_contracts.py` | 54,987 | `3a5ca6e47b4ed09a1df16cbaa7c6a9e781fc46fb8119a69b6e276105eb4fa27a` |
| `tests/test_text_embeddings_notebook.py` | `ARCHIVE_ONLY` | `docs/achive/single_source_refactor_20261103_01/tests/test_text_embeddings_notebook.py` | 15,067 | `9d6fde5b69dfe3adf826c25cba85622dd0c51b08d3b09ef53361b6174f89d0e4` |

The `test_computing_refactor.py` and `test_smoke_contracts.py` **paths remain
active** with canonical imports/fixtures; their original bodies also have
local archive copies. They are not counted as removed tests. Former
`test_pilot_backend.py` semantic coverage is ported to
`test_modular_backend.py`, with named-stage and current-source contracts.
The text-extraction Notebook test is archived because extraction is not an
active training responsibility; immutable-input coverage remains in canonical
content/ID/feature tests.

The mixed `test_colab_migration.py` retains input-whitelist, interrupted-copy,
checksum, stage failure/recovery, persistence, resource, Notebook and real-model
reload checks. Only obsolete ZIP packaging assertions/imports were removed.
`test_git_workflow.py` compares actual root-Notebook stage cells on the canonical
Git path with the same nongit canonical path, then checks reuse/recovery;
it no longer imports a second legacy smoke pipeline. Documentation tests now
check leading module docstrings/file inventory, not redundant per-statement
comment density. Exact commands/results belong to the implementation report.

## 3. Split and consolidated APIs

This table disambiguates files whose initial mechanical move could otherwise
suggest that unrelated responsibilities stayed together. Symbols refer to
actual AST definitions or explicit named runtime phases. Removed APIs have no
active compatibility implementation in the old namespace.

| Former API / responsibility | Canonical owner / replacement | Adaptation and constraint |
|---|---|---|
| `contracts.Config`, IDs, split/hash helpers | `core.contracts` | Same IDs/cutoff/seed/split math; repository root resolved for the deeper package |
| `data.Transaction`, `Sample`, CSV inspection | `data.sample` | Typed bounded/raw-record contracts retained |
| Legacy `data.load_sample` / cohort materialization | `data.history.canonical`, `History`; `data.streaming.Partition` | Production smoke/full use one bounded SQL/Parquet path; no unbounded full-data object list |
| `pilot.data.History`, `canonical`, rosters | `data.history` | Full truth independent of candidate/model caps; separate authorized Test roster adapter |
| `pilot.streaming.TargetStream`, `Partition`, `Shards` | `data.streaming` | Deterministic bounded target order and complete roster/group contracts |
| `recall.Hit`, `top` | `retrieval.base` | One-based source ranks and score/article-ID ties |
| `recall.Candidate.rrf`, native merge inside `Snapshot.retrieve` | `fusion.native.Candidate`, `merge`; `fusion.pipeline.fuse` | Exact `1/(60+rank)` source rule; no injected positives or extra RRF truncation |
| `Snapshot.popularity` | `retrieval.trend` | Rule method delegates to one canonical popularity/retrieval implementation |
| `Snapshot.repurchase[_reference]` | `retrieval.repurchase` | Shared reference/columnar adapters preserve raw counts/decay; not smoke/full algorithms |
| `Snapshot.sibling[_reference]` | `retrieval.sibling` | Actual product-code variant rule and seed self-exclusion |
| `Snapshot._itemcf`, combined snapshot ItemCF build | `retrieval.itemcf`; `retrieval.snapshot.build_statistics` | Statistics and ItemCF build are separated so rule cells do not silently build ItemCF |
| `pilot.snapshot.GlobalSnapshot` | `retrieval.snapshot.GlobalSnapshot` | Current-user history adapter; ItemCF load is explicit |
| `learned.confidence`, `fold_user`, `fold_users_batch` | `retrieval.wals.model` | Single confidence scaling and genuine frozen-factor user solves |
| Legacy `learned.WALS.fit`, `pilot.models.fit_wals` | `retrieval.wals.training.fit_wals` | One legal base-window sparse fitting path for smoke/full; no duplicate legacy fitter |
| `learned.WALS.retrieve` | `retrieval.wals.model.WALS.retrieve`; `retrieval.learned.wals_batch` | Fixed mappings, active eligibility, original inner product and batched query adapter |
| Legacy `learned.ExactIndex` and `pilot.vectors` | `retrieval.vectors.BlockIndex`, `ActiveBlocks`, `block_topk` | One bounded exact-search/index persistence contract; source content FAISS remains separate |
| `learned.TowerRetriever`, `pilot.features.PilotTowerRetriever` | `retrieval.learned.TowerRetriever`, `tower_batch` | One frozen learned-vector query adapter; no duplicate trainable model |
| Legacy `build_tower_index` / `encode_items` | `two_tower.training.item_vectors`; named `index_towers` runtime phase | Encode the learned item catalog and persist its own index/map; no original-index replacement |
| `towers.TwoTower`, `positive_mask`, `logq_loss` | `retrieval.two_tower.model` | One trainable architecture and multi-positive LogQ objective |
| `towers.event_history`, `TowerEncoder`; `pilot.towers.SqlEncoder` | `retrieval.two_tower.inputs` | Raw-input assembly with strict same-day exclusion and shared tensor semantics |
| Legacy `sample_targets` / `fit_towers` loop | `data.streaming.TargetStream`; `two_tower.prepared`; `two_tower.training.fit` | Separate old smoke loop retired; canonical production target order/internal selection/checkpoint path shared by profiles |
| `pilot.towers.device_for`, `move`, `item_vectors`, `internal_recall`, `cached_internal_recall`, `fit` | `retrieval.two_tower.training` | Actual device, frozen content/trainable projections, full internal selection and complete-state resume |
| `pilot.towers.encode` / legacy `encode_queries` | `retrieval.two_tower.inference.encode`, `query_inputs.QueryInputs` | Batched cutoff-specific frozen inference and reusable model-independent query inputs |
| `compute.prepared`, `query_inputs`, `runtime` | `two_tower.prepared`, `query_inputs`, `execution` | Preserve caches, measured host/CUDA events, prefetch and static raw item bank; no learned-vector training cache |
| `features.FeatureBuilder`, `frozen_clusters` | `features.builder` | One ordered 49-feature implementation and frozen lookup |
| `pilot.models.preprocessing` | `features.preprocessing.preprocessing` | One pre-base frozen preprocessing/schema fit |
| Combined `pilot.features.generate` | `retrieval.pipeline.run_channel` → `fusion.pipeline.fuse` → `features.pipeline.generate` | Separate persisted boundaries; feature stage consumes union rather than rerunning six channels |
| `compute.shards.FeatureShards` | `features.storage.FeatureShards` | Whole-query columnar batches and streaming ledger |
| `ranking.select_training_rows`, matrices, ordering | `ranking.model` | Train-only 1:20 hard/random sampler, contiguous groups and deterministic Top-12 |
| `pilot.ranking.train`, `reload` | `ranking.pipeline` | One jointly grouped fit; complete native Validation; fresh-process score/feature parity |
| `evaluation` formulas/accumulator | `evaluation.metrics` | Single macro full-truth definitions; zero-hit roster preserved |
| Legacy local CLI / worker orchestration | `runtime.local_cli`, `runtime.modular`, `runtime.worker` | Thin current flags and same production stages; no parallel legacy algorithm tree |
| `colab.runner` / stage state | `runtime.runner`, `runtime.modular`, `runtime.state` | Explicit dependency graph, one named stage, leases, identities and verified persistence |
| `colab.files`, checkpoints, device, Notebook bridge | Corresponding `runtime` modules | Existing verification/progress/process isolation retained with canonical imports |
| ZIP `bootstrap`, `release`, `scripts/build_release.py` | Local archive only; pinned Git checkout is the active bootstrap | No archive import/extraction is a runtime prerequisite |

## 4. Import and caller provenance

The baseline inventory stores every recorded project import for each of its
83 paths. Current source AST is the basis for the README's primary-caller map;
no archived module is loaded to derive this table. In the current inventory,
**94 active Python files** contain **441 static project-import
occurrences**, with **zero** imports from the former `hm_recsys.pilot`,
`hm_recsys.colab`, `hm_recsys.compute` namespaces or `docs.achive`.
This is an AST import check, not a claim to infer every dynamic call. Explicit
worker module strings and Notebook commands use canonical runtime paths.

| Former caller chain | Current caller chain |
|---|---|
| `scripts/local_smoke.py` → `hm_recsys.cli` | Same thin script → `runtime.local_cli` → canonical modular stage graph |
| Notebook → `colab.cli/runner` → pilot/old smoke paths | Root Notebook → `scripts/run_pipeline.py` → `runtime.cli` → `runtime.modular.run_one` |
| `colab.worker` and root `worker` with separate dispatch | One `runtime.worker` → `runtime.stages` → canonical package functions |
| `pilot.towers` → root `towers`, `compute.*`, `pilot.data` | `two_tower.training` → `model/inputs/prepared/execution`, `data.history/streaming` and runtime checkpoint helpers |
| `pilot.features.generate` → live six-channel retrieval inside features | `retrieval.pipeline` persisted channel outputs → `fusion.pipeline` persisted union → `features.pipeline` |
| `pilot.ranking` → root `ranking/evaluation/features` | `ranking.pipeline` → `ranking.model`, `features.storage/builder`, `evaluation.metrics` |
| Compute benchmark → old pilot/root model namespaces | `runtime.benchmark` / `features.benchmark` → the same canonical trainer/input/feature modules |
| Tests import old root/pilot models or ZIP helpers | Tests import concrete canonical packages; Git/bootstrap/state coverage replaces obsolete ZIP-client assertions |

The new independently executable stage names are `prepare`, `trend`,
`repurchase`, `sibling`, `itemcf`, `wals`, `twotower`, `fusion`, `features`,
`ranker`, `validation`, `reload`, with separately gated `final_test`.
`stage --stage NAME --stage-action run/resume/status` does not implicitly fit
missing prerequisites. Source/model/feature/split identity changes require a
new run, not mutation of a completed old manifest.

## 5. Archive byte preservation and Git index

### 5.1 Original and intermediate copies

The following are independent preservation facts:

- `before_source/` in the local evidence root preserves all **83** original
  inventory files byte-for-byte.
- `docs/achive/single_source_refactor_20261103_01/source_before/` preserves all
  **51** original Python source files, including the removed namespaces.
- Four additional exact originals are now under that same `source_before/`:
  `scripts/build_release.py` and
  `tests/{test_computing_refactor,test_pilot_backend,test_smoke_contracts}.py`.
- The four original Notebook copies under archive `notebooks/`, the pilot
  config under `superseded/configs/`, and the extraction test under archive
  `tests/` match their original inventory hashes.
- `superseded/scripts/build_release.py` and the three `legacy_tests/*.py`
  copies are **intermediate import-edited versions**, not mislabeled originals.
  Their original-byte counterparts above remain separately available.
- Intermediate `superseded/src/hm_recsys/runtime/{legacy_cli,legacy_worker,
  bootstrap,release}.py` files and `source_namespaces/*/__init__.py` preserve
  migration history; none is imported by active source.

No completed runtime input/model/checkpoint tree was moved as part of these
source/Notebook archive operations. Archive copies are ignored local evidence,
not a public distribution or an automatic compatibility fallback.

### 5.2 Ten pre-existing archive index entries

`archive_untracking.json` explicitly records `index_only=true` and
`commit_or_push=false`. Each of these ten files remained present with the
following bytes/SHA256 after its index-only untracking:

| Index path removed | Preserved bytes | Preserved SHA256 |
|---|---:|---|
| `docs/achive/hm_mvp_full_reports_colab_v1.zip` | 550,613 | `8851a405750bea6811a1d11654adc6f8f1ab3a8475b2dc0b0d8a7c4634f8d89c` |
| `docs/achive/mvp_notebooks_colab/00_Data_Preparation.ipynb` | 91,919 | `a4e7c4e5337219c2ab3ec4f80fdfb855738f1f0b6a05880bdd303d966673a430` |
| `docs/achive/mvp_notebooks_colab/01_CPU_Retrievers.ipynb` | 102,437 | `8b962af427c98af0e4888b398862cae85717d802c2533b938b5ca98dace3bbec` |
| `docs/achive/mvp_notebooks_colab/01_CPU_Retrievers_v2.ipynb` | 144,138 | `bbacd1dc35c9052645fb72a90a61c114f3d075be185f75945bc210f199a48515` |
| `docs/achive/mvp_notebooks_colab/02_GPU_Retrievers v2.ipynb` | 228,948 | `c0d8123d59baadb51e43dda46cbdff9873dcf0d5bab9b4da9db608865b1f0e6d` |
| `docs/achive/mvp_notebooks_colab/02_GPU_Retrievers.ipynb` | 117,522 | `7fc48702e1a3f2eeede2da3cdea4e2a649cd07e500e16c6cfded87cf6ca46afd` |
| `docs/achive/mvp_notebooks_colab/03_Fusion_Ranking v2.ipynb` | 317,833 | `46b74c0d2ecd41c912714823c65bc1a349f02231741bdd800bcf30bb930b3f99` |
| `docs/achive/mvp_notebooks_colab/03_Fusion_Ranking.ipynb` | 127,241 | `a879904b6e7ab7c9854f95a5a50d54bda39b206c6886bfb5f433fdd9af94dfd6` |
| `docs/achive/mvp_notebooks_colab/04_Offline_Evaluation v2.ipynb` | 466,891 | `199ece486d34539b9b9c8e7fb6bdf47ed80aed34731510fb80bf124a76589644` |
| `docs/achive/mvp_notebooks_colab/04_Offline_Evaluation.ipynb` | 113,524 | `7df09077a7a5a4683f7c9c11c4285d75ed3b2ec84c744a5791f379905cd81f26` |

The four former `notebooks/*.ipynb` index paths were also removed while their
exact archive copies were retained. Read-only `git ls-files -- docs/achive
notebooks` returns no tracked paths; `git diff --cached --name-status --
docs/achive notebooks` shows the ten archive plus four Notebook deletions from
the **index**. This is not deletion of their archived worktree bytes and not a
commit or history rewrite. The ten-path JSON above is not misrepresented as a
fourteen-path operation log.

## 6. Mapped active destination fingerprints

These are current working-tree SHA256 values for active destinations referenced
by the old-file/API map, recorded separately from original hashes. They bind
this mapping to actual source, not to an archived release name. New-module
roles and the exhaustive current active inventory remain in the README.

| Active mapped destination | Current SHA256 |
|---|---|
| `configs/colab_full.json` | `2b61f7f3070f7148c02a7d69dc1644613e7a7c430641599ae49fd93840a737f0` |
| `configs/colab_smoke.json` | `a6bd2a37d6fcf1e9f9c608558ad0f10fd5f42bfa650c8d2dfd73fe989fc2bc17` |
| `configs/compute_optimized.json` | `2bd394ee0803e189e2a92a778615cd47c01859b445afa47357fa6f8c0b17e473` |
| `configs/smoke.json` | `52f805b9c5de6f23595a5af883bb5ac90fb88d3572eb331f1f5da05d7d62fd8a` |
| `scripts/audit_git.py` | `d8de90bba815d974b978fbef72a3e987e8da9f0e326d56d1d207a54cb8913ea5` |
| `scripts/benchmark_compute.py` | `ad3682747fc3aff77cff8c7f5010853a81c4855202fac856b4d7085f2c0ae4bf` |
| `scripts/local_smoke.py` | `79e000effc30cde461e565b6f64e7ba6332fc7381f6234836521d2de9420dc49` |
| `scripts/probe_device.py` | `c0f38be962cb00c2f7917800fa22987b3ba42abdd2185d5244c08a7df9ba0f46` |
| `scripts/run_pipeline.py` | `1af55167fb234674fa4e37b5984b93b2bdec5f220700e3cfc29c233f5aea0d8b` |
| `src/hm_recsys/__init__.py` | `5a88a21aa773fddaab7b8fc0880774ea274f6406f616fe403733e2380084e2bd` |
| `src/hm_recsys/core/contracts.py` | `f8a896bab5d83a88b1b121d2e9d61afdda465a2259ea8b917a4a21672ae9aba7` |
| `src/hm_recsys/core/progress.py` | `02f26b886879cc24a74ab0f15992f0e1d9ecf2e52ea85b00403fcc68410bf373` |
| `src/hm_recsys/core/synthetic.py` | `6550182ecdfa230d2671b11187fc1f10049095cbf2123f4e847ea8c6cfbc3c1b` |
| `src/hm_recsys/data/history.py` | `36e5ee5b362670d1a920db9fa1abdc653865183879722592b036c9a7efa7b0cc` |
| `src/hm_recsys/data/sample.py` | `4f0d62f1b65f5d7078de9d5226b9ade3e71291726fd17d60aca017656b9ae958` |
| `src/hm_recsys/data/streaming.py` | `572c2adf804cc13100960efe806e1f6bef9049277166ebc2b7e9f15d0e908817` |
| `src/hm_recsys/evaluation/metrics.py` | `e96e62eafab9f1ceb63a953502ba67e2a74d6bc7c5301881521eb8a40973dc73` |
| `src/hm_recsys/features/benchmark.py` | `38cfc70ecd15a41a77113b080057705c993807f12ca73fcfe05f286bcd37e950` |
| `src/hm_recsys/features/builder.py` | `8c1573267e97bd935f4b3f4e42d7a6b1d713c21ee9f9b6a0bed1b09eaee25a1d` |
| `src/hm_recsys/features/content.py` | `9539594daccefb9c380a424286d7a49197a66487a8c546f3839e5fb92747d4cf` |
| `src/hm_recsys/features/pipeline.py` | `fa959f86ceade30cc0852221912ee9ee2f6234bc2b65eaf7bc58f0714588402f` |
| `src/hm_recsys/features/preprocessing.py` | `2499e7b4c21976cd4f9c021c3e6524da0411750e7573e290416a7e57b2d9d638` |
| `src/hm_recsys/features/storage.py` | `9f2a87444f001ce34f13340830ee440d2d8e83732444caec52470edab008389f` |
| `src/hm_recsys/fusion/native.py` | `d7cbdfad2f9cb1ee5b01e043f6d10a88b043179684c298fa5e231c50aaeb0d17` |
| `src/hm_recsys/fusion/pipeline.py` | `0fde7e46c782c274ae6700e2396c81285f6357e6c19f383eefc0eacbbee59ce4` |
| `src/hm_recsys/ranking/model.py` | `edbe727dd55d176ab484e8de69e8bb3025d9add6bae252845175c9d638e94af3` |
| `src/hm_recsys/ranking/pipeline.py` | `c1bb6af05c58f19626b6f47eab74592ec9114ef18d68eafcdda605f51297f2e8` |
| `src/hm_recsys/retrieval/base.py` | `3a12110dd3850d16139382ddbf8b7c9030806b5a40c55d5e2e6e9f8a36818424` |
| `src/hm_recsys/retrieval/itemcf.py` | `29b0c718871b8f88dc9344ce1deab0d3ff66ce11aa2c9217929fddc4c1b83291` |
| `src/hm_recsys/retrieval/learned.py` | `47782d21d65e2719f0930f96cc576718e63160906e297a3f54ebf0944e143bb3` |
| `src/hm_recsys/retrieval/pipeline.py` | `bb55de53cdf8ec9466587421d1f890cfe7dadef37bcff362a5b23ad49a0f6dd5` |
| `src/hm_recsys/retrieval/repurchase.py` | `c9d4beb68cad9f01ce383d8fd149d298b5e1b287b15069c2518a1f3a9dd910eb` |
| `src/hm_recsys/retrieval/sibling.py` | `c6f53b90902cb88eedb54cb6495e6095760b231cf1fabb6ee2d9870758e2b8d6` |
| `src/hm_recsys/retrieval/snapshot.py` | `32cfc7e03160f29760f090182c410718397a49dd8f509a550834d27c153307ed` |
| `src/hm_recsys/retrieval/trend.py` | `2289720d63d2d3c0cd7d81c4a3de1f0bb5e14333f634b1c0e7404e6470d6c141` |
| `src/hm_recsys/retrieval/two_tower/execution.py` | `e44ca24785e82e848f29dbb7d9c8fac31c6af6b0009e0d7b1d70fcd6e014058a` |
| `src/hm_recsys/retrieval/two_tower/inference.py` | `22e7fb96d2208a9b847e527907be321c7792a0a0333110d0245c00df91bb1109` |
| `src/hm_recsys/retrieval/two_tower/inputs.py` | `f98054dd4318345b6989b2a815b862883449de0731d4afd498caa16009a7fc58` |
| `src/hm_recsys/retrieval/two_tower/model.py` | `a06a934436804cd31054f9c638e687c0640e6ccdbfe85dc1f537eec0e2ccdc46` |
| `src/hm_recsys/retrieval/two_tower/prepared.py` | `c4283d0b31f4f41f0a1d88f7b90864e5fce0020df49ce5bc28aea0f52852eb71` |
| `src/hm_recsys/retrieval/two_tower/query_inputs.py` | `a01b21078a66f20eaf1724fd6afe1d919488cf9941957db69aadc58011cd7ff2` |
| `src/hm_recsys/retrieval/two_tower/training.py` | `5bff4b1e7c8df7c2fa92a154c2ad1781cefdfc4745f9cbe5c3c51a630a33f782` |
| `src/hm_recsys/retrieval/vectors.py` | `96057b144dac2f34c42d69cc2d6b563135706eeb43c0785f15e8a35472c83c23` |
| `src/hm_recsys/retrieval/wals/model.py` | `8885c925062fb29d5b62bea61d594e738f80ff8d05948751f153b537df7717c0` |
| `src/hm_recsys/retrieval/wals/training.py` | `a97ebe96463a50c193ec1acc0189dbfef3f3856d697abc7a2fdfc06091b0c5a7` |
| `src/hm_recsys/runtime/acceptance.py` | `66db0d4c8ddca45f29538f4480d01e74fff84773444df3fe6c80e9746adfb934` |
| `src/hm_recsys/runtime/benchmark.py` | `e7978ad4c7c984aff04d91b3d69a28ae4dec857c6c995e477892d3369ce71410` |
| `src/hm_recsys/runtime/capacity.py` | `fb1e610b71e6024853a34f591cd916daf0dd0ff8c925862e50127b3bc93fd7a2` |
| `src/hm_recsys/runtime/checkpoints.py` | `2f603a38fe928308bc1169bdca1ae324e69d8d38c8073ec643ba72d48f2f9ebc` |
| `src/hm_recsys/runtime/cli.py` | `7975a18e0947f871d5c736a47e4e6bcbb44ff8e9b345513de76efae7379406b4` |
| `src/hm_recsys/runtime/device.py` | `222fa3bc1f50cddb8a6cae89d9aafafd6b15e4a6e7577e63253ce2621de36d9e` |
| `src/hm_recsys/runtime/files.py` | `0886c98fc89ed42a9c4bcccbd339388c5e9d19bd72ed8d0a0cb73332433f0145` |
| `src/hm_recsys/runtime/git_checkout.py` | `40ef595c46ce73c074739aa59a5fe0c0f357033e8c69aef01ea92248e73566ad` |
| `src/hm_recsys/runtime/git_safety.py` | `8750004f5097c68684105709ee9a5c65e0a6ccf7fda64cc6ff47c909db4c6a35` |
| `src/hm_recsys/runtime/local_cli.py` | `1cd759b8c3bbb7f29e83b2642f63873137ffad7286527691962afdfa77d0bb61` |
| `src/hm_recsys/runtime/modular.py` | `bfc695e8f3e238bdb92ec158d616fa6ec92dcc1214183dc244b8101fac70cd4b` |
| `src/hm_recsys/runtime/notebook.py` | `f497df88b2d6039e5258ac0943a4e0f81496ce5b05607825c3a3a5b23976730e` |
| `src/hm_recsys/runtime/profiles.py` | `25e04bc5d705f6233500077be236c3345dfe9842299a41fc9358d44d2ec90a9e` |
| `src/hm_recsys/runtime/resources.py` | `6a14398412444161add3213a7d252f7a9929197b16567ae176911b1e32905d0e` |
| `src/hm_recsys/runtime/review.py` | `fc9785bb8d5f5ef0714ffe80fe63a1842d21299331ff109c00c0aab5d1d7d79f` |
| `src/hm_recsys/runtime/runner.py` | `071680423ac377b9a243ee586a1c37b4e86b213299ef9cec18b0ea6d1984cf90` |
| `src/hm_recsys/runtime/settings.py` | `f713f6a402d08155e9e1775b449da667fd0d555ffb30c8b3c6f2be45fe22a871` |
| `src/hm_recsys/runtime/stages.py` | `96c8f49cd40850045514cb3191edfd01d0a5ec8128981acc55c5a89db9149f0f` |
| `src/hm_recsys/runtime/state.py` | `a28e8b2b982150820578dd3da8d524f86f1cbdf3b503ef6afd08acde0b59b15b` |
| `src/hm_recsys/runtime/testing.py` | `c1e73362b35f31f17ba1b2da096c4d2fbc790f473ae1652a47d15465c70b25c8` |
| `src/hm_recsys/runtime/worker.py` | `ba0c7c96d1a83450887fc8d14024b902a89a88e4254ddcca94c43c6c09f91c65` |
| `tests/conftest.py` | `4c844748842e3577d00c50a5c445fb0ca8d32bef3990c892cb1af14112238cf9` |
| `tests/test_colab_checkpoints.py` | `117458eae490abb836ca3fefd71bab3fe7ea998900abaeaa5f73166716ce8140` |
| `tests/test_colab_migration.py` | `4a12adbab91b314f49c39a7bf34708fb6b2fa73a1ca56873b0edbf63dc511ca6` |
| `tests/test_compute.py` | `c2abbe33124bea2f4aa4562a56107b2f0a1c23ec7ccbb7cfa31a73c0c9371ca4` |
| `tests/test_computing_refactor.py` | `8720ca9b1f697b9886d7773d267baec44d311a4ac91817d25397ef140a74d9bb` |
| `tests/test_documentation.py` | `6de8bd6d20409af28e2e0c46d922b57d6ba43fec1ab7cb8679d7321637403c45` |
| `tests/test_full_backend.py` | `115babf129d5ba64cb3183efb6e088b8beed55b101f85734971b6758013d6dce` |
| `tests/test_full_checkpoints.py` | `1cd756ecea32537389e2e48a8b89c84cf1ff3febcfc3766c548970f4fcfc0767` |
| `tests/test_git_checkout.py` | `cb855ee0c438c07227d1f058e82080f92e4c14675db1a0b3624883158459c28a` |
| `tests/test_git_safety.py` | `6f8a999f42f6c30ba97a8a22694f7b96e591e1857e63a1b2e744a484cf55b51f` |
| `tests/test_git_workflow.py` | `a33b394b22aae0c58f1734a6d6b9282c008ce2899341b42c9058aab893fdff14` |
| `tests/test_github_notebook.py` | `db192c28a7073bf40dfcbc5ca295c9e38b24270218a2c2767436e7f0336c79fb` |
| `tests/test_modular_backend.py` | `2a0838982677243cb0b88313aba2a3dd182790f2d43075d129c87905fa05ae93` |
| `tests/test_pilot_checkpoints.py` | `4c4c28f427be8a60a3253957cbb33fdf5e81d7ce2fde425f9222d0bfabacfdbe` |
| `tests/test_progress.py` | `0753451e77a2d0e60b168c86fa7302d53a795b5c6fbf451724974cb06ae5e729` |
| `tests/test_smoke_contracts.py` | `485d9aa3f5d0f9845a12c84b50411786bbcd10442d4a11743c685e226fcd0ef7` |

Fingerprints identify the map's working-tree state; any subsequent source/test
fix must refresh this section and be included in the implementation report's
actual verification. They are not external signatures or proof of model quality.

## 7. Documentation snapshots

The four documentation files were copied with original bytes to `before_docs/`
before their authorized edits. Unlike a restored historical architecture, their
new content describes current canonical ownership and preserves the existing
learning/evaluation semantics.

| Prior document | Pre-edit SHA256 | Current responsibility |
|---|---|---|
| `README.md` | `f2036187c701c963a6e0058d29ccec9d8027794eb96e74bef115382cfadc07d5` | Complete active file map, inputs/models, explicit Notebook/CLI, resources/recovery and publication limits. |
| `AGENTS.md` | `2dde5fcf045aa16dd00f3c40ea5383306248008a638f3dee5529af7a52092da0` | Canonical package/root-Notebook rules, module documentation, ignored archives and protected Test boundary. |
| `docs/archi/architecture.md` | `8080beab331392b4de97bc1d04cd4d885c61556fb4ae909d89700a8f99679996` | Actual package/call/stage/data graph and unchanged temporal/model contracts. |
| `docs/archi/github_colab_runbook.md` | `eee5cd1e81a1a675fcb3db7e2acbb8a6c8c08af9458c359f4f8c3958b54fcab5` | Pinned Git smoke/full setup, independent stage cells, resume, reports and separate final-Test approval. |

Current proposal/protocol semantic rules were not changed by this map. Old
prompt/report locations, where superseded, are preserved under local archive
`historical_docs/`; current design pages must not import or require them from a
public checkout. The implementation report records any additional reference-only
link updates.

## 8. Privacy and acceptance boundaries

The whole `docs/achive/` and runtime `data/` trees are ignored and excluded from
active lint/type/test traversal. That prevents new normal additions, but **does
not purge committed history**. The existing Git audit found historical/private
content in earlier committed revisions. Index-only untracking and a clean new
source tree do not make that history safe to push. No commit, push, remote
upload, force-add, history rewrite or customer-data exfiltration was performed.
The repository owner must review/resolve the recorded history blockers before
publication; the implementation report provides the final audit status.

This map proves file classification, path/API lineage and the preservation/hash
checks listed here. It does not assert that all tests passed, that a real Colab
runtime recovered, or that full H&M/final Test reran. See the
[implementation report](code_architecture_refactor_report.md) for actual commands,
results, bounded before/after parity/performance and explicitly unrun work.
The [README](../../README.md#active-python-file-map),
[architecture](architecture.md) and [runbook](github_colab_runbook.md) are the
active operational references; archives remain historical and unmaintained.
