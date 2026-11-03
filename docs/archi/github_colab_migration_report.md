# GitHub Colab Workflow Migration Report

## Contents

- [Scope and inspected state](#scope-and-inspected-state)
- [Implementation](#implementation)
- [Preserved contracts](#preserved-contracts)
- [Verification](#verification)
- [Privacy and publication gate](#privacy-and-publication-gate)
- [User handoff](#user-handoff)
- [Unverified environments](#unverified-environments)

## Scope and inspected state

This task replaces the main Notebook's source-archive upload/extraction workflow
with a credential-free GitHub source URL and verified detached Commit checkout.
The existing repository was inspected in place, not initialized or replaced.
Origin is `https://github.com/HaokunLiu-Vincent/ecommerce-recsys.git`, branch main,
starting HEAD `54c24e54b543490b334b68daeece91c29bfe3689`
(`colab run - smoke and pilot pass`). Numerous uncommitted
full/compute/progress changes were already present and retained. No commit or push of this working repository, history rewrite, remote change,
model/data deletion or real full job was performed. Temporary Git fixtures create
commits and push only to their own local bare remotes. This dirty workspace is not advertised as a remotely fetchable
Commit. A new manual reviewed commit is required for Colab to obtain these edits.

Read authority: root AGENTS.md v0.3.2; docs/proposal.md v0.3.2; current
architecture, implementation decisions and evaluation protocol v2.0; current
Colab/full/live-progress runbooks; and the complete GitHub workflow task. The
latest workflow changes execution provenance, not the recommendation protocol.
No applicable nested AGENTS.md was found.

## Implementation

| File/function | Previous behavior | Implemented behavior |
|---|---|---|
| `.gitignore` | Partial artifact/credential exclusions | Expanded data/model/index/checkpoint/cache/credential exclusions; actual source inputs retained |
| `src/hm_recsys/colab/git_checkout.py` | No shared Git bootstrap | Stdlib URL/ref/SHA/destination validation, detached checkout, clean/origin/HEAD verification, explicit resume pinning, ephemeral authentication, honest local development metadata |
| `src/hm_recsys/colab/git_safety.py` | No automated tracked/history audit | Offline index/history/ignored-tracked/large/secret/output findings without printing matched private values |
| `scripts/audit_git.py` | No local privacy gate | Metadata report and nonzero publication gate for unresolved findings; never untracks/deletes/pushes |
| `src/hm_recsys/colab/cli.py::main` | Source hashes only | Optional complete Git identity flag trio, actual checkout inspection on plan/run/resume/status; historical CLI retained |
| `src/hm_recsys/colab/runner.py::execution_identity` | Version-one source/config/input/schema/runtime contract | Version-two Git provenance plus explicit protocol/lock hash; current source re-inspected rather than trusting a bootstrap marker |
| `runner.execute/_execute_profile` | No explicit Commit provenance | Existing execution/store/checkpoint logic receives verified Git identity; worker manifest and aggregate result record actual Commit/data digest |
| `src/hm_recsys/cli.py::execute` | Local smoke hardcoded an unavailable-Git marker | Actual local HEAD/cleanliness recorded; dirty development is explicitly nonreproducible |
| `notebooks/01_Colab_Training.ipynb` | Uploaded ZIP/checksum/manifest bootstrap | Direct pinned GitHub checkout; smoke/full only; existing live progress, locked environment, data staging and stage execution retained |
| `tests/test_git_checkout.py`, `test_git_safety.py`, `test_git_workflow.py`, `test_github_notebook.py` | No Git-workflow coverage | Local-remote, privacy, actual Notebook command, Git identity/persistence and genuine shared-model regression fixtures |
| Existing Notebook contract tests (`test_full_backend.py`, `test_colab_migration.py`, `test_pilot_backend.py`) | ZIP configuration and brittle probe-cell selector | Current Git configuration and actual CLI parser contract |
| `src/hm_recsys/colab/notebook.py::_stop_group`, `tests/test_progress.py` | Intermittent macOS KILL permission denial immediately after TERM | At most 0.5 s retry on the same owned group; persistent denial still raises; live relay/orphan and explicit transient/persistent tests pass |
| `README.md`, `docs/archi/github_colab_runbook.md` | Upload/build steps primary | Local smoke/audit, manual publication, Colab GitHub links, pinned startup/private auth/full gates/recovery |
| Existing Colab/full/live runbook notices | Archive setup appeared current | New primary GitHub guide linked; archive instructions explicitly historical compatibility |

Git version-two identities cannot reuse version-one archive/local-development
runs. Original artifacts remain untouched. Ref is recorded informationally but
retained in the exact identity, so recovery preserves its original configured
value even if the branch moves. No automatic migration between formats or
arbitrary cross-run upstream reuse is introduced. Dirty/incompatible existing
checkouts are refused; a fresh runtime is the documented safe remedy. No reset,
pull-during-training or automatic replacement of user code exists.

## Preserved contracts

Six channels and quotas 60/30/30/50/50/50, the native deduplicated union, one
LambdaMART, time boundaries, full ground truth, zero-hit queries, multi-positive
LogQ loss, FP32 batch 512, model budgets, Train-only 1:20 negatives, the shared
49-column feature schema and fresh-process reload remain unchanged. WALS,
Two-Tower, ItemCF, feature and ranker execution code were not rewritten for Git.

Workers still stream actual structured progress while running. Unknown totals
remain unknown. Memory/resource rejection, isolated native processes, local
training, twelve-file Drive staging, checksum-verified stage deltas, full-state
checkpoints, previous valid remote pointers and explicit stale-lease recovery
remain active. Final Test remains disabled. Pilot/benchmark CLI and optional
historical archive tooling remain backward compatible but are absent from the
main Notebook UI. No DVC/MLflow/CI/cloud service was introduced.

## Verification

Final evidence is retained under
`data/runs/benchmark/github_colab_20261103_01/`. Commands use the existing
`uv run --locked` environment, `UV_CACHE_DIR=/tmp/hm-smoke-uv-cache`, and native
thread limits of two. No dependency declarations or lockfile were changed.
Python was 3.12.16 and Git was 2.54.0 (Apple Git-157). On this desktop, guarded
subprocess tests used authorized process-metadata access because the sandbox
blocks the existing psutil RSS monitor; the monitor was not disabled.

| Actual command/check | Exit | Inspected result |
|---|---:|---|
| `uv run --locked ruff check src scripts tests notebooks/01_Colab_Training.ipynb` | 0 | Maintained scope clean |
| `uv run --locked ruff format --check src scripts tests notebooks/01_Colab_Training.ipynb` | 0 | 75 files already formatted |
| `uv run --locked pyright` | 0 | Zero errors/warnings |
| `uv run --locked pytest -q --basetemp data/runs/benchmark/github_colab_20261103_01/accepted_pytest_tmp --junitxml data/runs/benchmark/github_colab_20261103_01/pytest_accepted.xml` | 0 | **84 passed, 2 skipped**, 194.47 s; skips explicitly require CUDA |
| `uv run --locked python scripts/local_smoke.py synthetic --run-id github_migration_synthetic_20261103_01` | 0 | Existing genuine CPU model pipeline and fresh-process reload passed; 10.24 s wall time |
| `nbformat.validate` and `compile` for every Notebook code cell | 0 | 21 cells, 11 code cells; no saved outputs/execution counts; full Notebook not executed |
| `uv run --locked python scripts/run_pipeline.py plan --profile full --persistent-root data/runs/benchmark/github_colab_20261103_01/not_executed_persistence` | 0 | Formal fold/stage plan only; no training |
| `uv run --locked python scripts/audit_git.py --output data/runs/benchmark/github_colab_20261103_01/git_safety_final.json` | 1 | Expected publication block: 33 findings, complete bounded history coverage |
| `git diff --check` | 0 | No whitespace defects |
| `uv run --locked ruff check .` | 1 | 122 existing notebook lint issues retained |
| `uv run --locked ruff format --check .` | 1 | Six preserved notebooks and four supplied instruction/audit Markdown files need formatting; 10 total, unchanged by this task |

Exact argv, durations and logs are linked by `static_accepted_commands.json`,
`pytest_accepted_commands.json`, `smoke_accepted_commands.json`,
`verification_summary.json` and `pytest_accepted.xml` in the evidence root.
Root-wide failures were not suppressed or fixed by editing unrelated notebooks
or the supplied task documents.

The bounded synthetic smoke selected **24 users, 40 items and 4,187 raw purchase
lines** from its generated fixture. WALS performed two iterations with 16 factors
and 960 nonzeros. Two-Tower performed 38 actual optimizer steps in one epoch,
processing 1,200 targets. One LambdaMART fit performed 30 rounds on 3,840 rows
in 96 contiguous training groups, with 408 native positive rows. Each of the
five partitions retained 24 queries and 960 native feature rows, with the shared
49-column schema. Validation retained its one zero-hit query.

Actual Validation output rows by channel were Trend 960, Repurchase 528,
Sibling 288, ItemCF 960, WALS 960 and Two-Tower 960. Channels were genuinely
executed; no fixed predictions or popularity substitutions were introduced.
Fresh-process reload scored 960 candidates for 24 queries, with exact Top-12
parity and maximum score difference zero. These numbers describe generated
fixtures, **not H&M benchmark quality**.

Git-path execution was compared with the existing local reference using the
same deterministic fixture/settings. All twenty candidate/feature/query/truth
files across F1–F4/Validation had identical hashes; Top-12 and full-ground-truth
metrics matched exactly. Same-Commit recovery restored verified stages into a
new local runtime directory. Failed-stage recovery, wrong Commit, identical
code under a different Commit, config/data/schema/runtime mismatches and
cross-format resume rejection were exercised. See `smoke_parity_summary.json`.
Actual Notebook bootstrap was executed against a temporary bare remote, not
just a duplicate helper; private askpass, pre-Git input rejection, generated
CLI arguments and status-without-training were tested. Existing real nested
worker latency, unknown totals, bounded failure logs and orphan cleanup tests
also passed.

Thirteen standalone guarded stages all exited zero. The sampled peak process-tree
RSS was **383,680,512 bytes (365.91 MiB)**, with minimum observed system available
memory 13,894,877,184 bytes. Existing 5 GiB RSS abort, 2 GiB available-memory floor,
two native threads and 0.2 s sampling remained active. The live available-memory
measurement is not an expanded training budget. These are sampled measurements,
not OS hard limits or full-data capacity estimates. Evidence remains beneath
`data/runs/smoke/github_migration_synthetic_20261103_01/`.

Earlier attempts are retained rather than relabelled as success: the first full
suite had 79 passed/2 failed/2 skipped (repository-error text and probe-cell
selector assumptions); the next had 81 passed/1 failed/2 skipped (intermittent
macOS owned-group signal denial). Correct assertions and the bounded cleanup
retry fixed these issues; transient and persistent denial cases were added.
Persistent permission failures still raise explicitly. The final full-suite
result above was actually rerun after those changes.

## Privacy and publication gate

The actual offline audit exited **1**, correctly blocking publication. It
inspected 88 tracked/index entries and two reachable local commits (94 historical
path versions); history coverage was complete within the stated bounds. It
reported 33 scope/category findings, not 33 distinct private files:

- One ignored-but-tracked archive:
  `docs/achive/hm_mvp_full_reports_colab_v1.zip`. Its opaque contents were not
  unpacked or certified; it remains tracked despite the new archive ignore rule.
- Seven current/historical notebooks retain execution outputs: five archived
  MVP notebooks, `notebooks/01_Data_Dowload&EDA.ipynb` and
  `notebooks/02_image_embeddings_extraction.ipynb`. Findings occur separately
  in working tree, index, HEAD and history.
- No original input data or recognized credential indicators were found by the
  bounded rules. This is not proof that retained outputs contain no private data.

Evidence: initial and final `git_safety_*.json` reports in
`data/runs/benchmark/github_colab_20261103_01/`; both report 33 findings.
Existing historical notebooks/archive and Git history were preserved. Publishing
is **blocked pending review of retained outputs and the tracked archive**; the
implementation does not claim this history is safe to push. Review those bytes
before any publication and decide deliberately whether historical remediation
is needed. Adding `.gitignore` or making a new clean commit alone cannot remove
old blobs. No automatic untracking, output clearing or history rewrite occurred.

## User handoff

1. Resolve reported privacy blockers and review the changes.
2. Run local checks/smoke and stage the complete reviewed current source/config
   dependency set, including pre-existing uncommitted full/compute/progress work;
   publishing only the new Notebook/Git adapters would omit required modules.
   Commit and push manually after privacy review.
3. Open the Notebook from GitHub at that new actual full SHA.
4. Configure the same repository/SHA with a new smoke RUN_ID; execute in order.
5. After smoke acceptance, choose a new full RUN_ID, explicitly allow full and
   request CUDA, keeping RUN_FINAL_TEST=False.
6. Resume only the original Commit/profile/device/config/data/runtime identity;
   use the same RUN_ID and RUN_ACTION="resume". Never rename an old ZIP/pilot/full
   artifact directory to match a Git run.

No ZIP, checksum or release manifest needs uploading in this path. See the
[runbook](github_colab_runbook.md) for directly usable configuration, exact roots,
private-secret handling and troubleshooting.

## Unverified environments

Real GitHub authentication/push, Colab GitHub open/clone, Linux environment
installation, CUDA execution, actual Drive mount/FUSE recovery and real H&M
full training are not run by this task. The user previously reported successful
Colab CUDA pilot, which does not establish this new Git workflow's cloud result.
Final Test is not run. Local filesystem and temporary-remote tests are explicitly
separate evidence categories.
