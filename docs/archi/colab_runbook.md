# H&M Colab Runbook

The current primary workflow is [GitHub checkout](github_colab_runbook.md):
open the checked-in Notebook from GitHub, pin its actual Commit, and stage only
validated Drive inputs. No source archive upload is required. Existing archive
run identities and the archive setup examples below are historical compatibility
instructions; they do not apply to new Git runs. Keep the original Commit and
Git identity for new-run recovery. Model, resource and progress details below
remain applicable unless explicitly identified as historical.


## Contents

- [Scope](#scope)
- [Build and upload](#build-and-upload)
- [Start smoke](#start-smoke)
- [Stages and recovery](#stages-and-recovery)
- [Profiles and blocked scale paths](#profiles-and-blocked-scale-paths)
- [Environment and verification limits](#environment-and-verification-limits)

## Scope

The thin Colab entry point supports the existing CPU smoke and an executable,
explicitly opted-in pilot backend. See [Colab Pilot Runbook](colab_pilot_runbook.md)
for pilot release/configuration/commands and current verification limits. Full now has an explicit execution path; see [Full Runbook](colab_full_runbook.md).
Final Test, tuning and deployment remain disabled. No existing source embedding
or completed smoke artifact is replaced.

The six quotas remain 60/30/30/50/50/50. One LambdaMART uses four disjoint
training folds, native candidates, shared feature schema, Train-only 1:20
hard/random negative sampling, and complete ground truth within the configured
query roster. Missing recall remains a miss. Smoke sampling is explicitly
limited and is not a full-population metric.

## Build and upload

From the repository, using its existing locked environment:

```bash
uv run --locked python scripts/build_release.py
```

The command prints the content-based build ID and ZIP SHA256. Outputs are
`data/releases/hm_recsys_<build_id>.zip`, the matching `.sha256`, and
`.manifest.json`. The manifest records every member's size/hash, lock hash,
source content identity and informational Git revision. ZIP member order,
timestamps and modes are fixed. Git is not initialized or modified. A changed
source produces a new package name; incompatible existing output is refused.

Upload those three files to:
`/content/drive/MyDrive/projects/e_commerce_recsys/releases/`.
Open `notebooks/01_Colab_Training.ipynb` separately through Colab's Upload notebook
interface. The same notebook is included in the ZIP for traceability; opening it
does not require unpacking the ZIP first. The package includes source, configs,
tests, current documents and the existing text-extraction notebook needed by
its existing static tests. It excludes `data/`, environments, caches,
credentials and `docs/achive/`. It contains no real user histories or vectors.

Checksums detect corruption; obtain all three release files from the same
trusted build. They are not a digital signature or publisher authentication.

## Start smoke

Set `RELEASE_NAME` to the printed ZIP name. Keep these defaults:

```python
RUN_PROFILE = "smoke"
RUN_ID = "colab_smoke_v1"
RUN_ACTION = "run"
ALLOW_FULL_RUN = False
RUN_FINAL_TEST = False
```

Choose an unused ID for a new run. Execute cells in order. They mount Drive,
verify/extract the release to `/content/e_commerce_recsys/`, install pinned uv
and Python 3.12 into an isolated locked environment, and invoke the shared CLI.
The notebook kernel's PyTorch is not replaced. A failed install or command
stops its dependent cells. Full command logs are local under
`data/runs/colab/<run_id>/notebook_logs/`; stage logs and sampled memory evidence
are included in committed stage deltas on Drive.

Only these required input directories are staged from the fixed Drive data root:

- `original data/`: articles, customers and transaction CSVs.
- `text_features/`: vectors, ID map, manifest and original FAISS file.
- `image_features/`: vectors, ID map, manifest, failures and original FAISS file.

The explicit whitelist contains 12 files. Extra files and old runs are not
copied. Copies use bounded buffers and source/destination checksums, then publish
atomically. Interrupted copies are not completed inputs. Existing local files
are reused only after destination hashing and source size/mtime checks; use
`--reverify` to hash Drive source bytes again. Size/mtime is a fast change
detector, not proof against a source changed while preserving both attributes.
Corrupt local copies are preserved with a `.rejected-*` suffix before recopy.
Source arrays, maps, manifests, failures and indexes remain read-only. Original
content indexes are never treated as trained Two-Tower indexes.

Example commands in the extracted project's environment:

```bash
.venv/bin/python scripts/run_pipeline.py plan --profile smoke \
  --persistent-root /content/drive/MyDrive/projects/e_commerce_recsys/data/runs/colab
.venv/bin/python scripts/run_pipeline.py stage-inputs \
  --source-root /content/drive/MyDrive/projects/e_commerce_recsys/data \
  --data-root /content/e_commerce_recsys/data
.venv/bin/python scripts/run_pipeline.py run --profile smoke --run-id colab_smoke_v1 \
  --mount-root /content/drive \
  --persistent-root /content/drive/MyDrive/projects/e_commerce_recsys/data/runs/colab \
  --until reload
```

`preflight` runs input acceptance only. `plan` never trains. `status` verifies
hashes instead of trusting directories. Use `--until <stage>` to stop at a
committed boundary. A fresh run defaults to the complete smoke/reload path.

## Stages and recovery

The boundaries are `preflight`, `sample`, `wals`, `towers`, `index_towers`,
`encode`, `preprocessing`, `features_F1` through `features_F4`,
`features_Validation`, `ranker`, `reload_encode`, `reload_features`, `reload`.
Each executes in a fresh child; PyTorch, FAISS and LightGBM retain their separate
process lifetimes. Feature stages reuse frozen base models and build dynamic
past-only histories at their own cutoff.

Local training stays under `/content/e_commerce_recsys/data/runs/colab/<run_id>/`.
Every completed stage validates required outputs, publishes only changed files
under a unique attempt, verifies the Drive copies, and writes `complete.json`
last beneath:
`/content/drive/MyDrive/projects/e_commerce_recsys/data/runs/colab/<run_id>/`.
Its manifest binds dependencies, source content, config, code, schema definition,
cutoffs, lock and numerical runtime. Local completion and persisted completion
are separate. Failure metadata does not imply success. The final `artifacts.json`
maps logical model/schema/result names to committed relative paths; it is an
inspection convenience, while stage manifests remain the authority.

In a new runtime, open the same standalone notebook, use the same release,
profile and run ID, and set `RUN_ACTION = "resume"`. Run from the top. Inputs are
restaged; verified artifacts are restored from Drive. Changed source, code,
configuration, schema, cutoff or Python/numerical runtime is refused. A local
Mac run cannot be resumed as a CUDA/Linux run by renaming directories.

Only one writer may own a run. After an abrupt runtime loss, verify the previous
runtime has stopped, then use `RECOVER_LOCK = True` (CLI `--recover-lock`) once.
Old lease evidence is retained. Do not recover a lock while another worker is
active. Separate runs use separate IDs.

WALS, FAISS, preprocessing, each feature fold and LambdaMART restart at a stage
boundary after failure. Two-Tower saves complete-batch model, optimizer,
scheduler, Python/NumPy/Torch RNG, epoch, step, target offset and sampling/order
identity. Float32 uses no AMP scaler. The checkpoint is hashed before transfer;
weights alone are never accepted as a continuation checkpoint. Same-runtime CPU
continuation is tested separately from cloud persistence. CUDA continuation and
cross-device equivalence are not claimed.

Drive disconnection or disk/resource exhaustion stops publication. An already
successful local stage can finish persistence on resume without retraining.
An invalid completed artifact is rejected, not silently recomputed over the
existing evidence. Diagnose it and use a new run ID if necessary. The filesystem
protocol is tested on local directories; Drive FUSE atomicity and disconnect
behavior require actual Colab verification.

## Profiles and blocked scale paths

`configs/colab_smoke.json` exactly retains the existing smoke compute limits.
`colab_pilot.json` specifies legal-past stable-hash base selection (no activity
threshold), a bounded F1/Validation roster, and explicit training caps.
`colab_full.json` removes user/catalog/transaction/query/target/step caps,
requires all four folds and Validation, WALS 64 factors/15 iterations,
Two-Tower 10 epochs, K=128 visual clustering, and the formal LambdaMART settings.
Batch size is explicit: 512 changes the number of in-batch negatives relative
to the proposal's 2048 and must be recorded as a training configuration choice.
The pilot uses its dedicated bounded adapters; it does not run smoke adapters at scale.

`plan --profile pilot` describes the implemented SQL-backed global history,
bounded full-catalog learned indexes, F1/Validation feature shards and single
resource-estimated LambdaMART fit. Pilot requires `--allow-pilot` and a Colab
runtime; it never requires `--allow-full`. Its production budget is unchanged.
The pilot runbook describes actual stages and separate CPU/CUDA selection.

`plan --profile full` now describes the implemented full adapter. Full requires
`--allow-full`, a new full run ID, a Colab runtime and independent resource
acceptance. It executes all four training folds and complete Validation. Read
[Full Runbook](colab_full_runbook.md) for null-limit semantics, bounded readers,
resource estimates and recovery. Final Test remains disabled.

## Environment and verification limits

The notebook installs `uv==0.13.6` separately and uses `uv sync --locked` with
Python 3.12; it does not regenerate `uv.lock`. Locked Linux wheel declarations
exist, but their installation and CUDA ABI are not locally proven. The shared
Two-Tower probe can run `scripts/probe_device.py --device cuda --output
<project>/data/runs/colab/<new_probe_id>`; it performs one actual synthetic
forward/backward/optimizer update and weight reload in the project interpreter.
Unavailable CUDA raises instead of falling back. CPU smoke uses CPU explicitly.

The resource plan observes available RAM, reserves headroom and sets sampled
process-tree RSS abort thresholds. Mac thresholds never exceed 5 GiB; actual
available-memory floor and 0.2-second monitoring remain active. Disk checks
reserve 10 GiB beyond staged bytes; environment setup reserves another 15 GiB.
Measurements are sampled diagnostics, not OS-enforced memory guarantees. A new
runtime must pass current preflight; saved budgets do not establish capacity.

Local verification, simulated persistence, CPU synthetic execution, actual
Colab smoke, CUDA, pilot/full and final Test are separate evidence categories.
Read `colab_implementation_report.md` for measured status and command outcomes.
