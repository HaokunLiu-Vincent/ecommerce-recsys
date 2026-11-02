# Colab Migration Implementation Report

This document records the **earlier migration release**, including its original
blocked pilot status. The current executable pilot implementation and new-release
verification are documented in [Colab Pilot Implementation Report](colab_pilot_implementation_report.md).
The user subsequently reported that the earlier real Colab CPU smoke passed;
that user-supplied evidence is distinguished from local measurements there.

## Scope and delivery status

This is a partial migration: reproducible packaging, thin notebook bootstrap,
input staging, isolated CPU smoke stages, checkpointing and verified persistence
are implemented. Pilot/full profiles and plans are present, but execution is
blocked by explicitly identified scale adapters. No local pilot/full or real
Colab runtime was started. Existing smoke evidence is not a Colab measurement.

## Changes

- Added the standalone nine-section `01_Colab_Training.ipynb`, with empty outputs,
  safe defaults and shared CLI calls rather than model implementation.
- Added `hm_recsys.colab` helpers, three thin scripts and three explicit profiles.
- Reused existing model, data, feature and evaluation functions. Shared changes
  add per-fold execution, optional portable input identity and complete-batch
  checkpoints. The original `scripts/local_smoke.py` interface is unchanged.
- Added failure/recovery, incompatible identity, profile, release, notebook and
  staged synthetic integration tests. See the runbook for artifact contracts.
- Preserved dependencies and lockfile; no old notebook lint issues were changed.

## Verification

Actual local verification on 2026-11-02 used the existing Python 3.12 uv
environment. No dependency declaration or lockfile was changed. Commands used
`UV_CACHE_DIR=/tmp/hm-smoke-uv-cache /Users/vincent/.local/bin/uv run --locked`.
Logs and structured evidence are under
`data/runs/colab/local_validation_20261102_01/`.

| Command / check | Exit | Actual result |
|---|---:|---|
| `pytest -q` | 0 | 26 passed in 46.06 seconds |
| `ruff check src scripts tests notebooks/01_Colab_Training.ipynb` | 0 | Passed |
| `ruff format --check src scripts tests notebooks/01_Colab_Training.ipynb` | 0 | Passed |
| `pyright` | 0 | Zero errors, warnings and information messages |
| `ruff check .` | 1 | 122 existing notebook lint findings; preserved |
| `ruff format --check .` | 1 | Seven existing files need formatting; preserved |
| `nbformat.validate` and Python `compile` on new code cells | 0 | Passed; cells not executed |
| `python scripts/probe_device.py --device cpu --output data/runs/colab/local_validation_20261102_01/cpu_probe` | 0 | One real synthetic shared-model update and reload passed |
| `python scripts/run_pipeline.py plan --profile smoke ...` | 0 | Bounded CPU plan; no training |
| `python scripts/run_pipeline.py plan --profile full ...` | 0 | Explicit blocked_implementation plan; no training |
| `git diff --check` | 0 | No whitespace errors |

The root formatter includes three archived notebooks, three existing notebooks,
and the user-supplied migration prompt's example code. None was reformatted.
The original smoke tests remain enabled; no quality check was weakened.
Intermediate failures were a missing predictions directory in the new adapter
and an incorrect candidate-fixture extension in the new parity test. Both were
corrected and the complete suite passed afterward; intermediate logs are retained.

The integration test executed actual shared WALS, Two-Tower and LambdaMART on
**synthetic data**, not new H&M training: 24 users, 40 catalog items, 4,187
transactions. WALS ran 2 iterations with 16 factors; Two-Tower ran 38 optimizer
steps on 1,200 targets; LambdaMART ran 30 rounds on 3,840 rows in 96 groups.
All six channels emitted candidates in every fold and Validation. Validation's
24-query roster retained its one zero-hit query. These synthetic metrics are
contract evidence, not model-quality or full-data results.

The staged test deleted its local synthetic runtime after the learned index,
restaged inputs into a new directory, restored committed stages, completed all
five feature partitions and performed fresh-process reload. Repeating the run
preserved completion hashes. Earlier notebook cells can also revisit an already
completed run. Direct comparison against the original smoke entry point found
identical per-fold feature/candidate Parquet bytes, query/label JSON, negative
sampling, Top-12 and Validation metrics.

A separate process test interrupted the shared Two-Tower after step 2, restored
model/optimizer/scheduler/RNG and target position, and compared final weights,
learned vectors, losses and internal-validation output at exact zero tolerance.
It also resumed a final-batch checkpoint without requiring another update.
This is same-runtime CPU evidence, not CUDA continuation evidence.

Other tests cover interrupted copy, corrupt checksums, missing inputs, disk
limits, local completion followed by failed persistence, incompatible
code/config/input/schema/cutoff identities, dependency mismatch, writer leases,
mount-disconnection simulation, path traversal, full opt-in, local scale-up
rejection, disabled Test, notebook structure and deterministic release extraction.

Resource monitoring sampled process-tree RSS every 0.2 seconds. The staged
synthetic pipeline's observed maximum was **0.673 GiB**, including its test
supervisor; this can double-count shared pages and miss inter-sample peaks.
A separate plan observed 10.0 GiB available and retained the 5 GiB Mac abort
threshold. Neither is a hard-cap guarantee or a full-scale resource estimate.
The CPU probe changed the shared content projection by about 0.001003 and passed
weight reload. CUDA was not requested or tested.

The existing `data/runs/smoke/real_v1/` remains separate and untouched. No real
transaction table was copied or loaded for these migration tests, and no H&M
training was rerun in this task. Existing source embeddings/indexes, old
notebooks, dependency files and the user's migration prompt were preserved.

## Release evidence and handoff

Build with `uv run --locked python scripts/build_release.py`. The final package
is content-addressed under `data/releases/`; its archive identity is intentionally
not embedded in this report, which is itself a package member. The external
manifest and checksum provide the exact archive identity without a hash cycle.
`release_verification.json` under the local validation evidence directory records
the actual build, clean temporary extraction, entry-point help commands, and
bounded extracted-source tests. These are local package checks, not a Linux
installation or Colab bootstrap execution.

Upload the ZIP, checksum and manifest together into the Drive project's
`releases/`; open `notebooks/01_Colab_Training.ipynb` separately in Colab, fill in
the archive name, and run smoke with all safe defaults. Follow
`colab_runbook.md` for source paths, stage boundaries, recovery and results.

Actual Colab execution, locked Linux dependency installation, Drive FUSE
publication semantics, CUDA update/reload, pilot/full training, final Test and
deployment are **not run**. Pilot/full additionally remain
**blocked_implementation**, not merely untested. This release does not satisfy
the complete pilot/full migration milestone.

## Remaining implementation blockers

Streaming transaction/history access; disk-backed global cutoff ItemCF and
statistics; full-vocabulary learned-index lifecycle; bounded columnar ranker
training and complete Validation ingestion; integrated multi-epoch selection
and full-population K=128 preprocessing. These are enforced in the CLI, not
hidden by lower sampling limits. No cloud or full-scale completion is claimed.
