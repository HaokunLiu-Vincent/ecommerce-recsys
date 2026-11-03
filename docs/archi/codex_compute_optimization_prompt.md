# Codex Task: Optimize H&M Compute Without Changing the Experiment

## Contents

- [1. Mission and scope](#1-mission-and-scope)
- [2. Read first and preserve the working repository](#2-read-first-and-preserve-the-working-repository)
- [3. Non-negotiable model and data contracts](#3-non-negotiable-model-and-data-contracts)
- [4. Establish an attributable performance baseline](#4-establish-an-attributable-performance-baseline)
- [5. Optimize Two-Tower data preparation](#5-optimize-two-tower-data-preparation)
- [6. Optimize static item lookup and device transfers](#6-optimize-static-item-lookup-and-device-transfers)
- [7. Preserve the loss while removing batch-construction overhead](#7-preserve-the-loss-while-removing-batch-construction-overhead)
- [8. Separate compute from validation and persistence costs](#8-separate-compute-from-validation-and-persistence-costs)
- [9. Optimize shared retrieval and ranking features](#9-optimize-shared-retrieval-and-ranking-features)
- [10. Keep numerical and training changes separate](#10-keep-numerical-and-training-changes-separate)
- [11. Verification and resource limits](#11-verification-and-resource-limits)
- [12. Deliverables and acceptance](#12-deliverables-and-acceptance)
- [13. Official implementation references](#13-official-implementation-references)

## 1. Mission and scope

Implement and verify performance improvements in the existing H&M pipeline,
prioritizing Two-Tower training throughput, data preparation, static feature
reuse, internal validation, and checkpoint I/O. Reuse the working full backend.
This is an implementation task, not another architecture proposal or a request
to write a further prompt. Profile, make actual changes, test, and package them.

The user observes about 0.4 GB of GPU memory usage with slow training. Treat this
as an observation, not a diagnosis. Optimize elapsed time and useful throughput,
not GPU memory occupancy for its own sake.

The supplied discussion suggests data starvation, Python row assembly,
small-kernel overhead, GPU-resident item features, and larger batches. Evaluate
these as hypotheses. Do not claim 95% GPU idleness, a particular Colab host CPU,
20–50x speedup, or a fixed number of minutes without measurements. Do not infer
an underlying GCP VM type, physical-core count, bandwidth, or PCIe speed from the
Colab product name. Capture the actual runtime environment instead.

Previously inspected release: `hm_recsys_9286d0aa9ca4c204.zip`.
It already uses a small multimodal MLP, unique in-batch candidate items,
a multi-positive LogQ loss, and a 49-column LambdaMART feature matrix. It is not
an image encoder being trained end to end, a pairwise BCE model needing a new
negative sampler, or a ranker receiving hundreds of raw content dimensions.
Verify these facts against the current working tree before editing.

No real full execution, final Test, deployment, or parameter search is
authorized on the local Mac. Deliver a bounded Colab benchmark entry point for
the user to run, followed by the existing explicitly authorized full workflow.

## 2. Read first and preserve the working repository

Read applicable `AGENTS.md` files and the actual current versions of:

- `docs/proposal.md` (resolve its actual location; do not create a conflicting copy).
- `docs/archi/architecture.md`, `evaluation_protocol.md`, and
  `implementation_decisions.md`.
- Current full/pilot runbooks and implementation reports.
- Relevant compute-audit documents if present; their absence is not a blocker.
- `configs/colab_full.json`, `configs/colab_pilot.json`, and the existing smoke
  configurations, CLI, notebook, tests, and release builder.

Inspect current code rather than restoring the inspected ZIP. Preserve user
fixes, including run-ID handling, existing uncommitted changes, old notebooks,
raw data, and precomputed embeddings. Do not initialize Git, commit, or push.
Do not run archived notebooks or import their superseded architecture.

Initial code locations to inspect, subject to current repository changes:

| Location | Reason to inspect |
|---|---|
| `pilot/towers.py::SqlEncoder.users` | Queries a batch, then calls the shared encoder one query at a time and concatenates small tensors |
| `pilot/data.py::History.event_histories`, `positives`, `customers` | Repeated SQL history, same-day-positive, and profile lookups |
| `pilot/streaming.py::TargetStream`, `rows` | Stable target ordering, Python row conversion, and resumable offsets |
| `towers.py::TowerEncoder.items/users`, `positive_mask`, `logq_loss` | Static lookup, nested loops, and exact learning semantics |
| `pilot/towers.py::move`, `fit`, `internal_recall`, `item_vectors` | Host/device transfers, synchronization, validation, and catalog encoding |
| `colab/checkpoints.py::save_checkpoint` | Serialization, hashing, verified Drive copies, and retention |
| `features.py::FeatureBuilder.build` | Repeated item-cluster assignment and user-history computations |
| `recall.py`, `pilot/snapshot.py`, `pilot/vectors.py` | Repeated popularity ordering, active-item filtering, and exact retrieval |
| `pilot/ranking.py`, `pilot/capacity.py`, `resources.py` | Ranker resource estimates and shared safeguards |

Use English identifiers, comments, docstrings, errors, and repository documents.
Follow the existing Google-style documentation, Ruff, Pyright, pytest, and
Notebook standards. Keep the Notebook thin and the implementation under `src/`.

## 3. Non-negotiable model and data contracts

The default optimized execution must preserve:

1. Six recall channels and quotas `60/30/30/50/50/50`; native deduplication; one
   LambdaMART; the current feature schema/order/types, currently 49 columns.
2. The existing frozen base-model boundaries, dynamic past-only histories,
   disjoint F1–F4/Validation label windows, full ground truth, and zero-hit and
   zero-candidate Validation queries. Final Test stays disabled.
3. Native ranking candidates; no positive injection; unchanged Train-only 1:20
   hard/random negative sampling and metric denominators.
4. Two-Tower target population and stable SHA256 ordering, existing epoch order,
   complete/partial batch boundaries, batch size 512, epoch count, optimizer,
   learning rate, scheduler step rules, precision policy, and model dimensions.
5. Frozen category mappings, unknown-ID semantics, modality masks, padding,
   weighted pooling, and the shared article/product-ID dropout and its RNG use.
6. Training history: the same latest 20 **raw purchase lines**, with the existing
   time lower bound, strictly before the event date, same-day rows excluded,
   duplicates retained, and identical deterministic tie ordering. Do not replace
   this with 20 distinct items or a user's final cumulative history.
7. Candidate items in each contrastive batch: the same sorted unique article
   IDs. All known same-query/same-day positives remain positive, including legal
   positives outside a target cap. Keep invalid-row skipping and target-offset
   accounting unchanged.
8. Exact training-population Q and the existing multi-positive loss. With score
   `s(u,i) = dot(u,i)/0.07 - log(Q(i))`, each valid row uses the difference between
   logsumexp over all batch candidates and logsumexp over its positive mask.
   Do not replace this with diagonal-only cross-entropy or remove LogQ.
9. Full internal-validation population, daily truth, active-pool policy,
   Recall@50, per-epoch selection, and deterministic best-epoch tie behavior.
10. WALS confidence/folding-in and ItemCF formulas, retrieval ordering, resource
    rejection behavior, and verified artifact/checkpoint identities.

Performance settings are part of run provenance. Use a new release and new run
IDs; never edit an existing run's identity to force reuse. Exact numerical
identity across devices is not assumed. Any non-equivalent behavior must be an
explicit experiment, never silently enabled as a performance-only change.

## 4. Establish an attributable performance baseline

Add a small integrated profiling/benchmark facility rather than a second
training framework. Preserve a callable reference implementation for bounded
A/B comparisons, without copying the entire project into a second pipeline.
The reference is the current code before these optimizations, not an old model.

### 4.1 Measurements

Instrument these boundaries separately:

- Input/canonical preparation and one-time cache construction.
- Target reading, history query, positive-set query, and profile lookup.
- CPU batch assembly, host-to-device copies, and waiting for prepared batches.
- User/item forward passes, contrastive loss, backward, and optimizer update.
- Scalar reads or synchronizations that stall the host, such as loss logging.
- Complete-catalog encoding, internal retrieval, metrics, and epoch selection.
- Checkpoint snapshot/serialization, hashing, local write, remote copy,
  remote verification, and completion publication.
- Each recall channel, shared feature construction, negative sampling,
  ranker dataset construction, fit, validation scoring, and reload.

Produce a compact machine-readable summary with counts, bytes, p50/p95 step
latencies, examples/second, query/candidate rows per second, actual optimizer
steps, peak RSS, CUDA allocated/reserved peaks, and cache hit/miss counts.
Capture actual CPU affinity/quota where available, thread settings, RAM,
GPU model, CUDA/PyTorch versions, and local/persistent filesystem paths.
Optional sampled GPU utilization must be labelled as a sample, not an inferred
constant. Missing telemetry is unavailable, never fabricated.

Use `perf_counter` for host wall time and CUDA events/profiler for device timing.
Synchronize at measurement boundaries when required; do not put a global CUDA
synchronization around every operation in the production loop. Host/device
work can overlap, so do not sum overlapping durations into a false 100% pie.
Use short `torch.profiler` windows with explicit warmup and named regions.
Report profiler overhead separately; throughput comes from non-traced runs.

### 4.2 Comparable runs

Provide three bounded modes: data preparation only, model-only replay of valid
prepared batches, and end-to-end batches including normal I/O. Replay must be
labelled diagnostic, not real training population coverage.

Use identical events, order, candidate sets, starting model/optimizer/RNG,
precision, batch boundaries, update count, and environment for reference versus
optimized comparisons. Warmup must not secretly advance the measured training
state. Report cold-cache cost and warm-cache cost, including amortized cache
construction over the configured ten epochs. Measure enough distinct batches
to avoid an artificially tiny working set. Keep local tests much smaller.

A suggested user-launched Colab diagnostic budget is 10 warmup and up to 100
measured batches per variant, with a configurable cap, not an automatic full
run. Do not claim it measures full-epoch validation or full-data I/O unless it
does. Reference and optimized timings must cover the same stages and work.

## 5. Optimize Two-Tower data preparation

### 5.1 Remove repeated history and positive SQL from the steady-state loop

Implement a bounded prepared-input representation, or an indexed equivalent
that demonstrably eliminates repeated scans/joins. Do not merely cache entire
Python `Transaction` lists for every user.

A suitable design has:

- A stable integer catalog-row mapping, separately from learned categorical IDs.
- One row per unique training `(user, event_date)` context, shared by its items.
- Encoded profile IDs, latest-20 historical category IDs, weights, and padding.
- Ragged complete same-day positive item lists, using offsets plus flat IDs.
- A compact ordered target stream referencing context ID and target article row.

Precompute only model-independent inputs. History **IDs and weights** are
cacheable; outputs of learned embedding layers, trainable content projections,
pooled learned history vectors, and final tower vectors are not invariant during
training. Keep their forward computation and gradients live after every update.

Build contexts using bounded sorted passes, window logic, or indexed per-user
history slices. Avoid a giant target-by-all-past-transactions join, full Python
object materialization, and a dense `[all_events, all_items]` positive matrix.
Measure the query plan and actual rows read; do not assert a full Parquet scan
per batch merely because a SQL call is present.

Deduplicate only context records, not the learning population. Preserve raw-line
history duplication, exact sort order, full positive sets, selected target Q,
and existing partial-batch behavior. Keep internal-validation context caches
separate from training artifacts; label-derived positives must never become
inference features or enter training-time sampling probabilities.

### 5.2 Batch-native reads

Construct `[B,3]` profiles, `[B,20,3]` history codes, and `[B,20]` weights in
contiguous batches, matching current shapes. Remove the per-query
`super().users([query])` followed by many `torch.cat` calls from the optimized
path. Use vectorized NumPy/Arrow/tensor indexing, not `.iloc` loops or per-row
Tensor construction. The existing code is a custom SQL/stream loop; do not
invent a DataLoader bottleneck that is not present.

Use mmap/columnar shards and bounded gathers; do not duplicate content vectors
per interaction. Compact safe disk types are permitted, with explicit overflow
checks and the exact required model-input dtype after conversion. PyTorch
`torch.long` is int64; account for actual dtypes, not a universal four bytes.

Prefetch at most a configured bounded number of batches. Preserve consumption
order and RNG; checkpoint the last completed batch, never the producer's
prefetched position. Avoid sharing a DuckDB connection unsafely across workers
or forking an initialized CUDA process. Keep a zero-worker CPU implementation.
If using a DataLoader, use an appropriate batch-native API supported by the
locked PyTorch version, not an assumed list-valued `__getitem__` protocol.

### 5.3 Cache identity and memory budget

Cache identity binds source hashes, catalog/category maps, encoder version,
cutoffs, history ordering/length, masks, target population/order and relevant
configuration. No path-existence-only reuse. Publish atomic manifests; reject
partial/corrupt/stale caches. Integrate caches into actual stage persistence and
resume, rather than rebuilding them silently on every new runtime.

Estimate bytes from the real number of contexts, targets, positive edges and
item dimensions before allocation. Include prefetch, page cache, pinned buffers,
serialization and scratch headroom. Do not assume the whole training dataset
fits GPU memory just because a hypothetical ID-only table would fit.

## 6. Optimize static item lookup and device transfers

Build one frozen **input** table per catalog item: six categorical codes, text
vector, image vector, availability masks, and stable row mappings. Read actual
embedding dimensions from validated artifacts. Do not hardcode the discussion's
128/384 dimensions or confuse raw features with the learned 128d output.

When CUDA is requested and a measured budget permits, load these static tables
once onto that device and gather by integer catalog positions. Use a dedicated
provider or non-persistent buffers to avoid serializing the entire frozen input
bank in every optimizer checkpoint. Reconstruct the bank from hash-verified
sources on reload. `nn.Embedding.from_pretrained(..., freeze=True)` is optional;
plain buffers/indexing may be simpler. Do not freeze learned ID embeddings or
trainable text/image projection layers.

Keep distinct concepts separate: catalog position for content lookup versus
category code 0 for unknown learned IDs. Two unknown-category articles may still
have distinct valid content; they must not be mapped to the same content row.
Preserve missing-image/text masks and existing normalized vector semantics.

Budget GPU resident inputs, model, gradients, AdamW state, activations,
contrastive logits, temporary buffers and reserve. Do not rely only on model
parameter bytes. When the frozen bank cannot fit, use a logged bounded host
lookup/transfer path while keeping the requested model device CUDA. This is
input-placement fallback, not silent model-device fallback.

For host-resident inputs, implement reusable pinned staging buffers and bounded
prefetch where supported. `non_blocking=True` alone does not prove transfer and
compute overlap. Manage stream events and buffer lifetimes correctly; never
reuse a source buffer while DMA is reading it. Do not call `.pin_memory()` on
freshly allocated tensors inside every step and assume it is faster. Benchmark
this against simpler transfers and keep the CPU path fully functional.

## 7. Preserve the loss while removing batch-construction overhead

Vectorize complete-positive-mask construction using compact positive IDs and
batch candidate positions. Keep memory proportional to the bounded batch, its
unique candidates and relevant positive edges, not total dataset size.

The optimized batch must have identical candidate column order, positive mask,
Q values, valid-row selection and target count. Repeated users, same-day multiple
positives, duplicate target items, and all-positive candidate rows need explicit
tests. A diagonal label tensor is not an acceptable replacement.

Do not cache current learned item vectors across optimizer steps. Do not freeze
user-history pooled embeddings to save CPU time. Do not introduce random
per-epoch permutations or extra RNG draws in prefetch/profiling; current target
order and ID dropout randomness are part of the experiment.

Inspect per-step CPU/GPU synchronization from `valid.any()`, finite checks,
`.cpu()`, `.item()`, and logging. Consolidate redundant work where safe, but do
not remove numerical failure detection or silently update on a nonfinite loss.
Move checks on CPU-known masks before transfer when that preserves behavior.
Any deferred failure reporting must be documented and bounded, not hidden.

## 8. Separate compute from validation and persistence costs

### 8.1 Internal validation

Cache model-independent daily query rosters, profiles, histories, complete
truth and active item positions. Preserve every required internal query and the
same metric/denominator. Re-encode learned item/query vectors for the current
model at each required epoch; never reuse an earlier epoch's vectors as if the
weights had not changed.

Optimize exact retrieval in bounded blocks, initially by avoiding repeated
membership construction and unnecessary string conversions. A tested optional
GPU exact-matmul path is allowed, but no approximate ANN substitution or
validation subsampling. Preserve score semantics and deterministic ties. Test
near ties, missing history, and empty pools; float32 GPU matmul is not promised
to be bitwise identical to CPU. Keep the reference path where equivalence cannot
be established, and report any numerical or ranking differences explicitly.

### 8.2 Checkpoint I/O

The inspected full configuration saves and publishes roughly every ten updates.
Measure local serialization, hashing, remote copy and verification separately
before attributing all time to data loading.

Implement separate, configurable local save and remote publish cadences, rather
than automatically copying to Drive on every local save. Keep the legacy policy
available. Supply an explicit new performance configuration with starting policy:

- Save at a complete batch after 100 optimizer updates **or** 60 seconds since
  the previous local save, whichever condition is first met.
- Publish a complete local snapshot at a complete batch when 300 seconds have
  elapsed since the last successful remote publication.
- Force verified save/publication for the first successful update, epoch
  selection, normal finalization and a safely handled explicit stop.

These are configurable starting values, not measured optima or a guarantee of
at most five minutes lost work. State that runtime loss can lose everything
since the last **verified remote** snapshot, including publication latency.
Separate local and persistent latest pointers. Publish the remote pointer last;
retain the previous verified remote checkpoint until its successor is verified.
On publication failure, surface the failure and preserve the last recoverable
state; do not claim a successful durable checkpoint or silently continue without
bound. Do not promise saving on hard OOM, SIGKILL or runtime destruction.

Use immutable complete-batch model/optimizer/scheduler/RNG/order/epoch/offset/
best-model snapshots. Simpler synchronous publication at a lower frequency is
acceptable; do not add unsafe asynchronous `state_dict` references that mutate
while being saved. Avoid copying frozen input banks into snapshots and avoid
rehashing every original data file on each batch. Preserve meaningful loss and
training progress records without repeatedly copying unbounded Python logs when
an append-only bounded representation can maintain the same reporting contract.
Test cadence with a fake clock and injected failures, not real sleeps.

## 9. Optimize shared retrieval and ranking features

Implement inexpensive semantic-preserving caches in the shared feature path:

- Precompute each item's assignment to the frozen K=128 image cluster once after
  fitting/loading the centers. Match the old squared-distance and argmin tie
  semantics; missing images remain missing. Bind cache to center/vector hashes.
  Do not approximate the centers or change the fit population.
- Precompute per-cutoff/global and age-bucket popularity ordering; reuse it
  across queries while retaining active filtering and ID tie-breaking.
- Compute per-user/cutoff counts, category-affinity maps, prices, latest content
  summaries, and Repurchase/Sibling outputs once and reuse per candidate.
- Batch user-item dot products and static metadata lookups, with unchanged
  normalization, masks, NaNs, schema, order and dtype.

Different features have different history windows. Do not accidentally replace
ranker 90-day/all-past statistics or its last-five content summary with the
Two-Tower last-20 event cache. WALS folding-in's fixed Gram cache already exists;
retain it rather than building a duplicate solver.

Do not change the point at which ranking negatives are sampled in this task
unless exact row selection and required diagnostics can be proven unchanged.
Do not drop candidate/provenance output or zero-hit queries to improve timing.
Do not expand raw embeddings into LambdaMART, change tree parameters, or train
one LightGBM repeatedly on separate shards. Keep its single-fit memory guard.

Make thread budgets explicit per stage. Do not replace every thread limit with
`-1`, assume 12 physical cores, or permit nested BLAS/OpenMP/loader pools to
oversubscribe the runtime. On Colab, provide bounded thread comparisons where
useful; retain the Mac's two-thread limit and native-library process isolation.

## 10. Keep numerical and training changes separate

The first comparison is batch 512, existing FP32 settings, same learning rate,
loss, epochs, target order and negatives. Obtain evidence for engineering
improvements before changing the learning problem.

Do not automatically enable larger batches, TF32, AMP, `torch.compile`, CUDA
Graphs, a different optimizer, or a new negative sampler. They may be separate
explicit opt-in experiments after baseline equivalence, but are not required
for this delivery. Do not upgrade dependencies solely to gain these features.

In particular:

- Increasing the contrastive batch changes negative candidates, Q correction
  context, update count and learning dynamics. Gradient accumulation is not
  equivalent to a larger in-batch-negative matrix.
- With B unique candidates, just a float32 B-by-B logits array needs `4*B*B`
  bytes: B=8192 is 256 MiB; B=32768 is 4 GiB, before masks, gradients and
  other intermediates. Do not call giant batches free.
- A different batch/precision policy requires a new experiment identity and
  quality evaluation, not a performance-only parity claim. Do not scale the
  learning rate automatically.
- Kernel-launch overhead must be visible in profiling before considering
  compile/graph complexity; include compilation warmup and dynamic-shape costs.

Keep these experiments off by default. The user is asking for faster execution
of the established design, not a new recommendation model.

## 11. Verification and resource limits

### 11.1 Required correctness tests

Test reference versus optimized behavior on small fixtures:

1. Ordered target/context mappings, batch boundaries and exact masks/Q.
2. Same-day exclusion; repeated raw lines; 20-history truncation; deterministic
   ties; cold users; unknown category IDs; missing modalities; distinct articles
   sharing unknown learned IDs; empty/all-positive batches.
3. Encoded inputs, forward outputs, loss and gradients. With identical starting
   states and RNG, compare one update and a short multi-step run; check dropout
   RNG, skipped rows, optimizer/scheduler counts and partial final batches.
4. Prepared data cannot change when only future transactions are added; training
   caches do not use internal/outer Validation or final Test supervision.
5. Cached input features remain fixed while trainable history and content
   projections still update; no stale model-dependent vector reuse.
6. Interrupted/restarted optimized runs with prefetch recover the exact consumed
   position. Same-runtime deterministic CPU continuation should retain existing
   exact checks; CUDA tests are separate and cannot be marked passed if skipped.
7. Local versus remote checkpoint cadence, failures before/after publication,
   stale latest pointers, failed uploads, corrupt caches, cache invalidation and
   new-runtime reconstruction of non-persistent device input banks.
8. Feature schema stays 49 columns; categorical cluster IDs, source flags,
   candidate order, sampled training rows, metric denominators and Top-12 agree
   with the reference. Discrete fields are exact; continuous differences require
   documented existing or explicitly justified tolerances, never loose thresholds
   chosen only to pass. Keep CPU/GPU claims separate.
9. Resource rejection, bounded pinned/prefetch buffers, full authorization,
   unique run IDs, explicit CUDA-unavailable failures and disabled final Test.
10. Actual Notebook-generated CLI commands are parsed by real entry points.
    New settings are not silently ignored; the old smoke/pilot/full interfaces
    and stage dispatch retain their intended behavior.

Do not require fragile wall-clock speed assertions in normal CI. Instead test
that optimized steady-state execution avoids repeated history SQL, row-wise
encoder calls, per-candidate cluster assignment, and redundant static content
transfers. Use measured benchmark artifacts for performance acceptance.

### 11.2 Allowed execution

Local machine: M1 Max with about 7–8 GB available RAM. Use at most two threads,
small synthetic inputs and existing safe smoke limits. Preserve the process-tree
memory monitor, its existing Mac abort ceiling (no higher than 5 GiB), and
system-memory reserve. No full-data scan/prepack, real full training, global
ItemCF rebuild, or large benchmark is authorized locally.

A short fixture must execute the same optimized backend, not mocks replacing
models. CUDA tests can be skipped explicitly on the Mac. Without a real CUDA
runtime, deliver the implemented GPU path, tests and the user-run benchmark,
and say CUDA performance is not measured. Never present CPU timing as L4 timing.

Use the locked project environment. Run relevant pytest, Ruff, formatting and
Pyright checks; preserve unrelated legacy lint issues. Do not replace raw inputs,
install global packages, disable warnings without justification, or reset user
work. Protect existing runs while inspecting a possibly still-running job; do
not hot-edit its files or overwrite its checkpoints.

## 12. Deliverables and acceptance

Deliver actual implementation, not instrumentation alone:

1. Batch-native Two-Tower input preparation with a bounded persistent context
   representation and exact positive/Q semantics.
2. Static item lookup with a budgeted GPU-resident option and bounded host path.
3. Equivalent batch mask/loss inputs and live trainable projections.
4. Configurable local/remote checkpoint cadence with verified recovery.
5. Frozen visual-cluster and repeated user/cutoff statistic caching.
6. Baseline/optimized benchmark and short-profiler CLI integrated with the
   existing project, plus correctness/resource tests.

Do not introduce another top-level training profile that accidentally misses
full stage dispatch. Prefer a documented performance-settings section/overlay
on the existing smoke/pilot/full profiles. Preserve the legacy reference mode
for tests and bounded A/B diagnosis. Resolve performance settings into the run
identity and summaries; the shipped Notebook remains safe by default.

Produce:

- `docs/archi/compute_performance_report.md`: observations, hypotheses,
  measured findings, changes, tradeoffs, tests and remaining unverified work.
- `docs/archi/compute_performance_runbook.md`: actual copyable configuration
  and commands for bounded reference/optimized Colab benchmarking, reviewing
  results, and starting a new optimized full run after acceptance.
- An integrated thin Notebook update, matching new ZIP, SHA256 and manifest.
- Run artifacts under `data/runs/benchmark/<run_id>/` (or the existing managed
  run tree if required by persistence), including `performance_summary.json`,
  `parity_report.json`, optional bounded traces, and an aggregate review bundle.
- A comparison table: backend, work counts, batch, device, precision, cache
  preparation time, data wait/assembly, transfer, GPU compute, internal
  validation, checkpoint local/remote time, total wall time, throughput,
  RSS/VRAM and verification status. Mark unmeasured cells `not_run`.

The performance summary must distinguish: steady-state train throughput,
complete training including cache/validation/persistence, and whole-pipeline
time. Report speedup only as measured reference time divided by measured
optimized time for identical work. No promised multiplier or runtime estimate.

The Colab workflow must offer a bounded benchmark before another ten-epoch full
job. Benchmark outputs must never masquerade as full training artifacts or model
quality results. Training and staging stay on `/content/e_commerce_recsys/`;
Drive inputs remain read-only at
`/content/drive/MyDrive/projects/e_commerce_recsys/data/`. Do not retrain text/image
encoders, upload raw histories into reports, or overwrite the source embeddings.

New code/config uses a new release and run ID. Do not force-resume an older
optimizer checkpoint into changed training code. Any supported checkpoint
conversion is a separately tested, explicit migration; it is not required here.
Publish instructions for safely leaving the old run's evidence intact.

Final response: list actual changes, verified unchanged semantics, measured
bottlenecks/speedups and their scope, commands and exit codes, artifacts, exact
next Colab steps, and anything not executed. If the work remains partial, name
the missing operation rather than declaring a successful optimization.

Stop after implementation, bounded local verification and packaging. Real
Colab benchmarks and full training are launched explicitly by the user.

## 13. Official implementation references

Use the locked dependency version's supported APIs. These references explain
measurement/loading mechanics; they do not establish this project's bottleneck
or any speedup. Version the actual documentation consulted in the report.

- [PyTorch profiler](https://docs.pytorch.org/docs/stable/profiler.html).
- [PyTorch CUDA semantics and timing](https://docs.pytorch.org/docs/stable/notes/cuda.html).
- [Pinned memory and non-blocking transfers](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html).
- [PyTorch data loading and batched fetching](https://docs.pytorch.org/docs/stable/data.html).
- [Google Colab resource and Drive limitations](https://research.google.com/colaboratory/faq.html).
