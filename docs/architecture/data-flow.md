# Data flow, identities and recovery

## Immutable inputs

Create an `ExperimentSpec` from a registered trusted adapter ID, source revision plus dirty-diff hash, dataset/file hashes, preprocessing/tokenizer revision, split/order manifest, initial-state recipe, seed map, config, contract and resource budget. Canonical JSON must have deterministic key ordering and reject NaN/Infinity before hashing. Store schema and hash algorithm versions. A Git SHA without dirty-tree information is insufficient.

A `RunManifest` adds Python/package lock digest, wheel origin/build, PyTorch/CUDA/cuDNN/compiler versions, OS, GPU model/capability/VRAM/driver, CPU/RAM, thread settings, determinism flags, backend choices, process start method and timeout. Do not include raw environment variables, usernames, machine paths or tokens in portable reports. Store private path mappings locally and sanitize exports.

## Execution lifecycle

1. Validate JSON/schema and source paths without importing code. Resolve symlinks and reject traversal outside registered roots.
2. Inspect the readable change record and bound resources. Bind any required trusted-code acknowledgment or isolated-run approval to exact code/spec/plan hashes.
3. Create private scratch/output directories. Spawn a worker with an argument vector, minimal environment and no shell interpolation.
4. Establish reference self-consistency with two fresh controlled executions for differential checks. Record cold and warm measurement windows independently.
5. Enumerate candidates. Precondition rejection requires no training execution. Each eligible trial starts from the same reference recipe, never from another candidate's output.
6. Save ordered events and atomic result files. The supervisor validates results, commits metadata and runs required checks.
7. Produce a report with resource measurements, change diff, property evidence and unknowns. A selected plan may require a separate full-run confirmation before a complete-training claim.

## Run state machine

`created → validated → awaiting_approval → queued → running → succeeded | oom | timeout | cancelled | failed | interrupted`

For trusted builtin cases whose execution policy already authorizes the spec, `validated → queued` is allowed. Human approval is required for newly imported/untrusted executable code; the MVP can reject it entirely. Approval changes a policy state, not a fidelity status. `succeeded` means execution completed, not scientific validity.

Use compare-and-set state transitions and idempotency keys to prevent duplicate starts. Cancellation is idempotent. After crash/restart, mark running records interrupted and reconcile owned child processes using job identity; never kill processes merely matching a generic Python name. Do not silently resume incomplete optimizer state.

## Artifact schema

Each measurement stores name, value or null, units, scope/window, sampling interval, source and support/error reason. Each artifact stores relative logical URI, SHA256, byte length, media type and producer/run ID. Binary state may remain local; reports can point to keyed summaries and hashes. A hash proves identity of recorded bytes, not scientific correctness or that inputs were safe.

Required phase observations: initialization; loss numerator/count; accumulated gradient; post-clip gradient if used; successful optimizer update; scheduler/scaler/EMA update; buffers; evaluation. Counters distinguish samples, valid targets, microsteps, attempted optimizer steps and successful updates. Timing uses a monotonic clock; timestamps use UTC plus client timezone metadata.

## Recovery and retention

Write results to a temporary file then atomically rename within the run directory. On failure, retain manifest, exit code, bounded stderr, events and partial measurements with completion flags. Limit log size and file count; redact secrets before export. Never delete user datasets or the source checkout as cleanup. Only delete owned scratch paths after checking their absolute resolved location. Failed candidates remain visible in denominators.

Evidence reuse requires matching scope and input identities. A new dataset ordering, GPU backend, contract margin or strategy version invalidates incompatible evidence. Export JSON/Markdown first; add Parquet only for scale, without changing evidence semantics. SQLite transactions and MLflow records must refer to artifact IDs, not substitute for the immutable manifest.
