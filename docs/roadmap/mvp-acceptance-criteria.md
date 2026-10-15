# MVP and first coding task

## Smallest useful research MVP

One registered training adapter, complete-update profiling on CPU and optional CUDA, two supported native memory strategies, a finite deterministic planner, differential checks across loss/gradient/state/update phases, explicit unknowns, and reproducible JSON/Markdown records. Trusted code only. No required LLM, frontend, cloud, arbitrary source rewriting or distributed runtime.

## Immediate next task: deterministic profiling vertical slice

Implement the first native adapter and profiler, **before implementing optimization**.

Likely files:

- `src/atlas/experiments/contracts.py`: immutable experiment/resource/measurement schemas and schema version.
- `src/atlas/experiments/adapter.py`: protocol and registered factories; no user-controlled arbitrary imports.
- `src/atlas/experiments/runner.py`: spawned worker lifecycle, timeout and owned-child cleanup.
- `src/atlas/profiling/collector.py`: monotonic wall timing, process RSS, optional CUDA peaks.
- `src/atlas/cli.py`: `atlas profile --workload synthetic_mlp --config ... --output ...` after packaging/entry-point setup.
- `benchmarks/workloads/synthetic_mlp.py`: original offline fixture, fixed seeded inputs and two-layer model with SGD/Adam variants.
- `configs/profile-cpu.json` and `tests/test_profile_integration.py`, `tests/test_runner_lifecycle.py`.

These paths name **future work**. The current `benchmarks/semantic_probes.py` is research evidence, not the adapter or production executor.

### Expected behavior

Validate config without importing target code, construct the registered trusted fixture inside a fresh process, run a bounded warmup and measured window including optimizer updates, and write an atomic versioned report. Distinguish cold-start/first-step peaks from steady-state measurements. Capture seeds, configuration, code/lock/environment identity and wall time. CPU mode reports GPU counters as null with `unsupported_cpu`, never zero memory as if measured. An unavailable requested CUDA device fails clearly; no silent CPU substitution in a GPU benchmark.

### Acceptance tests

1. An offline CPU smoke run completes two measured logical updates and emits a schema-valid report with positive elapsed time, nonnegative observed RSS and exact update/sample counts.
2. Replaying the same CPU initialization/data recipe gives the same required state observations in the validated deterministic environment; timing is not asserted exactly equal.
3. An Adam fixture includes the first state-allocation update in the cold measurement and does not profile only a forward pass.
4. A hanging fixture times out, preserves a terminal reason/log and leaves no owned child. A child-spawning fixture tests tree cleanup separately on Windows and Linux.
5. An invalid contract or unregistered workload fails before its sentinel import/side effect executes.
6. Simulated telemetry-unavailable conditions produce explicit null/reason, not fabricated measurements. Mocking is appropriate for this availability behavior, not for the entire training computation.
7. Optional physical-CUDA test, skipped explicitly when unavailable: complete update, synchronized timing, allocated/reserved peaks, device identity and no accidental concurrent inference. A deliberately small bounded allocation tests counter availability without forcing host/device exhaustion.
8. Output-path traversal, stale run IDs and interrupted writes are rejected or safely reported; source files remain unchanged.

Do not call M1 done until CPU integration and lifecycle tests pass; CUDA support is marked unvalidated until its real-device test passes. Add dependencies only when needed, update the lock and record the tested environment.

## Later MVP acceptance

| Area | Required evidence |
|---|---|
| Planner | Identity plus at most declared plan cap, deterministic ordering, no contract relaxation, no-plan explanation |
| Accumulation | Same logical samples/targets/denominator and update clock; unequal/tail positive and negative controls; batch-coupled cases rejected/abstained |
| Checkpointing | Explicit eligible regions; RNG/state checks; BatchNorm/counterexample faults detected |
| Fidelity | Per-dimension result/status/scope; numerical tolerances distinct from statistical margins; missing evidence abstains |
| Tracking | JSON survives MLflow outage; SQLite transactions; large artifacts ignored by Git |
| Safety | Trusted-code policy enforced; no inherited credentials or arbitrary API shell/import fields; hard-isolation claims absent unless validated |
| Evaluation | Native/manual/composed baselines, failures included, held-out case definitions, actual hardware versus emulation labeled |

Success means the system can test the research hypothesis honestly, not that the hypothesis must be true.
