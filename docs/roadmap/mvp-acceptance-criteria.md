# MVP and first coding task

## Smallest useful research MVP

The next useful release is a registered adapter/profiler and scoped evidence-report pipeline. It measures complete logical updates, preserves immutable inputs and records explicit unknowns. Reuse compatible existing comparison primitives. A later finite native plan grid is engineering infrastructure; a distinct selection algorithm remains unselected under [ADR 007](../research/decisions/007-benchmark-first.md). No required LLM, frontend, cloud, arbitrary source rewriting or distributed runtime.

## Immediate next task: deterministic profiling vertical slice

Use the [updated research gate](../research/14-updated-recommendation.md) and [existing-checker integration note](../research/checker-integration-options.md) when implementing this slice. The independent research fixtures do not complete this production milestone, and generic buffer/state checks are already substantially covered by existing work. Prioritize a comparable native-plus-checker baseline before expanding mechanisms or UI.

The new [objective/worker boundaries](../architecture/objective-and-evidence-boundaries.md) and [claim-applicability design](../architecture/claim-applicability.md) govern the interfaces. Native QA/checker prototypes now execute, but the production adapter/API remains unbuilt. Historical environment profiles are for reproducing research; the MVP should require one CPU controller and at most one deliberately matched optional ML-worker profile, not all eight environments.

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
9. A Windows environment launcher and actual interpreter are both inside the owned boundary before training. Test wrong creation identity, failed assignment, missing startup acknowledgment, timeout, normal exit, cancellation and group-handle close. Exit zero after cancellation cannot produce a completed scientific verdict. Linux needs its own tested process-group profile; Windows evidence does not establish it.
10. A registered optional worker can use the compatible Python 3.11/Torch 2.13 stack while the controller remains Python 3.12. Resolve only fixed profile IDs; reject arbitrary interpreter/import paths. Record actual runtime, driver/cuDNN/device settings inside the measured worker rather than filling them from a post-study read.
11. LogicalUpdateContext computes reviewed global metadata denominators before microbatch backward. Multi-head/unequal-tail and empty-local positive controls match; undefined global means and unsupported model-dependent normalizers abstain explicitly. The schema distinguishes features/questions/tokens and auxiliary update frequency.
12. A known unstable-reference fixture preserves its literal comparison result but leaves transformation attribution unverified. A supported finite pair cannot satisfy a different horizon/field/relation. Strict parameter and functional contracts are separate; neither automatically overrides the other.
13. Capture request/process lifetimes and nested preparation, update, checking and artifact windows without double counting. Cold and warm scopes are named. Failed trials and reference repeats consume the same declared budget for every arm; unmeasured subdivisions remain unavailable.

Do not call M1 done until CPU integration and lifecycle tests pass; CUDA support is marked unvalidated until its real-device test passes. Add dependencies only when needed, update the lock and record the tested environment.

## Later MVP acceptance

| Area | Required evidence |
|---|---|
| Planner | Identity plus at most declared plan cap, deterministic ordering, no contract relaxation, no-plan explanation |
| Accumulation | Same logical samples/targets/denominator and update clock; unequal/tail positive and negative controls; batch-coupled cases rejected/abstained |
| Checkpointing | Explicit eligible regions; RNG/state checks; BatchNorm/counterexample faults detected |
| Fidelity | Reused supported checks plus distinct observation/predicate/claim/applicability records; exact precision/field scope; missing evidence abstains |
| Tracking | JSON survives MLflow outage; SQLite transactions; large artifacts ignored by Git |
| Safety | Trusted-code policy enforced; no inherited credentials or arbitrary API shell/import fields; hard-isolation claims absent unless validated |
| Evaluation | Native/manual/composed baselines, failures included, held-out case definitions, actual hardware versus emulation labeled |

Success means the system can test the research hypothesis honestly, not that the hypothesis must be true.
