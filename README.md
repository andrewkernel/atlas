# Atlas

A research and engineering specification for a local scientific-fidelity-aware ML compute optimizer.

**Verdict: Proceed with modifications—only to a bounded, fair comparison study.** Existing systems already cover memory optimization, state-aware numerical checks and conditional invariants. Atlas should test whether explicit experiment contracts improve valid, resource-feasible plan selection over a competent composition of those systems. Standalone novelty and useful expanded access remain unestablished.

Start with the [updated recommendation](docs/research/14-updated-recommendation.md), [research index](docs/research/README.md), [claim-to-evidence map](docs/research/evidence-map.md) and [assignment deliverable audit](docs/research/deliverable-audit.md). The package includes individual prior-art notes with read scopes, competitor analysis, scientific-fidelity framework, architecture/API/security blueprint, six ADRs and a phased roadmap. The [amendment ledger](docs/research/13-audit-and-amendments.md) preserves corrections and negative results.

## What the evidence establishes

- Original CPU semantic/state/update controls and statistical audits expose known failure mechanisms and assumption limits. The current CPU suite has 20 passing tests, including 40 generated accumulation partitions. These are authored controls, not independent field-risk estimates.
- [Synthetic GPU measurements](docs/research/experiments/cuda-memory-probe.md), a labeled [allocator-cap control](docs/research/experiments/cuda-budget-probe.md), and [twelve native selective-checkpoint trials](docs/research/experiments/native-selective-control.md) show resource/time tradeoffs on one physical 12 GB RTX 3060. Existing native selective checkpointing strengthens the baseline Atlas would need to beat.
- [Six five-epoch real-data CNN pilots](docs/research/experiments/fashion-pilot.md) show about 32.7% lower allocated memory and 66.4% longer summed synchronized update time for the combined native plan. Three paired validation differences are exploratory; both arms already fit.
- Reviewed [TrainParity controls](docs/research/experiments/trainparity-controls.md), adapted [Rockmate-family planning](docs/research/experiments/rockmate-compatibility.md) and [compatible-stack streaming integration](docs/research/experiments/interop-controls.md) execute. The streaming adaptation reuses existing snapshot/comparison code successfully; no new generic checker is justified by these results.
- [Execution-order](docs/research/experiments/planner-reuse-control.md), [comparison/health](docs/research/experiments/comparator-contract-control.md) and [shared-reference token-objective](docs/research/experiments/token-reference-control.md) controls clarify exact evidence boundaries. A differential PASS does not establish that its supplied reference implements the intended experiment.
- [Six same-seed energy replays](docs/research/experiments/board-energy-control.md) preserve training identities and retain full shared-board traces. Counter energy rises in each combined-plan pair, while integrated power and counters disagree materially. Independent arithmetic is checked; calibrated process/whole-host energy is not established.

**No primary held-out policy advantage, useful real-workload OOM rescue, powered full-training equivalence, local-LLM benefit or preservation of scientific conclusions has been established.** The application is not implemented. No independent expert labels were obtained; the [adjudication procedure](docs/research/adjudication-protocol.md) prepares that remaining gate.

## Reproduce the bounded CPU checks

With Python 3.12 and uv 0.13.1:

```text
uv sync --locked
uv run --locked python benchmarks/audit_specification.py
uv run --locked pytest -q
uv run --locked python benchmarks/semantic_probes.py --output artifacts/replayed-cpu-probes.json
```

The core lock uses the official PyTorch CPU index. [Separate dependency/environment records](docs/research/dependencies-and-licenses.md) identify CUDA, checker, planner, shared-stack and telemetry scopes; core CI does not rerun foreign packages or GPU experiments. Linux/Windows CI additionally reconstructs saved energy arithmetic/replay identities and checks the derived record. Data, weights, external reference source/PDFs and large artifacts remain outside tracked files.

## Immediate next implementation

Build the trusted adapter/profiler slice in the [acceptance criteria](docs/roadmap/mvp-acceptance-criteria.md): immutable typed contracts, registered factories, complete-update measurements, explicit unavailable metrics and tested owned-worker lifecycle. The [contract walkthrough](docs/architecture/evidence-contract-example.md) records comparison, objective, execution and measurement obligations. Compare competent native/existing-checker baselines before expanding mechanisms, FastAPI/MLflow, Ollama or the frontend.

This private repository was created separately from the read-only ChatGPT project mirror. Synced reference files were preserved. The [chronological log](docs/research/12-research-log.md) records actual work, attempted runs and unfinished evidence; it makes no artificial five-hour claim. Public licensing remains an owner decision.
