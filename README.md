# Atlas

A research-first specification for a local scientific-fidelity-aware ML compute optimizer.

**Assessment: Proceed with modifications.** Automatic memory optimization and training checking already have substantial prior art. Atlas should test whether explicit experiment contracts improve selection among existing memory-saving plans on a single consumer GPU. Novelty and practical benefit remain hypotheses.

Start with the [updated recommendation](docs/research/14-updated-recommendation.md), [executive assessment](docs/research/00-executive-assessment.md) and [research index](docs/research/README.md). The package includes a competitor matrix, individual paper notes, scientific-fidelity framework, architecture/API/security specification, benchmark protocol, six decision records and a phased roadmap. The [continuation audit](docs/research/13-audit-and-amendments.md) records corrections and closer prior art.

## Evidence status

Original offline CPU probes demonstrate loss-weighting, BatchNorm, RNG, checkpoint-state and update-timing failure modes. The expanded suite has 20 passing tests, including 40 generated accumulation cases, state-repair controls and synthetic statistical-design/assumption calibration. [Eight synthetic GPU trials](docs/research/experiments/cuda-memory-probe.md) measure a memory/time tradeoff on the physical12 GB RTX 3060; a separate [allocator-cap control](docs/research/experiments/cuda-budget-probe.md) labels emulation explicitly. **No real-workload OOM rescue, full-training equivalence, LLM benefit or scientific-conclusion preservation was established.** The application has not been implemented. See [actual experiments](docs/research/experiments/README.md) and [open questions](docs/research/11-open-questions.md).

[Six real-data five-epoch CNN pilots](docs/research/experiments/fashion-pilot.md) also completed: about 32.7% lower allocated memory for the combined native plan, longer update time, and small paired validation differences. This is exploratory three-seed evidence, not a conclusive equivalence or planner-superiority result.

The [evidence map](docs/research/evidence-map.md) separates measured findings, source inspections, mathematical controls and untested hypotheses. Local replay controls reproduce the lowest seed's final hashes and epoch metrics exactly within each arm; they add no independent seeds.

## Reproduce the bounded probes

With Python 3.12 and uv 0.13.1 installed:

```text
uv sync --locked
uv run --locked python benchmarks/audit_specification.py
uv run --locked pytest -q
uv run --locked python benchmarks/semantic_probes.py --output artifacts/replayed-cpu-probes.json
```

The lock explicitly uses the official PyTorch CPU index; it does not validate CUDA. Keep data/weights/large artifacts outside Git. [Dependencies and licenses](docs/research/dependencies-and-licenses.md) distinguish installed packages from untested optional components.

## Next engineering milestone

Implement a trusted synthetic workload adapter and complete-update profiler with CPU fallback, explicit GPU-measurement availability and bounded worker lifecycle. The [first coding task](docs/roadmap/mvp-acceptance-criteria.md) names files, expected behavior and acceptance tests. FastAPI/MLflow, Ollama and the frontend follow only when their phase is justified.

This repository was created separately from the read-only ChatGPT project mirror; synced references were preserved. The research log records actual work and unfinished evidence. Public licensing has not yet been selected by the owner.
