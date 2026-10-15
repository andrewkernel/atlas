# Technical feasibility

## Recommended approach

Use a **hybrid of explicit adapters, runtime profiling, deterministic finite planning and scoped verification**. A local LLM is optional and has no authority to establish scientific validity. Keep source rewriting out of the first research study.

| Approach | Strength | Limitation | Decision |
|---|---|---|---|
| Pure rules | Inspectable, cheap, reproducible | Cannot know dynamic memory peaks or undeclared semantics | Use for eligibility and enumeration |
| Static AST | Find likely batch size, dtype, optimizer and suspicious stateful calls without importing code | Aliases, indirect calls, generated code, custom ops and data-dependent behavior defeat completeness | Advisory findings with source spans; never safety certification |
| Dynamic profiling | Observes actual operators, allocator peaks and timing | Executes code; perturbs runtime; one trace misses other branches/shapes | Central to resource measurement on trusted adapters |
| LLM-generated patches | Can interpret unfamiliar source and propose changes | Hallucination, nondeterminism, verification cost and memory contention | Later optional controlled experiment |
| Constrained hybrid | Joins known mechanisms, measured costs and required checks | Adapter effort and incomplete coverage | Selected; small enough for one developer |

[FX symbolic tracing](https://docs.pytorch.org/docs/2.14/fx.html) does not generally handle input-dependent Python control flow. Tracing also executes Python, so it does not belong in untrusted static inspection. Compiler graph capture is not a proof of complete arbitrary-program semantics.

## Memory model

For a simple all-FP32 Adam setup with P trainable scalar parameters, parameter storage is about 4P bytes, gradients about 4P and two moment buffers about 8P: roughly 16P bytes once state is initialized. This is an analytic example, not a universal estimate. Master copies, frozen parameters, sparse state, tied storage, optimizer implementation and missing gradients alter it.

Use M_peak ≈ persistent state + simultaneously live activations + temporary workspace + runtime/context + allocator overhead. Sum unique storages rather than double-counting tensor views. Static counts give a conditional lower bound or rough component estimate; they do not bound custom workspaces or later dynamic shapes. Meta/FakeTensor analysis can help supported graphs but must not promise a reliable pre-execution total.

Profile complete forward/backward/**optimizer** updates, including the first optimizer-state allocation. Use a bounded representative shape set including the longest declared sequence and evaluation/checkpoint-save paths. Warmup and cold-start peaks are separate. In the [CUDA allocator API](https://docs.pytorch.org/docs/2.14/notes/cuda.html), allocated tensor bytes and reserved allocator bytes differ. Neither equals total process VRAM; use optional device telemetry for context/external allocations and state its attribution limits.

Do not use a linear microbatch extrapolation as a guarantee: workspaces and kernel dispatch can have discontinuities. A safety margin is a heuristic, recorded as such. Reconfirm the selected candidate on the actual target hardware.

## OOM diagnosis

Capture stage, requested allocation if available, allocated/reserved/free memory, shape, dtype, optimizer-state initialization, process RSS, driver/runtime, and other device usage. Distinguish CUDA allocator OOM, host OOM, timeout, unsupported operator, device error and measurement failure. Fragmentation is a hypothesis supported by allocator evidence, not inferred solely from reserved minus allocated bytes. Clearing cache cannot free live tensors or solve oversized persistent state.

Never retry indefinitely. Each profile/candidate runs in a disposable child process; on OOM or device error, save partial evidence, terminate the process tree, and start a clean process for another candidate. Do not continue from a half-updated training state. Limits and stop behavior require dedicated tests.

## Can short checks replace full training?

They can catch wrong loss weighting, altered update timing, dropout recomputation and mutated buffers. The [executed probes](experiments/README.md) support these examples. They cannot rule out late divergence, change in generalization, rare data branches, or scientific effect reversal. Statistical/conclusion evidence remains unverified unless the full relevant study is performed. Cost-aware verification must include the cost of the baseline, repeated runs and failures.

## Adapter versus arbitrary loops

Require an importable, registered `ExperimentAdapter` describing data/loss accounting, model construction, initialization replay, eligible checkpoint regions and update phases. Arbitrary `python train.py` remains unsupported for automatic optimization. A custom-loop author writes an adapter; AST/LLM hints can reduce navigation effort later but cannot fabricate missing semantics. The adapter itself is executable trusted code.

## Stack and compatibility

| Component | Decision | Verified here / remaining work |
|---|---|---|
| Python | 3.12 | Ran on 3.12.10; default host Python 3.14 is deliberately not used |
| PyTorch 2.x | Native CPU first, CUDA optional | Installed and tested 2.14.1+cpu; no CUDA build validated |
| uv / locks | Lock research dependencies | uv 0.13.1; `uv.lock` resolved and synced; dependency hashes recorded |
| NumPy/SciPy/pytest/Hypothesis/psutil | Scientific utilities and tests | Versions pinned in pyproject; executed semantic tests |
| FastAPI/Pydantic | Typed local service after command-line prototype | Architecture retained; packages/API not implemented or integration-tested |
| MLflow + SQLite | Local tracking and control-state storage | Retained, deferred; separate DB ownership and migrations specified |
| Ollama | Replaceable optional backend | Installed CLI; no models present or evaluated |
| NVIDIA telemetry | Optional NVML via maintained binding | Device inspected with NVIDIA CLI; energy API support untested |
| React/TypeScript/Vite/Tailwind | Inspection UI later | Retained; absent from research-critical path |
| Docker | Optional reproducible trusted execution environment | CLI present; daemon/isolation/GPU behavior not validated |

Do not assume latest packages can be combined merely because metadata admits Python 3.12. Add API/tracking extras and lock them when implemented; test the integration before claiming support. CPU and CUDA environments need explicit wheel selection and separate validated lock/profile records. Native Windows is the executed environment; Linux/WSL2 are proposed GPU-development options, not validated by these CPU tests.

## Undergraduate feasibility judgment

One developer can build the adapter/profiler, two deterministic strategies, bounded checks and a small benchmark. General source transformation, automatic intent inference, distributed optimization and hostile GPU sandboxing would each expand the project substantially. A staged 10–16 developer-week core study is an estimate conditional on skills, dataset access and working CUDA dependencies; the full optional roadmap is longer and should be cut at evidence gates.
