# Competitor comparison

**V** = explicitly documented in the linked primary source; **I** = Atlas assessment/inference. **ND** = no general scientific-conclusion verifier documented in the inspected material; this is not proof the feature is absent. No competing system was executed here. Consumer support below is a compatibility assessment, not a hardware certification. Exact inspected repository heads and license metadata are in the [snapshot](competitors/repository-snapshot.json).

## Mechanisms, architecture and selection

| System and primary source | Problem / architecture (V) | Mechanisms (V) | Automatic selection (V unless qualified) | Consumer hardware / requirements and limitations (I unless specified) |
|---|---|---|---|---|
| [PyTorch native](https://pytorch.org/blog/activation-checkpointing-techniques/) | Training runtime and compiler | AMP, checkpointing, selective recomputation, native attention | Kernel dispatch; compiler rematerialization and memory budget options | Single-device use plausible; backend/shape/version limits; internal APIs must not be treated as stable |
| [DeepSpeed/ZeRO](https://www.deepspeed.ai/training/) | Training engine, partitioned state, offload runtime | Sharding, accumulation, mixed precision, offload, checkpointing | [Autotuner](https://github.com/deepspeedai/DeepSpeed/blob/master/deepspeed/autotuning/README.md) searches selected configuration fields | Single GPU offload supported; cited 10B example uses 32 GB V100 (V); not an 8 GB guarantee; extension/toolchain setup costs |
| [Accelerate](https://huggingface.co/docs/accelerate/en/usage_guides/gradient_accumulation) | Training-loop adapter over devices/backends | Accumulation; integrates distributed/precision backends | Config assistance; not a general semantic optimizer in inspected guide | CPU/single GPU useful; adapter integration needed; delegates mechanisms |
| [Colossal-AI](https://colossalai.org/docs/basics/booster_plugins/) | Booster plugins and heterogeneous state management | Gemini/ZeRO, parallelism, precision, offload | [Colossal-Auto](https://colossalai.org/docs/Colossal-Auto/get_started/run_demo/) checkpoint search | CUDA extensions and plugin constraints; documentation has accumulation caveats; check installed version rather than assuming combinations work |
| [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) | Specialized transformer training stack | Tensor/pipeline/data parallelism, recomputation, distributed optimizer | Configuration-driven; performance options, not arbitrary-program search | Primarily cluster/large-transformer use; single-GPU research configurations possible; integration expensive for MVP |
| [FSDP2](https://docs.pytorch.org/docs/2.14/distributed.fsdp.fully_shard.html) | Per-parameter state sharding and hook-managed communication | Sharding, mixed precision, CPU offload | User wrapping policy; automatic communication scheduling | Multi-rank benefit; no rank-partitioning capacity benefit at world size one; host/transient peaks still matter |
| [Checkmate](https://arxiv.org/pdf/1910.02653) | Profile-guided rematerialization solver | Save/recompute plans | MILP and approximation | Published TensorFlow/Keras pipeline (V), Gurobi (V); not a drop-in current PyTorch adapter |
| [DTR](https://arxiv.org/abs/2006.09616) | Runtime tensor eviction metadata | Online rematerialization | Greedy online policy | Prototype/runtime integration; current package compatibility untested; dynamic graph support does not imply arbitrary side-effect safety |
| [Rockmate](https://proceedings.mlr.press/v202/zhao23b.html) | PyTorch graph/block analysis | Hierarchical rematerialization | Block and sequence optimization | Closest single-device memory comparator; current compatibility and supported model patterns need testing |
| [Capuchin](https://www.microsoft.com/en-us/research/publication/capuchin-tensor-based-gpu-memory-management-for-deep-learning/) | Runtime tensor-access-pattern management | Eviction, prefetch, recomputation | Runtime pattern-driven decisions | TensorFlow prototype (V); CPU bandwidth/host RAM constraints; modern PyTorch port out of scope |
| [Alpa](https://arxiv.org/abs/2201.12023) | Compiler/runtime for parallel plans | Inter/intra-operator parallelism | Hierarchical search | Distributed JAX/XLA-oriented system; repository archived (metadata V); poor MVP dependency |
| [FlashAttention](https://arxiv.org/abs/2205.14135) | IO-aware exact attention algorithm | Tiling, recomputation, fused operations | Native backend selection exists through PyTorch; paper is an algorithm | Hardware/dtype/head-shape limits; avoiding attention matrix storage is not reducing all model memory |
| [ActNN](https://arxiv.org/abs/2104.14129) | Compressed saved activations | Stochastic low-bit activation compression | Adaptive precision allocation | CUDA/layer integration; approximate gradients; modern compatibility untested |
| [TrainCheck](https://orderlab.io/TrainCheck/technical-doc/) | Trace instrumentation, inferred invariants, online/offline checker | Proactive training checks | Automatically infers relations and preconditions | Tutorial allows CPU (V); full artifact asks two GPUs with ≥12 GiB each (V); does not make all study protocols automatic |
| [TrainParity](https://github.com/intelland/TrainParity) | Differential fresh-process execution | Resume, accumulation, sample-coverage checks | User declares relation; deterministic checker | CPU viable; inspected metadata limits Python <3.12 and Torch <2.14 (V); compare in a separate environment |
| [Deep500](https://github.com/deep500/deep500) | Layered benchmark infrastructure | Operator/gradient validation and training benchmarks | Benchmark composition, not general memory plan search | CPU/GPU workloads; old ecosystem integration needs review; distributed paths use MPI |
| [KernelBench](https://github.com/ScalingIntelligence/KernelBench) | Generated-kernel evaluation harness | Functional checks plus runtime comparison | Candidate generation external; feedback can guide search | GPU/toolchain required for target kernels; not full-training scientific inference |
| [MLflow](https://mlflow.org/docs/latest/ml/tracking/) | Local/server experiment tracking | Parameters, metrics, artifacts | No memory optimization search in tracking API | Local machine supported; tracking overhead/storage; not a verifier |
| [Chisel](https://people.csail.mit.edu/rinard/paper/oopsla14.pdf) | Approximate-kernel optimization under specifications | Approximate operation selection | Constraint-aware optimization | Approximate hardware model; conceptual prior art, not commodity PyTorch backend |

## Fidelity and overlap

| System | Fidelity considered in evidence | Verification of scientific meaning | Atlas already covered | Remaining distinction to test (I) |
|---|---|---|---|---|
| Native PyTorch | Numerical caveats, RNG handling, operator correctness | ND; checkpoint metadata check is limited | Most initial mechanisms and profiler | Workload contract and decision evidence |
| DeepSpeed/ZeRO | Training algorithm/configuration semantics | ND | Profiling, memory estimates, search, execution | Constraint-conditioned acceptance and interactions |
| Accelerate | Update/accumulation semantics in guide | ND | Loop adaptation | Contract-based plan selection |
| Colossal-AI | Intended training behavior and compatibility | ND | Multiple mechanisms and automated plans | Explicit scope and verification costs |
| Megatron/FSDP2 | Distributed optimizer/training semantics | ND | State-memory mechanisms | Commodity single-device selection study |
| Checkmate/DTR/Rockmate/Capuchin | Intended recomputation/data dependency semantics | ND for arbitrary protocol and conclusions | Automated memory management | Mutable-state/RNG conditions and composed-plan evaluation |
| Alpa | Parallel execution semantics | ND | Automated execution plans | Different hardware/scope; not an implementation dependency |
| FlashAttention | Exact intended attention mathematics; kernel numerical differences possible | ND | Attention memory mechanism | Contract-specific numerical and training evidence |
| ActNN | Theory and task-quality evaluation for approximation | ND for unrelated scientific questions | Approximate memory reduction | Allowed-change policy; not strict-parity default |
| TrainCheck | Inferred invariants and silent errors | Explicit low-level checks, not domain conclusions | A large portion of a generic correctness engine | Does integration with selection beat simple composition? |
| TrainParity | Declared differential relation and abstention | Scoped observations, not general model quality | Much of accumulation verification | Checkpoint composition and selection benefit |
| Deep500/KernelBench | Correctness/performance benchmarking | Scoped computations; ND for scientific conclusions | Verification-plus-performance evaluation pattern | Training-contract selection task |
| MLflow | Provenance/measurement | ND | Experiment records | Evidence semantics and decision logic |
| Chisel | Accuracy/reliability constraints | Kernel-level specification guarantees | General constrained-optimization idea | Different domain and evidence boundary |

## Fair comparison requirements

Continuation additions: [HiLoRe](papers/hilore.md) is a direct conceptual comparator; [FairScale](competitors/fairscale.md) supplies a relevant existing state-aware checkpoint implementation. Neither was executed. Their omission from the original metadata snapshot is preserved as historical scope, not hidden by rewriting that snapshot. Include both in the next capability study; record unsupported workload/precision combinations explicitly.

Deeper continuation review also adds [HiRemate/OffMate](competitors/rockmate-family.md) and confirms [TrainParity's worker/snapshot state coverage](competitors/trainparity.md). Original matrix entries are a starting classification; these additions govern current comparator selection. Buffer/state/None-gradient capture is not an unoccupied generic-verifier gap. No external system was executed.

Use current native PyTorch as a strong baseline; do not intentionally disable an already-used efficient attention kernel and then attribute the difference to Atlas. Match optimizer, data sequence, initialization, precision contract and update count. Compare Atlas with **native mechanisms plus applicable existing checks**, not only unmodified code. For mismatched framework/Python versions, report separate capability and runtime comparisons; do not attribute version differences to the planner.

Migration effort is a cost. Record adaptation hours and modified lines, but do not interpret lines as a direct measure of scientific reliability. Failed installations and unsupported workloads belong in coverage results, with versions and causes. Do not silently remove difficult systems from the table or use an incompatible configuration to make Atlas look better.
