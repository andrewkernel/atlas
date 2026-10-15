# Literature review and search method

## Search protocol

This is a targeted scoping review, not a systematic review or a proof of absence. Searches used the web tool for academic/official sources and authenticated read-only GitHub metadata queries. Query families included automatic memory optimization, tensor rematerialization, training equivalence, scientific fidelity, invariant checking, reproducibility, approximate computing, equivalence testing, local coding models and budget-aware experimentation. Backward searches followed references in the closest systems and TrainParity's comparison with TrainCheck.

Priority went to primary papers, proceedings, official repositories and API documentation. Search snippets were used to locate sources; major conclusions use opened primary pages or paper sections. Read depth is stated in [sources](sources.md) and individual [paper notes](papers/README.md). We did not read every paper in full, reproduce the cited systems, or perform an exhaustive 2026 citation-graph search. Repository metadata snapshots do not validate advertised functionality.

## Findings by lineage

### Memory optimization and automatic selection

[Checkmate](papers/checkmate.md), [DTR](papers/dtr.md), [Rockmate](papers/rockmate.md), and [Capuchin](papers/capuchin.md) cover static and dynamic recomputation, eviction and offload. Their design space already includes profile-guided or automatic decisions. [DeepSpeed Autotuning](https://github.com/deepspeedai/DeepSpeed/blob/master/deepspeed/autotuning/README.md) and [Colossal-Auto](https://colossalai.org/docs/Colossal-Auto/get_started/run_demo/) overlap directly with a profiler-to-plan architecture. Atlas should consume native mechanisms, not claim their invention.

[ZeRO](papers/zero.md), [Megatron](papers/megatron.md), [FSDP2](https://docs.pytorch.org/docs/2.14/distributed.fsdp.fully_shard.html), and [Alpa](papers/alpa.md) address model state or distributed execution. Sharding state across ranks does not provide the same capacity gain at world size one. CPU offload remains relevant but moves the bottleneck to host memory and transfers; it is not free compute.

[Mixed precision](papers/mixed-precision.md), [FlashAttention](papers/flashattention.md), [ActNN](papers/actnn.md), [Adafactor](papers/adafactor.md), and [QLoRA](papers/qlora.md) illustrate distinct kinds of changes. Exact attention mathematics, approximate saved activations, changed optimizer state, and a restricted fine-tuning parameterization require different contracts. Similar accuracy on a paper's tasks cannot authorize a universal preservation claim.

### Correctness and reproducibility are already systems concerns

[EasyScale](papers/easyscale.md) explicitly studies consistency under changing resource allocations. [Deep500](papers/deep500.md) combines correctness-oriented and performance benchmarking. [TrainCheck](papers/traincheck.md) uses training invariants, and [TrainParity](competitors/trainparity.md) tests declared relations. This rules out novelty claims based on merely adding state checks or recording seeds.

[Accounting for Variance](papers/variance.md) shows why initialization alone is not the entire benchmark uncertainty. [MLflow](https://mlflow.org/docs/latest/ml/tracking/) stores experiment records; storage does not establish valid experimental design. Atlas needs both provenance and a justified comparison protocol.

### Cost-aware and approximate computation

[Hyperband](papers/hyperband.md) reallocates evaluation resources; [compute-optimal training](papers/compute-optimal.md) chooses model/data allocation. These optimize a research objective that can intentionally change the original protocol. They are alternatives for a researcher willing to redesign the study, not guaranteed reproductions of a fixed experiment.

[Chisel](papers/chisel.md) already makes optimization conditional on accuracy/reliability specifications. [FPBench](https://github.com/FPBench/FPBench) and [Herbie](https://fpbench.cs.washington.edu/) separate numerical error from algebraic form. The idea of optimizing subject to correctness or accuracy constraints is much older than Atlas. End-to-end scientific inference is a more difficult, domain-dependent boundary.

### Compilers, learned optimization and code agents

[Ansor](papers/ansor.md) uses a learned cost model to search tensor programs. [Learned optimizers](papers/learned-optimization.md) learn update rules, which generally changes a fixed optimizer experiment. [KernelBench](papers/kernelbench.md) evaluates correctness and speed of generated kernels; a useful model-generated change must pass independent checks. None establishes that a small local model can safely rewrite unfamiliar training projects.

## Synthesis

Continuation searches added the phrases `fidelity constrained training memory optimization`, checkpoint/BatchNorm state handling and paired-equivalence power. Primary follow-up found [HiLoRe](papers/hilore.md), [FairScale's specialized mitigation](competitors/fairscale.md), and exact-power guidance in [TOSTER](https://aaroncaldwell.us/TOSTERpkg/reference/power_t_TOST.html). These searches changed the novelty assessment and corrected the experimental budget; see the [amendment ledger](13-audit-and-amendments.md). A secondary index was used only to discover HiLoRe; the paper itself supports the retained comparison.

The potentially useful gap is **decision quality at the boundary between existing memory mechanisms and explicit, workload-specific training constraints**, including interactions, refusal and verification cost. It is a candidate empirical contribution. We found no source in this bounded search demonstrating precisely the proposed held-out, single-device selection study, but that observation is not evidence of global absence.

Before claiming academic originality, extend the review to derivative work on rematerialization safety, compiler validation, checkpoint side effects, contract-based ML testing, and transformation composition. Run the closest open systems on a shared case set. The [novelty analysis](06-novelty-analysis.md) defines what would invalidate the candidate claim.
