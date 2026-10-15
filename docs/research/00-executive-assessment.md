# Executive assessment

**Verdict: Proceed with modifications.** Atlas is credible as a small empirical systems project about constrained selection of memory-saving transformations. The original broad product is too ambitious for one undergraduate, and its broad novelty is contradicted by existing work. Do not describe Atlas as the first fidelity-aware optimizer or a verifier of scientific truth.

## What the evidence changes

Automatic rematerialization is established in [Checkmate](https://arxiv.org/abs/1910.02653), [DTR](https://arxiv.org/abs/2006.09616), and [Rockmate](https://proceedings.mlr.press/v202/zhao23b.html). [DeepSpeed Autotuning](https://github.com/deepspeedai/DeepSpeed/blob/master/deepspeed/autotuning/README.md) already combines profiling, memory estimates and configuration search. Native PyTorch has [selective checkpointing and memory-budget-driven rematerialization](https://pytorch.org/blog/activation-checkpointing-techniques/). An integration of these features is useful engineering, but that alone is not a research contribution.

Preserving behavior is also established: [EasyScale](https://arxiv.org/html/2208.14228v2) targets consistency under changing resources; [TrainCheck](https://www.usenix.org/conference/osdi25/presentation/jiang) checks training invariants; [TrainParity](https://github.com/intelland/TrainParity) tests declared training relations. These must be explicit comparators rather than omitted neighbors.

The continuation audit found an even closer recent neighbor, [HiLoRe](papers/hilore.md), which directly weakens general fidelity-constrained memory-planning novelty. [FairScale](competitors/fairscale.md) also predates Atlas's checkpoint-state example and has specialized handling. The [audit amendments](13-audit-and-amendments.md) narrow the claim and correct a statistical planning error.

## Selected question

**On held-out single-device PyTorch workloads and checkpointing/accumulation plans, does contract-constrained selection accept fewer plans that violate declared training semantics, without reducing valid-feasible workload coverage, than native optimizations plus existing checks under the same eight-trial/30-minute search budget?**

This question is falsifiable even if the result is that Atlas adds no value over the composed baseline. Resource-only selection is an additional ablation. The budget is the initial proposed protocol and may be amended after a disjoint pilot, before confirmation. The claim concerns observed contracts and tested workloads, not all scientific conclusions or arbitrary Python.

## Proposed contribution

A small, auditable benchmark of transformation interactions; an explicit experiment contract; and an abstaining planner that chooses among existing native mechanisms using measured resource cost and scoped evidence. The research artifact is the evaluation of selection quality and cost. The contract format and reporting are supporting engineering. Novelty remains provisional until deeper code comparisons and an advisor review.

## Smallest useful system

A Python 3.12 command-line library with a trusted adapter, CPU/CUDA profiler, finite candidate registry, config-only patch records, gradient/state/update comparisons, and JSON/Markdown reports. Initially allow smaller microbatches with preserved logical batches and activation checkpointing on declared pure regions. Keep BF16 as an opt-in empirical extension. Use SQLite and an MLflow adapter once runs need tracking. Add FastAPI/Pydantic after the command-line path works; Ollama and React come later.

## Actual evidence from this session

The [CPU probes](experiments/README.md) demonstrate unequal-loss weighting errors, BatchNorm sensitivity, checkpoint/dropout RNG effects, buffer changes despite matching gradients, and clipping/update-timing faults in a checkpoint-plus-accumulation composition. The continuation adds a scoped state-repair positive control, RNG-interleaving controls and synthetic equivalence-power calibration;17 tests pass, including40 generated accumulation cases. These are bounded controls, not evidence for the primary selection hypothesis. No full training, GPU memory optimization, LLM comparison, energy study, or human expert labeling was performed.

A 12 GB RTX 3060 is present on the host, but the isolated research environment uses a CPU-only PyTorch wheel. No physical 6–8, 16 or 24 GB tiers were tested. No result supports a general claim that a 24 GB experiment can be reproduced on an 8 GB GPU.

## Three reasons to stop or pivot

1. **Insufficient incremental value:** native tools plus TrainCheck/TrainParity may achieve the same selection quality. Compare directly and pivot to an evaluation contribution if so.
2. **Verification is too expensive or too weak:** small probes may miss late drift; full paired runs can exceed the saved compute. Measure verification cost and report inconclusive outcomes.
3. **Access benefit is small:** memory savings may merely replace VRAM scarcity with CPU RAM, runtime, or integration burden. Measure end-to-end time and adaptation effort, and avoid institutional-equity claims without a user study.

The [proposal](10-research-proposal.md), [architecture](../architecture/overview.md), and [acceptance criteria](../roadmap/mvp-acceptance-criteria.md) define the next stage. All complete-training and consumer-GPU benefits remain hypotheses.
