# Executive assessment

**Current verdict: proceed with the evaluation package; require a distinct algorithm before policy superiority.** A scoped empirical/engineering project is feasible. Broad optimizer/verifier novelty is contradicted by prior work, and benchmark novelty still needs independent review. Do not describe Atlas as the first fidelity-aware optimizer or a verifier of scientific truth.

The [updated recommendation](14-updated-recommendation.md), [treatment audit](15-comparison-identifiability.md) and [revised proposal](10-research-proposal.md) govern current scope. The earlier question below is conditional future work: an Atlas next-action algorithm remains unselected. Native/existing-checker BERT controls and selected reference-noise measurements improve evidence while narrowing what it supports. Historical test counts below describe earlier checkpoints; the current repository index records later validation.

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

The [CPU probes](experiments/README.md) demonstrate unequal-loss weighting errors, BatchNorm sensitivity, checkpoint/dropout RNG effects, buffer changes despite matching gradients, and clipping/update-timing faults in a checkpoint-plus-accumulation composition. The continuation adds scoped state repair, RNG interleaving, unused-gradient/successful-update clocks and statistical calibration;20 tests pass, including 40 generated accumulation cases. These are bounded controls, not evidence for the primary selection hypothesis. A later [gross shared-board energy replay](experiments/board-energy-control.md) adds scoped measurements and an unresolved instrument discrepancy; no calibrated energy inference, LLM comparison or human expert labeling was performed.

A separate CUDA environment now passes a native smoke test and [eight synthetic GPU trials](experiments/cuda-memory-probe.md) on the physical12 GB RTX 3060. The combined plan lowers this fixture's allocated peak from about 973 to 378 MiB, with slower updates and small first-update numerical differences. No OOM real workload was rescued. No physical6–8,16 or24 GB tiers were tested, and no result supports a general24→8 GB reproduction claim. CPU CI and the original CPU lock remain separate.

The [allocator-cap control](experiments/cuda-budget-probe.md) is explicitly emulated; two plans complete one update before failing later. The [composition audit](composition-evidence.md) also limits novelty: present controls do not yet demonstrate an emergent failure of individually valid transformations. Continue only to a bounded strong-baseline comparison before expanding the product.

[Six real-data CNN pilots](experiments/fashion-pilot.md) now complete the frozen five-epoch protocol with three paired seeds. They show lower allocated memory, longer update time and small validation differences, with matching paired initialization/data order. This is exploratory training evidence; three pairs do not establish outcome equivalence or scientific-conclusion preservation.

## Three reasons to stop or pivot

Direct existing-system controls now include [TrainParity accumulation checking](experiments/trainparity-controls.md) and [adapted Rockmate-family GPU planning](experiments/rockmate-compatibility.md). These strengthen practical prior-art overlap, while retaining compatibility failures, exact scope and workflow measurement limits. They do not replace the independent held-out policy study.

1. **Insufficient incremental value:** native tools plus TrainCheck/TrainParity may achieve the same selection quality. Compare directly and pivot to an evaluation contribution if so.
2. **Verification is too expensive or too weak:** small probes may miss late drift; full paired runs can exceed the saved compute. Measure verification cost and report inconclusive outcomes.
3. **Access benefit is small:** memory savings may merely replace VRAM scarcity with CPU RAM, runtime, or integration burden. Measure end-to-end time and adaptation effort, and avoid institutional-equity claims without a user study.

The [proposal](10-research-proposal.md), [architecture](../architecture/overview.md), and [acceptance criteria](../roadmap/mvp-acceptance-criteria.md) define the next stage. All complete-training and consumer-GPU benefits remain hypotheses.
