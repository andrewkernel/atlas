# Updated recommendation after continued research and audit

**Proceed only with a bounded benchmark/comparison pilot.** A local engineering tool is feasible. A standalone novel optimizer or scientific-verification system is not established by the current evidence. Expanding the application before the closest-baseline comparison would invest in the least certain part of the project.

## What is now supported

- The physical12 GB RTX 3060 runs a separate PyTorch 2.14.1/CUDA 13.0 environment. Native checkpointing/microbatching show measured memory/time tradeoffs in [synthetic profiles](experiments/cuda-memory-probe.md), a clearly labeled [allocator-cap control](experiments/cuda-budget-probe.md), and [six five-epoch real-data CNN pilots](experiments/fashion-pilot.md).
- Original CPU controls establish specific denominator, BatchNorm, RNG assignment, clipping, absent-gradient and update-clock hazards. They also include positive/repair controls. These are finite fixtures, largely established mechanisms.
- [Direct TrainParity controls](experiments/trainparity-controls.md) now reproduce three positive, six faulted, one refusal and one setup-error outcome in a supported separate CPU stack, after a retained adapter-setup correction. This confirms practical generic-checker overlap.
- Statistical audits corrected an actual power-formula error and demonstrated why valid normal-model power is not enough to justify the outcome model. Full-scientific claims need substantially more than small numerical checks.

The CNN reference fits easily. Its combined plan reduces allocated peak by about 32.7% while summed synchronized update time increases about 66%. Three paired validation differences range-.21 to+.05 percentage points; there is no powered equivalence conclusion. This is useful feasibility/debugging evidence, not expanded access or primary-policy superiority.

## What became less credible

The [TrainParity source inspection](competitors/trainparity.md) confirms direct overlap in buffers, keyed/None gradients, optimizer/scheduler/scaler state and known accumulation fault controls. [FairScale](competitors/fairscale.md) already addresses a motivating state hazard. [HiLoRe](papers/hilore.md) directly overlaps broad memory/fidelity-constrained planning. [Rockmate's successors](competitors/rockmate-family.md) include hierarchical planning and integrated offload; comparing only the original2023 abstraction would miss relevant capabilities.

The current composed-update fixtures do **not** yet demonstrate an emergent failure of individually valid transformations. [That distinction](composition-evidence.md) weakens an interaction-specific novelty claim. A new report format, explicit contract schema, local UI or optional LLM does not by itself supply missing research novelty.

## Narrow claim worth testing next

Does a transparent contract-conditioned selection workflow improve the safety/valid-feasible-coverage/cost tradeoff on independently adjudicated held-out training cases over a competent composition of native mechanisms and existing checkers?

Use the [precise units and denominators](evaluation-estimands.md), shared adapter/contract information, comparable software, and equal search budgets. The benchmark and comparison are the central artifact. A negative result that existing composition suffices is an acceptable outcome; it must not be hidden by choosing weak baselines or relabeling known faults as new interactions.

## Bounded next-stage gate

Before adding more mechanisms, an LLM or a frontend:

1. The basic compatible external-checker execution gate is now met for original small CPU cases. Extend that reviewed adapter to the actual selected resource workload; the held-out selection comparison is still required.
2. Obtain independent labels on a small positive/invalid/indeterminate corpus and identify whether any actual combined-case coverage is missing. If all value reduces to known rules, pivot to benchmark consolidation/upstream contributions or explicitly call the project engineering.
3. Select a meaningful memory-bound workload with justified dimensions and frozen input/protocol, then measure a genuine useful completion under a declared budget. Do not enlarge a toy solely to manufacture an impressive access claim.
4. Compare decisions and total workflow cost with the strong baseline. Freeze practical decision criteria after a disjoint pilot; record no-plan, unsupported, failure and uncertainty outcomes.

If the existing composition matches the tradeoff or verifier cost overwhelms its benefit, stop the standalone planner research expansion. If a concrete repeated gap survives, implement only the minimal deterministic workflow required to evaluate it. The current application directories are still boundaries, not a finished product.

## Remaining limits

Only a narrow TrainParity CPU control suite was executed among external checkers; no external memory-planning system was run. No independent annotators/advisor reviewed the cases, no primary selection experiment ran, no real OOM workload was rescued, and no 6–8/16/24 GB card was tested. There is no energy or local-model inference result, no general scientific-conclusion certification and no validated hostile-code sandbox. WikiText/pretrained workloads remain conditional. The research log records actual duration; the continuation is not retroactively called a five-hour investigation.
