# Updated recommendation after continued research and audit

**Proceed only with a bounded benchmark/comparison pilot.** A local engineering tool is feasible. A standalone novel optimizer or scientific-verification system is not established by the current evidence. Expanding the application before the closest-baseline comparison would invest in the least certain part of the project.

## What is now supported

- The physical12 GB RTX 3060 runs a separate PyTorch 2.14.1/CUDA 13.0 environment. Native checkpointing/microbatching show measured memory/time tradeoffs in [synthetic profiles](experiments/cuda-memory-probe.md), a clearly labeled [allocator-cap control](experiments/cuda-budget-probe.md), and [six five-epoch real-data CNN pilots](experiments/fashion-pilot.md).
- Original CPU controls establish specific denominator, BatchNorm, RNG assignment, clipping, absent-gradient and update-clock hazards. They also include positive/repair controls. These are finite fixtures, largely established mechanisms.
- [Direct TrainParity controls](experiments/trainparity-controls.md) now reproduce three positive, six faulted, one refusal and one setup-error outcome in a supported separate CPU stack, after a retained adapter-setup correction. This confirms practical generic-checker overlap.
- [Adapted Rockmate-family GPU controls](experiments/rockmate-compatibility.md) now execute existing planning/scheduling, including tighter-budget schedules with repeated forwards and zero recorded numerical errors over eight updates. Failed interfaces and framework interactions are retained. This is limited compatibility evidence; observed whole allocation is higher than native in this diagnostic, and no general performance comparison is justified.
- Statistical audits corrected an actual power-formula error and demonstrated why valid normal-model power is not enough to justify the outcome model. Full-scientific claims need substantially more than small numerical checks.

The CNN reference fits easily. Its combined plan reduces allocated peak by about 32.7% while summed synchronized update time increases about 66%. Three paired validation differences range-.21 to+.05 percentage points; there is no powered equivalence conclusion. This is useful feasibility/debugging evidence, not expanded access or primary-policy superiority.

## What became less credible

The [TrainParity source inspection](competitors/trainparity.md) confirms direct overlap in buffers, keyed/None gradients, optimizer/scheduler/scaler state and known accumulation fault controls. [FairScale](competitors/fairscale.md) already addresses a motivating state hazard. [HiLoRe](papers/hilore.md) directly overlaps broad memory/fidelity-constrained planning. [Rockmate's successors](competitors/rockmate-family.md) include hierarchical planning and integrated offload; comparing only the original2023 abstraction would miss relevant capabilities.

The current composed-update fixtures do **not** yet demonstrate an emergent failure of individually valid transformations. [That distinction](composition-evidence.md) weakens an interaction-specific novelty claim. A new report format, explicit contract schema, local UI or optional LLM does not by itself supply missing research novelty.

The [reuse follow-up](experiments/planner-reuse-control.md) now finds a concrete deferred-forward execution failure while streaming controls pass. Its persistence without planned recomputation weakens, rather than establishes, a novel emergent-transform claim. It motivates making the verifier's execution relation explicit and giving the composed baseline a supported streaming adaptation. No unsafe scientific acceptance was observed.

## Narrow claim worth testing next

Does a transparent contract-conditioned selection workflow improve the safety/valid-feasible-coverage/cost tradeoff on independently adjudicated held-out training cases over a competent composition of native mechanisms and existing checkers?

Use the [precise units and denominators](evaluation-estimands.md), shared adapter/contract information, comparable software, and equal search budgets. The benchmark and comparison are the central artifact. A negative result that existing composition suffices is an acceptable outcome; it must not be hidden by choosing weak baselines or relabeling known faults as new interactions.

## Bounded next-stage gate

Before adding more mechanisms, an LLM or a frontend:

1. The basic executable-baseline gate now covers CPU controls, adapted GPU planner cases and a shared supported-stack integration. Existing capture/comparison primitives can verify the known streaming relation. Extend this to a meaningful selected resource workload; the independently labeled held-out selection comparison is still required.
2. Obtain independent labels on a small positive/invalid/indeterminate corpus and identify whether any actual combined-case coverage is missing. If all value reduces to known rules, pivot to benchmark consolidation/upstream contributions or explicitly call the project engineering.
3. Select a meaningful memory-bound workload with justified dimensions and frozen input/protocol, then measure a genuine useful completion under a declared budget. Do not enlarge a toy solely to manufacture an impressive access claim.
4. Compare decisions and total workflow cost with the strong baseline. Freeze practical decision criteria after a disjoint pilot; record no-plan, unsupported, failure and uncertainty outcomes.

If the existing composition matches the tradeoff or verifier cost overwhelms its benefit, stop the standalone planner research expansion. If a concrete repeated gap survives, implement only the minimal deterministic workflow required to evaluate it. The current application directories are still boundaries, not a finished product.

## Remaining limits

External execution is limited to selected TrainParity CPU controls and adapted Rockmate-family pure GPU cases; no full published performance suite, HiRemate or OffMate study was replicated. No independent annotators/advisor reviewed the cases, no primary selection experiment ran, no real OOM workload was rescued, and no 6–8/16/24 GB card was tested. Only a [gross shared-board energy replay](experiments/board-energy-control.md) now exists, with unresolved counter/power disagreement; there is no calibrated process/whole-host energy or local-model inference result, no general scientific-conclusion certification and no validated hostile-code sandbox. WikiText/pretrained workloads remain conditional. The research log records actual duration; the continuation is not retroactively called a five-hour investigation.

[Twelve native selective-checkpoint controls](experiments/native-selective-control.md) further strengthen the comparator: existing per-operation saving/recomputation produces a distinct intermediate tradeoff on the original pure fixture. The bounded study should include a supported fixed selective configuration when applicable, under the same registry/search budget; it remains an existing native mechanism, not an Atlas contribution.

The [token-reference control](experiments/token-reference-control.md) also confirms that finite, reproducible differential agreement cannot establish the supplied reference implements the intended objective. Make denominator units/masks and author intent explicit, share that information with the baseline, and obtain independent adjudication. This known oracle limitation is not evidence that only Atlas can enforce the contract.
