# Updated recommendation after continued research and audit

**Proceed with the benchmark package; define the distinct policy treatment before comparison.** The [identifiability amendment](15-comparison-identifiability.md) and [ADR 007](decisions/007-benchmark-first.md) refine the earlier recommendation: an equally informed competent baseline needs an explicitly different Atlas algorithm to make superiority testable. A local engineering tool is feasible; standalone optimizer/verifier novelty remains unestablished.

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

External execution is limited to selected TrainParity CPU controls and adapted Rockmate-family pure GPU cases; no full published performance suite, HiRemate or OffMate study was replicated. No independent annotators/advisor reviewed the cases, no primary selection experiment ran, no real OOM workload was rescued, and no 6–8/16/24 GB card was tested. Only a [gross shared-board energy replay](experiments/board-energy-control.md) now exists, with unresolved counter/power disagreement; there is no calibrated process/whole-host energy or local-model inference result, no general scientific-conclusion certification and no validated hostile-code sandbox. WikiText and full pretrained-model fine-tuning/evaluation remain conditional; the bounded pretrained QA segments below have executed. The research log records actual duration; the continuation is not retroactively called a five-hour investigation.

[Twelve native selective-checkpoint controls](experiments/native-selective-control.md) further strengthen the comparator: existing per-operation saving/recomputation produces a distinct intermediate tradeoff on the original pure fixture. The bounded study should include a supported fixed selective configuration when applicable, under the same registry/search budget; it remains an existing native mechanism, not an Atlas contribution.

The [token-reference control](experiments/token-reference-control.md) also confirms that finite, reproducible differential agreement cannot establish the supplied reference implements the intended objective. Make denominator units/masks and author intent explicit, share that information with the baseline, and obtain independent adjudication. This known oracle limitation is not evidence that only Atlas can enforce the contract.

The second continuation executes [sixteen pretrained BERT QA resource segments](workload-screen/qa-resource-pilot.md), replacing the blanket absence of a pretrained workload screen. It does not run full fine-tuning or QA evaluation. Native SDPA is a strong baseline; eager completion above nominal device capacity exposes a Windows residency boundary. Repeated-hash differences require reference-noise diagnosis, while a separate ownership control invalidates launcher-only worker-memory readings. [Composite-loss controls](experiments/composite-objective-control.md) add objective/health boundaries and repaired positives. These advance the evaluation and architecture, not a primary-policy win.

## Latest comparison and implementation implications

[The built-in BERT constructor already selects SDPA](workload-screen/qa-default-attention-control.md). Its full-batch resource condition is a competent native default in this pinned stack; the explicit eager condition cannot be used to manufacture an Atlas benefit. [Actual profiler events](workload-screen/qa-dispatch-control.md) identify the observed efficient CUTLASS backend in two shapes, not FlashAttention or every configuration.

[Owned-worker existing-checker integration](workload-screen/qa-numerical-control.md) passes repeat/checkpoint controls and detects deliberate changes. Microbatching fails the fixed predicate even with dropout zero. [Native AdamW replays](workload-screen/qa-adam-sensitivity-control.md) explain selected update sensitivity without changing the original experiment. [Fresh-process logical-32 reference pairs](workload-screen/qa-reference32-control.md) also have small selected parameter differences exceeding a diagnostic bound, despite matching initial/input/RNG records. Keep literal observations separate from fault attribution; do not widen a tolerance to force a desired outcome. [QA-head relation controls](experiments/qa-gauge-control.md) motivate distinct prospective functional and strict-parameter contracts, not retrospective waivers.

The [claim-applicability design](../architecture/claim-applicability.md) and revised [MVP acceptance criteria](../roadmap/mvp-acceptance-criteria.md) now require exact observation fields/arithmetic/scope, reference stability, logical multi-head objective context, actual worker ownership, per-worker runtime provenance and nested cost windows. The [cost ledger](workload-screen/qa-workflow-costs.md) totals 25 windows/34 model executions/144 updates and 536.676 seconds of reported supervised windows across five studies; this is neither complete workflow cost nor GPU-active time. [Post-study reconciliation](workload-screen/post-study-environments.md) covers eight profiles and 54 selected installed source files, with explicit timing/attestation limits.
