# Faculty-facing research proposal

## Title

**Contract-Constrained Selection of Memory-Saving PyTorch Training Plans on a Single Consumer GPU**

## Problem and candidate gap

Memory-saving transformations can make a training job fit while changing its intended update or mutable state. Existing runtimes already supply sophisticated optimization mechanisms, and validation tools already check important training properties. The open question for this project is whether making explicit contract evidence part of **plan selection**, especially for combinations, yields useful decisions beyond composing those existing capabilities.

This is a provisional empirical gap, not an established first-of-its-kind claim. The [competitor matrix](03-competitive-landscape.md) includes direct counterexamples to broad novelty; the closest additional checks are [TrainCheck](https://www.usenix.org/conference/osdi25/presentation/jiang) and [TrainParity](https://github.com/intelland/TrainParity).

The continuation's [HiLoRe comparison](papers/hilore.md) further weakens the general planner claim; [FairScale](competitors/fairscale.md) precedes the motivating state hazard. Treat the benchmark and measured incremental benefit as the research test. The [power correction](experiments/equivalence-calibration.md) also makes full-outcome equivalence more expensive than the initial illustrative calculation suggested.

## Research question and hypothesis

On held-out single-device PyTorch workloads and checkpointing/accumulation plans, does contract-constrained selection accept fewer plans that violate declared training semantics, without reducing valid-feasible workload coverage, than native optimizations plus existing checks under the same eight-trial/30-minute search budget?

Primary hypothesis: the policy reduces incompatible acceptance without reducing valid-feasible coverage on the held-out set. Resource-only selection is an additional ablation. **Null/practical failure:** the composed baseline matches or exceeds that tradeoff, or added verification cost makes a gain unusable. The initial budget may change only after a disjoint pilot and before confirmation. The experiment is valuable even if the hypothesis fails.

## Method

1. Define a small contract schema with invariant data/objective/update fields and scoped comparison rules.
2. Implement trusted adapters and profile complete logical updates on CPU and a validated CUDA environment.
3. Enumerate at most 8 native strict plans per workload; apply precondition rules and bounded differential checks, including buffers and update clocks.
4. Compare selection policies under the same wall-time/trial budgets. Record every failed, rejected and abstained candidate.
5. Evaluate final outcomes only under prespecified repeated full-run protocols; leave conclusion fidelity unverified when those runs are unavailable.

The method proposes no new optimizer, rematerialization solver or automatic theorem prover. Its engineering output is an auditable evidence trail; its research output is measured selection behavior and a reusable interaction benchmark.

## Baselines and evaluation

Use unmodified/native PyTorch, manual configuration, compatible native automated checkpointing and Rockmate/DeepSpeed subsets, an explicit native-plus-checker composition, deterministic Atlas, optional LLM ranking, and fidelity-disabled Atlas. Primary semantic evaluation uses positive/negative/unknown cases with held-out code/fault families and two qualified annotators. Resource evaluation uses a small CNN, two decoder sizes and pretrained fine-tuning, conditioned on source/license resolution.

Metrics: valid feasible coverage, incompatible acceptance, false rejection/abstention, memory peaks, total time, search/check overhead, failure rate and paired outcome uncertainty. Numerical equivalence and scientific margins are distinct. See the complete [experimental design](07-experimental-design.md), including actual-versus-planned evidence and power limits.

## Expected contribution, if supported

- An explicit mapping from declared training requirements to plan eligibility and necessary evidence.
- An audited benchmark of checkpointing/accumulation interactions and unknown cases.
- An empirical demonstration of incremental selection benefit **or** a well-supported negative result explaining why existing composition suffices.

The benchmark could be a more durable contribution than a large application. No publication, impact or acceptance is guaranteed.

## Limitations and risks

Adapters constrain generality and impose author effort. Small numerical checks do not establish late-training behavior. Identical outcomes on a few workloads do not establish equality across scientific domains. One GPU cannot establish hardware portability. A missing full baseline can leave fidelity unknown. Verification and LLM costs may erase savings. An ordinary local process is not a sandbox for hostile training code.

## Evidence before approaching an advisor

Bring this source-backed proposal, the competitor comparison, executable CPU counterexamples, one working trusted profiler adapter, an explicit comparison against native mechanisms plus existing checks, and a realistic compute inventory. Ask for help with the novelty boundary, experiment-contract semantics, independent labels and meaningful margins. It is reasonable to approach an advisor before positive results; present uncertainties plainly rather than arrive with a premature claim of novelty.

## Possible academic paths

An undergraduate research symposium or supervised independent-study thesis is a realistic first target. The [ACM Student Research Competition](https://src.acm.org/?searchterm=src) provides an undergraduate research forum at participating conferences; eligibility and the current host CFP need checking at submission time. [MLSys](https://mlsys.org/Conferences/2026) and its relevant workshops are topical places to study accepted work and eventual fit, not a claim that this specification is ready for the main conference. [JOSS](https://joss.theoj.org/about) may be relevant only after substantial, openly licensed research software with tests, documentation and demonstrated research use exists; this Markdown specification alone is not a software-paper submission.

No live deadline or guaranteed suitable workshop is asserted. Establish the research result and advisor guidance before selecting a venue.

## Immediate next step

The continuation [updated recommendation](14-updated-recommendation.md) and [checker-integration options](checker-integration-options.md) govern this stage: use the minimal profiling path to establish a shared strong-baseline comparison, rather than rebuilding generic checking as a novelty claim. Real-data pilot evidence now exists, but independently adjudicated cases and incremental benefit remain unproven.

Implement the deterministic adapter/profiler vertical slice described in [first coding milestone](../roadmap/mvp-acceptance-criteria.md). Do not start a frontend, generic code agent or distributed runtime. Reassess the project after one complete native baseline profile and an observed cost/coverage comparison on the first interaction cases.
