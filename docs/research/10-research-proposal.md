# Faculty-facing research proposal

## Title

**Evaluating Scoped Fidelity Evidence for Native PyTorch Training Plans on a Consumer GPU**

## Problem and candidate gap

Memory-saving transformations can change an intended update or mutable state, while finite checks observe only particular relations/windows. Existing runtimes and validation tools already supply substantial capabilities. The immediate project is a reproducible evaluation package examining which claims those observations justify and what resource/verification costs they incur on limited hardware.

Benchmark novelty itself remains unestablished. The [competitor matrix](03-competitive-landscape.md), TrainCheck/TrainParity and [constrained-acquisition/translation-validation lineages](16-policy-and-validation-lineage.md) prevent broad first-of-its-kind claims. [ADR 007](decisions/007-benchmark-first.md) prioritizes evaluation and existing-tool integration, with an explicit stop/pivot option.

The continuation's [HiLoRe comparison](papers/hilore.md) further weakens the general planner claim; [FairScale](competitors/fairscale.md) precedes the motivating state hazard. Treat the benchmark and measured incremental benefit as the research test. The [power correction](experiments/equivalence-calibration.md) also makes full-outcome equivalence more expensive than the initial illustrative calculation suggested.

## Research question and hypothesis

Selected immediate question: **On a frozen independently adjudicated set of workload/contract cases, what fraction of required claims can a competent native-plus-existing-checker implementation support within a declared observation horizon and total budget?** Record the supported count and all required case-claim pairs, contradicted/indeterminate counts and cost. Distinguish literal finite comparisons from objective adequacy, reference stability, execution scheduling and observer applicability. This finite-set estimate is not a field-wide risk rate, and the current investigator-created discovery cases do not supply its independent labels.

A concrete falsifiable sufficiency hypothesis is that this existing composition covers every independently supported positive case-claim pair in that fixed set under the declared budget. An observed unsupported pair refutes that bounded implementation/budget hypothesis; it does not prove the upstream tools are incapable of an adaptation or establish that Atlas supplies one. A zero-gap result is useful. Freeze cases, required claims, adaptation allowance, evidence-sufficiency procedure, horizon and budget before the comparison. Resource/host-memory/workflow tradeoffs are secondary measured outputs, not substitutes for the claim denominator.

The earlier selection-superiority hypothesis is conditional future work. [The treatment audit](15-comparison-identifiability.md) identifies the missing distinct Atlas algorithm: equal facts/checks and an equal decision function do not define distinct treatments. No confirmatory policy comparison should start until algorithm, objective, information and budgets are frozen after a disjoint pilot. A negative result that existing composition suffices is acceptable. Current investigator-created cases are development evidence, not an independent holdout or field-prevalence estimate.

## Method

1. Freeze intent, reference protocol, executor/observer and required relation separately; record denominator units, precision, fields and horizons.
2. Use trusted adapters and existing capture/checker primitives in a mutually supported stack. Profile full logical updates and validate the actual owned worker boundary.
3. Build positive/invalid/indeterminate development cases, retaining reference-noise and observer-failure cases. Obtain independent adjudication before using a new corpus for confirmatory evaluation.
4. Compare native recipes and applicable checking scopes with all failures, missing evidence and costs retained. Different relations answer different questions; do not call their disagreement a classifier error without a shared intended target and independent labels.
5. Run outcome/quality studies only with a separately frozen evaluation method, margins and sufficient repeated full runs. Leave unavailable domain conclusions unverified.

The method proposes no new optimizer, solver or theorem prover. A reusable benchmark/evidence package is the current engineering deliverable. A research contribution depends on a meaningful independently reviewed corpus, repeatable findings beyond established toy mechanisms, or a later distinct policy that survives the competent baseline. Current compositions do not establish a new emergent interaction.

## Baselines and evaluation

The essential baseline is native full-batch, supported efficient attention, checkpointing/accumulation and the same applicable existing checks with equal adapter facts. Existing Rockmate-family subsets supply scoped compatibility evidence; their historical paper performance is not replicated. Optional algorithms/LLM ablations wait for a defined policy. Independent semantic review requires at least two qualified annotators and explicit relation labels; none has been obtained.

Executed development evidence includes real-data CNN pilots and pretrained BERT/QA resource/first-update controls. The logical-32 BERT reference itself varies in selected fields, while reduced-batch checkpointing passes the existing comparator. These refine the study, not a policy advantage. Full QA fine-tuning/evaluation, decoder studies and other physical GPU tiers remain unexecuted or conditional. See the [resource evidence](workload-screen/qa-resource-pilot.md), [reference-noise diagnostic](workload-screen/qa-reference32-control.md), [cost-window limits](workload-screen/qa-workflow-costs.md) and [prospective quality-evaluation specification](workload-screen/qa-quality-evaluation-spec.md).

Immediate metrics are fixed-set case-claim support/disposition counts, independently supported positive-claim coverage, observed resource completion and complete acquisition/integration/check cost. Later policy metrics include valid-feasible coverage, incompatible acceptance, false rejection/abstention, memory peaks, search/check overhead, failure rate and paired outcome uncertainty. Do not mix claim-pair denominators with the [policy-unit estimands](evaluation-estimands.md). Numerical equivalence and scientific margins are distinct. See the complete [experimental design](07-experimental-design.md), including actual-versus-planned evidence and power limits.

## Expected contribution, if supported

- A versioned corpus of declared relations, assumptions, reference/observer limitations and independently reviewed labels if obtainable.
- Reproducible native/existing-tool measurements with complete failure and cost accounting.
- A defensible empirical case study or negative result; incremental selection benefit is an additional conditional claim requiring a distinct algorithm.

The benchmark could be a more durable contribution than a large application. No publication, impact or acceptance is guaranteed.

## Limitations and risks

Adapters constrain generality and impose author effort. Small numerical checks do not establish late-training behavior. Identical outcomes on a few workloads do not establish equality across scientific domains. One GPU cannot establish hardware portability. A missing full baseline can leave fidelity unknown. Verification and LLM costs may erase savings. An ordinary local process is not a sandbox for hostile training code.

## Evidence before approaching an advisor

Bring this source-backed proposal, the competitor comparison, executable CPU counterexamples, one working trusted profiler adapter, an explicit comparison against native mechanisms plus existing checks, and a realistic compute inventory. Ask for help with the novelty boundary, experiment-contract semantics, independent labels and meaningful margins. It is reasonable to approach an advisor before positive results; present uncertainties plainly rather than arrive with a premature claim of novelty.

## Possible academic paths

An undergraduate research symposium or supervised independent-study thesis is a realistic first target. The [ACM Student Research Competition](https://src.acm.org/?searchterm=src) provides an undergraduate research forum at participating conferences; eligibility and the current host CFP need checking at submission time. [MLSys](https://mlsys.org/Conferences/2026) and its relevant workshops are topical places to study accepted work and eventual fit, not a claim that this specification is ready for the main conference. [JOSS](https://joss.theoj.org/about) may be relevant only after substantial, openly licensed research software with tests, documentation and demonstrated research use exists; this Markdown specification alone is not a software-paper submission.

No live deadline or guaranteed suitable workshop is asserted. Establish the research result and advisor guidance before selecting a venue.

## Immediate next step

The [updated recommendation](14-updated-recommendation.md), [treatment audit](15-comparison-identifiability.md) and [claim applicability](../architecture/claim-applicability.md) govern this stage. Existing-tool prototypes now run on real BERT; production interfaces and independent adjudication remain uncompleted. Preserve the discovered metadata gap: post-study driver/cuDNN reads do not replace contemporaneous per-worker capture.

Implement the narrow trusted adapter/profiler/report slice in the [revised coding milestone](../roadmap/mvp-acceptance-criteria.md), and prepare independent-review pilot packets without presenting the current cases as untouched holdout. Do not start a frontend, generic verifier or distributed runtime. Reassess research value before developing a new ranking algorithm.
