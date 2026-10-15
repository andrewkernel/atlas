# Atlas research and engineering specification

**Verdict: Proceed with modifications.** Study contract-conditioned selection of native memory-saving plans for trusted single-device workloads. Broad automatic-memory-optimization and training-correctness novelty is already contradicted by prior art; incremental novelty remains to be established.

Started 2026-10-14 America/Chicago (2026-10-15 UTC). This is a bounded initial investigation, not a claim of five hours, an exhaustive review or a completed primary evaluation.

## Main documents

| Document | Purpose |
|---|---|
| [00 Executive assessment](00-executive-assessment.md) | Verdict, research question, architecture and risks |
| [01 Problem definition](01-problem-definition.md) | Scope, formal selection problem and unavailable-reference boundary |
| [02 Literature review](02-literature-review.md) | Search method, research lineages and synthesis |
| [03 Competitive landscape](03-competitive-landscape.md) | Detailed mechanisms, hardware, fidelity and overlap matrix |
| [04 Scientific fidelity](04-scientific-fidelity.md) | Property/status framework, mathematical conditions and evidence |
| [05 Technical feasibility](05-technical-feasibility.md) | Memory estimates, OOM diagnosis, adapters and stack |
| [06 Novelty analysis](06-novelty-analysis.md) | Six ranked candidates, selected contribution and falsifiers |
| [07 Experimental design](07-experimental-design.md) | Baselines, workloads, physical tiers, statistics and threats |
| [08 Local LLM evaluation](08-local-llm-evaluation.md) | Models/licenses, task boundaries and equal-budget ablation |
| [09 Critical review](09-critical-review.md) | Three authored adversarial perspectives and revisions |
| [10 Research proposal](10-research-proposal.md) | Faculty-facing narrow research package |
| [11 Open questions](11-open-questions.md) | Unfinished evidence and stop/pivot requirements |
| [12 Research log](12-research-log.md) | Observed chronology and execution limits |
| [13 Audit and amendments](13-audit-and-amendments.md) | Corrections, new prior art and evidence audit |
| [14 Updated recommendation](14-updated-recommendation.md) | Current decision after deeper comparison and real-data pilots |
| [15 Comparison identifiability](15-comparison-identifiability.md) | Distinct-treatment requirement and benchmark-first scope amendment |
| [16 Policy/validation lineage](16-policy-and-validation-lineage.md) | Established constrained acquisition and bounded translation validation |

## Supporting evidence and implementation

- [Paper notes](papers/README.md), [source register](sources.md), [competitor notes/snapshots](competitors/README.md).
- [Actual CPU probes and limits](experiments/README.md), [raw results](experiments/cpu-semantic-probes.json).
- [Strategy-by-strategy consequences](strategy-analysis.md), [planned constructors](../../benchmarks/planned-workloads.md).
- [Exact evaluation units/denominators](evaluation-estimands.md), [composition-evidence boundary](composition-evidence.md), [historical failure provenance](real-world-failures.md).
- [Measured cost/Pareto analysis](cost-and-pareto-analysis.md), [local-model component/cache audit](local-model-capacity.md), [statistical assumption controls](experiments/statistical-assumption-audit.md).
- [Verification-pattern memory control](experiments/activation-liveness-probe.md), [existing-checker integration options](checker-integration-options.md).
- [Reliability/multiplicity/stopping audit](statistical-claim-selection.md).
- [Bibliography availability/metadata audit](bibliography-audit.md), [fresh-process training repeatability](experiments/fashion-repeatability.md).
- [Claim-to-evidence map](evidence-map.md) for reviewing the supported conclusions and limits.
- [Dependency/license status](dependencies-and-licenses.md), [dataset/model revisions](asset-snapshot.json).
- [Architecture decisions](decisions/README.md), [component architecture](../architecture/overview.md), [security](../architecture/security.md), [API contracts](../architecture/api-design.md).
- [Implementation phases](../roadmap/implementation-plan.md), [milestones](../roadmap/milestones.md), [first coding task/tests](../roadmap/mvp-acceptance-criteria.md).
- [Repository handoff and validation status](repository-status.md).

## What exists versus what is proposed

Executed: original CPU semantic/composition/state/update-clock controls, statistical calibration,20 pytest tests including 40 generated accumulation partitions, eight synthetic CUDA trials, four published allocator-cap trials and six five-epoch Fashion-MNIST CNN pilots on a physical12 GB RTX 3060, dependency locking, source/repository inspection and documentation checks. Proposed: Atlas application, primary optimization-selection comparison, confirmatory full-training equivalence, expert labels, other physical hardware tiers, calibrated energy/attribution study and LLM evaluation. Later narrow existing-tool integration and gross-board replays are indexed below. The [real-data pilot protocol](experiments/fashion-pilot-protocol.md) was frozen before execution; [results](experiments/fashion-pilot.md) remain exploratory.

The strongest local observation is that checkpointing can match gradients while changing BatchNorm buffers/evaluation. That motivates evidence design; it does not validate the proposed planner or certify scientific conclusions.

[Gross-board energy controls](experiments/board-energy-control.md) add six exact same-seed training replays, independent arithmetic, and a passive telemetry diagnostic. Both raw APIs are retained despite substantial disagreement. These are not calibrated process energy or additional independent seeds.

[Evidence-contract walkthroughs](../architecture/evidence-contract-example.md) collect the resulting implementation obligations without claiming an implemented decision engine. The [deeper TrainCheck note](papers/traincheck.md) now retains selected primary-PDF read/visual scope and comparator-cost limits.

[Independent adjudication packet specification](adjudication-protocol.md) prepares the uncompleted external-label gate, including reviewer information, disagreement retention and discovery/holdout separation. No external reviewer has been contacted or labels obtained.

[Assignment deliverable audit](deliverable-audit.md) maps requested outputs to documents and distinguishes the completed initial blueprint from unfinished empirical studies.

[Native selective-checkpoint control](experiments/native-selective-control.md) tests another existing native baseline on the unchanged pure fixture, with all twelve trials and first-update observations retained.

[Token-objective/reference adequacy](experiments/token-reference-control.md) tests a shared-reference boundary independently of nonfinite health: two internally consistent executions can agree on the wrong stated objective.

The second continuation adds [pretrained BERT/QA resource segments](workload-screen/qa-resource-pilot.md), source/asset/feature identities, a separately locked QA environment and a retained launcher-monitor defect. [Composite objective controls](experiments/composite-objective-control.md) refine per-term normalization and health. The current CPU/Windows lifecycle suite passes 56 tests; earlier 20-test counts describe historical checkpoints. [ADR 007](decisions/007-benchmark-first.md) and the [objective/evidence interface](../architecture/objective-and-evidence-boundaries.md) now govern scope. The policy algorithm, independent labels and primary study remain uncompleted.

[Repaired Windows job lifecycle](workload-screen/windows-job-control.md) and [reduced-batch BERT capture](workload-screen/qa-numerical-control.md) now execute through a new verified process boundary. The existing comparator passes native checkpointing and detects deliberate perturbations; microbatch results retain both stochastic differences and a smaller dropout-zero numerical failure without changing tolerance.

[Device-event dispatch](workload-screen/qa-dispatch-control.md) identifies the observed memory-efficient attention backend. [Selected adaptive-update replay](workload-screen/qa-adam-sensitivity-control.md) explains selected parameter/moment changes given the recorded small gradients; bias invariance and remaining weight differences prevent a max-parameter-error shortcut to scientific conclusions. The [lineage amendment](16-policy-and-validation-lineage.md) further narrows acquisition/proof claims.

Latest evidence: [owned BERT checks](workload-screen/qa-numerical-control.md), [default attention](workload-screen/qa-default-attention-control.md), [actual dispatch](workload-screen/qa-dispatch-control.md), [AdamW sensitivity](workload-screen/qa-adam-sensitivity-control.md), [logical-32 reference noise](workload-screen/qa-reference32-control.md), [QA-head relation controls](experiments/qa-gauge-control.md), [cost ledger](workload-screen/qa-workflow-costs.md) and [post-study environments](workload-screen/post-study-environments.md). The [claim-applicability design](../architecture/claim-applicability.md) connects these boundaries to the revised implementation specification.

The [prospective QA quality-evaluation specification](workload-screen/qa-quality-evaluation-spec.md) records newly inspected decoder/metric source and missing-data/version guards. It is not an implemented decoder, full-training protocol freeze or quality result.
