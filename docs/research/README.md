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

## Supporting evidence and implementation

- [Paper notes](papers/README.md), [source register](sources.md), [competitor notes/snapshots](competitors/README.md).
- [Actual CPU probes and limits](experiments/README.md), [raw results](experiments/cpu-semantic-probes.json).
- [Strategy-by-strategy consequences](strategy-analysis.md), [planned constructors](../../benchmarks/planned-workloads.md).
- [Exact evaluation units/denominators](evaluation-estimands.md), [composition-evidence boundary](composition-evidence.md), [historical failure provenance](real-world-failures.md).
- [Dependency/license status](dependencies-and-licenses.md), [dataset/model revisions](asset-snapshot.json).
- [Architecture decisions](decisions/README.md), [component architecture](../architecture/overview.md), [security](../architecture/security.md), [API contracts](../architecture/api-design.md).
- [Implementation phases](../roadmap/implementation-plan.md), [milestones](../roadmap/milestones.md), [first coding task/tests](../roadmap/mvp-acceptance-criteria.md).
- [Repository handoff and validation status](repository-status.md).

## What exists versus what is proposed

Executed: original CPU semantic/composition/state/update-clock controls, synthetic equivalence-power calibration,19 pytest tests including40 generated accumulation partitions, eight synthetic CUDA trials plus four published allocator-cap trials on a physical12 GB RTX3060, dependency locking, source/repository inspection and documentation checks. Proposed: Atlas application, real-workload optimization selection, full-training equivalence, expert labels, other physical hardware tiers, energy study and LLM evaluation.

The strongest local observation is that checkpointing can match gradients while changing BatchNorm buffers/evaluation. That motivates evidence design; it does not validate the proposed planner or certify scientific conclusions.
