# Problem definition and claim boundaries

## User and task

A researcher has a **specific experimental protocol**, a trusted local training implementation, and limited GPU memory, host RAM, or wall time. They want to know which supported execution changes are compatible with their declared scientific constraints and whether any such plan fits the available machine. They do not merely want maximum throughput at any model quality.

The university-resource motivation is plausible but not measured here. Atlas cannot supply missing data, expertise, research time, accelerator throughput, or access to a large reference run. The claim that software reduces institutional inequality would require separate evidence about adoption and completed research.

## Formal scope

Write a reference experiment as E = (code, data, initialization, training protocol, evaluation, environment). A contract C names required relations and allowed changes. A candidate transformation T is selected from a finite supported registry. Hardware budget B contains device memory, host memory, timeout and disk limits.

For each T, Atlas produces measurements R(T(E)), evidence V(E,T(E),C), and explicit unknowns U. A practical planner minimizes measured end-to-end cost among candidates that satisfy required checks and fit B. It may return **no justified plan**. It must not trade a violated hard constraint against a scalar quality score.

The optimization is constrained and empirical. A short successful profile establishes feasibility for that observed execution window, not a bound for every later batch. A mathematical derivation establishes a relation under stated assumptions, not the absence of every implementation defect.

## Properties that must remain separate

| Property | Operational question | Evidence boundary |
|---|---|---|
| Resource efficiency | Did peak allocation or time decrease? | Measurement window, metric and hardware |
| Feasibility | Did the declared workload complete within all budgets? | Complete observed run; pilot-only status otherwise |
| Structural/methodological preservation | Were architecture, objective, data, optimizer and protocol maintained? | Contract diff plus observations; adapter declarations may be incomplete |
| Mathematical equivalence | Is the intended update algebra unchanged? | Explicit assumptions about separability, state and randomness |
| Numerical parity | How do outputs, gradients and states differ? | Defined dtype, inputs, devices and tolerances |
| Statistical equivalence | Is a population effect contained in a justified equivalence interval? | Repeated runs and a specified estimand |
| Conclusion fidelity | Does the original scientific claim still follow? | Domain and protocol-specific human assessment plus experimental evidence |

## In scope for the first research iteration

- Single-process, single-device supervised training with registered adapters and known loss accounting.
- A supplied reference initialization and deterministic replay inputs for short comparisons.
- Activation checkpointing and microbatch decomposition without changing the intended logical update.
- Conditions including BatchNorm, dropout, unequal denominators, tail batches, clipping and scheduler timing.
- A finite plan space, bounded trial count and explicit refusal/abstention.

## Out of scope

Arbitrary-repository rewriting, automatic extraction of scientific intent from papers, distributed training, inference-only quantization, new CUDA kernels, architecture search, optimizer substitution, cloud scheduling and automatic approval of untrusted code. BF16/FP16 change arithmetic and need a separately allowed empirical contract; they are not enabled silently.

## Baseline unavailable on the target device

An OOM baseline creates an identification problem: there are no comparable baseline gradients or outcomes at that full setting on that device. Options are a trusted external reference artifact with matching provenance, a slower reference on CPU with cross-device caveats, or an explicitly smaller matched study. None proves the unavailable full experiment. Report resource feasibility separately from unestablished fidelity. Do not compare an optimized large run to a smaller reference and call them equivalent.

## Failure and refusal are useful outputs

Atlas should reject known-incompatible plans, abstain when a precondition is unknown, and preserve execution errors separately. It should explain why no supported plan was found, including whether model state alone appears to exceed memory. Human waiver creates a new contract and experiment revision; it must not overwrite a previous failure.

These boundaries follow from existing [reproducibility limits](https://docs.pytorch.org/docs/2.14/notes/randomness.html), [accuracy-consistent systems](https://arxiv.org/html/2208.14228v2), and the need to distinguish [equivalence evidence from nonsignificance](https://doi.org/10.1177/1948550617697177). The formal problem and product boundary above are proposed Atlas design decisions.
