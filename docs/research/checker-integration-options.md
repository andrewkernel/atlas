# Reuse and integration options before building a new checker

The [deeper source audit](competitors/trainparity.md) establishes substantial overlap in observation/capture/comparison. The [original liveness analogue](experiments/activation-liveness-probe.md) also shows why a verification execution can consume different resources from a target plan. Neither finding justifies assuming that Atlas already outperforms the tool.

## Preferred next comparison

Start with a reviewed shared workload and a mutually compatible stack for both policies. Give existing checks the same initialization, input/contract facts and named observations. Measure the checker workflow as it actually runs, plus an explicitly documented memory-conscious adaptation if necessary. Record human adaptation time and modified behavior. Keep runtime comparisons separate when environments differ.

Reuse existing capture/comparison semantics where possible instead of implementing another generic state checker and presenting it as a new contribution. Any vendoring requires provenance/license compliance and maintenance responsibility; a separate reviewed source-path adapter now executes the [bounded CPU controls](experiments/trainparity-controls.md). This is not an integrated Atlas product. A small executable comparison is not a substitute for the eventual independently labeled shared-workload selection study.

## Streaming a known global denominator

For a separable objective `L = sum_j S_j / D`, if D is known from batch metadata before forward and is independent of trainable parameters, a standard memory-conscious executor can backpropagate `S_j/D` immediately for each microbatch, with fixed parameters and one final update. Detached numerator/count summaries do not need to retain activation graphs. Collect the same gradient/state observations at the logical-update boundary, with clipping/scaling on the declared clock.

This is ordinary gradient accumulation, not a new optimization. It preserves the conditional algebra, while reduction order and finite precision still need measured comparisons. Auxiliary regularizers, shared parameters, stochastic assignment and batch coupling remain explicit obligations.

## When the denominator is unavailable

Do not silently run an extra no-grad forward to discover D: BatchNorm buffers, custom state and RNG can change, and the pass has a real cost. A model-output-dependent or differentiable denominator needs its own derivation and replay conditions. If D depends differentiably on parameters, scaling summed numerator gradients afterward generally misses the denominator derivative.

For a stop-gradient denominator, later normalization of accumulated numerator gradients can be algebraically possible, but it changes arithmetic/scaling behavior and can interact with AMP overflow, clipping and auxiliary terms. It is not a default safe substitution. The smallest initial adapter should require the simple metadata-known denominator or abstain. Prefer a clear unsupported case to a generic fallback that changes the contract.

## Missing reference problem

When the full reference OOMs on the target budget, neither Atlas nor an existing differential checker magically acquires that observation. An uncapped or larger-device reference, a CPU reference with cross-device caveats, or a narrowed independently identified study can help. Smaller-batch checks are bounded evidence for the smaller execution; they do not certify an unavailable full experiment. Conditional mathematical support and empirical outcome support must remain separate.

## Contribution decision

An [executed reuse control](experiments/planner-reuse-control.md) now passes native/planner streaming but raises a shape error for two deferred-forward patterns on the pinned family wrapper. Keeping outputs and a non-recomputing schedule do not repair it. This tests the inspected checker's scheduling pattern, not an actual combined-tool run. A fair adapted baseline must preserve a supported target execution relation and retain unsupported/error outcomes; this loud failure supplies no unsafe-acceptance or novel-rematerialization claim.

The research test is whether integrating eligibility, resource measurement, evidence and refusal yields useful decisions beyond this competent composition. If the answer is no, a well-documented integration/benchmark or upstream improvement may be the appropriate artifact. No upstream message, issue or pull request was sent in this session; this is a local reviewable integration note only.

The [shared GPU integration](experiments/interop-controls.md) now implements this competent adaptation for a known pure/separable case using existing lower-level snapshot capture and comparison. The default runner error is retained, and adapted initialized/post-update comparisons pass with a detected perturbation negative. Lower-level API stability, exact tolerance conventions, observation/health scope and wider workloads still need explicit policy handling. This is a reusable baseline experiment, not an Atlas product or new verification algorithm.

The [comparison-policy audit](experiments/comparator-contract-control.md) now concretely tests an additional adapter obligation: tensor/scalar rules and baseline/candidate scale differ from the proposed Atlas formula. Use the actual named relation, independently enforce declared health/configuration predicates, and do not silently relabel existing results as satisfying another contract.
