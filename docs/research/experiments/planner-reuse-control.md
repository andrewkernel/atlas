# Reuse and backward scheduling are part of the tested execution

The adapted pinned planner passes single-forward and streaming-accumulation controls. On the same pure workload, both tested patterns that defer backward until after two forwards raise a gradient-shape error. Keeping outputs does not repair it. The error also occurs in a schedule with **zero planned repeated forwards**, so this is not established as an emergent rematerialization interaction.

## Frozen execution and controls

The [protocol](planner-reuse-protocol.md) and [original fixture](../../../benchmarks/planner_reuse_control.py) fix seed 1901, the earlier four-block Linear/ReLU architecture, two different equal-size inputs/targets, exact global MSE denominator, one external SGD update, a 32 MiB Rotor configuration and numerical tolerances. Construction uses the previously tested scoped RNG guard and restores initial parameters. The native reference builds both loss terms before a joint backward; native streaming independently verifies the separable objective. The single-forward positive has its own one-microbatch reference, not the two-microbatch objective.

An initial reporting error under the minimal child environment masked the first deferred exception and stopped the launcher. [The incomplete attempt](planner-reuse-reporting-attempt.json) retains two completed positives, one reporting failure and two unstarted modes. Amendment `95b5cfe` repairs reporting without changing computation/settings/tolerances, continues after worker failures and labels any nonfinite scalar field explicitly. The corrected run includes all five intended modes.

|Execution pattern |Default 32 MiB |32 MiB with `keep_outputs=True` |64 MiB with `keep_outputs=True` |
|---|---|---|---|
|Native streaming |Matches native joint reference |Not repeated |Not repeated |
|Planner streaming |Matches |Matches |Matches; zero planned repeated forwards |
|Planner two forwards, separate backwards |Runtime error |Same error |Same error; zero planned repeated forwards |
|Planner two forwards, joint backward |Runtime error |Same error |Same error; zero planned repeated forwards |
|Planner one forward |Matches its one-forward reference |Not repeated |Not repeated |

Sources: [corrected five-mode run](planner-reuse-control.json), [output-retention follow-up](planner-reuse-retention-control.json) and [non-recomputing follow-up](planner-reuse-nonrecompute-control.json). All successful comparisons have zero measured max-absolute and relative-L2 gradient/parameter/momentum errors. All trial input/target digests agree. Failed modes report `invalid gradient at index 0 - got [0] but expected shape compatible with [128, 64, 64]`; they do not complete the candidate optimizer update and have no candidate outcome-equivalence result.

## What the controls rule out

The native and streaming positives make a general loss-normalization or data mismatch explanation less plausible on this fixture. The retained-output option rules out that single proposed option as a sufficient repair. The higher-budget schedule with no repeated forward operations shows that the observed failure does not require the selected recomputation operations. Neither control establishes the exact internal cause. Shared runtime storage and output/gradient lifetime are implementation hypotheses, not demonstrated diagnoses.

The [HiRemate paper](../papers/hiremate.md), Appendix C.2.2, discusses multiple forwards before backward in its own implementation/environment. These current generated Rotor configurations and contemporary source/stack do not reproduce that historical setting. The observation must not become “the paper is refuted,” “all HiRemate models fail,” or a newly established Atlas algorithmic advantage. No upstream issue or message was sent.

## Consequence for verification and selection

The inspected TrainParity worker also collects all microbatch losses before their backwards. This experiment tests that **execution pattern**, not an actual combined TrainParity/Rockmate run in a mutually compatible CUDA environment. Its result is an integration warning: a checker must preserve the target's relevant forward/backward ordering, or name the different relation it tests. A correct streaming plan can encounter an execution error under a deferred verification schedule.

For a metadata-known separable denominator, a competent baseline can use ordinary streaming accumulation and capture observations at the same logical-update boundary. Nonseparable objectives cannot be made streaming merely to avoid a failure. Record scheduling requirements and unsupported patterns in registered adapter/plan metadata; an unverified pattern yields abstention/error rather than an unsafe scientific-acceptance claim.

These are loud execution failures caught by the runtime, not observed silent scientific corruption or unsafe accepted final decisions. They strengthen the benchmark's execution-contract requirements and the case for testing actual composed workflows. They do not yet satisfy the independently labeled held-out primary study or prove a failure of two individually valid transformations.
