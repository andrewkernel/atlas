# Measured tradeoffs, planning cost and break-even conditions

This analysis derives only from the [synthetic CUDA fixture](experiments/cuda-memory-probe.md), not the proposed real-workload evaluation. All four measured configurations are non-dominated on allocated memory versus update time: each memory reduction trades against higher median update cost. There is no single universally best plan.

## Observed finite set

Using the average of the two fresh-process medians for each plan:

|Plan |Allocated peak(MiB) |Average process-median update(ms) |
|---|---|---|
|Full16, no checkpoint |972.58 |74.53 |
|Micro4, no checkpoint |508.50 |83.45 |
|Full16, checkpoint |452.44 |87.74 |
|Micro4, checkpoint |378.45 |97.40 |

For a hypothetical **observed allocated-byte** budget of1000 MiB, the full plan is fastest;600 MiB selects micro4 without checkpoint;500 MiB selects full16 with checkpoint;400 MiB selects the combined plan. These are arithmetic choices from the measured set, not fresh capped executions or hard capacity guarantees. Reserved memory, fragmentation, native non-allocator use and later shapes can change actual success. The separate450 MiB [allocator-cap experiment](experiments/cuda-budget-probe.md) supplies an actual bounded control at one cap only.

The fixture reference already fits the physical device. Thus memory reduction here is not evidence of expanded experimental access. Every measured memory-saving plan is slower. Calling the result a general compute-saving optimization would obscure the actual objective: meeting a resource constraint with an acceptable declared method/cost.

## Costs outside timed updates

Across the published eight child trials, summed owned-child wall time is39.22 seconds. The80 timed update samples total6.86 seconds; first updates total1.43 seconds and model/input preparation1.54 seconds. The remaining wall time includes imports, warmups, CPU state snapshots/comparison, serialization, process startup and other overhead. These windows are nested/different scopes, not independent additive estimates of a production pipeline.

This39.22-second sum is not a measured Atlas planning cost: it includes two repetitions, research diagnostics and no selection policy. It does demonstrate why reporting only the kernel/training window can omit material workflow cost. Dependency installation/downloads and the author's investigation time are outside those child windows. Include whichever startup costs the future deployment scenario actually incurs, and report cold versus repeated use separately.

## Break-even reasoning

For K otherwise comparable repeated executions, baseline cost T0 per run, candidate cost T1, and extra search/verification/inference cost S+V+I, net time saving requires `K*(T0-T1) > S+V+I`. If T1≥T0 and overhead is nonnegative, no positive K produces time savings. Memory feasibility can still be the goal; name that benefit accurately.

Evidence reuse is limited by its contract/code/data/environment scope. Do not amortize a verification result across changed workloads, scientific margins or stochastic relations without justified reuse. When the baseline cannot run, there may be no observed T0; report feasible completion and cost instead of inventing a speedup against an unavailable run. External-reference and adaptation costs remain visible.

## Implication for the optional LLM

A fixed eight-plan registry bounds what ranking can discover. If every admissible candidate is exhaustively measured within the common wall/trial budget, reordering cannot improve the best observed feasible candidate; it can affect time to first useful plan, early stopping and explanation. An LLM must therefore demonstrate a benefit under those actual budget conditions after inference/unload/review costs. More fluent explanations or a newer model family do not by themselves justify a planning-performance claim.
