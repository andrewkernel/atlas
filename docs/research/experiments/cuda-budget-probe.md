# Allocator-budget sensitivity — explicitly emulated

This experiment uses a450 MiB **PyTorch allocator limit** on the physical12 GB RTX3060, not a450 MiB GPU or a physical8 GB comparison. It is a resource-control experiment for the original synthetic MLP, not a real research workload or primary policy evaluation.

[Source](../../../benchmarks/cuda_budget_probe.py), [raw structured evidence](cuda-budget-probe.json), and [unconstrained fixture](cuda-memory-probe.md). The imported fixture's source hash and the locally generated reference tensor file's hash are recorded independently. Four fresh owned child processes have180-second timeouts. No device settings are changed. The cap uses the [native memory-fraction API](https://docs.pytorch.org/docs/2.14/generated/torch.cuda.memory.set_per_process_memory_fraction.html).

| Plan | Result under450 MiB allocator cap | Completed optimizer updates |
|---|---|---|
|Microbatch16, checkpoint off |Allocator OOM |0 |
|Microbatch16, checkpoint on |Allocator OOM |1 |
|Microbatch4, checkpoint off |Allocator OOM |1 |
|Microbatch4, checkpoint on |Completed bounded fixture |16 |

The two one-update-then-OOM results reinforce why a successful first optimizer update is insufficient for measured-window feasibility. The cap also involves allocator reservations/fragmentation; comparing a single allocated-byte estimate to a budget does not fully predict a request's success. There was no global device exhaustion: the driver reported about10 GiB free in these OOM excerpts.

## Driver-diagnostic quality audit

The first exploratory run's OOM messages printed impossible per-process memory of17,179,869,184 GiB. That value is consistent with an unsigned sentinel being formatted as bytes, but the internal cause was not established. We do not interpret it as actual process memory or diagnose a driver bug from the string alone. The published run repeats the case with a completed-update counter, retains only the non-process message excerpt, and records whether the impossible diagnostic recurs. All three OOM trials flag it.

This is direct evidence that an OOM error string is not authoritative device telemetry on this tested Windows configuration. Use named allocator queries for their actual scope and typed supported telemetry fields for other metrics; represent unsupported/sentinel measurements as null with a reason. Do not infer host/GPU process use by parsing that diagnostic.

## Limits

The count wrapper records an optimizer step only after the reviewed AdamW step returns. It adds no model operation and is specific to this fixture; it is not a generic event observer. A completion covers16 repeated-input updates, not a training horizon or arbitrary shape. Only the combined plan has completed-state observations under this cap. Failed plans retain OOM rather than being dropped from denominators. No power, energy, smaller-card performance, total-GPU isolation or scientific-outcome result follows.
