# Native selective checkpointing strengthens the baseline

Frozen at `d2e4343` before execution, the [protocol](native-selective-protocol.md) runs the **unchanged original synthetic CUDA fixture** with native identity, ordinary checkpointing and a fixed selective policy. [Source](../../../benchmarks/native_selective_control.py) and [all twelve raw trials](native-selective-control.json) retain identities, contemporary controls, complete first/later measurement windows and numerical observations. All complete sixteen logical AdamW updates in separate owned workers. No application or novel optimizer is implemented.

| Native configuration | Microbatch / logical 16 | Later allocated peak MiB, both repetitions | Mean of two process-median update times, ms |
|---|---:|---:|---:|
|Identity|16|972.582|74.793|
|Identity|4|508.503|83.898|
|Ordinary non-reentrant checkpoint|16|452.442|87.517|
|Ordinary non-reentrant checkpoint|4|378.446|97.686|
|Selective matmul-save checkpoint|16|740.360|76.926|
|Selective matmul-save checkpoint|4|450.446|86.154|

Selective checkpointing saves the outputs of three declared matrix-multiply operations and prefers recomputation otherwise. The first logical update records 16 actual `aten.addmm.default` saved-op callbacks at full batch or 64 with microbatch 4; mm/bmm are not observed here. These establish the policy path, not a traced GPU-kernel count. Cache mutation checks stay enabled. The observer collects these counts only in the first update, outside the later timed window.

At full batch, identity/plain/selective first-update gradient, parameter and separate optimizer-state errors are all zero against the owned local reference. All microbatch-4 configurations reproduce the earlier rounding observations: parameter maximum absolute error 6.24e-8, gradient relative L2 about 6.93e-7, second-moment relative L2 about 1.14e-6; all recorded values are finite. This supports the stated finite first-update relation, **not final-state or whole-training equivalence**. Later updates are timed/completed, not independently state-compared by this observer.

Existing native mechanisms offer distinct memory/time points without an Atlas planner. Here selective full-batch checkpointing lies between identity and ordinary checkpointing: less memory than identity, less measured time than ordinary checkpointing, and more memory than ordinary checkpointing. Include such a supported configuration in the development baseline rather than comparing Atlas only to a deliberately restricted ordinary wrapper. A primary registry must remain finite and share the same search budget/information across policies.

These are descriptive observations from two process repetitions per configuration and ten dependent timing steps per process on a shared RTX 3060. No confidence interval or robust dominance assertion is justified; small differences are not established population speedups. Summed owned-child wall time is 59.62 seconds, including startup/checks/warmups, not just timed updates or measured Atlas search cost. This pure fixed-input fixture has no dropout, mutable BatchNorm state, attention, data pipeline, useful OOM rescue, energy study or scientific-outcome analysis. Contemporary controls prevent silently using historical timing as the comparator.

The [official PyTorch API](https://docs.pytorch.org/docs/2.14/checkpoint.html) supplies the selective context and per-operation policy; this study contributes scoped local execution evidence only. No torch.compile or automatic memory-budget API is exercised. The original historical source/results remain unchanged.
