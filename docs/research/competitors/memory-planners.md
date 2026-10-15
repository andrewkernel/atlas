# Memory planners — comparison duties

Primary capability evidence is linked in the [matrix](../03-competitive-landscape.md); paper-specific read depth is in the [notes](../papers/README.md).

## Closest applicable comparators

Native PyTorch checkpointing/selective recomputation and Rockmate are the closest execution-level comparators for the first one-device study. DeepSpeed Autotuning is a strong profiler/search comparator where its framework and hardware requirements genuinely fit. Checkmate, DTR and Capuchin are major intellectual prior art even if their historical runtime integration cannot be installed fairly in the chosen environment.

Do not recreate a rematerialization solver simply to make Atlas look novel. Start with native flags/declared regions, then test whether constraint evidence changes decisions or exposes useful failure cases. Native memory-budget knobs can be internal/version-specific; record their exact API and supported behavior before using them as a baseline.

## Required parity for a resource comparison

Same initial state, samples/masks, objective, logical update count and allowed arithmetic contract. Record actual backend selection, graph capture/compiler mode, warmup, first-update peak, adaptation time and candidate budget. Framework migration that changes optimizer or data semantics is a separately modified experiment, not a neutral implementation swap.

An unsupported installation is an availability result, not proof of scientific inferiority. Keep its error/version record and compare a compatible subset or leave performance unknown. World-size-one state sharding is not equivalent to obtaining the memory of multiple GPUs.
