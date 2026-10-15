# ZeRO: Memory Optimizations Toward Training Trillion Parameter Models

- **Authors:** Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, Yuxiong He.
- **Year / venue:** 2019 preprint; 2020 published work; SC 2020; paper URL is preprint.
- **Verified primary URL:** [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054).
- **Read depth:** Abstract; DeepSpeed documentation inspected separately.
- **Objective:** Reduce redundant distributed model state.
- **Technical method:** Partition optimizer state, gradients and parameters across ranks.
- **Main finding used:** State-memory scaling is already a major training-system capability.
- **Limitations / Atlas assessment:** Distributed gains do not imply equivalent capacity improvement with one rank.
- **Relevance:** Diagnose persistent-state versus activation bottlenecks.
- **Code status:** DeepSpeed implementation exists; not run here.
- **Open question:** When is offload the only useful one-GPU option?
- **Requires further verification:** Paper-scale results cannot be extrapolated to 8 GB cards.

Retrieved in this research session; no independent replication is implied.
