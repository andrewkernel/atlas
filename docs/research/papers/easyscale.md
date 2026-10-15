# EasyScale: Elastic Training with Consistent Accuracy and Improved Utilization on GPUs

- **Authors:** Mingzhen Li, Wencong Xiao, Hailong Yang, Biao Sun, Hanyu Zhao, Shiru Ren, Zhongzhi Luan, Xianyan Jia, Yi Liu, Yong Li, Wei Lin, Depei Qian.
- **Year / venue:** 2023 (preprint 2022); SC 2023.
- **Verified primary URL:** [EasyScale: Elastic Training with Consistent Accuracy and Improved Utilization on GPUs](https://arxiv.org/html/2208.14228v2).
- **Read depth:** Design and consistency/overhead evaluation sections inspected.
- **Objective:** Preserve training consistency under resource elasticity.
- **Technical method:** Virtualized training threads, state management and determinism configurations.
- **Main finding used:** Explicit consistency-preserving resource adaptation already exists.
- **Limitations / Atlas assessment:** Heterogeneous deterministic execution can be costly; guarantees are system-scoped.
- **Relevance:** Strong counterexample to broad fidelity novelty.
- **Code status:** Implementation discussed; repository/build not independently audited.
- **Open question:** How does single-device composition differ from resource elasticity?
- **Requires further verification:** Do not transfer its guarantees to arbitrary Atlas transformations.

Retrieved in this research session; no independent replication is implied.
