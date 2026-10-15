# ActNN: Reducing Training Memory Footprint via 2-Bit Activation Compressed Training

- **Authors:** Jianfei Chen, Lianmin Zheng, Zhewei Yao, Dequan Wang, Ion Stoica, Michael W. Mahoney, Joseph E. Gonzalez.
- **Year / venue:** 2021; ICML 2021.
- **Verified primary URL:** [ActNN: Reducing Training Memory Footprint via 2-Bit Activation Compressed Training](https://arxiv.org/abs/2104.14129).
- **Read depth:** Abstract only; no theorem audit.
- **Objective:** Compress saved training activations.
- **Technical method:** Stochastic quantization with gradient-variance/convergence analysis and adaptive precision.
- **Main finding used:** Approximate activations can preserve measured quality on evaluated tasks.
- **Limitations / Atlas assessment:** Gradients are altered; empirical results do not authorize strict equivalence.
- **Relevance:** Approximation needs an allowed-method-change policy.
- **Code status:** Paper describes framework; current repository/build not audited.
- **Open question:** Can low-cost checks predict harmful approximation?
- **Requires further verification:** Task generality and present compatibility unverified here.

Retrieved in this research session; no independent replication is implied.
