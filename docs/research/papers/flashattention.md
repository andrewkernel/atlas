# FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness

- **Authors:** Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré.
- **Year / venue:** 2022; NeurIPS 2022.
- **Verified primary URL:** [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135).
- **Read depth:** Abstract and native SDPA documentation.
- **Objective:** Reduce attention IO and memory.
- **Technical method:** Tiled exact attention avoids materializing the full score matrix.
- **Main finding used:** Exact intended mathematics can use a different execution schedule.
- **Limitations / Atlas assessment:** Floating-point values and backend eligibility still differ.
- **Relevance:** Distinguish mathematical equivalence from numerical parity.
- **Code status:** Implementations exist; native SDPA inspected, external build not run.
- **Open question:** Which backend differences affect a declared experiment?
- **Requires further verification:** No speedup or fit claim tested here.

Retrieved in this research session; no independent replication is implied.
