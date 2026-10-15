# Capuchin: Tensor-based GPU Memory Management for Deep Learning

- **Authors:** Xuan Peng, Xuanhua Shi, Hulin Dai, Hai Jin, Weiliang Ma, Qian Xiong, Fan Yang, Xuehai Qian.
- **Year / venue:** 2020; ASPLOS 2020.
- **Verified primary URL:** [Capuchin: Tensor-based GPU Memory Management for Deep Learning](https://alchem.usc.edu/portal/static/download/capuchin.pdf).
- **Read depth:** Author-hosted paper metadata and first-party overview.
- **Objective:** Reduce training memory pressure.
- **Technical method:** Runtime tensor access patterns guide eviction, prefetch and recomputation.
- **Main finding used:** Joint runtime memory management predates Atlas.
- **Limitations / Atlas assessment:** Published TensorFlow prototype; hardware/interconnect assumptions matter.
- **Relevance:** Offload/recompute planning is not novel by itself.
- **Code status:** Not independently confirmed from inspected source.
- **Open question:** How costly is CPU capacity on commodity systems?
- **Requires further verification:** Current availability and implementation details remain unchecked.

Retrieved in this research session; no independent replication is implied.
