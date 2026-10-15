# Capuchin: Tensor-based GPU Memory Management for Deep Learning

- **Authors:** Xuan Peng, Xuanhua Shi, Hulin Dai, Hai Jin, Weiliang Ma, Qian Xiong, Fan Yang, Xuehai Qian.
- **Year / venue:** 2020; ASPLOS 2020.
- **Verified primary URL:** [Capuchin first-party overview](https://www.microsoft.com/en-us/research/publication/capuchin-tensor-based-gpu-memory-management-for-deep-learning/); [publisher DOI](https://doi.org/10.1145/3373376.3378505).
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

Continuation retrieval could not fetch the former author-hosted PDF URL through the direct HTTP audit. The first-party overview remains available through browsing. Its first author is displayed as “Quan Peng”, while the publisher-deposited [Crossref record](https://api.crossref.org/works/10.1145/3373376.3378505) says Xuan Peng; this note follows the publisher record. This metadata discrepancy does not change the technical assessment, and title availability alone would not have caught it.
