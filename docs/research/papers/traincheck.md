# Training with Confidence: Catching Silent Errors in Deep Learning Training with Automated Proactive Checks

- **Authors:** Yuxuan Jiang, Ziming Zhou, Boyu Xu, Beijie Liu, Runhui Xu, Peng Huang.
- **Year / venue:** 2025; OSDI 2025.
- **Verified primary URL:** [Training with Confidence: Catching Silent Errors in Deep Learning Training with Automated Proactive Checks](https://www.usenix.org/conference/osdi25/presentation/jiang).
- **Read depth:** Paper introduction/design and official technical/AE guides.
- **Objective:** Detect silent training errors.
- **Technical method:** Instrument traces; infer relations/preconditions; check execution invariants.
- **Main finding used:** Existing tooling already covers proactive semantic checks.
- **Limitations / Atlas assessment:** Observed invariant coverage is not a proof of all scientific conclusions.
- **Relevance:** Essential comparator or reusable component.
- **Code status:** https://github.com/OrderLab/TrainCheck
- **Open question:** Does contract-conditioned selection add anything beyond composition?
- **Requires further verification:** Reproduce relevant faults and measure current overhead before comparison.

Retrieved in this research session; no independent replication is implied.
