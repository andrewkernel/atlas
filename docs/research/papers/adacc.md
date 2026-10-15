# Adacc

Ping Chen et al. **Adacc: An Adaptive Framework Unifying Compression and Activation Recomputation for LLM Training.** [arXiv2508.00806v2](https://arxiv.org/abs/2508.00806v2),8 August2025; v1 has a different title. Abstract and metadata inspected; no implementation or theorem audit.

The authors describe tensor-level retention/compression/recomputation, outlier-aware activation compression, MILP scheduling and adaptive policy evolution, with model-accuracy comparisons. These are paper-reported capabilities, not replicated measurements.

**Atlas assessment:** mixed compression/recomputation search is already established prior art. Atlas's strict initial registry excludes compression, so this is a conceptual and extension comparator rather than a mandated drop-in baseline on every workload. Future approximate mode must compare its actual mechanisms and fidelity protocol, not merely contrast names or hardware tiers. An abstract-level review cannot establish that Adacc lacks particular safety checks.
