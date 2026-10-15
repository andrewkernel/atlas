# Rockmate: an Efficient, Fast, Automatic and Generic Tool for Re-materialization in PyTorch

- **Authors:** Xunyi Zhao, Théotime Le Hellard, Lionel Eyraud-Dubois, Julia Gusak, Olivier Beaumont.
- **Year / venue:** 2023; ICML 2023, PMLR 202.
- **Verified primary URL:** [Rockmate: an Efficient, Fast, Automatic and Generic Tool for Re-materialization in PyTorch](https://proceedings.mlr.press/v202/zhao23b.html).
- **Read depth:** Proceedings abstract, selected graph/alias analysis and AppendixD implementation/numerical notes; pinned repository README. No execution.
- **Objective:** Control PyTorch training activation memory.
- **Technical method:** Analyze blocks and combine block-level and sequence-level scheduling.
- **Main finding used:** Automatically generates recomputation plans for supported structures.
- **Limitations / Atlas assessment:** Current-model coverage and mutable-state assumptions need examination.
- **Relevance:** Closest single-device memory-planning comparator.
- **Code status:** https://github.com/topal-team/rockmate
- **Open question:** Can existing plans plus independent checks match Atlas selection?
- **Requires further verification:** No performance or compatibility replication here.

Retrieved in this research session; no independent replication is implied.

Continuation [paper inspection](https://proceedings.mlr.press/v202/zhao23b/zhao23b.pdf) confirms explicit handling of shared-storage/view/in-place dependencies and saving/restoring random states for deterministic recomputation. Its implementation notes acknowledge finite-precision differences and report float64 equality on tested models; do not promote the paper's broad equivalence language into an arbitrary-program guarantee. These mechanisms also precede Atlas. The current repository includes [HiRemate and OffMate](../competitors/rockmate-family.md), so a comparison restricted to the original2023 algorithm would omit relevant successors.
