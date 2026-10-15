# Rockmate: an Efficient, Fast, Automatic and Generic Tool for Re-materialization in PyTorch

- **Authors:** Xunyi Zhao, Théotime Le Hellard, Lionel Eyraud-Dubois, Julia Gusak, Olivier Beaumont.
- **Year / venue:** 2023; ICML 2023, PMLR 202.
- **Verified primary URL:** [Rockmate: an Efficient, Fast, Automatic and Generic Tool for Re-materialization in PyTorch](https://proceedings.mlr.press/v202/zhao23b.html).
- **Read depth:** Proceedings abstract; repository metadata.
- **Objective:** Control PyTorch training activation memory.
- **Technical method:** Analyze blocks and combine block-level and sequence-level scheduling.
- **Main finding used:** Automatically generates recomputation plans for supported structures.
- **Limitations / Atlas assessment:** Current-model coverage and mutable-state assumptions need examination.
- **Relevance:** Closest single-device memory-planning comparator.
- **Code status:** https://github.com/topal-team/rockmate
- **Open question:** Can existing plans plus independent checks match Atlas selection?
- **Requires further verification:** No performance or compatibility replication here.

Retrieved in this research session; no independent replication is implied.
