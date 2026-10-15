# OffMate

Xunyi Zhao, Lionel Eyraud-Dubois, Théotime Le Hellard, Julia Gusak and Olivier Beaumont. **OFFMATE: full fine-tuning of LLMs on a single GPU by re-materialization and offloading.**2024 preprint, HAL04660745. [Author-team report](https://radar.inria.fr/report/2024/topal/index.html), section8.12, and [pinned repository description](https://github.com/topal-team/rockmate/blob/c2f678455393c5d872a1adb099949fe25f682bd0/README.md) inspected. The HAL landing page returned an anti-bot page; the paper itself was not read. Do not imply a full-paper evaluation.

The team describes integrated rematerialization, weight/activation offload and CPU optimization for single-GPU full fine-tuning. The repository exposes an Offmate specialization that also handles optimizer stepping.

**Atlas assessment:** automatic single-GPU memory reduction with offloading is already a direct engineering neighbor. Any later Atlas offload extension must compare semantics, host RAM, transfer cost and optimizer integration against this capability rather than claiming novelty from local hardware alone. Team-reported numerical preservation and resource gains are not independently validated here; no consumer-card capacity guarantee follows.
