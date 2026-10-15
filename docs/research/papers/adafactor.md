# Adafactor: Adaptive Learning Rates with Sublinear Memory Cost

- **Authors:** Noam Shazeer, Mitchell Stern.
- **Year / venue:** 2018; ICML 2018, PMLR 80.
- **Verified primary URL:** [Adafactor: Adaptive Learning Rates with Sublinear Memory Cost](https://proceedings.mlr.press/v80/shazeer18a.html).
- **Read depth:** Proceedings abstract and introduction.
- **Objective:** Reduce adaptive optimizer state.
- **Technical method:** Factor second-moment estimates and adjust update behavior.
- **Main finding used:** Low auxiliary memory is possible with an altered optimizer.
- **Limitations / Atlas assessment:** Comparable task results do not make it Adam-equivalent.
- **Relevance:** Do not silently substitute optimizers in strict mode.
- **Code status:** Not independently audited in this session.
- **Open question:** When is methodology redesign preferable to reproduction?
- **Requires further verification:** Atlas must not inherit task-quality claims without evaluation.

Retrieved in this research session; no independent replication is implied.
