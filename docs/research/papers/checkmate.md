# Checkmate: Breaking the Memory Wall with Optimal Tensor Rematerialization

- **Authors:** Paras Jain, Ajay Jain, Aniruddha Nrusimha, Amir Gholami, Pieter Abbeel, Kurt Keutzer, Ion Stoica, Joseph E. Gonzalez.
- **Year / venue:** 2020 (preprint 2019); MLSys 2020.
- **Verified primary URL:** [Checkmate: Breaking the Memory Wall with Optimal Tensor Rematerialization](https://proceedings.mlsys.org/paper_files/paper/2020/hash/0b816ae8f06f8dd3543dc3d9ef196cab-Abstract.html).
- **Read depth:** Paper formulation and evaluation setup inspected; not replicated.
- **Objective:** Fit training under memory budgets.
- **Technical method:** Profile-guided rematerialization, MILP and approximation; published TensorFlow/Keras execution pipeline.
- **Main finding used:** Automatic memory/time tradeoff planning is established.
- **Limitations / Atlas assessment:** Graph/model/runtime assumptions and solver costs restrict direct reuse.
- **Relevance:** Directly defeats broad automatic-memory-planner novelty.
- **Code status:** https://github.com/parasj/checkmate
- **Open question:** How much incremental value can contract-conditioned selection add?
- **Requires further verification:** Modern compatibility and side-effect handling need code-level testing.

Retrieved in this research session; no independent replication is implied.
