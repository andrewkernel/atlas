# HiRemate

Julia Gusak, Xunyi Zhao, Théotime Le Hellard, Zhe Li, Lionel Eyraud-Dubois and Olivier Beaumont. **HiRemate: Hierarchical Approach for Efficient Re-materialization of Neural Networks.** [ICML2025, PMLR267:21418–21443](https://proceedings.mlr.press/v267/gusak25a.html). Proceedings abstract, metadata and repository description inspected; not replicated.

The work recursively partitions computation graphs, solves subgraphs under multiple budgets and merges schedules for single-GPU activation rematerialization. It broadens the original Rockmate structure assumptions.

**Atlas assessment:** this is a relevant successor, not a peripheral omission. Benchmark the currently appropriate family member on supported workloads instead of selecting an older restricted algorithm to make Atlas appear more general. No current-PyTorch compatibility, numerical coverage or performance advantage was independently established here. The repository README has example-name inconsistencies; verify actual exports and pinned code before copying a usage fragment.
