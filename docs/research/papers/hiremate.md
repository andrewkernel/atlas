# HiRemate

Julia Gusak, Xunyi Zhao, Théotime Le Hellard, Zhe Li, Lionel Eyraud-Dubois and Olivier Beaumont. **HiRemate: Hierarchical Approach for Efficient Re-materialization of Neural Networks.** [ICML2025, PMLR267:21418–21443](https://proceedings.mlr.press/v267/gusak25a.html).

**Read depth:** initial abstract review, then selected PDF methods/evaluation/appendices; equations and tables visually checked. [PDF identity and page scope](hiremate-deeper-read.json).

Hierarchical partitioning, solver options and shared-input liveness corrections are substantive prior art. Appendix C.1.3 excludes persistent model-input bytes from managed accounting. C.2.2 discusses multiple forwards before backward.

The performance study uses five warmups/ten measured iterations on an older Torch/CUDA/Gurobi stack. Table 4 adds forward/backward sanity checks; its largest Transformer observed-runtime cell is blank, verified visually. These are not multi-seed full-training equivalence evidence.

**Atlas assessment:** compare the appropriate family member and memory definition. Our [adapted family controls](../experiments/rockmate-compatibility.md) do not replicate HiRemate's hierarchical study or historical environment. Reusable-module behavior needs its own control; serial numerical parity is insufficient. Broader supported graphs and corrected liveness accounting weaken generic planning novelty, while leaving the primary contract-conditioned selection question open.
