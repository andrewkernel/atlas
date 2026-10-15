# TorchProbe

Qidong Su, Chuqin Geng, Gennady Pekhimenko and Xujie Si. **TorchProbe: Fuzzing Dynamic Deep Learning Compilers.** [arXiv2310.20078v1](https://arxiv.org/abs/2310.20078v1),30 October2023. Abstract and metadata inspected; no system execution.

The work uses semantics-preserving program transformations to exercise dynamic Python features and differential behavior in PyTorch/Triton compilation.

**Atlas assessment:** transformed-program testing is established prior art, so a benchmark of transformations needs a precise new scope and empirical value. Atlas studies training-plan semantics and resource decisions rather than compiler robustness alone. That distinction must be demonstrated; it does not create a new testing paradigm. The repository/package with a similar lowercase name found in search is an unrelated inspection utility and was not treated as this paper's implementation.
