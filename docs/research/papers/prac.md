# PRAC

Yanyi Li, Yimu Zhang and Cong Fang. **PRAC: Principal-Random Subspace for LLM Activation Compression and Memory-Efficient Training.** [arXiv2602.23111v1](https://arxiv.org/abs/2602.23111v1),26 February2026. Abstract/metadata only; theorem assumptions not audited and results not replicated.

The abstract describes a principal subspace plus a scaled random orthogonal-tail approximation, with conditional unbiased-gradient/minimum-variance claims and memory/quality experiments.

**Atlas assessment:** unbiasedness is a statement about an expectation under conditions, not equality of realized gradients or trajectories. Approximation variance and downstream outcomes still matter. This is relevant prior art for any future activation-compression extension; it does not justify enabling compression under a pathwise strict contract. Read the actual estimator and proof before borrowing its guarantees; do not infer universal unbiasedness from the abstract.
