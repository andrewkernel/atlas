# Recompute-state and RNG audit controls

Executed original reviewed CPU float64 fixtures with PyTorch2.14.1+cpu on2026-10-15; no external framework code was executed. [Source](../../../benchmarks/state_replay_probes.py), [raw results and source hash](state-replay-probes.json).

## BatchNorm positive and negative controls

The native non-reentrant checkpoint fixture again matches gradients but increments the BatchNorm counter twice and changes evaluation output(max absolute difference.052464). An explicitly scoped recomputation context that restores existing registered buffer values returns the counter to1 and matches gradient/buffer/evaluation observations in this fixture.

This demonstrates that the hazard can be repaired for a known module; it is not evidence that arbitrary mutable regions must always be rejected. [FairScale already contains specialized handling](../competitors/fairscale.md), so the issue itself is established prior art. The MVP may conservatively exclude such regions, but a novelty or coverage comparison must recognize existing repairs.

The control does not cover buffer replacement/aliasing, state saved for backward, Python attributes, caches, custom RNGs, hooks, I/O, nested wrappers, multiple backwards, concurrent forwards or distributed effects. Restoring buffers can itself invalidate autograd when a buffer is a saved input. Do not promote this fixture context into a generic transaction mechanism.

## Partitioning stochastic execution

Both arms start with identical parameters, inputs and CPU seed and consume the same logical samples. In the one-dropout-layer fixture, full-batch and4+4 partitions match outputs, gradients within2.78e-17 and final RNG state. This is a positive observation for one implementation/shape, not a general guarantee.

Adding a second dropout layer changes the interleaving of draws between layers and samples. The full versus4+4 fixture differs in forward output by.289018 and gradients by.472836 **while final RNG states are equal**. Equality of a seed or final generator state therefore cannot certify assignment of randomness to each semantic operation. Shape/call-order or explicitly replayed stochastic inputs must be part of a pathwise comparison.

Failure of this pathwise comparison does not establish a changed dropout distribution or degraded final accuracy. A contract allowing independent identically distributed masks asks a different question and needs distributional assumptions plus outcome evidence. No full-training, GPU or general distributional equivalence result follows from these fixtures.
