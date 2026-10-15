# FairScale checkpoint-state handling

Inspected official repository at commit `146f160241651e1211c4247979f159a4ef43b54a` on2026-10-15. This is a targeted source read, not execution or a compatibility certification.

- [checkpoint_utils.py](https://github.com/facebookresearch/fairscale/blob/146f160241651e1211c4247979f159a4ef43b54a/fairscale/nn/checkpoint/checkpoint_utils.py) defines `patch_batchnorm`: hooks suppress running-stat tracking in the no-grad forward.
- [checkpoint_activations.py](https://github.com/facebookresearch/fairscale/blob/146f160241651e1211c4247979f159a4ef43b54a/fairscale/nn/checkpoint/checkpoint_activations.py) invokes that patch in its wrapper and exposes recomputation context for guarding side effects. Its custom autograd path has a no-grad forward and a grad-enabled recomputation.

The native non-reentrant PyTorch probe in this repository uses a different execution path. A fix based solely on `torch.is_grad_enabled()` cannot be transferred between these paths without checking its assumptions. Treat a patched FairScale wrapper and the unpatched native fixture as different implementations. This known hazard and specialized mitigation precede Atlas; reporting them as newly discovered would be inaccurate.

An [original buffer-restoration control](../experiments/state-replay-probes.md) repairs the narrow local fixture. It does not execute or replicate FairScale and is not a proposed generic production wrapper. A fair comparison must include suitable existing state-aware handling where compatible.
