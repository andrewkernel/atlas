# Planner preparation is part of the state contract

On the completed pure fixture, constructing the adapted Rotor plan changes the default CUDA RNG state. Recorded model parameters, gradients, training mode, Python/NumPy/Torch CPU RNG and the empty buffer set remain unchanged. A scoped Torch RNG context restores the observed CUDA state. The eight-update numerical controls still have zero measured errors.

The [protocol](planner-state-control-protocol.md) and [original observer](../../../benchmarks/rockmate_state_control.py) were frozen at `e6e491b` before execution. [Raw state hashes/results](rockmate-state-control.json) bind the same reviewed source, environment and compatibility fixture. States are sampled before construction, immediately after it inside the guard, and after guard exit. No raw RNG state, model values or generated random data are exported.

|Observed dimension |Changed during construction |Matches original after Torch RNG guard |
|---|---|---|
|Torch CUDA global RNG, device 0 |Yes |Yes |
|Torch CPU global RNG |No |Yes |
|Python global RNG |No |Yes |
|NumPy global RNG |No |Yes |
|Named parameter values |No |Yes |
|Named gradients, including None |No |Yes |
|Training mode |No |Yes |
|Named buffers |None exist in this model |Empty set remains empty |

The guard is the existing [`torch.random.fork_rng`](https://docs.pytorch.org/docs/2.14/random.html) API with device 0 specified; it preserves Torch CPU and specified-device global RNG. It does not restore arbitrary Python/NumPy/custom-generator or model state. Those other dimensions simply did not change in this execution. This is not a new restoration algorithm or generic transaction wrapper.

The fixture fixes its inputs/targets and has no stochastic layers, so changed constructor RNG cannot affect its recorded training outputs. It would be incorrect to infer universal profiling noninterference from those earlier zero errors. This observation establishes a change in this scoped initial-state dimension, not distributional bias, an invalid scientific conclusion, a new upstream defect, or an emergent failure of two individually valid transformations.

For Atlas's proposed execution adapter, capture a declared scientific start state separately from planning/profiling state. After preparation, verify or replay the initialization, buffers, update state and required RNG relation before the first scientific update. A fresh process alone does not establish that relation if profiling advances state inside that process. A pure-region numerical probe cannot replace a stochastic-assignment check. Opaque state or custom generators remain unsupported unless the adapter defines and validates them; failed replay must abstain.
