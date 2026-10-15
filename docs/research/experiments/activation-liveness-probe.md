# Verification-pattern activation liveness control

This original experiment tests an execution pattern observed during [selected TrainParity source inspection](../competitors/trainparity.md). **No TrainParity code was imported or run**, so these are not its measured performance or global capability results. [Source](../../../benchmarks/activation_liveness_probe.py), [raw evidence](activation-liveness-probe.json).

The inspected worker evaluates a list of microbatch loss terms before looping over backward calls. A memory-conscious training executor normally can run backward after each microbatch, retaining the accumulated gradients while freeing saved activations. [PyTorch autograd documentation](https://docs.pytorch.org/docs/2.14/notes/autograd.html) explains saved-tensor lifetime; this is an established mechanism, not an Atlas invention.

## Controlled analogue

Use the same original residual-MLP geometry/initialization as the earlier CUDA fixture, logical batch 16 split into microbatches4, AdamW, no dropout or checkpointing. Eight repeated-input updates per process. Compare:

- **Streamed:** forward, normalized loss and backward for one microbatch before the next forward.
- **Deferred:** build all four normalized losses and their graphs, then run each backward before one optimizer step.

Both preserve the intended separable-loss gradient under the fixture's assumptions. First-update parameters, gradients and Adam moments are checked against the locally generated full-batch reference whose bytes are hashed. All completed arms have the same first-update discrepancies as each other relative to that reference: parameter max6.24e-8, gradient max4.73e-11; moments finite and correspondingly close. This compares finite observations, not all-state/all-input equivalence.

|Execution pattern |Allocator cap |Outcome |Completed updates |Allocated peak(MiB) |
|---|---|---|---|---|
|Streamed |None |Completed |8 |506.51 |
|Deferred |None |Completed |8 |922.59 |
|Streamed |600 MiB |Completed |8 |506.51 |
|Deferred |600 MiB |Allocator OOM |0 |586.31 before failure |

The cap is emulation on the same physical12 GB RTX 3060. The uncapped memory difference is about 416 MiB. Minor source-variable lifetimes differ from the earlier profile script, so compare patterns within this experiment rather than treating every byte as interchangeable with an older measurement. No timing comparison, full-training outcome or physical smaller-card result is claimed.

## Implication for the baseline and integration

A verifier can preserve observed update numerics yet retain more activation graphs than the target execution. Its resource behavior must therefore be measured separately; a semantically comparable checker execution is not automatically the same memory plan. A reference that cannot fit still requires a trusted external/CPU reference or an explicit unverified scope.

Give the composed baseline a fair memory-conscious adaptation when valid, with the same observations and declared loss denominator. Do not deliberately retain this pattern to manufacture Atlas superiority. The [integration note](../checker-integration-options.md) describes safe scope choices. Current source capability and this original analogue motivate an experiment; they do not establish an unfixable limitation of the existing tool or new algorithmic novelty.
