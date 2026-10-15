# Optimization strategy analysis

All memory/runtime expectations here are conditional engineering predictions, not measured savings. “Initial” means support after the profiling prototype; no transformation engine is implemented in this research repository. Hardware-specific behavior must be detected at runtime.

## Resource behavior and implementation boundary

| Strategy | What changes | Memory / runtime expectation | Hardware | Initial support |
|---|---|---|---|---|
| Gradient accumulation | Split one logical update into sequential forward/backward calls | Lower activation peak; persistent state stays; more launches and possibly slower utilization | CPU or single GPU | Yes, registered separable losses and explicit update events |
| Smaller microbatch alone | Reduce samples per step | Lower activation peak; may change optimization noise and total update count | CPU/GPU | Only paired with logical-batch preservation; otherwise modified methodology |
| Activation checkpointing | Recompute selected activations in backward | Lower saved activation memory; added compute; inputs/workspaces remain | CPU/GPU | Yes, explicitly eligible pure regions |
| BF16 autocast | Selected operations use BF16 | Often lowers activation storage; FP32 state may remain; acceleration depends on kernels | Capability query and a real operation test, CPU support varies | Optional empirical extension after strict prototype |
| FP16 autocast | Selected operations use narrower-range FP16 | Similar possible savings; scale/update overhead; overflow risks | Compatible GPU/kernel path | Defer until scaling-state checks exist |
| Gradient/loss scaling | Scale loss before backward; unscale, overflow-check, possibly skip update | Usually no material independent VRAM gain; helps low-precision arithmetic | AMP-supported device/runtime | Only with later FP16 strategy, not a standalone saving |
| Optimizer offload | Transfer state and possibly update computation to CPU | Less GPU state; more host RAM and transfer time; CPU bottleneck | GPU + adequate host RAM/interconnect | Defer; first state-bottleneck alternative to investigate |
| Quantization / QLoRA | Quantized weights/activations or low-rank adaptation | Large targeted memory reductions; dequantization and training-method changes | Backend/kernel dependent | Exclude from strict-preservation set; separately declared research mode only |
| Fused operators | Combine operations or optimizer implementations | Fewer intermediates/launches, but not guaranteed lower peak | Backend/compiler/toolchain dependent | Defer; native use may be preserved in a supplied baseline |
| Alternative attention | Replace attention execution backend | Avoid materialized attention matrix; often faster; eligibility matters | Native SDPA compatibility or separate library requirements | Later native SDPA backend comparison; no new kernel |
| Data loading changes | Workers, pinning, prefetch or ordering | Host-memory increase possible; faster input delivery; no direct model-state reduction | CPU, RAM, storage | Only preserving ID/content trajectory with controlled transforms; not initial optimization search |
| Memory-efficient optimizers | Factor/quantize optimizer state or choose a different algorithm | Less state; different update rule and possible convergence effects | CPU/GPU/backend dependent | Exclude optimizer substitution from strict mode |
| Model parallelism | Partition model across devices | Per-device state/activation changes, communication and bubbles | Multiple devices and supported links | Out of MVP; cannot assume one-laptop availability |
| Gradient compression | Approximate communicated gradients | Mostly communication savings; residual buffers can add memory | Distributed setup | Out of MVP; not relevant to primary single-device bottleneck |

## Numerical, mathematical and methodological obligations

| Strategy | Conditions and consequences | When conclusions can change | Required evidence |
|---|---|---|---|
| Accumulation / microbatching | Correct global denominator, batch separability, fixed parameters and update clocks; reduction order changes | BatchNorm, dropout path, cross-example losses, tail errors, per-microstep weight decay/clipping | Sample/mask accounting; loss numerator/count; gradients, buffers, optimizer/scheduler state; repeated full runs when required |
| Checkpointing | Recomputed functions must preserve relevant state and RNG relation; repeated side effects are unsafe | Running statistics, cache writes, stochastic recomputation, custom autograd state | Region declaration; state/RNG observations; gradient and buffer checks; positive/negative controls |
| BF16 | Same intended operations but reduced significand precision; wide exponent range is not numerical equivalence | Small effects, unstable optimization, sensitive subgroup/ranking comparisons | Dtype trace, nonfinite monitoring, paired updates and full outcome study; empirical classification |
| FP16 / scaling | Range/rounding changes; unscale and clip at logical-update boundary; skipped updates alter clocks | Overflow/underflow, scaler changes within a window, scheduler advancement on skipped steps | Log scale, overflow, attempted/successful update counts and scheduler/EMA events; full-run checks |
| Offload | Same formulas require preserved dtype, order and state; CPU optimizer implementation may differ | Changed rounding, asynchronous races, incomplete transfer/state restoration | Compare optimizer state and update deltas; profile host peak, copy time, memory failures |
| Quantization | Rounding/compression and sometimes trainable subspace change | Any question about full-precision/full-parameter optimization | Structural change record, explicit new contract, downstream quality study; never equivalence by default |
| Fusion | Algebraic intent can match while reduction/FMA/order changes | Numerically sensitive trajectories or fused optimizer semantics | Keyed operator/update comparisons, dtype/backend records, regression cases, repeated outcomes |
| Attention | Exact dense-attention mathematics requires same scale/mask/causality/dropout; backend numerics differ | Wrong mask convention, dropout during evaluation, different causal alignment, precision drift | Adversarial shape/mask tests; forward/backward comparisons; native backend identity and eligibility |
| Data loading | Same stable sample IDs are necessary but not sufficient; transform RNG/content matters | Worker seeding, dropped/repeated samples, order-dependent curriculum, changed augmentation | IDs and content/augmentation provenance where possible; worker seeds; epoch coverage and ordering checks |
| Optimizer changes | Adafactor, quantized state or a low-memory algorithm need not implement AdamW updates | Optimizer comparison studies become confounded | Mark modified methodology; only evaluate as a new optimizer experiment |
| Parallelism | Synchronization, global batch, RNG partition, optimizer state and arithmetic order must be specified | Stale/asynchronous updates, synchronization or reduction differences | Distributed trace/state checks and hardware study, beyond MVP |
| Compression | Usually approximate updates; error-feedback state is part of algorithm | Convergence, rare events, ranking, bias | Algorithm-specific analysis and full-training comparisons; no strict certification |

## Source anchors and cautions

- [Native AMP guidance](https://docs.pytorch.org/docs/2.14/notes/amp_examples.html): scaling, clipping and update order.
- [Checkpoint API](https://docs.pytorch.org/docs/2.14/checkpoint.html): recomputation and RNG caveats.
- [SDPA API](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html): masks, dropout and backend-dependent values.
- [ZeRO-Offload](https://www.deepspeed.ai/tutorials/zero-offload/): host optimizer path. Its example is not a consumer-hardware guarantee.
- [Adafactor](https://proceedings.mlr.press/v80/shazeer18a.html), [ActNN](https://arxiv.org/abs/2104.14129), [QLoRA](https://arxiv.org/abs/2305.14314), and [PowerSGD](https://papers.neurips.cc/paper_files/paper/2019/hash/d9fbed9da256e344c1fa46bb46c34c5f-Abstract.html): different approximation/algorithm families, not interchangeable safe switches.

Autocast does not necessarily halve total training memory: ordinary PyTorch autocast can keep FP32 model/optimizer state. Likewise, sharding has no state-partition benefit with a single rank. These distinctions must appear in estimates and reports.

`zero_grad(set_to_none=True)` can change how an optimizer handles unused parameters relative to zero gradients; it is not an unconditional harmless memory tweak. Custom loss functions, dynamic shape decisions and custom operators trigger conservative abstention unless an adapter supplies the necessary semantics.
