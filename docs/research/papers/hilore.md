# HiLoRe

Xinrui Chen, Mengyang Li, Ou Wu and Ji Zhang. **HiLoRe: What to Store, Compress, or Recompute for Efficient GRPO Training.** [arXiv2609.33570v1](https://arxiv.org/abs/2609.33570), submitted 27 September 2026, under review. Selected [HTML methods/appendices](https://arxiv.org/html/2609.33570v1) inspected; [byte identity/read scope](hilore-deeper-read.json). Not executed; the anonymous code page supplied no inspectable content.

Its recovery allocator combines precise storage, compression and recomputation under memory and calibrated gradient-risk constraints. Exact H/R recovery is conditional on matched states, deterministic operators and accumulation order. The risk budget constrains a surrogate, not a certified probability of preserving scientific conclusions. Joint development checks constrain mean accumulated-gradient error.

Table 17 reports relative gradient error 0.0030 with zero risk budget and no low-precision actions; the declared training configuration disables strict determinism. This observation and the conditional proposition have different scopes; the cause of residual error is not independently isolated here. Neither a paper defect nor unconditional bitwise parity follows.

Evaluation primarily uses LoRA/four-rank 48-GB setups, with three-seed downstream comparisons; full-parameter evidence is narrower. It directly weakens broad memory-plus-fidelity novelty. Atlas needs a specific incremental selection result; forcing this GRPO system onto an unrelated CNN would be an unfair comparator.
