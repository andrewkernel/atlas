# ADR 002 — Finite deterministic planning over native mechanisms

**Status:** accepted for MVP.

**Context:** Atlas needs to test constraint-conditioned selection, not invent a general memory optimizer. Established tools already perform sophisticated planning.

**Alternatives:** new solver/compiler; unrestricted LLM patch generation; fixed unconditional optimizations; finite native strategy registry with eligibility checks and profiling.

**Evidence:** [Checkmate](https://arxiv.org/abs/1910.02653), [Rockmate](https://proceedings.mlr.press/v202/zhao23b.html) and [DeepSpeed Autotuning](https://github.com/deepspeedai/DeepSpeed/blob/master/deepspeed/autotuning/README.md) limit broad novelty. Local counterexamples motivate checking eligibility rather than assuming every memory change is safe.

**Decision:** initially enumerate microbatch accumulation and declared-region checkpointing; include identity and explicit no-plan output. Profile bounded trials; verify composed plans. Precision/offload are later evidence-gated extensions.

**Tradeoffs:** simple, inspectable and reproducible, but limited capacity when persistent state dominates. No optimality claim outside the finite tested set.

**Revisit when:** the evaluation identifies missed useful plans and there is evidence a new mechanism/search method improves valid feasible coverage after overhead. Compare against reusing a mature planner before implementing a new one.
