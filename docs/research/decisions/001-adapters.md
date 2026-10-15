# ADR 001 — Require registered workload adapters

**Status:** accepted for MVP.

**Context:** arbitrary PyTorch programs hide loss accounting, side effects, dynamic shapes and update clocks. Reliable optimization requires knowing which behavior a researcher intends to preserve.

**Alternatives:** arbitrary AST/LLM source rewriting; graph-only FX transformation; manual configuration without a contract; explicit trusted adapter.

**Evidence:** [FX limitations](https://docs.pytorch.org/docs/2.14/fx.html) include dynamic Python control flow. Existing [Deep500](https://github.com/deep500/deep500) and [TrainParity](https://github.com/intelland/TrainParity) demonstrate the usefulness of explicit comparison interfaces. Our CPU probes expose semantics that superficial syntax inspection misses.

**Decision:** a registered adapter declares data/loss/update semantics and checkpoint regions; a standard executor owns the first supported loop. AST findings remain advisory; importing/tracing is execution.

**Tradeoffs:** less automatic adoption and more integration effort; clearer scope and testability. An adapter's declaration is trusted, not a proof of arbitrary code behavior.

**Revisit when:** held-out adoption studies show adapter work dominates value, and a narrower automatic extraction method has measured correctness/coverage that improves this tradeoff. Do not expand scope solely because a model proposes a plausible patch.
