# ADR 003 — Property-specific evidence, never a scientific-validity score

**Status:** accepted.

**Context:** lower memory, matching gradients and similar final metrics answer different questions. Missing full baselines and underpowered comparisons are common practical limitations.

**Alternatives:** one confidence score; final-metric threshold; exact equality everywhere; vector evidence with scoped outcomes/statuses.

**Evidence:** our checkpoint BatchNorm probe matches gradients while changing buffers/evaluation. [Numerical-accuracy guidance](https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html) distinguishes mathematically equivalent executions from bitwise equality. [Equivalence-test principles](https://doi.org/10.1177/1948550617697177) require more than nonsignificance.

**Decision:** report dimensions separately, with pass/fail/abstain/error and explicit scope, assumptions and artifacts. Numerical tolerances differ from scientific equivalence margins. Human judgment owns scientific intent and interpretation.

**Tradeoffs:** more verbose reports and more inconclusive cases, but fewer misleading global claims. Exact checks remain appropriate for counts/structure and some deterministic controls, not automatically for all low-precision GPU computation.

**Revisit when:** a particular domain has a formally specified conclusion predicate and validated verification procedure. Extend that narrow domain; do not generalize its assurance to all ML research.
