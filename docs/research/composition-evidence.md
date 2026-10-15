# What the current composition evidence does and does not show

The current fixtures are useful controls, but they do not establish a new class of emergent transformation-interaction bugs. This limitation matters because the proposed contribution emphasizes interactions.

## Three different kinds of case

1. **Single-transform incompatibility:** training BatchNorm under microbatching, or a mutable checkpoint region with duplicated buffer updates. Each can fail without a second transformation.
2. **A known defect inside a composed execution:** the two-update fixture combines checkpointing and accumulation, then intentionally clips each partial gradient or steps after each microbatch. This is a faulty update implementation; the same defect need not require checkpointing. Existing accumulation checkers may already catch it.
3. **An emergent composition failure:** each transformation individually satisfies the same declared relation on a reference domain, but the combined implementation does not. To claim this category, demonstrate both isolated positive controls, the combined failure, and the relevant changed precondition or implementation mechanism. Matching only one convenient input does not prove the transformations individually valid on all inputs.

The first two categories are currently evidenced. The third remains an open requirement, not an existing finding. Do not describe the current clipping/optimizer controls as proof of a uniquely compositional failure. The resource study does show a combined memory/time tradeoff; resource interaction and semantic interaction are different claims.

## Conditional composition reasoning

Let a contract relation R be transitive on a stated domain D. If T1 preserves R for every E in D, T1 maps D into D, and T2 preserves R for every E in D, then transitivity gives R(E,T2(T1(E))). Individual tests of R(E,T1(E)) and R(E,T2(E)) at one E do not establish those universal/domain-closure premises. Approximate tolerance relations also need care: an elementwise error bound is generally not transitive at the same tolerance, and errors/costs can accumulate.

For the initial pure-region checkpointing and separable-loss accumulation design, the mathematical obligations are comparatively straightforward when domain closure, state, randomness and update clocks are actually preserved. The difficult engineering work is observing/enforcing them on real implementations. Atlas currently has no new composition theorem or proof system.

## Contribution gate

Before claiming an interaction benchmark as a distinct research contribution, obtain independently adjudicated cases spanning actual shared-workload compositions, compare existing checkers with the same observations, and show what is missing beyond known single-transform rules. Include cases where composition succeeds, where one transform is already invalid, and where the relation is indeterminate. Keep those categories separate in detection results.

If the closest checker composition catches the current controls equally well, that is the expected strong-baseline result, not a reason to weaken the baseline. A useful negative study, benchmark consolidation or upstream case contribution may be the appropriate result. Proceed to a bounded comparison pilot; a standalone new optimizer or uniquely compositional checker publication claim is unsupported at this stage.
