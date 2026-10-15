# Adversarial review and resulting revisions

These are three deliberately distinct **review perspectives authored in this session**, not independent external reviews, expert endorsements or a multi-agent consensus.

## Reviewer 1 — ML systems

**Strongest objection:** Atlas largely composes known tools. Native checkpoint/AMP/SDPA mechanisms, Rockmate's automatic rematerialization, DeepSpeed's profiler/search, and TrainCheck's checking leave little broad novelty. Calling a rule engine “AI” or adding a UI would not change this. Sources: [native memory budgeting](https://pytorch.org/blog/activation-checkpointing-techniques/), [Rockmate](https://proceedings.mlr.press/v202/zhao23b.html), [DeepSpeed Autotuning](https://github.com/deepspeedai/DeepSpeed/blob/master/deepspeed/autotuning/README.md).

**Potentially fatal:** a simple composition of native strategies and existing checkers gives the same result with less code. Another fatal issue is claiming arbitrary-loop support while requiring carefully authored adapters for all successful demos.

**Resolving experiments:** compare selection under equal trial and wall-time budgets against that composition; include manual/native memory-budget baselines and unsupported workloads. Hold out code and condition families. Report adaptation effort, false rejection, failure modes and total overhead.

**Required revision:** single-device, adapter-based, two-strategy study; no new optimization algorithm claim; existing-checker composition is an essential baseline. Integrate/reuse checker ideas where compatible rather than claim invention. A negative incremental-value result is acceptable and may imply upstream contributions instead of a standalone product.

## Reviewer 2 — reproducibility and statistics

**Strongest objection:** “scientific fidelity” risks implying that a final metric or gradient test certifies an entire scientific conclusion. Intended mathematical equivalence, observed numerical parity, distributional equivalence and hypothesis preservation are distinct. [EasyScale](https://arxiv.org/html/2208.14228v2) already treats consistency explicitly. [Equivalence testing](https://doi.org/10.1177/1948550617697177) requires justified margins and sufficient power; failure to reject a difference is inadequate.

**Potentially fatal:** no full reference is available on the target machine, yet the optimized full run is called a reproduction; or few seeds with broad margins create a false equivalence claim. Self-authored mutations and labels can also make a checker look sound by construction.

**Resolving experiments:** baseline self-consistency; keyed gradient/buffer/optimizer comparisons; held-out faults and external cases; expert adjudication; paired full outcomes for the precise claim. For rankings, measure method×transformation interaction, not only one model's mean accuracy. Report all unknowns and inconclusive intervals.

**Required revision:** vector evidence statuses, separate `ABSTAIN`/`ERROR`, no scalar scientific score, no automatic domain-conclusion approval. The actual CPU BatchNorm example demonstrates why gradients alone are inadequate, but does not prove the new verifier complete. Use power-informed confirmation and preregister analysis after a disjoint pilot.

## Reviewer 3 — resource-constrained computing

**Strongest objection:** reducing VRAM can increase runtime, host-RAM use, energy and setup burden. Verification may cost more than it saves. A local LLM that occupies the GPU undermines the access objective. A12 GB card is one hardware point, not evidence for6 GB laptops or institutions.

**Potentially fatal:** repeated baseline/verification runs are unaffordable for the intended users, or only activation-heavy toy workloads benefit while persistent state prevents useful work. Container and CUDA setup can dominate a student's available time.

**Resolving experiments:** measure total time including setup/search/checks/failed trials, host peaks, real completion under memory/time limits, and adaptation effort. Compare physical tiers when available; label allocator caps as emulation. Use device or external energy measurements where supported rather than extrapolate from time. Seek user evidence later for institutional-impact claims.

**Required revision:** CPU-development path, no mandatory model/backend service, explicit no-plan result, optional one-time characterization and reuse only within a valid evidence scope. Start with the observed12 GB host after CUDA validation. Larger-scale access benefits remain untested.

## Proposal revision ledger

| Initial idea | Evidence/objection | Final revision |
|---|---|---|
| General scientific optimizer | Extensive memory/search prior art | Contract-conditioned selection study |
| Arbitrary research-code analysis/patching | Dynamic semantics and execution risk | Trusted registered adapters; AST advisory |
| Gradient/accuracy verifies preservation | RNG/state counterexamples and statistical limits | Separate dimensions, phases and unknowns |
| LLM central to planning | Finite registry and contention | Optional later ranking/explanation ablation |
| Broad memory-saving toolbox | Too many numerical/methodological interactions | Accumulation + pure-region checkpointing first |
| Consumer hardware generalization | One physical device observed | One-device evidence; other tiers planned |
| Full verification on cheap pilot | No universal short-to-full implication | Full-run conclusions require full-run evidence |

**Unresolved major objections:** novelty over composed tools; real-world rate of useful eligible plans; affordability and adequacy of verification; dataset rights/provenance; and lack of independent labels. None is dismissed by architectural neatness. These are stop/pivot gates in the roadmap.
