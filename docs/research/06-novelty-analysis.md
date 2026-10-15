# Novelty analysis and research candidate ranking

## Verdict on the broad idea

The broad concept is an intersection of established areas: memory-constrained execution, accuracy/semantics-constrained optimization, training validation and reproducibility. There is no evidence that the phrase “scientific fidelity” names one unified established subfield with the exact proposed product boundary. There is abundant evidence that its component research questions already exist. Rebranding them is not originality.

In particular, “automated memory planning” conflicts with [Rockmate](https://proceedings.mlr.press/v202/zhao23b.html) and native [memory-budget optimization](https://pytorch.org/blog/activation-checkpointing-techniques/); “preserving training with changing resources” conflicts with [EasyScale](https://arxiv.org/html/2208.14228v2); “automatically finding training invariants” conflicts with [TrainCheck](https://www.usenix.org/conference/osdi25/presentation/jiang); and “checking accumulation parity” overlaps [TrainParity](https://github.com/intelland/TrainParity).

The continuation found [HiLoRe](papers/hilore.md), a closer counterexample to the broad framing. Its existence strengthens the case for making the interaction benchmark the primary artifact. [FairScale's state handling](competitors/fairscale.md) also means the BatchNorm counterexample is a known hazard, not an Atlas discovery. Candidate1 remains a falsifiable study, with weaker confidence in standalone planner novelty; candidate2 is the preferred durable output if the composed baseline is competitive.

## Candidate ranking

Ratings are qualitative planning judgments, not measured probabilities of publication. Hardware refers to the proposed evaluation, not a verified requirement.

| Rank | Narrow contribution | Originality and closest work | Depth / significance | Feasibility / hardware | Main failure mode / interpretability |
|---|---|---|---|---|---|
| 1 | **Contract-conditioned selection under composed checkpointing and accumulation** | Provisional incremental gap versus Rockmate + TrainCheck/TrainParity | Moderate: evidence-aware decision pipeline and interaction study | One 8–12 GB GPU, CPU controls; 10–16 core developer weeks | Existing composition is equally effective; still interpretable as a negative result |
| 2 | Audited benchmark of unsafe transformation interactions | Dataset/evaluation novelty must be checked; related to Deep500, TrainCheck and KernelBench | Moderate empirical depth, reusable cases | CPU first; one GPU for resource effects; 5–9 weeks | Contrived faults or leakage; hold out workload and fault families |
| 3 | Cost-aware sequential verification budget allocation | Overlaps Hyperband and cost-aware testing; distinct objective must be formalized | Potentially high, but requires enough trials to model misses | One GPU plus many repeated runs; 8–14 weeks beyond profiler | Weak low-cost proxies; interpretable only if miss/cost tradeoff evaluated |
| 4 | Short-probe prediction of final paired equivalence | Overlaps reproducibility and learning-curve research; very uncertain novelty | Statistically demanding and scientifically important | Many seeds/full runs, likely beyond one GPU's practical budget | Rare late divergence and underpowered equivalence; high inconclusiveness |
| 5 | Local-LLM candidate ranking versus deterministic enumeration | Crowded code-agent optimization area; KernelBench adjacent | Useful ablation, limited standalone novelty | CPU or 6–12 GB GPU, serialized inference/training; 3–6 extra weeks | Finite registry makes LLM redundant; easy negative result |
| 6 | Arbitrary-PyTorch transformation proof system | Related formal numerical verification; broad claim untenable | Very high theoretical/engineering scope | Hardware secondary; language semantics dominates | Dynamic Python, custom ops and side effects; low undergraduate feasibility |

Choose candidate 1, with candidate 2 as its required evaluation artifact. Do not develop all candidates concurrently. Candidate 3 is a later pivot only if there is evidence that verification cost limits value.

## Check each suggested direction against prior art

- **Fidelity-constrained selection:** general concept is established by approximate computing, e.g. [Chisel](https://people.csail.mit.edu/rinard/paper/oopsla14.pdf). Candidate distinction is the workload-specific single-device training decision study.
- **Verification of training-preserving changes:** partially implemented by TrainCheck, TrainParity and numerical tests. New value must be demonstrated on composed memory plans and held-out failure conditions.
- **Compute-budget experiment design:** Hyperband and compute-optimal scaling already address budget allocation. Changing a study's design is a different service, explicitly outside strict reproduction mode.
- **Numerical reproducibility under memory changes:** EasyScale and native reproducibility controls are strong neighbors. Atlas has no new numerical theorem at present.
- **Per-workload planning:** already common in profilers, compilers and rematerialization systems. A new planner must outperform or clarify an existing workflow, not merely produce a recommendation.
- **Scientific-constraint recommendations / unsafe improvement detection:** plausible application boundary; not an established novelty claim without direct checker and composed-baseline comparisons.

## What would falsify the selected gap

1. An existing tool exposes the same contract-conditioned candidate selection with equivalent interaction coverage and costs.
2. The composed baseline rejects the same invalid candidates and reaches the same feasible plans at equal or lower total cost.
3. Atlas's apparent improvement disappears when its fault cases, thresholds or workloads are held out.
4. Its safe set is too conservative to yield a useful plan, or its checks accept empirically invalid plans at unacceptable rates.

If (1) or (2) holds, publish an honest evaluation if warranted, contribute cases upstream, or narrow the effort to a reproducibility study. If memory feasibility increases only by changing scientific constraints, call it methodology redesign rather than preservation.

## Evidence still needed for a novelty claim

Read the closest implementations, reproduce representative cases, search forward citations and recent workshop literature, ask an advisor with ML-systems or scientific-computing expertise, and document a concrete behavior the closest baseline cannot already provide. This session establishes a defensible investigation target, not a confirmed novel algorithm.
