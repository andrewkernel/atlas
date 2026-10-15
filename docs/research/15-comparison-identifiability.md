# Comparison audit: what is the Atlas treatment?

**The current primary superiority hypothesis is not yet an executable comparison.** Giving Atlas and the competent native-plus-checker baseline the same contracts, adapters, preconditions, candidates and observations is necessary for fairness. It also removes any automatic advantage from merely saying Atlas is contract-aware. The documents have not specified an additional selection algorithm that distinguishes the two treatments. Do not implement a product or launch a confirmatory policy study around that ambiguity.

This amendment narrows [H1 and H3](07-experimental-design.md) and the [updated recommendation](14-updated-recommendation.md). The benchmark, integration obligations and measurement audit remain useful even if an independent planner contribution disappears. Nothing here is an empirical policy-win result or a new decision-theory theorem.

## The identical-policy boundary

For one fixed unit let C be its candidate registry, K its supplied contract/adapter facts, O its allowed observations and B its budget. A policy maps its current observation history to: reject a candidate, acquire an observation, execute a check/profile, accept one candidate, or return no plan/error. Its ordering, stopping, eligibility and final tie-break are all part of the policy.

If two deterministic policies have the same initial state and decision function, and receive the same observations/costs after every chosen action, they choose the same actions and final decision. Renaming one Atlas or adding an evidence report cannot change this. In an exhaustive fixed observation table with shared eligibility, objective and tie-break, both return the same optimum of that finite eligible set. This follows directly from the definitions, not a research contribution.

Real executions need not supply a fixed table: caching, allocator history, background load, numerical nondeterminism and failed observations can make order matter. Identical expected recipes do not imply bitwise identical measurements or runtimes. Record those effects; do not reinterpret a noisy tie as a policy effect. The [QA resource screen](workload-screen/qa-resource-pilot.md) supplies a concrete self-consistency warning.

## Separate four questions

| Question | Distinct treatments needed | What it could support |
|---|---|---|
| Do explicit semantic facts matter? | Resource-only ablation versus semantic eligibility, both with native mechanisms | Value of supplied information/constraints; not superiority over an equally informed baseline |
| Does observation/plan ordering help under a finite budget? | Frozen fixed/random order versus an explicitly different adaptive order; same candidates, facts, checker and budget | Incremental selection/cost evidence if it survives held-out testing; a known type of optimization problem |
| Does an adapter/checking interface reduce engineering effort? | Independently performed matched integration tasks with measured time/failures and equivalent requirements | Engineering usability/reliability evidence; external participants and a separate study are needed |
| Is a semantic interaction missing from existing checks? | Independently valid single transforms, an invalid composition, and an existing-checker comparison on the same supported relation | A concrete coverage gap if demonstrated; current controls have not demonstrated this |

Do not substitute the resource-only ablation for the strong baseline in the primary comparison. Do not count a manually supplied Atlas fact as automatically inferred or charge its acquisition only to the baseline. Existing lower-level capture primitives plus an authored adapter remain part of the competent comparator.

## Minimum executable policy specification

Before held-out evaluation, publish a versioned policy object with:

1. Exact candidate list and transformation implementations; software profile; precondition rules; which rules are supplied, inferred or assumed.
2. Exact initial information for every arm, including permitted historical observations and discovery/holdout provenance.
3. A deterministic next-action function, tie-break and termination rule. “Profile intelligently” is insufficient.
4. Required evidence and acceptance predicates for each contract relation. An unobserved full-training outcome cannot become valid because a short profile completes.
5. Budget accounting for preparation, reference repeats, profiles, capture/comparison, failed trials, ranking and final choice. Record user cost and cost omitted from a narrower controlled estimand separately.
6. A fixed final-choice objective, including resource scopes and an explicit tradeoff or Pareto reporting rule. Do not choose the fastest plan in one arm and the smallest allocation in the other.

The original eight-trial/30-minute cap is a pilot planning cap. It does not identify action costs. State whether a reference repeat consumes a trial, whether an early precondition refusal consumes only time, and whether final confirmation is outside the search budget. Count the same work under the same definition for both arms. A timeout is a censored/failed observation, not a free hint of intrinsic infeasibility.

## A next comparison, if ordering is pursued

The following is a design candidate, **not** a frozen implemented study:

- Shared eligibility/evidence engine: native mechanisms and supported existing capture/checker adaptations. Both arms receive the audited adapter facts and contract. Required unknown evidence blocks acceptance in both arms.
- Fixed-order baseline: a sequence declared before holdout, chosen with the same disjoint pilot allowance. A seeded fixed random-order distribution is secondary; ranking seeds are separate from scientific training seeds.
- Adaptive arm: a published next-action function using only its allowed history and disjoint-pilot parameters. Charge its added profiling/estimation. Its algorithm is currently **unselected**; this document does not claim one exists.
- Stop when either the time or action cap is reached. Choose among sufficiently evidenced candidates observed before that point; no-plan/errors remain in the workload denominator. Use the same objective/tie-break.
- Freeze source and practical criteria after a disjoint pilot. Independently adjudicate required semantics, retaining indeterminate cases. Pair arms by workload/contract/hardware/scientific seed and randomize order without giving one arm the other's hidden observations.

An exhaustive profile table can describe the observed frontier or permit a retrospective fixed-table replay. It cannot establish online workflow cost, discover unobserved behavior, provide independent outcome labels, or supply a free oracle only to Atlas. The sixteen QA resource trials are a development screen, not this policy comparison.

## Research decision

Make the current deliverable a reproducible **contract-and-resource evaluation package with existing-tool integrations**. Keep the application architecture as a later engineering option. Reopen a novel-planner claim only after a distinct algorithm or repeatable coverage gap survives the competent baseline. A useful negative result is that native mechanisms plus existing checks already cover the cases.

The label “scientific contract” does not establish scientific intent. Objective/preprocessing declarations need author review and independent adjudication. Byte identity does not validate those declarations. The external-label gate remains incomplete; no reviewer contact or endorsement is implied.
