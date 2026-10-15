# Independent adjudication packet specification

No independent labels or reviews have been collected. This document prepares that next-stage work; it does not designate the investigator's existing controls as a held-out independently labeled corpus. No packet has been sent to external people.

## Unit and information

Use the [selection-study unit](evaluation-estimands.md): implementation, scientific contract, input/seed manifest and resource-budget cell. Make a case packet before policy evaluation with code/data/environment digests, declared scientific question, logical samples and loss denominator, update clocks, stochastic relation, state fields, intended execution relation, precise resource scope, allowed method changes, provenance and reference-availability limits. Adjudicators need this information; blinding must not conceal scientifically necessary evidence.

Provide neutral candidate identifiers and inspectable configuration differences. Withhold policy name, ranking, acceptance verdict and investigator's desired label. Keep raw traces accessible when needed, with a separate record of which evidence the adjudicator viewed. A minimal numerical observation can support only that phase/window, never an unsupported full-outcome label. Historical issue identity is provenance, not automatic truth.

## Label sheet

For every required property, record valid / invalid / indeterminate, its exact relation and scope, supporting evidence identifiers, reasons, known assumptions and whether the information suffices. Separate mathematical eligibility, finite numerical observations, methodology, statistical outcome and domain conclusion. Resource completion/feasibility gets its own scope-matched observed/failed/unknown label. Preserve a missing reference as unknown where comparison is required.

Record one candidate-level semantic label only after retaining those property labels. An invalid required property is sufficient for invalidity under that contract; an indeterminate required property prevents strict valid acceptance. Report evidence of a contract violation separately from mere lack of sufficient evidence. Exact method constraints cannot be relaxed by a numerical tolerance. Reviewer confidence, if collected, is metadata and never a scientific safety score.

## Independence, disagreements and leakage

Ask at least two appropriately qualified reviewers to label independently before showing each other's decisions or policy outcomes. Record relevant expertise and conflicts. A third adjudicator can resolve disputes with reasons; retain both originals and the unresolved label where evidence remains insufficient. Domain-conclusion judgments need domain expertise, not only ML software familiarity. Agreement measures describe consistency, not truth. Report raw agreement and category counts; if kappa is used, state prevalence effects and do not use it as a validity certificate.

Existing hand-authored probes and published tool behavior are discovery/pilot information. Do not turn them into an untouched test set by renaming identifiers. Split whole implementations/projects and fault/condition families where feasible, and record which developer/reviewer saw what before freezing policies. A small curated fixed set permits descriptive comparison; it does not establish population risk. If genuine independent holdout is infeasible, label the study investigator-constructed and narrow the claim.

## Freeze and review sequence

1. Select the target claim and workload population or explicit fixed set. Separate contract-level author intent from empirical outcome evidence. Freeze the adjudication procedure and disjoint pilot/confirmation split.
2. Pilot packet clarity, reviewer burden and disagreement on development cases. Allow every baseline the same correction/adaptation opportunity; record effort.
3. Freeze policy implementations, software, candidate registry, all comparison/health predicates, budgets and decision criteria. Commit packet identities and the rule for new or unresolved evidence.
4. Obtain independent labels using the frozen procedure before exposing recommendations; keep a restricted label copy if necessary. Record labeling changes as amendments, with chronology.
5. Run all policies on the same declared units. Preserve no-plan, error, unsupported and timeout outcomes. Reveal labels for final evaluation only under the agreed procedure.
6. Reconstruct all-unit unsafe-acceptance bounds, valid-feasible coverage and opportunity coverage from raw unit records. Report unknowns, reviewer disagreements, total acquisition/review/compute cost and all exclusions.

If a case is discovered during final evaluation, quarantine it for a later study or declare the current analysis exploratory. A new benchmark version is preferable to silently editing the test set. The current evidence does not satisfy this independent-label gate, regardless of how many local checks pass.
