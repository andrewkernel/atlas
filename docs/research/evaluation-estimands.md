# Selection-study units, denominators and unknown labels

This amendment makes the primary policy comparison precise. It is an analysis protocol proposal, not a reported Atlas result. Risk/coverage tradeoffs and rejection are established concepts, for example [SelectiveNet](https://proceedings.mlr.press/v97/geifman19a.html); Atlas does not invent abstention evaluation.

## Unit and independent labels

A policy evaluation unit is a prespecified workload implementation + contract + input/seed manifest + hardware/budget cell. Each policy receives the same unit, candidate registry, adapter facts, allowed observations and search budget, and returns at most one selected plan or an explicit no-plan/error outcome. A candidate trial is not an independent workload. Repeated timing steps and tensors are not independent policy decisions.

Separately label each candidate's required semantic relation as **valid**, **invalid**, or **indeterminate**, with property scope and reasons. Resource feasibility is a different label: completed within the declared resource window/budgets, failed, or unknown. A numerical check passing on a small window cannot label full-training equivalence valid. Independent qualified annotators adjudicate semantics using evidence outside the policy's own verdict. Neither annotator agreement nor a fixture author's intention proves universal truth.

Freeze labels or the adjudication procedure before inspecting policy recommendations. Retain disagreements and ambiguous cases. Unknown is not silently reassigned to invalid to increase detection rates, or to valid to increase coverage. For policy eligibility, an indeterminate required relation blocks strict acceptance; that is a decision rule, not a discovered semantic violation.

## Primary pair of endpoints

Let N be all declared units, A the number with an accepted plan, U the accepted plans independently labeled invalid, K the accepted plans with an indeterminate required relation, and V the accepted plans independently supported as valid and resource-feasible.

- **Unconditional known-unsafe acceptance:** U/N, with U and N. An ignorance interval for the actual unsafe fraction is [U/N,(U+K)/N] if all other decisions are labeled; these are identification bounds, not confidence intervals.
- **Valid-feasible coverage:** V/N, retaining no-plan, errors, unsupported cases and timeouts in N. A refusal-only policy has zero coverage.

The main safety/coverage comparison uses this pair at a fixed budget. **Selective risk U/A** is a secondary endpoint, accompanied by [U/A,(U+K)/A] and acceptance coverage A/N. When A=0, selective risk is undefined/null; do not report0%. Do not change the denominator according to how many candidates each policy chose to inspect. Reporting U/A alone can improve merely by accepting more easy valid plans while accepting the same unsafe plans.

Also report valid-plan opportunity coverage conditional on units known to contain at least one valid-feasible candidate, with its separate denominator and count of indeterminate opportunity labels. This diagnoses conservative failure but does not replace the all-unit primary denominator. A finite-set exhaustive profile is only an observed oracle for its declared window, not a global optimum or a proof of future completion.

Candidate-level check detection is a separate task: invalid accepted / independently invalid candidates, false rejection / independently valid candidates, indeterminate disposition, and errors. Do not pool candidate-level and final-policy denominators or use multiple mutations of one implementation as independent workload samples.

## Arithmetic example — invented, not empirical

Four units contain: U1 a valid slower and invalid faster plan; U2 only known invalid plans within budget; U3 a valid plan; U4 a feasible plan with indeterminate semantics.

| Illustrative policy | Accepted | Known invalid | Indeterminate accepted | Valid-feasible | U/N | V/N | Selective risk bounds |
|---|---|---|---|---|---|---|---|
|Resource-only choice |4 |2 |1 |1 |.50 |.25 |[.50,.75] |
|Contract-respecting choice |2 |0 |0 |2 |0 |.50 |[0,0] |
|Refuse every unit |0 |0 |0 |0 |0 |0 |undefined |

The zero-width identification bound [0,0] means no indeterminate labels in this example; it does not give a statistical confidence bound of zero risk in an unseen population. Even zero observed errors on a small sample leaves substantial uncertainty. Four invented units prove no real policy benefit.

## Baseline fairness and inference

Give the native-plus-checker baseline the same workload declarations, reference snapshots, contract information and adaptation allowance. A competent composition may use existing state-aware wrappers and appropriate check settings. Record human integration time separately. Comparing a carefully adapted Atlas against an uninformed/default checker only measures adaptation advantage. Freeze each baseline before held-out evaluation and permit the same pilot budget.

Prefer a mutually compatible software stack for the primary causal policy comparison. TrainParity's inspected restrictions exclude the original Python 3.12/Torch 2.14 research environment. A separate [Python 3.11/Torch 2.13 shared-stack control](experiments/interop-controls.md) now executes both pinned sources and a streaming capture adaptation on one pure fixture; this does not complete compatibility validation for the intended resource workloads. Freeze a reviewed shared stack for every compared primary policy or restrict claims to separated capability studies. Shared traces help only for properties they actually observe; they cannot replace timing or unsupported dynamic behavior.

Report paired per-unit differences and all counts. Cluster by workload implementation/fault family where appropriate. Few fixed hand-selected units support descriptive comparisons, not broad population reliability. Freeze practical superiority/noninferiority margins only after a disjoint pilot and domain justification; observed non-reduction is not automatically a powered coverage noninferiority result. Any threshold sweep is prespecified or explicitly exploratory, with no best-test-set threshold selected afterward.
