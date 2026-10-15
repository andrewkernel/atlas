# Reliability bounds, candidate selection and stopping rules

This note audits future inferential claims. It reports no measured Atlas field-risk estimate. The [selection estimands](evaluation-estimands.md) define counts/unknown labels; the [assumption audit](experiments/statistical-assumption-audit.md) shows why a valid sampling model is a separate requirement.

## Zero observed errors is not zero unknown risk

For n independent Bernoulli trials from a specified population with 0 observed errors, solving `(1-p)^n = .05` gives the exact one-sided95% upper bound `p_upper = 1-.05^(1/n)`. This is a direct binomial derivation; [NIST's confidence-interval reference](https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm) distinguishes exact small-count intervals and one/two-sided conventions.

|Independent population trials,0 errors |One-sided95% upper bound |
|---|---|
|10 |25.89% |
|24 |11.73% |
|100 |2.95% |
|300 |.994% |

These are illustrative calculations, **not bounds on Atlas**. Twenty-four designed mutations, forty generated partitions, thousands of tensors, or repeated steps from one run do not become independent population draws by being counted. Correlated workload/fault families require an explicit sampling/cluster interpretation. Indeterminate labels create identification uncertainty in addition to sampling uncertainty.

A balanced fault corpus is useful for detection comparison, but its injected-fault prevalence is not the incidence in users' projects. Precision and unsafe-acceptance risk depend on the deployed workload mix. Report corpus composition and class-specific counts; do not export a benchmark percentage as an unconditional real-world reliability guarantee.

## Selecting among statistical passes

Suppose eight non-equivalent candidates each undergo a valid level.05 equivalence decision and a workflow chooses any candidate that passes. With independent decisions, the probability of at least one false pass is `1-.95^8≈.3366`; without independence, the simple union bound is at most.40. These arithmetic examples assume each underlying test is valid. Shared references and adaptive candidate construction require more careful dependence/selection analysis.

This is different from TOST's two one-sided components: requiring both components to reject yields the single equivalence test, rather than two independent discoveries. It is also different from one global conjunction that requires every prespecified required property to pass: if at least one property null is true, accepting the whole conjunction is contained in rejection of that true null, provided each component test has the stated validity. Do not mechanically divide alpha by every number in the interface without defining the claim family.

The simpler Atlas protocol profiles/searches on disjoint pilot information, freezes one finalist and analysis, then uses fresh confirmatory seeds. It does not search among eight favorable full-outcome p-values and report the winner. If multiple candidate-specific inferential claims are intended, prespecify the family and appropriate multiplicity control(e.g. Bonferroni/Holm when justified) or use independent confirmation. The present first-update numerical comparisons and fixed tolerances are finite computational checks, not formal population tests to which these p-value calculations automatically apply.

## Repeated looks and early stopping

Repeatedly recomputing a fixed-n confidence interval after each seed and stopping as soon as it lies inside the equivalence margin generally does not preserve its nominal fixed-horizon coverage. Resource early stopping and inferential stopping are different. The initial protocol should keep the confirmatory run count fixed and distinguish a resource/timeout stop from a successful scientific decision.

If cost-aware sequential verification later becomes a research candidate, it must use a justified sequential design and assumptions. [Howard et al.](papers/confidence-sequences.md) provide established time-uniform confidence-sequence methodology; citing it does not validate a chosen bound for arbitrary ML outcomes. Confidence sequences can remain wide when data are scarce or tails poorly controlled. Do not introduce sequential machinery just to make the pilot look conclusive.

## Required reporting

Name the population or fixed test set, unit, estimand, required properties, label scope, stopping rule, candidate-selection procedure, alpha/coverage convention and multiplicity family. Separate descriptive numerical/state evidence from population inference. Retain inconclusive, divergent and failed runs. No scalar “scientific confidence” score follows from any of these quantities.
