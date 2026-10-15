# Statistical-assumption and design-sensitivity audit

This is a synthetic exploratory experiment, not observed ML outcomes or a claim that training failures occur at any particular frequency. [Source](../../../benchmarks/statistical_assumption_audit.py), [raw evidence](statistical-assumption-audit.json), and the separately hashed [normal-model calibration](equivalence-calibration.md).

## Power is conditional on a design model

Under independent normal paired differences, symmetric margin±.5 and each one-sided alpha.05, the minimum count for 80% joint TOST power changes substantially with paired SD and the true mean difference:

| Paired SD |True mean0 |True mean+.25 |
|---|---|---|
|.5 |11 pairs |27 pairs |
|1.0 |36 pairs |101 pairs |
|1.5 |79 pairs |224 pairs |
|2.0 |139 pairs |398 pairs |

These are numerical integrations under the declared normal model. The mean+.25 is still inside the margin, but closer to a boundary and harder to distinguish from it. Do not plan only for a zero-effect best case and then claim the same power for all equivalent effects.

For an illustrative three-pair normal pilot with observed paired SD 1, a95% chi-square SD interval is approximately[.52,6.28]. This is not an observed Atlas pilot. It shows why three seeds give a very uncertain variance estimate even when normality is granted. Paired SD depends on the actual covariance of outcomes; matching integer seed labels does not guarantee a strongly correlated pair.

## Same mean/variance, different interval validity

The second experiment compares an ordinary t-interval TOST at n36 under two independent models, each with population SD 1. One is normal. The other is a deliberately constructed mixture: Gaussian noise SD.05 plus a1%-probability jump, centered/scaled to mean mu and SD 1. The jump is about 10.04; no observed training distribution motivates that number.100,000 experiments are simulated for each condition with recorded seeds.

|True mean |Normal equivalence-decision fraction |Rare-mixture fraction |
|---|---|---|
|0 |.80643 |.69512 |
|-.5, null boundary |.05064 |0 |
|+.5, null boundary |.04907 |.69680 |

At either null boundary, declaring equivalence is a false positive. The normal controls are near the nominal.05 rate. For this asymmetric mixture, the upper-bound false-equivalence fraction is about.697. This does not mean TOST itself is invalid; the ordinary normal-theory t-interval used here is unreliable under this small-sample mixture. Matching population mean and variance is insufficient to validate its finite-sample inference.

The chance of seeing no rare component in 36 independent draws is.99^36≈.6964. Those samples look tightly concentrated inside the margin and miss the component that moves the population mean to the boundary. An ordinary bootstrap of such a sample cannot invent the unseen component. A nonsignificant normality test on a small sample also does not establish the required model. Roughly299 independent draws would give95% probability of seeing at least one event at a **known assumed**1% frequency; that is an event-detection calculation, not an equivalence sample-size recommendation or an estimated real failure frequency.

## Protocol consequence

Before inferential training-outcome claims, justify the estimand and sampling model from domain knowledge and disjoint evidence. Record convergence failures, censored/timeout runs and prespecified tail or failure endpoints instead of deleting them from an accuracy mean. Do not replace failures with a convenient score without defining the resulting estimand. Run a model-appropriate operating-characteristic study when normal approximations are doubtful, and accept inconclusive results when the available compute cannot justify the conclusion.

There is no universal cheap replacement that certifies arbitrary rare events. A distribution-free bound with a broad known outcome range may need far more data; resampling assumptions also require justification. This audit supports abstaining from unsupported full-science claims, not automatically choosing a different test after seeing favorable data. The primary semantic-selection pilot can still proceed with clearly bounded contract properties and descriptive counts.
