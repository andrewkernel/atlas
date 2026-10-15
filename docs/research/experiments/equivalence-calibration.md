# Equivalence power audit — synthetic statistical experiment

The continuation audit found a substantive error in the original planning formula: it used an80% one-sided power quantile for a decision requiring both one-sided rejections. This document corrects that calculation. It does not change the scientific margin to suit the available compute.

## Derivation

Suppose independent paired differences are normal with true mean0, population SD sigma, sample size n and symmetric equivalence bounds ±Delta. With known SD, TOST at each one-sided alpha accepts equivalence exactly when

`-Delta + z_(1-alpha)*sigma/sqrt(n) < sample_mean < Delta - z_(1-alpha)*sigma/sqrt(n)`.

Standardizing the sample mean gives joint power `max(0, 2*Phi(Delta*sqrt(n)/sigma-z_(1-alpha))-1)`. Setting power to 1-beta yields `n = ((z_(1-alpha)+z_(1-beta/2))*sigma/Delta)^2`. This is a derivation under the stated model, not a general power formula for every experimental design.

For unknown SD, the sample mean and sample variance are independent under normality. [Original audit code](../../../benchmarks/equivalence_calibration.py) integrates the conditional acceptance probability over a chi-square variance distribution. A separate deterministic-seed simulation draws those sufficient statistics directly. [TOSTER](https://aaroncaldwell.us/TOSTERpkg/reference/power_t_TOST.html) documents exact one-sample/paired TOST power and defines paired SD as the SD of differences; this code was independently written and the R package was not executed.

## Recorded result

[Raw JSON](equivalence-calibration.json) records source hash, environment, seeds and200,000 repetitions per condition. Here sigma=1, Delta=.5 and each one-sided alpha=.05.

| Independent pairs | Integrated unknown-SD power | Simulated fraction |
|---|---|---|
|3 |.0190 |.0185 |
|5 |.0203 |.0203 |
|10 |.0655 |.0661 |
|25, old approximation |.5668 |.5694 |
|35, corrected known-SD approximation |.7900 |.7907 |
|36 |.8052 |.8047 |
|50 |.9344 |.9348 |

Boundary controls at true mean±Delta have equivalence-decision probability at most.05; an outside-bound control is lower. The JSON gives binomial Wilson intervals describing Monte Carlo error only. A single95% Monte Carlo interval need not include the integrated value in every repeated simulation. Integration precision, an independent sampling route and type-I boundary controls are more informative than treating one fixed interval as a universal software assertion.

## Design consequences

The earlier25-pair example understated cost. Ten pairs can be dramatically underpowered for this margin/variance ratio. Domain margins must be fixed independently of observed candidate results. Estimate variance using external or disjoint pilot information, examine plausible larger SDs, and freeze a realistic count before confirmation. Correlated seeds, heavy tails, heterogeneous workloads, multiple confirmatory claims and uncertain pilot variance require additional design work. This experiment establishes none of those assumptions for Atlas workloads, and no actual training equivalence was tested.
