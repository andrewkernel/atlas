# Comparison conventions and health predicates

Reused comparison code needs an exact policy identity. The executed controls confirm that identical rtol/atol values can yield different decisions under tensor, scalar and proposed baseline-scaled rules. Matching nonfinite values can also pass an equality policy without establishing a healthy reference. Neither finding is, by itself, an upstream defect or a new Atlas verification algorithm.

The [source/protocol](comparator-contract-protocol.md) was frozen at `dcc1818`; [results](comparator-contract-control.json) identify Python 3.11.17/Torch 2.13.0+cpu, float64 tensors and the pinned reviewed checker. Ten fixed constructed controls run without GPU kernels or ML data. These are boundary examples, not a practical error-rate estimate.

## Three distinct finite rules

For baseline a and candidate b, the proposed Atlas text uses `abs(a-b) <= atol + rtol*abs(a)`. The inspected tensor comparator calls [`torch.isclose(a,b)`](https://docs.pytorch.org/docs/stable/generated/torch.isclose.html), whose relative scale is the second argument, here candidate b. Scalar comparison uses [`math.isclose`](https://docs.python.org/3.11/library/math.html#math.isclose): the maximum magnitude and a max of absolute/relative bounds, rather than their sum. These documented conventions are individually meaningful, but not interchangeable.

|Representation and values |Parameters |Existing result |Proposed baseline-additive result |
|---|---|---|---|
|Tensor a=1, b=1.11 |rtol=.1, atol=0 |PASS |Fail |
|Tensor a=1, b=.905 |rtol=.1, atol=0 |FAIL |Pass |
|Scalar a=1, b=1.11 |rtol=.1, atol=0 |PASS |Fail |
|Scalar a=1, b=1.15 |rtol=.1, atol=.1 |FAIL |Pass |
|Tensor a=1, b=1.00001000005 |rtol=1e-5, atol=0 |PASS |Fail |

The final example deliberately sits near a tiny boundary; it is not evidence these differences commonly affect training. Earlier finite-run comparisons retain their recorded implementation/settings; they are not retroactively relabeled or claimed to implement the proposed formula exactly.

## Equality is distinct from health

Matching positive-infinity tensors pass exact and tolerance policies. Identically constructed NaN tensor bytes pass exact comparison; default tolerance fails, and explicit equal-NaN tolerance passes. The separate `both_finite` predicate is false for all of these. The default NaN failure's infinite diagnostic error summaries are retained as null with two explicit field locations, not silently turned into finite errors or invalid JSON.

A caller requiring finite loss, gradients, parameters or optimizer moments must check that declared health predicate independently. Some attention masks legitimately contain negative infinity, so a blanket “all captured tensors must be finite” policy would itself be wrong. A repeatable broken reference does not establish a valid experiment. Similarly, tolerant equality of optimizer hyperparameter floats cannot replace an exact methodology/configuration check when that property is required.

## Integration requirement

Bind formula, reference side, scalar/tensor treatment, dtype, absolute-bound combination, nonfinite policy, field scope and comparator version to evidence. Keep configuration invariants, reference-health requirements and numerical relation checks separate. A thin policy adapter or explicitly chosen existing convention may be sufficient; do not rebuild generic capture/comparison under a novelty claim. Until conventions match, report the actual implemented relation and leave any broader contract predicate unverified.
