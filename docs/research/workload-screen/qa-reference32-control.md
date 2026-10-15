# Fresh-process logical-32 reference variation

All four [frozen workers](qa-reference32-protocol.md) complete one native SDPA/AdamW update, with two fresh-process repetitions for dropout .1 and two for the modified zero-dropout condition. There is **no candidate transformation in these pairs**. [Raw record](qa-reference32-control.json), [secondary scalar/fsum reconstruction](qa-reference32-reconstruction.json), [local-artifact verifier](../../../benchmarks/verify_qa_reference32_artifacts.py).

| Dropout | Position-embedding gradient max difference | Position-embedding parameter max difference | Parameter elements exceeding declared diagnostic |
|---|---:|---:|---:|
| .1 | 2.85218e-9 | 3.19444e-7 | 5 of 393,216 |
| 0 | 2.32831e-9 | 2.16067e-7 | 2 of 393,216 |

Both conditions have equal initialized whole-model/group/input/start-RNG identities, equal reported loss/norm within each pair and matching end RNG hashes. Whole post-model and post-clip-gradient hashes differ. Among the selected arrays, QA-head parameters match exactly and query-weight differences stay within the declared diagnostic. All selected gradient and moment arrays have zero exceeded elements. Equal final RNG does not establish equal floating-point execution.

The diagnostic is `abs(a-b) > 1e-7 + 1e-5*abs(b)` computed independently in float64. It is not an executed TrainParity call or a scientific-equivalence margin. A pointwise acceptance rule with this bound would already encounter reference variation in these declared fields before attributing a candidate difference to its transformation. Do not increase tolerance after observing these results and call the same data confirmatory. The implementation-specific numerical source of the variation is not isolated; deterministic-algorithm mode was false.

## Retained coverage and audit

Four selected named parameters contain 984,578 elements, about .90% of this model's parameter elements. Each exports parameters, clipped gradients and both AdamW moments. The four ignored local NPZ artifacts total 63,033,176 bytes, with file/array identities in the raw record; no pickle or full-state tensor export is used. Whole-state hashes do not supply distances for the remaining parameters or missing optimizer observations.

The second implementation reads those identity-matched artifacts and reconstructs maxima/indices, exceeded counts, equality, finite health and norm summaries through Python scalar arithmetic and `math.fsum`, rather than NumPy's vector reductions. All 32 array comparisons/7,876,624 compared elements agree within the declared arithmetic-reconstruction tolerance; no additional training is run. This checks the saved selected-state arithmetic, not independent scientific labels or unobserved fields.

Actual RSS peaks range about 1.59–1.69 GiB while job peak commit ranges 10.28–10.53 GiB. They are different OS measurements, not alternative estimates of one number or physical GPU residency. Every owned group is empty after cleanup. This first-update reference diagnostic does not establish full-run consistency, QA quality, deterministic-kernel guarantees or another hardware tier's feasibility. It closes the previously unmeasured *selected-distance* part of logical-32 reference noise while retaining its limited coverage.
