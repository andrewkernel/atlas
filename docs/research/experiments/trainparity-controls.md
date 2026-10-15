# Direct TrainParity controls

The pinned existing checker executed successfully in its supported CPU stack. Three positive controls PASS, six deliberately faulted controls FAIL, unsupported metadata ABSTAINS, and an invalid plan returns ERROR. All 11 outcomes match the authored predictions. This directly strengthens the case for reusing existing checking machinery; it does not demonstrate Atlas superiority.

## Provenance and attempted-run denominator

The [protocol](trainparity-controls-protocol.md), [configuration](../../../configs/trainparity-controls.json), original [cases](../../../benchmarks/trainparity_cases.py) and [launcher](../../../benchmarks/run_trainparity_controls.py) were frozen at `c5a0c93` before external imports/execution. The first attempt exposed an investigator adapter error: instance names instead of required class names. Its 11 setup ERROR results remain in [the attempted-run record](trainparity-setup-errors.json); no accumulation workers ran. Amendment `df967a4` fixed only the import targets and removed unused instances before the second attempt. Do not merge this setup attempt into a tool detection-rate estimate or silently omit it from workflow history.

TrainParity source revision is `79a1fc23c36073f09b97aa99c2daad7e96e7bce4`, package version `0.1.1.dev0`. The [scoped review manifest](../competitors/trainparity-execution-review.json) identifies every downloaded source file by URL/hash and bounds the reviewed execution path. Original source/license remain in ignored local reference storage, not vendored into Atlas. The [separate environment](checker-environment.json) records Python 3.11.17, PyTorch 2.13.0+cpu, NumPy 2.4.6 and hash-pinned dependencies. Atlas's Python 3.12/PyTorch 2.14 lock is unchanged.

## Recorded results

Raw [results](trainparity-controls.json) retain every result dictionary, all differences, version/settings and source identities. All cases use original float64 linear-model fixtures, one logical MSE/SGD update, fixed seed 1701, and rtol=1e-10/atol=1e-12. The seven-example case splits into four and three; the balanced case has six examples. These tolerances are numerical control settings, not full-training or scientific equivalence margins.

|Control |Outcome |First reported phase/path or reason |
|---|---|---|
|Balanced accumulation with explicit accounting |PASS |All declared update observations match |
|Unequal accumulation with explicit accounting |PASS |All declared update observations match |
|Balanced accumulation without explicit accounting |PASS |`loss_normalization_captured=false` remains visible |
|Unequal means instead of explicit global accounting |FAIL |`loss_accounting.denominator`, missing candidate field |
|Clip partially accumulated gradients |FAIL |`gradient.weight`, tensor value |
|Optimizer step per microbatch |FAIL |`gradient.weight`, type/None mismatch |
|Scheduler step per microbatch |FAIL |`optimizer.param_groups[0].lr`, value |
|Omit final microbatch |FAIL |`loss_accounting.denominator`, value |
|Erase gradients before observation |FAIL |`gradient.weight`, type/None mismatch |
|Unsupported string metadata in default batch splitter |ABSTAIN |Unsupported batch leaf |
|Zero microbatch count |ERROR |Plan validation before worker execution |

The wrong-mean result first identifies an accounting-schema difference, not an isolated numerical estimate of normalization error. It must not be advertised as such. The known fault still changes the intended normalization, but phase localization is explicitly an observed divergence, not a causal diagnosis. The scheduler fault emits the expected PyTorch warning about stepping the scheduler before the optimizer; this was retained, with machine paths excluded from the portable report.

Nine ordinary checks launch three fresh workers each, the unsupported split launches one, and the invalid plan launches none: 28 worker launches. Summed check-call wall time is 120.18 seconds; summed outer per-control wall time including checker-parent startup/imports is 143.71 seconds. These are observations of this small diagnostic workflow, not measured Atlas planning overhead, portable checker latency, a GPU-memory benchmark or a fair timing comparison against earlier PyTorch 2.14 probes. Review/adaptation/dependency acquisition time is outside these windows; the chronological log records that work.

## What changed in the recommendation

The previous source-only assessment now has a direct compatible-environment replication of selected accumulation controls. Existing checks catch named-gradient, state/update-clock, accounting and omitted-tail problems. The initial strong-baseline gate's basic executable adapter is therefore satisfied for these narrow CPU cases.

The no-accounting positive also illustrates an integration obligation: a PASS can coexist with absent normalization evidence. A caller requiring explicit denominator evidence must inspect that flag and the declared relation. The tool does not claim to infer arbitrary loss semantics, and this positive case's balanced mean is algebraically justified. Missing evidence is not automatically a false acceptance.

No independently labeled held-out corpus, full-training outcome, activation-checkpoint selection, sample-coverage API, resume API, local model, GPU allocation or complete Atlas policy was tested. All expected labels are investigator-authored. Eleven designed outcomes do not estimate field sensitivity, specificity or safe-decision probability. These results provide stronger evidence of overlap, not a new checker contribution.
