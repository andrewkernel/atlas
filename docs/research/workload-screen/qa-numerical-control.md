# Reduced-batch BERT capture and numerical boundaries

The [frozen protocol](qa-numerical-protocol.md) at 3181ea7 runs two owned workers and ten fresh-model one-update executions on three real QA features. Both workers complete with verified group cleanup. [Raw results](qa-numerical-control.json), [authored execution/capture adaptation](../../../benchmarks/qa_numerical_control.py). The pinned existing TrainParity capture/comparison implementation runs in the compatible QA stack; no new checker is implemented.

Each condition uses an identical full-batch reference, identical repeat, native checkpointing, microbatch 2+1 and the combined plan. All 37 inspected Dropout modules use either .1 or zero as declared. The zero condition is a modified deterministic-positive diagnostic, not a repair claimed to preserve the original stochastic method. All initial model/group/input/start-RNG identities agree. Snapshots observe 199 parameter and gradient tensors, two buffers, 199 AdamW steps and both 199-tensor moment families; RNG is reported separately and excluded from the comparison predicate.

| Dropout | Comparison to three-feature reference | Existing predicate | Descriptive gradient relative L2 | Descriptive model max absolute difference |
|---|---|---|---:|---:|
| .1 | repeat | PASS, captured tensor bytes equal | 0 | 0 |
| .1 | checkpoint | PASS, captured tensor bytes equal | 0 | 0 |
| .1 | microbatch / combined | both FAIL | .790479 | 6.00815e-5 |
| 0 | repeat | PASS, captured tensor bytes equal | 0 | 0 |
| 0 | checkpoint | PASS, captured tensor bytes equal | 0 | 0 |
| 0 | microbatch / combined | both FAIL | 1.89633e-6 | 7.95641e-6 |

The frozen predicate remains rtol 1e-5 / atol 1e-7, equal_nan false, with the previously audited existing tensor/scalar conventions. No tolerance is relaxed after the result. With dropout .1 the first reported difference is a gradient. With dropout zero the first difference is a post-update position-embedding parameter, while overall gradient error is much smaller: maximum 1.30851e-7. AdamW first/second moments and steps are separately observed. This is consistent with finite-precision changes carrying through an adaptive update; the causal contribution of each arithmetic operation has not yet been isolated. It does not establish a changed mathematical objective, a domain-quality failure or an appropriate scientific margin. Investigate rather than converting a finite numerical FAIL into an unsupported scientific conclusion.

Within each condition, checkpointing and its corresponding microbatch composition yield matching recorded metrics. No new emergent interaction is demonstrated. Both separately captured .01 weight perturbations are detected at `model.qa_outputs.weight`, establishing sensitivity for this real-model capture path.

Actual-worker sampled RSS peaks around 4.98/4.97 GiB and job commit around 8.06/7.73 GiB; these are different measurements. Each worker's outer lifetime is about 36.3/36.4 seconds. Individual full-value captures take .398–.848 seconds here, excluding later comparison/diagnostic arithmetic. Those values are not a fair workflow-speed comparison. Snapshots remain in RAM and are released; multi-GB state files are not written. Job membership/cancellation guards remain soft trusted-code supervision, with no hostile-code or hard-quota claim.

The three-feature repeat is bitwise consistent in captured tensors. This does not repair or quantify the differing hashes observed for logical batch 32 in the separate resource screen. Shapes, horizon and observation conditions differ. Small-batch results cannot supply an unavailable larger reference or certify full fine-tuning, QA quality, distributional equivalence or Atlas policy advantage. The main result strengthens the existing-tool comparator and defines the next numerical-diagnosis question.
