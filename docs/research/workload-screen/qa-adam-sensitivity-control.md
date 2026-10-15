# Selected-component AdamW sensitivity diagnosis

The [frozen follow-up](qa-adam-sensitivity-protocol.md) at f4e24be completes both three-feature dropout-zero executions and the declared CUDA vector replays. [Raw record](qa-adam-sensitivity-control.json), [execution/replay source](../../../benchmarks/qa_adam_sensitivity_control.py), [scoped optimizer review](qa-optimizer-review.json). All retained process identities and group members exit; actual RSS peaks about 5.34 GiB and job commit about 8.08 GiB. The original supervisors and their earlier byte identities remain unchanged.

Ten components are selected by the declared largest-post-update-difference rule, at most three from a named parameter. They are diagnostic selections from one paired update, not independent samples or held-out evidence. Their scalar initial values, gradients, post-update values and first/second moments are retained. Native CUDA FP32 vector AdamW with the original hyperparameters reproduces **all 60 selected parameter/moment values at zero ULP distance**: ten components × two executions × three fields. Thus the replay isolates these selected updates given the observed gradients. It does not isolate the source of the gradient rounding or prove an entire-model causal explanation.

| Fixed-gradient counterfactual | Maximum selected pair difference |
|---|---:|
| AdamW eps 1e-8, original | 7.95641495e-6 |
| AdamW eps 1e-6 | 1.26690754e-7 |
| AdamW eps 1e-4 | 1.27401377e-9 |
| SGD, same lr/decay groups | 1.27408582e-13 |

Equal-gradient replays match exactly in all four settings. Changing epsilon or optimizer changes the method: these are interventions for diagnosis, **not** fidelity-preserving repairs or full-training recommendations. The original tolerance and training hyperparameters are unchanged.

## What the largest difference actually denotes

The largest component is end-head `qa_outputs.bias[1]`, initialized to zero. Its reference/candidate clipped gradients are about 4.95478e-9 and 7.07825e-10, respectively; AdamW moves it to about -9.93952e-6 and -1.98311e-6. Small absolute gradients near the optimizer epsilon can therefore create the observed parameter difference through the installed adaptive update. The first-step ideal real-arithmetic expression is `p*(1-lr*decay) - lr*g/(abs(g)+eps)`; it is explanatory, not a bitwise substitute for native floating-point operations.

Each QA head's bias adds the same scalar to all of its position logits. In real arithmetic, `softmax(z+c)` equals `softmax(z)` because the common exponential factor cancels. A strict parameter predicate can therefore detect a change in this nuisance direction without demonstrating a probability or answer change. Finite-precision behavior still needs its own observation. Eight other selected components are ordinary weights, so the model difference cannot simply be dismissed as entirely bias invariance. No post-update functional/quality study has been completed by this replay.

The raw selection magnitude uses float32 subtraction. The table independently subtracts exported float32 values in Python double arithmetic; their tiny last-bit reporting difference is not an additional training discrepancy. The [original full-value comparison](qa-numerical-control.md) remains FAIL under its frozen predicate. A different functional relation must be declared and adjudicated prospectively, shared with the baseline, and checked on its own scope rather than used as an after-the-fact waiver.
