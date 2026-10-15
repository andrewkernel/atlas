# Scientific-fidelity framework

## Fidelity is a vector of scoped claims

Atlas must not output a single “scientific confidence” score. For each candidate it reports structural/methodological, mathematical, numerical, statistical and conclusion dimensions separately. Resource savings are separate measurements. A property entry contains: claim, scope, reference and candidate identities, assumptions, evidence IDs, comparison rule, result, limitations, and reviewer disposition if applicable.

| Dimension | Example obligation | Automation and limits |
|---|---|---|
| Structural / methodological | Same trainable parameters, data split/order, masks, optimizer hyperparameters, logical batch and update schedule | Hash/config diffs and runtime events can establish declared fields; arbitrary code semantics and hidden data dependencies remain outside scope |
| Mathematical | Accumulation computes the same intended gradient estimator | Conditional derivation plus adapter preconditions; Python syntax matching is not proof |
| Numerical | Outputs, loss numerator/denominator, gradients, parameters, buffers, optimizer state and counters meet declared comparisons | Empirical finite-input checks; record absolute/relative error, dtype and nonfinite values; no universal implication |
| Statistical | Prespecified outcome difference lies inside scientifically justified bounds | Repeated full runs, estimand and valid interval/test; small-probe parity is insufficient |
| Conclusion | The experiment still tests the original hypothesis and supports its claimed inference | A human must approve the protocol mapping and interpret the result; formalizing a narrow conclusion predicate only automates that predicate |

## Two independent classifications

An individual check has `PASS`, `FAIL`, `ABSTAIN`, or `ERROR`. `ABSTAIN` means insufficient evidence or unmet observability; `ERROR` means execution/measurement failure. Neither is a pass. These outcomes resemble existing scoped-checking practice in [TrainParity](https://github.com/intelland/TrainParity); they are not claimed as a new invention.

An evidence claim has one of these statuses:

| Status | Required support | Example |
|---|---|---|
| Verified under explicit conditions | Stated predicate/domain, all necessary recorded conditions, reproducible check or justified derivation | Optimizer step counter advanced exactly once over this observed window; algebraic accumulation identity under declared assumptions |
| Empirically supported | Protocol, measurements and uncertainty/tolerance, finite scope | Gradient parity over 40 generated CPU cases; full-run equivalence interval when justified |
| Methodologically modified | Concrete approved deviation from reference protocol | Replacing AdamW with Adafactor; shorter context; changed trainable parameter set |
| Unverified | Missing evidence or unsupported scope | Full-run conclusions after only a ten-step probe |
| Known incompatible | A demonstrated or logically established conflict with a required contract property | BatchNorm training statistics differ when the contract requires the original batch behavior |

“Verified under conditions” must never be shortened to “verified scientifically.” A tested tensor equality is an observed equality, not proof for every input. The planner uses only the dimensions required by a specific contract, and reports unchecked dimensions in every accepted plan. A methodologically modified plan can be considered only under a new explicit contract; it is never silently promoted into the original strict set.

## Conditional accumulation derivation

Let a logical batch have loss L = (sum_i l_i(theta))/D, where D is the total declared sample weight or valid-target count. If each microbatch j contributes numerator S_j and denominator D_j with D = sum_j D_j, compute each backward contribution as S_j/D and sum the gradients before exactly one optimizer update. In exact arithmetic, linearity gives sum_j grad(S_j/D) = grad(L).

This requires all of the following:

1. The same samples, targets, masks and weights occur exactly once in the logical update, with the same loss definition. An auxiliary regularizer is accounted for once with its intended coefficient.
2. Parameters remain fixed within the logical batch. Optimizer, scheduler, EMA, clipping and weight-decay clocks match the reference update semantics. Clip the completed gradient, not each partial gradient.
3. Forward/loss computations are separable across the partition, or their cross-example operations have an explicitly equivalent implementation. BatchNorm, batchwise contrastive negatives, batch-hard mining and mixture routing can violate this.
4. Randomness has the relation the contract requires. Same seed does not ensure the same dropout masks under changed shapes or RNG call order. Distributional similarity is weaker than pathwise parity.
5. Partial windows and variable valid-target counts use the correct actual denominator. Equal averaging of per-microbatch means is valid only under matching denominators.

Floating-point summation order can still differ. [PyTorch numerical-accuracy documentation](https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html) explains why mathematically equivalent batched and sliced computations need not match bitwise. In distributed work, rank-averaging and token normalization require an additional derivation; distributed execution is excluded from the MVP.

The [Hugging Face accumulation correction](https://huggingface.co/blog/gradient_accumulation) is a concrete real-world precedent for denominator errors. The session's own unequal-partition probe independently exhibits one.

## Checkpointing obligations

Treat an eligible region as a recomputable function whose externally visible effects are accounted for. Model buffers, Python caches/counters, hooks, custom generators, autocast context, device transfers and custom autograd behavior are part of the analysis. No training BatchNorm or unknown mutable module inside a strict eligible region in the first implementation. Excluding such regions is conservative, not a claim that no correct specialized implementation exists.

Use `use_reentrant=False` explicitly, preserve RNG state, record checkpointed module paths and region input/output shapes. The [checkpoint API](https://docs.pytorch.org/docs/2.14/checkpoint.html) warns about recomputation differences and limited device RNG handling; its default metadata comparison is not value verification. Custom generators and side effects need additional checks or abstention.

**Observed local counterexample:** the [BatchNorm checkpoint probe](experiments/cpu-semantic-probes.json) has zero gradient difference but running-mean error approximately 0.0361 and evaluation-output error approximately 0.0262. This motivates checking buffers before and after a complete update and evaluation. It does not estimate how often that failure occurs in real projects.

## Numerical comparisons

Save corresponding observations at declared phases: initial state; loss accounting; pre-clip gradients; post-clip gradients; post-update state; post-scheduler state; evaluation. A missing observation yields `ABSTAIN`, not equality of absent fields. Compare keyed tensors by semantic name, shape and dtype; treat missing/extra parameters and changed trainable flags as structural differences. Exclude neither buffers nor optimizer counters.

Use an explicitly declared elementwise criterion |a-b| <= atol + rtol*|a| for empirical comparisons, with baseline as the named reference. Also report max absolute error, relative L2 error (with a defined zero-reference rule), gradient cosine where norms are nonzero, and nonfinite counts. Cosine agreement alone can conceal scaling errors. ULP measures are optional for matching types, not a substitute for a scientific margin.

For the small float64 controls in this repository, 1e-12 is a **fixture discrimination threshold**, chosen to distinguish numerical rounding from large injected differences. It is not a universal training tolerance. A future GPU/FP32 contract needs separate calibrated numerical bounds and positive controls. Never tune the tolerance after seeing a candidate's result to make it pass. Reference nondeterminism must be measured before attributing a difference to the transformation.

## Statistical and conclusion evidence

Let d_s = outcome_candidate,s - outcome_reference,s over independent paired seeds. Testing whether E[d_s] is practically negligible requires a scientifically defended margin Delta and a fixed analysis protocol. [TOST](https://doi.org/10.1177/1948550617697177) at alpha .05 corresponds to a 90% confidence interval wholly inside (-Delta,+Delta) under the model's assumptions. A nonsignificant ordinary difference test does not establish equivalence. A mean-equivalence result does not establish equal variance, tail behavior, subgroup effects, calibration or identical trajectories.

If the original scientific question compares methods A and B, evaluate both with and without the execution change: the relevant interaction is (A-B)_candidate - (A-B)_reference. Testing only one model's accuracy cannot establish ranking or effect preservation. Declare which outcome, subgroup and direction matter before running the study; domain judgment is necessary to choose the margin and interpret inference.

Atlas can automate data collection and a prespecified test. It cannot infer the author's scientific intent reliably or make an underpowered experiment conclusive. See [experimental design](07-experimental-design.md) for separation of pilots and confirmation.

## Architecture consequence

Each strategy declares preconditions, affected dimensions and required checks. Compositions must be checked as compositions; passing isolated transformations is insufficient. Evidence is bound to code, contract, plan, environment and input identities. Any of these changing invalidates reuse unless an explicit reusable scope covers the change. The LLM cannot edit the contract, set tolerances, waive unknowns, or approve its own patch.
