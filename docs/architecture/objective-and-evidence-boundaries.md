# Objective, execution and evidence boundaries

This is a proposed interface refinement grounded in the [QA resource screen](../research/workload-screen/qa-resource-pilot.md), [composite-objective controls](../research/experiments/composite-objective-control.md) and earlier [reference-adequacy control](../research/experiments/token-reference-control.md). It is not an implemented schema, general semantic analyzer, proof system or novel checker. [ADR 007](../research/decisions/007-benchmark-first.md) limits expansion to the evaluation package until the distinct-treatment gate is met.

## Four identities before comparison

Keep these separately versioned and linked:

| Record | Contains | Does not establish |
|---|---|---|
| Scientific intent | Author-stated objective, sampling unit, method constraints, outcomes and acceptable relation | That the reference implements it |
| Reference protocol | Preprocessing, model/data bytes, loss semantics, initialization, update/evaluation recipes | That its outcome is scientifically correct |
| Execution recipe | Mechanisms, scheduling relation, backend, worker profile, preparation/start boundary | That an observer ran that exact relation |
| Evidence | Actual observer/executor versions, observations, predicates, scope and dependencies | Unobserved inputs, horizons or scientific conclusions |

A source hash validates identity under its byte convention, not author intent. A comparator's PASS concerns its specified relation and captured fields. An internally consistent wrong objective remains possible. Independent review is a separate incomplete gate, not a field a policy can mark complete for itself.

## LogicalUpdateContext and loss terms

Refine the proposed adapter with `prepare_logical_batch(batch, semantics) -> LogicalUpdateContext` and `loss_terms(state, microbatch, context) -> LossTerms`. For the initial supported class, preparation computes denominator metadata from the complete CPU logical batch before any microbatch backward. It does not run a full GPU forward to discover the denominator. Each term declares:

- Stable term ID and objective version, differentiable summed numerator, coefficient and denominator ID.
- Denominator value, unit, provenance, mask/ignore/clamp rules, and whether it is independent of trainable parameters under the declared method.
- Required scope: logical batch, dataset, or update. Auxiliary regularizers declare their update frequency; they are not silently multiplied once per microstep.
- Empty-local and empty-global policies. Zero local count with positive global count can contribute a zero sum. A globally undefined mean requires an author-defined method rule or an explicit unsupported outcome.

For metadata-constant positive denominators, the supported separable objective is

`L = sum_t alpha_t * (sum_j S[t,j]) / D[t] + R`.

Each microbatch contributes `sum_t alpha_t * S[t,j] / D[t]`. Do not additionally multiply an already globally normalized contribution by its example fraction. A common scalar denominator can represent a correctly authored composite expression, but its semantic context and term-specific coefficients must remain explicit. A scalar interface is not fundamentally incapable of this; existing capture/checker tools can consume the same correct adapter.

For parameter-dependent D, `grad(S/D) = grad(S)/D - S*grad(D)/D^2`. Detaching D changes that derivative while preserving the forward value. The initial executor rejects unsupported prediction-dependent normalizers; the finite quotient-rule control does not supply a generic streaming implementation. `requires_grad=False` alone does not prove parameter independence: an adapter may already have detached a model-derived quantity. Declared stop-gradient behavior is part of the method, not an automatic repair.

Retain feature examples, questions, valid endpoint labels, tokens, microsteps, optimizer attempts and successful updates as different counters. A token count cannot substitute for examples. A class-weighted cross-entropy denominator may be a sum of target weights; it must not inherit the unweighted valid-count rule without review. [PyTorch documents](https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html) the reduction/ignore semantics; the installed implementation and declared options determine the actual method.

## QA-specific method facts

The frozen resource adapter has 512 sampled questions, 518 features, 270 source articles and four CLS-target windows. Training samples **features**. Questions with more windows receive more feature occurrences; uniform-question weighting would be a different method. Neither the question count nor article count is the current loss denominator.

Context window advance 128 and overlap 128 are distinct. Tokenizer revision, normalization/offset convention, query truncation, feature ordering, answer selection and out-of-window labels are scientific-method inputs. Do not treat a preprocessing change as a memory-only transformation. The local protocol deliberately does not claim an exact TensorFlow training reproduction.

The inspected builtin QA implementation clamps endpoint labels into [0, sequence_length], ignores the upper endpoint sentinel, and averages start/end means separately. Negative -100 is therefore clamped to position zero, unlike a common language-model ignore convention. All current pilot labels are in [0,383], including CLS zero, so both head denominators equal feature count and example-fraction accumulation is justified for that objective. Future ignored-label adapters require separate global head counts; never generalize this special case from one observed batch.

## Preparation, scientific start and worker profile

Preparation may construct models, map legacy checkpoint names, plan, trace, allocate, warm caches and consume random numbers. Scientific start follows verification/replay of the declared parameter/buffer, optimizer/counter and RNG relation. Hash the observed initialized state after loading; configuration text alone cannot reveal a silently missed encoder key. Record preparation cost independently but include it in user-facing workflow cost.

A registered worker profile binds an allowed interpreter/environment and reviewed adapter set. The Python 3.12 CPU controller lock is separate from the Python 3.11 QA/checker-compatible worker lock. Transport/schema compatibility is not library compatibility. Never accept arbitrary interpreter paths, import strings or shell commands from an API to bridge the two. Record the exact installed versions and selected file/build identities; a matching package version alone is insufficient.

The Windows ownership control demonstrates that a launch PID can name a venv redirector rather than the interpreter. Future supervision must verify the actual worker and process boundary before training, apply guards to that boundary and verify cleanup before a subsequent trial. Test failed assignment, startup timeout, cancellation, natural exit and descendant creation. A job object can manage a process group; it is not a hostile-code sandbox, and RSS sampling is still a soft observation rather than a hard memory limit. The original QA supervisor remains a frozen defective measurement example and must not be reused for further training.

## Evidence applicability and invalidation

Each evidence node needs: producer/observer/executor/predicate IDs and versions; input/code/config/contract/data/environment identities; relation and field scope; phase/horizon; assumptions; status/reason; dependencies; units and window. A dependency must resolve to immutable bytes. Canonical JSON rules apply only to the declared JSON schema; do not silently normalize tensor dtype, data Unicode, scientific strings or preprocessing bytes to manufacture identity.

| Observation | Permitted claim | Still unavailable |
|---|---|---|
| Eight QA updates complete | Segment completion under observed settings | Full fine-tuning or QA quality |
| CUDA allocated peak is below a threshold | Allocator observation in that window | Exclusive physical residency or another card's fit |
| Full-batch reference has differing repeated hashes | Bitwise repeatability is not established | Magnitude, tolerance failure or root cause |
| Small-batch differential check passes | Captured relation for that shape/window | An unavailable full-batch reference's relation |
| An emergency host-pressure stop occurs | Trial stopped for the recorded condition | Intrinsic model infeasibility or CUDA OOM |
| Existing comparator matches captured tensors | Its actual predicate/fields pass | Reference adequacy, uncaptured state or full-training conclusions |
| Worker has a nonfinite loss but finite gradients | Loss-health failure is observable independently | Permission to ignore loss health because gradients look finite |

Any change to preprocessing, logical-batch semantics, masks, update clocks, RNG relation, backend, execution scheduling, observer, predicate or relevant environment invalidates incompatible evidence. Reuse requires an explicit applicability check, not a matching model name. Missing/stale evidence is unknown, never a cached PASS.

## Budget and reference availability

Record nested cost windows without double counting: supervisor end-to-end contains child lifetime; child lifetime contains preparation, updates, capture, hashing and writes. A sum of per-update timers omits other work. Report inclusive and exclusive costs with the containment relation. Observer/capture overhead is user cost even when excluded from a controlled training-kernel estimand.

Physical VRAM, Torch allocation/reservation, board-wide telemetry, per-process working set/private commit and host available RAM are different measurements. Windows fallback policy/residency is unknown in the QA screen. Small launcher RSS is an observer-scope error, not extraordinary model efficiency. Preserve invalid raw observations with an amendment; a derived report must not promote them to valid worker measurements.

If a full reference is unavailable, keep that boundary explicit. A smaller reference, external artifact, mathematical precondition or reduced deterministic control can provide its own scoped evidence, not an invented full-scale oracle. An external artifact needs provenance and compatibility review; its digest does not prove scientific correctness. Resource-only/short-operational contracts may report unverified statistical/conclusion dimensions when those are not required. They must not be presented as stricter contracts that passed.

## Executed boundary refinements

The [Windows job control](../research/workload-screen/windows-job-control.md) now validates a scoped suspended-launch/assignment/resume sequence, retaining the earlier denied assignment attempts. The [three-feature BERT integration](../research/workload-screen/qa-numerical-control.md) uses that boundary for real training and existing full-value capture. These are benchmark infrastructure, not a production API/sandbox. The original resource supervisor remains historical and must not be reused.

Cancellation can produce exit code zero on observed job-handle close. A completed scientific run therefore needs a complete identity-bound worker payload, update/phase obligations and no conflicting supervisor stop reason; process success alone is insufficient. The BERT control also shows that dropout-zero microbatch gradients can be close while an adaptive post-update parameter exceeds a frozen numerical predicate. Keep predicates per required field/phase and report the first observed divergence without promoting it to a root cause or a scientific-quality conclusion.

[Selected optimizer replay](../research/workload-screen/qa-adam-sensitivity-control.md) now reproduces the observed selected updates/moments exactly, but the largest difference lies in a QA-head constant-logit bias. Strict parameter equality and a functional relation are different contracts. Deriving an invariance does not waive an existing required tensor predicate; a new relation needs prospective scope/assumptions and equal availability to the baseline. Other weights also differ, so no blanket functional-equivalence result follows.
