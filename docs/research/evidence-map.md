# Evidence map for the current recommendation

Use this table to distinguish measured results, source claims, derivations and unfinished hypotheses. It is not a scalar confidence score. Exact source/environment identities are in each linked record.

|Claim |Evidence type and scope |Supported conclusion |Unsupported extension |
|---|---|---|---|
|Loss/state/RNG/clock hazards occur |[Original CPU fixtures](experiments/README.md), generated finite partitions |Specific implemented relations can fail despite a plausible memory-saving intent |Field prevalence, universal detection or new-bug discovery |
|A known mutable-region hazard can be repaired |[Scoped buffer control](experiments/state-replay-probes.md); [FairScale source](competitors/fairscale.md) |A conservative exclusion is not proof no specialized repair exists |Generic buffer transactions or arbitrary side-effect safety |
|Native plans lower allocation |[Eight synthetic CUDA trials](experiments/cuda-memory-probe.md), one physical GPU |This fixture has a memory/time tradeoff |All models, all GPUs, energy saving or expanded access |
|Combined native plan trains on real data |[Six five-epoch CNN pilots](experiments/fashion-pilot.md), three paired seeds |32.7% lower allocated peak here, longer update time, finite endpoint differences |Conclusive equivalence, a rescued real OOM run or planner superiority |
|The pilot is repeatable locally |[Two lowest-seed replays](experiments/fashion-repeatability.md) |Exact final hashes and epoch metrics within each recorded arm/environment |Extra independent seeds or future-platform determinism |
|Resource success depends on execution window |[450 MiB cap](experiments/cuda-budget-probe.md) |Two plans complete one update then fail later |A first-step peak is a whole-run guarantee |
|Verification pattern can retain more memory |[Streamed/deferred analogue](experiments/activation-liveness-probe.md) |Same observed first-update numerics can have different peaks/capped outcomes |An executed TrainParity benchmark or an unfixable tool limitation |
|Generic checking has strong prior art |[TrainParity source](competitors/trainparity.md), [direct CPU controls](experiments/trainparity-controls.md), [TrainCheck](papers/traincheck.md) |Selected existing accumulation checks execute and match designed controls in a compatible stack |Independent field accuracy, complete suite coverage, GPU performance or Atlas superiority |
|The planner field is crowded |[Rockmate family](competitors/rockmate-family.md), [HiLoRe](papers/hilore.md), [paper notes](papers/README.md) |Broad first-of-kind framing is unsupported |Proof no narrower empirical contribution exists |
|Existing planning code can execute here with adaptation |[Rockmate-family GPU controls](experiments/rockmate-compatibility.md), exact source/environment identities and retained failures |Pure fixed-batch plans include repeated forwards with zero recorded update errors |Published-suite replication, arbitrary-model support, total-memory guarantee or Atlas advantage |
|Preparation can change scientific start state |[Constructor state hashes and guard control](experiments/rockmate-state-control.md) |Observed CUDA RNG advances even when pure update comparisons match; a scoped existing guard restores it |All profilers, every state dimension, arbitrary rollback or a changed outcome distribution |
|The initial power estimate was too small |[Integration/simulation](experiments/equivalence-calibration.md) |36 pairs under the illustrative normal estimated-SD model, versus the earlier25 approximation |A domain margin, observed ML SD or universal seed count |
|Inference assumptions matter |[Constructed mixture control](experiments/statistical-assumption-audit.md) |Same mean/variance does not ensure finite-sample t-interval validity |An estimated real training failure probability |
|Selection/stopping alter error guarantees |[Claim-selection derivations](statistical-claim-selection.md) |Define claim family, unit and stopping before inference |All finite numerical checks are p-value tests |
|Model size is insufficient for fit |[Metadata/cache arithmetic](local-model-capacity.md) |Components, context, cache and runtime must be separated |Observed inference speed, accuracy, fit or benefit |
|A standalone Atlas adds value |Proposed [strong-baseline study](evaluation-estimands.md) only |A falsifiable next question |A completed primary study or established novelty |
|Scientific conclusions are preserved |No domain conclusion validated; no independent expert labels |Keep this dimension unverified |A preserved-science badge from gradients or accuracy |

The [updated recommendation](14-updated-recommendation.md) follows from the combination: engineering feasibility has direct evidence; distinct incremental research value and useful expanded access remain unresolved.

[Reuse/accumulation controls](experiments/planner-reuse-control.md) additionally show that a supported streaming execution can fail under a deferred verification pattern on the tested wrapper. The failure persists without planned recomputation and is a loud runtime error, not evidence of new emergent-transform behavior or unsafe scientific acceptance. A later [shared-stack integration](experiments/interop-controls.md) now executes both sources and a competent streaming capture adaptation; no primary policy comparison is implied.
