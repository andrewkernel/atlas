# Components and interfaces

Proposed contracts below describe implementation targets, not existing public APIs. Use Python protocols/dataclasses for core types; Pydantic at configuration/API serialization boundaries with strict validation and unknown fields forbidden.

| Module | Responsibility | Input → output | Failure behavior |
|---|---|---|---|
| `analysis/` | File inventory, hashes, advisory AST findings | Source snapshot → findings with locations | Syntax error/unsupported analysis is explicit; never import code for static scans |
| `experiments/contracts.py` | Experiment, budget and allowed-change schemas | JSON → validated immutable contract | Reject unknown fields, inconsistent denominators/budgets and unsupported schema version |
| `experiments/adapter.py` | Trusted training interface | Registered adapter ID + config → logical updates and observations | Missing required observations cause abstention |
| `profiling/` | Time, allocator and host/device telemetry | Authorized worker run → profile with windows/units | Device unavailable, sampling unavailable and OOM are distinct |
| `optimization/` | Registry, preconditions, enumeration and ranking | Contract + profile → finite plans + exclusions | Return no justified plan; never loosen contract |
| `fidelity/` | Dimension-specific checks and aggregate decision | Reference/candidate observations + contract → evidence vector | Pass/fail/abstain/error per check, not a guessed score |
| `experiments/runner.py` | Process lifecycle and artifacts | Approved immutable run spec → bounded worker result | Timeout/process-tree termination; preserve partial logs |
| `inference/` | Optional local advice | Sanitized context + allowed IDs → proposals | Parse/model failure falls back to deterministic ranking |
| `api/` | Later local transport | Typed HTTP payloads → core operations | No arbitrary import strings or shell commands from requests |

## ExperimentAdapter v0

Require these capabilities, with serialization implemented by a registered factory:

```python
class ExperimentAdapter(Protocol):
    def describe(self) -> WorkloadSemantics: ...
    def build(self, config: ExperimentConfig) -> TrainingState: ...
    def batches(self, replay: ReplayManifest) -> Iterator[Batch]: ...
    def loss_terms(self, state: TrainingState, batch: Batch) -> LossTerms: ...
    def checkpoint_regions(self, state: TrainingState) -> list[RegionDeclaration]: ...
    def observe(self, state: TrainingState, phase: Phase) -> StateObservation: ...
```

`LossTerms` supplies differentiable summed numerators, per-term denominator units/masks and auxiliary-term semantics. The [objective/evidence amendment](objective-and-evidence-boundaries.md) refines this sketch with `prepare_logical_batch` and an explicit `LogicalUpdateContext` passed to microbatch loss construction. The initial executor supports reviewed metadata-constant positive global denominators; it must not infer a universal weighting rule from one scalar mean. The executor owns logical update boundaries; custom loops remain unsupported until equivalent phase observations are available.

`TrainingState` includes model parameters/buffers, optimizer, optional scheduler/scaler/EMA, counters and RNG recipe. `Batch` includes stable sample IDs, content or augmentation digest where available, target mask and count. Stable IDs alone do not prove matching content.

Represent absent, None, zero and nonfinite gradients distinctly, keyed by parameter identity. Record optimizer attempts and actual completed updates separately; scheduler/EMA declare their intended clock. An optimizer returning None does not reveal whether it ran. [CPU controls](../research/experiments/update-clock-probes.md) motivate these fields without claiming a generic observer is implemented.

`WorkloadSemantics` declares batch coupling, stochastic modules/generators, side effects, dynamic shape bounds and unsupported features. Declarations are trusted inputs with observed sanity checks, not automatically proved facts.

## OptimizationStrategy v0

Expose `id`, `version`, `affected_dimensions`, `check_preconditions(contract, semantics)`, `enumerate_parameters(config,budget)`, `apply_config(config,params)` and `required_checks`. Initial strategies are `microbatch_accumulation` and `activation_checkpoint`. An immutable `Plan` contains ordered operations, config diff, reasons, precondition evidence and expected affected properties. Apply only in a worker's copied config/model construction; source mutation is excluded.

Include the required forward/backward scheduling relation and supported live-forward-context pattern in registered execution metadata. A verifier must execute that relation or explicitly report a different one. The [reuse control](../research/experiments/planner-reuse-control.md) passes a streaming plan but fails deferred-forward patterns on the tested existing wrapper, even without planned recomputation. Do not turn a verification executor's different schedule into a verdict about an unexecuted target plan. Streaming is eligible only when the objective/accounting supports it; opaque or nonseparable cases remain unsupported.

The proposed planner includes identity, deduplicates candidates and rejects incompatible compositions. Predicted memory is heuristic. A future ranking policy must specify its exact next-action function and cost budget; “conservative bottleneck rules then observed cost” is not an implemented algorithm or a distinct experimental treatment. [ADR 007](../research/decisions/007-benchmark-first.md) requires this specification before primary comparison. A proposed final tie-break uses strategy IDs and canonical parameter JSON. Cache only matching identities and applicable evidence scopes.

## Verification interface

Checks consume named phase snapshots, not arbitrary logs or LLM prose. Compare baseline self-consistency first. A `CheckResult` contains predicate ID/version, dimension, outcome, status, scope, assumptions, evidence references, tolerance/test specification and failure location. “First observed divergence” is an observation, not a root-cause diagnosis.

Aggregate eligibility requires all contract-required checks to pass and no incompatible preconditions. Unchecked statistical/conclusion dimensions remain explicitly unverified when not required by a short-probe contract. No process that generated a candidate may authorize a waiver.

## Tracking and persistence

Use Atlas SQLite tables `experiments`, `contracts`, `plans`, `runs`, `checks`, `artifacts`, `approvals`, `events` with schema version and foreign keys. One supervisor writes transactions; workers append to private JSONL files and hand off results. MLflow receives selected metadata, metric series and artifact references via an adapter; never write directly into its internal tables. Tracking failure must not destroy evidence: leave a pending sync event and keep local JSON. Large artifacts are local and ignored by Git; small sanitized aggregate results may be committed deliberately.

The [token-reference control](../research/experiments/token-reference-control.md) illustrates why LossTerms must declare denominator units and mask/ignore semantics, not only a number. Internally consistent reference/candidate adapters can agree on a different objective from the stated token-weighted intent. Check those method facts against the contract independently and give the composed baseline the same information.
