# Experiments actually executed

## Scope and environment

On 2026-10-15 at 04:43:58 UTC, ran original offline CPU probes with Python 3.12.10 and PyTorch 2.14.1+cpu on Windows 11, one PyTorch thread, deterministic algorithms enabled. No external datasets/models/checkpoints or research-repository code were executed. A12 GB RTX 3060 is installed on the host, but these probes use no CUDA.

Code: [`benchmarks/semantic_probes.py`](../../../benchmarks/semantic_probes.py). Raw small structured evidence: [CPU results](cpu-semantic-probes.json). The result records the source SHA256, seeds, environment and measurement scope. Dependency versions/hashes are in the root lock; the [dependency snapshot](../dependency-snapshot.json) records metadata and untested optional components.

## Observations

| Controlled comparison | Maximum absolute difference observed | What it demonstrates in this fixture |
|---|---|---|
| Correct sample-weighted accumulation, unequal3+4 partition | Gradient 2.22e-16 | Algebraic control matches to floating-point rounding |
| Equal weighting of unequal microbatch means | Gradient 0.113781 | Wrong denominator changes the update gradient |
| Equal4+4 partition, separable loss | Gradient 5.55e-17 | Positive control for mean-of-means under equal counts |
| Training BatchNorm with 4+4 accumulation | Gradient 1.563317 | Correct weighting alone cannot restore full-batch behavior |
| Checkpoint dropout, preserve RNG | Gradient 0; forward0 | Positive recomputation control |
| Checkpoint dropout, do not preserve RNG | Gradient 0.064949; forward0 | A matching forward result can hide backward inconsistency |
| Checkpoint BatchNorm region | Gradient 0; running mean0.036057; evaluation output0.026179 | Gradient parity can hide changed mutable buffers |
| Float32 `(1e8-1e8)+1` versus `1e8+(-1e8+1)` |1 versus 0 | Arithmetic order matters independently of methodology |

BatchNorm's tracked-batch counter was1 for the reference and 2 for the checkpointed region. These values are implementation-specific observations of this fixture, not a universal statement about every checkpoint configuration.

## Checks

Eight pytest tests passed in the initial run; the expanded suite has eleven passing tests. One Hypothesis test exercised 40 generated sample partitions with a deterministic tiny float64 model. Other tests check positive controls and deliberately incompatible cases. The 1e-12 threshold distinguishes rounding from injected errors here; it is not a universal scientific tolerance.

## Additional composition controls

At 05:05:10 UTC, ran [composition_probes.py](../../../benchmarks/composition_probes.py); [raw results](cpu-composition-probes.json) record its independent source hash and environment. These use two logical AdamW updates, a pure tiny model, and a candidate combining 3+4 microbatch accumulation with activation checkpointing.

| Case | Final parameter max absolute error | Update counter |
|---|---|---|
| Correct weighting, checkpointing and final-gradient clipping |5.55e-17 |2 reference /2 candidate |
| Clip after each partial gradient |0.033198 |2 /2 |
| Step optimizer after each microbatch |0.024045 |2 /4 |

These controls show that matching update counts alone misses clipping errors, while the optimizer-timing fault violates both counters and parameters. They do not implement a planner, validate arbitrary compositions, or estimate full-training effects. Reproduce separately with `uv run --locked python benchmarks/composition_probes.py --output artifacts/replayed-composition-probes.json`.

Reproduce after installing uv 0.13.1 and Python 3.12:

```text
uv sync --locked
uv run --locked pytest -q
uv run --locked python benchmarks/semantic_probes.py --output artifacts/replayed-cpu-probes.json
```

The replay command writes to ignored local artifacts so the original result remains unchanged. Exact numerics are only expected within the validated environment; timestamps and elapsed times differ.

## Continuation audit controls

The [equivalence-power calibration](equivalence-calibration.md) corrects the initial sample-size example using direct integration and synthetic simulation. The [state/RNG controls](state-replay-probes.md) add a narrow repair positive control and a stochastic interleaving counterexample. These bring the local suite to 17 passing tests. They do not convert synthetic calibration into observed full-training evidence.

## What was not established

The original CPU experiments establish no optimizer/planner performance, full-training effect, energy, consumer-tier comparison, external framework superiority, human-labeled classification accuracy, or LLM benefit. No confidence interval over real training outcomes can be inferred from 40 generated partitions.

## Separate physical GPU fixture

The continuation's [synthetic CUDA experiment](cuda-memory-probe.md) now records allocated/reserved peaks, synchronized update timing and first-update state comparisons for four native configurations with two fresh-process repetitions each on the physical12 GB RTX 3060. The [raw evidence](cuda-memory-probe.json) and [separate environment](cuda-probe-environment.json) document its scope. This supersedes the initial absence of any CUDA measurement; it does not establish primary policy efficacy, real-workload OOM rescue or scientific outcomes.

The [450 MiB allocator-cap control](cuda-budget-probe.md) records three OOMs and one bounded completion, including completed-update counts and invalid driver-diagnostic flags. This is emulation on the 12 GB board. The [CPU update-clock fixtures](update-clock-probes.md) record absent-versus-zero gradient behavior and a scaler-skip clock error;19 CPU tests now pass. Older test-count statements above refer to their dated checkpoints.

## Real-data pilot and final calibration expansion

[Six five-epoch Fashion-MNIST CNN pilots](fashion-pilot.md) now complete a protocol frozen before data acquisition/execution. Three paired pilot seeds show a32.7% allocated-memory reduction with slower updates and small differing validation endpoints; no conclusive equivalence or access claim. [Data manifest](fashion-data-manifest.json) records original training-file bytes; the canonical test set was untouched. This supersedes the initial absence of any real-data training, while retaining the limits of every earlier synthetic experiment.

[Statistical assumption controls](statistical-assumption-audit.md) add model/SD/effect sensitivity and a rare-mixture counterexample; the current CPU suite has 20 passing tests. Its constructed probabilities are not estimated ML failure frequencies.

[Two fresh-process replays](fashion-repeatability.md) of the lowest pilot seed reproduce each arm's final parameter digest and all epoch validation metrics exactly. They are repeatability controls, not extra independent seed pairs. [Four activation-liveness controls](activation-liveness-probe.md) separately show why a verification execution pattern can retain more memory than a streamed target plan; no external checker is run.

[Derived summary arithmetic](derived-summary.json), produced by [the standard-library calculation](../../../benchmarks/derive_research_summary.py), independently recalculates headline percentages and paired differences from the raw JSON. Canonical input hashes avoid text-newline ambiguity. It confirms32.7137% lower allocated memory and66.4385% higher synchronized update time in the CNN pilot; this is descriptive arithmetic, not new independent evidence or inference.

## Direct external-checker controls

[TrainParity controls](trainparity-controls.md) execute a reviewed pinned checker on original CPU cases in a separate compatible stack. The setup failure, adapter amendment, protocol, all 11 outcomes and environment are retained. This confirms selected existing capabilities, not Atlas selection superiority.

## Direct external memory-planner controls

[Rockmate-family compatibility and budget controls](rockmate-compatibility.md) execute adapted pinned source on an original pure GPU workload. Frozen attempts retain factory/framework/parser/no-schedule outcomes; larger/intermediate budgets complete finite numerical controls. Static schedules and resident-memory diagnostics bound the interpretation. No Atlas policy, published performance suite or scientific outcome is replicated.

[Planner-construction state control](rockmate-state-control.md) records CUDA RNG advancement before training and a scoped existing Torch guard that restores it. Other recorded dimensions remain unchanged on the pure fixture; this is not generic state rollback or a distributional result.

[Reusable-module controls](planner-reuse-control.md) add native/single-forward/streaming positives and deferred-forward runtime failures. Existing output retention does not repair them, and a no-recomputation control rules out the claim that recomputation is necessary for the failure. The initial reporting failure and all intended modes remain inspectable.

[Shared-stack integration](interop-controls.md) now runs the native checker, planned deferred checker, and a competent streaming adaptation using existing snapshot/comparison primitives in a mutually supported GPU stack. The latter passes initialized/post-update comparisons and detects a deliberate parameter change. The different control/process work is not a fair speedup benchmark.
