# Experiments actually executed

## Scope and environment

On2026-10-15 at 04:43:58 UTC, ran original offline CPU probes with Python 3.12.10 and PyTorch 2.14.1+cpu on Windows 11, one PyTorch thread, deterministic algorithms enabled. No external datasets/models/checkpoints or research-repository code were executed. A12 GB RTX 3060 is installed on the host, but these probes use no CUDA.

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
| Float32 `(1e8-1e8)+1` versus `1e8+(-1e8+1)` |1 versus0 | Arithmetic order matters independently of methodology |

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

## What was not established

No optimizer or planner performance, full-training effect, GPU memory saving, GPU runtime, energy, consumer-tier comparison, external framework superiority, human-labeled classification accuracy, or LLM benefit was measured. No confidence interval over real training outcomes can be inferred from 40 generated partitions. This experiment supports failure-mode analysis and test design, not the primary research hypothesis.
