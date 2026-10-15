# TrainParity — close overlap requiring direct comparison

Primary evidence: [repository README](https://github.com/intelland/TrainParity), [comparison with TrainCheck](https://github.com/intelland/TrainParity/blob/main/docs/comparison-with-traincheck.md), and selected [accumulation source](https://github.com/intelland/TrainParity/blob/79a1fc23c36073f09b97aa99c2daad7e96e7bce4/src/trainparity/accumulation.py) sections. Captured head/metadata are in [repository snapshot](repository-snapshot.json). The initial inspection was source-only; the later [direct controls](../experiments/trainparity-controls.md) now execute selected accumulation checks in a separate compatible CPU environment.

## Verified documented scope

Tests declared fresh-process resume and accumulation relations and finite sample-coverage policies; separates pass/fail/abstain/error and localizes observed divergence. Declared observations span loss accounting, gradients and update-related state. Its guarantees are explicitly finite and it does not claim model-quality certification or arbitrary-loop rewriting.

The inspected README permits Python>=3.11,<3.12 and PyTorch>=2.6,<2.14. That conflicts with this repository's Python 3.12/Torch 2.14 CPU environment. Treat integration as a separate-environment comparison, not a package to force-install into the current lock. Current license metadata is MIT.

## Atlas assessment

Generic accumulation verification and abstention are not Atlas novelty. The candidate difference is contract-conditioned memory-plan selection and interactions with activation checkpointing; this must be demonstrated rather than inferred from missing marketing language. Review implementation/tests before claiming absent coverage. Do not confuse saved-checkpoint resume with activation recomputation checkpointing.

## Deeper continuation inspection

[Selected source/test identities](trainparity-source-inspection.json) now cover accumulation orchestration, its worker, snapshot capture, the accumulation test module and a fixture at the same pinned commit. That inspection preceded execution; the later direct-control run is separately identified below.

The worker explicitly captures named model buffers with parameters, keyed gradients including None, canonical optimizer state, scheduler/scaler state and loss numerator/denominator observations. The generic snapshot also supports RNG and extra state. Execution-plan fields include per-microbatch optimizer/scheduler stepping, partial-gradient clipping, omitted tails and AMP ordering faults. The inspected tests include clean/faulted fresh-process controls and explicit loss-accounting divergence; listing a fault field is not a runtime replication of every fault.

This overlaps more of the proposed generic verifier than the original README-level review established. Buffer capture, abstention and state/update snapshots cannot be treated as Atlas distinctions. Its worker computes loss terms for all microbatches before their backwards; a different memory-conscious executor needs a fair adaptation/capability analysis, not an assumed drop-in equivalence. Do not infer unsupported full-training or checkpoint-composition guarantees from these bounded observations.

## Direct experiment and remaining comparison

The [frozen direct controls](../experiments/trainparity-controls.md) now reproduce selected clean/fault/refusal/error behaviors in Python 3.11/PyTorch 2.13. Three PASS, six FAIL, one ABSTAIN and one ERROR match the authored predictions after a retained adapter-setup repair. This is a narrow executable capability comparison, not an independent replication of the repository's entire advertised validation suite.

Include native strategies plus this checker as a composed baseline for the remaining independently labeled held-out selection study. Measure adaptation and total orchestration cost; do not reuse the CPU control times as GPU or cross-version performance evidence.
