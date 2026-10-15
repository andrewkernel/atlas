# TrainParity — close overlap requiring direct comparison

Primary evidence: [repository README](https://github.com/intelland/TrainParity), [comparison with TrainCheck](https://github.com/intelland/TrainParity/blob/main/docs/comparison-with-traincheck.md), and selected [accumulation source](https://github.com/intelland/TrainParity/blob/79a1fc23c36073f09b97aa99c2daad7e96e7bce4/src/trainparity/accumulation.py) sections. Captured head/metadata are in [repository snapshot](repository-snapshot.json). No code was executed.

## Verified documented scope

Tests declared fresh-process resume and accumulation relations and finite sample-coverage policies; separates pass/fail/abstain/error and localizes observed divergence. Declared observations span loss accounting, gradients and update-related state. Its guarantees are explicitly finite and it does not claim model-quality certification or arbitrary-loop rewriting.

The inspected README permits Python>=3.11,<3.12 and PyTorch>=2.6,<2.14. That conflicts with this repository's Python 3.12/Torch 2.14 CPU environment. Treat integration as a separate-environment comparison, not a package to force-install into the current lock. Current license metadata is MIT.

## Atlas assessment

Generic accumulation verification and abstention are not Atlas novelty. The candidate difference is contract-conditioned memory-plan selection and interactions with activation checkpointing; this must be demonstrated rather than inferred from missing marketing language. Review implementation/tests before claiming absent coverage. Do not confuse saved-checkpoint resume with activation recomputation checkpointing.

## Next experiment

Use shared authored loss/update cases and compare documented observation coverage in a compatible environment. Include native strategies plus this checker as a composed baseline; measure adaptation and orchestration costs. Current advertised validation numbers were not independently replicated here.
