# Continuation audit and amendments

The owner asked for continued research and simultaneous double-checking after the initial specification. This ledger distinguishes corrections from newly gathered evidence. Earlier commits and original result files remain available; amended prose governs the current protocol.

| ID | Claim under review | Finding and disposition | Evidence / remaining limit |
|---|---|---|---|
| A01 | Illustrative paired TOST needs about25 pairs for80% power at margin.5 SD | **Corrected.**35 for the known-SD approximation;36 with estimated SD under independent normal differences.25 gives about56.7% power. | [Derivation, integration and simulation](experiments/equivalence-calibration.md); not real training variance or domain margin |
| A02 | Candidate gap between memory management and explicit fidelity | **Narrowed.**Recent [HiLoRe](papers/hilore.md) makes broad planner novelty less credible. Prioritize a specific interaction benchmark and incremental selection comparison. | Selected primary paper sections; no runtime replication |
| A03 | BatchNorm checkpoint-state discrepancy motivates verifier | **Retained as a fixture, novelty limited.**[FairScale](competitors/fairscale.md) already has specialized mitigation. An original positive control repairs our narrow fixture. | [State-replay experiment](experiments/state-replay-probes.md); no generic mutable-region guarantee |
| A04 | Identical seed/final RNG can support pathwise equivalence | **Insufficient.**Two-layer dropout control has equal final RNG states but different gradients after partitioning. | One CPU implementation/shape; distributional behavior and outcomes untested |
| A05 |17 passing tests support Atlas efficacy | **Rejected interpretation.**They check research fixtures and calibration, not primary selection benefit. | No application, expert-labeled corpus or competing-system execution yet |
| A06 | No CUDA support or measurement has been validated | **Updated with new evidence.**Separate2.14.1+cu130 environment passes native smoke and eight synthetic physical-GPU trials. | [Measured fixture](experiments/cuda-memory-probe.md), not real-workload or planner validation |
| A07 | First complete update captures maximum training allocation | **Insufficient.**Later windows peak higher in the fixture after persistent optimizer state exists during forward/backward. | Both windows recorded; no guarantee about unseen future shapes |
| A08 | One optimizer relative norm adequately summarizes state parity | **Refined.**Compare steps and each moment family separately so step magnitudes cannot hide small-state differences. | Published CUDA run repeats the corrected instrumentation; absolute errors retained |

## Audit discipline

For each substantive claim, distinguish paper-reported results, primary implementation behavior, original measured fixtures, conditional derivation and proposed work. A source absence or failed page fetch is not a negative capability result. Record software identities; a fix for a reentrant wrapper cannot be transferred blindly to native non-reentrant checkpointing.

Statistical calibration was checked through a separate mathematical integration and sampling route with boundary controls. These routes share a stated probability model; agreement does not validate that model for actual ML outcomes. The state probes include a negative control and a repaired positive control. Retain results that match unexpectedly, such as the single-dropout partition case, rather than selecting only counterexamples.

The most important uncompleted audit remains the primary empirical comparison: native supported mechanisms plus existing state-aware checks versus Atlas under matched budgets. Until that is run, the selected contribution is provisional.
