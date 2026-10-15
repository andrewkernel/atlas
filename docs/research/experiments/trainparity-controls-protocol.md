# Frozen direct-checker controls

This protocol and the original fixture/configuration are saved before importing or executing the pinned checker. The purpose is to test the practical overlap of existing checks with Atlas's proposed generic verifier. It is not the held-out policy comparison required by the research gate.

## Scope and review

TrainParity revision `79a1fc23c36073f09b97aa99c2daad7e96e7bce4` is reviewed as text. The scope is the imported module graph and accumulation execution path: case import, tensor-tree splitting, accounting, update phases, immutable state capture, serialization, comparison, subprocess dispatch and reporting. Command-oriented resume and quickstarts are not exercised. Module-level statements in the import graph were inspected; static inventory is supporting evidence, not an automatic security proof. Dependencies are official installed binaries. No package build hook, downloaded test case, remote command, external checkpoint or third-party quickstart is run.

The checker uses dynamic import for a user-specified case. This experiment permits only the named original cases in `benchmarks/trainparity_cases.py`. It uses small float64 CPU tensors and a two-parameter linear model. The outer launcher supplies only explicit system/interpreter/temporary paths and determinism settings, avoiding inherited credentials. This is scoped trusted-code execution, not OS network/filesystem isolation or a validated untrusted-code sandbox. No privileged operation is required.

## Environment and controls

Separate Python 3.11.17, PyTorch 2.13.0+cpu and NumPy 2.4.6 satisfy the source's declared Python/PyTorch bounds. NumPy 2.5.4 cannot install on Python 3.11; the compatible version is recorded rather than changing Atlas's Python 3.12 CPU lock. All arms of these checks run in the same external environment. Do not attribute timing differences against earlier PyTorch 2.14 probes to checker overhead.

The configuration declares 11 controls in order, a fixed seed 1701 and tolerance rtol=1e-10/atol=1e-12. These are small numerical-control tolerances, not scientific equivalence margins. All controls remain in the denominator, including setup errors and abstentions. Each ordinary check launches two baseline controls and one candidate; worker timeout is 30 seconds, with a 110-second per-control outer timeout. The authored code spawns no grandchildren beyond the checker's owned workers. The timeout path is not an independently tested Windows process-tree containment guarantee.

Predictions: clean balanced/unequal accumulation and balanced accumulation without explicit accounting pass; six deliberate observation/update/accounting faults fail; unsupported string metadata abstains; zero microbatch count errors. The wrong-mean configuration also changes the observed accounting schema, so its first failure may be a schema/flag difference rather than a numerical normalization error. The no-accounting positive checks the outcome and separate `loss_normalization_captured` flag; it does not imply the checker infers the objective's semantics.

## Interpretation

Record exact reported outcomes, first differing phase/path, all report differences, wall time, source/config/review identities and package versions. Preserve warnings and all failures. Predictions are authored labels, not independent adjudication, sensitivity/specificity estimates or a random workload sample. No GPU allocation, full-training quality, sample-coverage API, resume behavior, activation-checkpoint transformation selection or Atlas application is measured here. Success primarily strengthens the obligation to reuse and compare competent existing checks.
