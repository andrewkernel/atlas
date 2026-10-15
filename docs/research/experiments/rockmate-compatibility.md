# Executed Rockmate-family compatibility and budget controls

Existing planner configurations now execute a pure GPU workload through an adapted interface. Four coarse-budget runs and two intermediate-budget Rotor runs complete eight updates each with zero measured maximum absolute differences in the recorded first gradients/parameters/momentum and final parameters. This confirms limited practical prior-art coverage. It does not establish an Atlas advantage, useful OOM rescue, full-training equivalence or replication of the published Rockmate/HiRemate/OffMate performance studies.

## Provenance, amendments and unsuccessful attempts

The [frozen protocol and amendment history](rockmate-compatibility-protocol.md), [source/review inventory](../competitors/rockmate-execution-review.json), [separate environment](planner-environment.json) and [original fixture](../../../benchmarks/rockmate_compatibility_probe.py) identify scope and exact dependencies. Pinned Rockmate/rkgb revision is `c2f678455393c5d872a1adb099949fe25f682bd0`; Python 3.12.10/PyTorch 2.14.1+cu130/CUDA 13.0 run on the physical RTX 3060. Source was imported after a scoped review, with no package build hook, C extension, external model/checkpoint, TwRemat or DAG-partitioning execution. This is trusted authored-code execution, not a hostile-code sandbox.

All five earlier two-arm attempts remain inspectable:

|Frozen source |Observed outcome |Interpretation |
|---|---|---|
|`cc04359` |[Direct factory and precision failures](rockmate-initial-attempt.json) |PureRotor's partitioner factory passes an instance into a string selector; its warning refers to undefined `self`. The other arm hits an export/precision interaction. |
|`5b936ba` |[Precision failure persists](rockmate-export-attempt.json) |Changing global/backend new-API settings did not repair the installed export path. |
|`d967b1d` |[Dynamo parser assertion](rockmate-dynamo-attempt.json) |Legacy-only full-FP32 settings pass export; raw parser rejects its assignment/node count. |
|`4cbb07b` |[JIT fallback representation failure](rockmate-jit-attempt.json) |Raw JIT extraction supplies parameter-name strings where ForwardGraph expects pairs. |
|`fd3832a` |[No schedule at 16 MiB](rockmate-no-schedule-attempt.json) |A named-input wrapper enables construction; neither configured solver finds a schedule at this budget. Later calling forward without one raises a missing-compiler error. This is not a global infeasibility proof. |

The [independent native export control](precision-export-control.json) imports no planner: default and legacy-only settings complete a tiny CPU export, while new backend/operator IEEE settings fail at a legacy cuDNN query in this binary stack, even though both conv/RNN getters report IEEE. Keep that framework interaction separate from Rockmate's algorithms. Its warning about missing Triton concerns optional Triton FLOP counting; no Triton kernel is part of the Linear/ReLU fixture.

Native exported-code diagnostics identify the input-name adapter: direct Sequential `forward(input)` produces 17 noncleanup assignments for 16 call-function nodes, including `input_1 = input`; the original `forward(x)` wrapper produces 16 and 16 with no alias assignment. No upstream source patch is applied. Stripping the wrapper's parameter-name prefix yields the same initialization digest as the original interface, and input/target bytes also match. This is a bounded workaround, not arbitrary-model compatibility.

## Completed execution and selected schedules

The [coarse bracket](rockmate-compatibility.json), frozen at `b3c0460`, tests generated Rotor and Rockmate configurations at nominal 64/128 MiB activation-planning budgets. Both configurations use a sequence/base partitioner and the Python Rotor solver; the Rockmate configuration permits a small CBC bottom-solving grid. This does not establish that CBC actually solved a hierarchical subproblem on this pure chain. Internal optimizer calibration/integration is disabled, the external SGD update is retained, and actual minor-node count is zero.

The [schedule diagnostic](rockmate-schedule-diagnostics.json) finds no repeated enabled forward operations in the 64 MiB Rotor plan. Therefore those coarse successes alone must not be called demonstrated rematerialization. The subsequent [fixed 32/48 MiB exploratory follow-up](rockmate-budget-frontier.json), frozen before execution at `199879f`, records planned repeated forward operations and successful numerical controls. These repetitions are static schedule operations, not a captured GPU-kernel trace.

|Rotor nominal planning budget |Outcome |Reported schedule peak (bytes) |Planned repeated forward operations |Observed whole allocated peak (MiB) |
|---|---|---|---|---|
|16 MiB |No schedule found |Unavailable |Unavailable |No candidate training |
|32 MiB |Eight updates completed |33,554,944 |7 |116.77 |
|48 MiB |Eight updates completed |50,332,160 |2 |126.77 |
|64 MiB |Eight updates completed |54,526,464 in diagnostic |0 in diagnostic |126.77 |
|128 MiB |Eight updates completed |Not captured |Not captured |126.77 |

Each corresponding native reference reports 102.79 MiB allocated peak. The 32/48 MiB reported schedule peaks exceed their nominal budget by 512 bytes; preserve those exact values rather than rounding them into an exact-cap guarantee. The cause was not isolated. Solver memory accounting, interfaces, allocator alignment and whole-process measurements are distinct. No hard physical-device quota was imposed.

All completed configurations show zero maximum absolute errors in the recorded numerical comparisons at the fixed eight-update horizon. That is finite numerical evidence for this pure, fixed-batch, non-stochastic workload. It does not establish bitwise identity of every state, arbitrary RNG/buffer handling or scientific outcome equivalence. The same seed/model/data repeated across configurations is not a sample of independent workloads or training seeds.

## Memory and time interpretation

The instrumented 64 MiB follow-up records 4,460,544 allocated bytes before the native training window and 71,572,480 before the candidate window after construction. The compatibility workflow runs its native reference first, then profiling/planning and candidate training in the same worker. Startup, retained framework/planner allocations and lazy workspaces can differ across these windows. No particular source of the difference was isolated. Subtracting those starting values does not create a fair causal memory-saving estimate.

The native later-update medians are about 1.77–2.09 ms in the coarse bracket; candidate medians about 3.33–3.65 ms. Construction is about .15 seconds after imports, native reference work and export diagnostics. Summed outer wall time for the four bracket workers is 28.47 seconds. These are small diagnostic observations, not a production overhead estimate or a fair isolated memory/runtime benchmark. The intermediate 32 MiB candidate is slower here as well. No energy was measured.

Most importantly, observed total allocation did not fall below the native reference in this workflow, even when the selected schedule plans recomputation. This is a useful reason to measure the complete execution and retain resident-state costs; it is not evidence the algorithm cannot save memory on other models or in a properly controlled deployment.

## Consequence for Atlas

The closest memory-planning baseline is more than a literature label: a reviewed adapted execution exists, and it can select schedules with repeated forwards while preserving these recorded updates. Generic recomputation planning plus finite numerical fidelity checks remains established territory. The remaining research gate is independently adjudicated, held-out contract-conditioned selection under comparable budgets and full workflow costs. No primary Atlas selection experiment has run.

The later [construction-state control](rockmate-state-control.md) observes CUDA RNG advancement despite this pure fixture's unchanged parameters and zero numerical errors. The proposed adapter therefore needs a separately verified scientific-start replay after planning; an existing scoped Torch RNG guard is sufficient only for the observed dimension.
