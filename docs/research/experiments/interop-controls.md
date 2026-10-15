# Actual compatible-stack checker/planner integration

The existing checker passes the native GPU accumulation control, reports ERROR for the adapted planner under its deferred execution, and its existing capture/comparison machinery passes an authored streaming adaptation. A deliberate parameter perturbation is detected. This supplies a competent reusable baseline, not evidence Atlas needs a new generic verification engine.

## Provenance and fixed comparison

The [protocol](interop-protocol.md), [original cases](../../../benchmarks/interop_cases.py), [runner](../../../benchmarks/run_interop_controls.py) and [environment](interop-environment.json) were frozen at `23fab8f` before this execution. TrainParity `79a1fc23c36073f09b97aa99c2daad7e96e7bce4` and Rockmate/rkgb `c2f678455393c5d872a1adb099949fe25f682bd0` run together on Python 3.11.17/PyTorch 2.13.0+cu130/NumPy 2.4.6. This meets the inspected declared version bounds; actual results, not metadata alone, validate the finite controls. The physical device remains the RTX 3060. Other historical environments/core lock remain unchanged.

The registered case fixes a pure four-block Linear/ReLU model, logical batch 256, actual MSE denominator, SGD lr .01/momentum .9, seed 2001 and numerical API parameters rtol=1e-5/atol=1e-7. A conditional adapter routes full-batch reference calls through the original model and 128-example candidate calls through its compiled Rotor wrapper. The scientific parameters have one registered namespace, shared with the compiled runtime. All builds prepare the same wrapper and use a scoped Torch RNG guard; no arbitrary project rewriting or unreviewed workload is run.

## Observed outcomes

|Mode |Result |Evidence and scope |
|---|---|---|
|Unmodified TrainParity accumulation runner, native model |PASS |Three fresh workers; verified initial state; explicit loss normalization captured |
|Same runner, conditional planner case |ERROR |Candidate worker raises the same invalid-gradient shape error under deferred forwards/backwards |
|Existing snapshot/comparison primitives, authored streaming execution |PASS for initialized and post-update snapshots |Named parameters, gradients, SGD state, explicit denominator and loss agree under the specified policy |
|Post-comparison .01 perturbation of one parameter |FAIL |Existing comparison localizes `model.core.layers["0"].weight`; max absolute difference about .01 |

[Raw results](interop-controls.json) preserve the complete runner reports, four successful snapshot-capture outcomes and positive/negative comparison details. No scalar nonfinite field was replaced during serialization. For the planned ERROR, `verified_equal_initial_state=false` is an early-return report default; it must not be misread as observed unequal initialization. The runner stopped before completing that comparison workflow.

The streaming mode uses a known separable numerator/global-denominator objective and captures post-state with RNG deliberately excluded. It is not a general loop converter, complete state proof or scientific conclusion check. The capture function is imported from the inspected lower-level `trainparity.snapshot` module; top-level API stability is not assumed. A real integration needs pinning, policy-convention matching, license/maintenance review and supported observation adapters.

## Costs and interpretation

Check-call wall times are 14.41 seconds for native accumulation and 17.36 seconds for the planned runner error. The streaming capture workflow takes 3.22 seconds inside its worker; respective outer times are 16.77, 19.64 and 5.95 seconds. These workflows perform different control/capture work and use different process counts: the runner performs two fresh references and a candidate, while the authored adaptation uses two builds in one worker and no second independent reference-control run. **Do not turn these times into a like-for-like speedup or verification-quality comparison.** No memory/energy study or whole-training outcome is measured.

This extends the earlier scheduling-pattern observation to actual imported existing tools in a mutually compatible GPU stack. It also shows why stopping at the default-runner error would be an unfair baseline: its reusable capture/comparison machinery can check a supported streaming execution. The runtime error is loud, not an observed unsafe scientific acceptance. Independent held-out labels, actual Atlas selection decisions, useful OOM rescue and complete workflow superiority remain untested.
