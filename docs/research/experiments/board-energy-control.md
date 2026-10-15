# Gross-board energy replays and telemetry disagreement

The [protocol](board-energy-protocol.md) and original replay/measurement code were frozen at `8fbab26` before six measurement runs. They replay unchanged five-epoch Fashion-MNIST seed 1101 in its original CUDA environment, three randomized-order pairs. These are resource repetitions of one seed, not additional independent training seeds. All six complete, all four initialization/data/order/final hashes match their original arm, and all five epochs' validation/update values match exactly. Raw [full traces](board-energy-control.json) and [independent derivation](board-energy-derived.json) are retained.

| Pair | Native counter J | Combined counter J | Native window s | Combined window s | Counter increase |
|---|---:|---:|---:|---:|---:|
|0|1187.809|1584.644|16.296|21.622|33.41%|
|1|1210.018|1587.587|16.235|21.559|31.20%|
|2|1217.424|1600.339|16.299|21.579|31.45%|

The measured window includes two seconds before, the entire child workflow, and two seconds after. Startup, first-step comparison, training, validation and shared display/background activity count. Mean gross counter readings are 1205.084 J native and 1590.857 J combined. These observations support neither energy savings nor isolated process/whole-computer energy, carbon, population inference or primary selection superiority. All original runs already fit this 12 GB GPU.

## Independent arithmetic and instrument limitation

A separate standard-library derivation reconstructs cumulative mJ differences and power integrals using exact rational arithmetic on raw integer timestamps/readings. It independently verifies source-bound training identities and every selected epoch value. The analytic 100 W / 2 s integration control gives 200 J. Both routes agree to floating rounding; this checks arithmetic, not sensor accuracy.

Integrated reported power is **23.61–25.00% higher** than the cumulative counter in every workload window. Native mean integral is 1501.018 J, combined 1976.963 J. Neither is substituted for the predeclared counter endpoint. Maximum sample gaps are 57.9–63.0 ms despite the nominal 50 ms target. Many consecutive counter readings are unchanged: faster polling does not supply independent or equally fresh measurements. The documented one-second averaged power signal on non-GA100 Ampere also precludes treating these samples as instantaneous independent draws.

After observing the disagreement, a separately [frozen passive diagnostic](passive-energy-protocol.md) at `b15c90a` reads the same APIs without an owned training child. Its [19.991-second capture](passive-energy-control.json) reports 840.603 counter J versus 888.734 integrated-power J, a 5.726% excess (42.049 W versus 44.457 W). Background use is uncontrolled; this is not an idle calibration. Disagreement is therefore not confined to the training-window implementation, but its changing magnitude and cause remain unisolated. No driver, sensor or measurement-domain defect is established. No known accuracy bound or corrected energy value follows.

The [official NVML API reference](https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlDeviceQueries.html) specifies cumulative energy in mJ since driver reload and board/circuitry power in mW, averaged over one second on this architecture family. Its ±5% statement concerns older Fermi/Kepler readings; it is not an accuracy certificate for this RTX 3060 experiment. An independently calibrated external meter and controlled workloads would be needed to validate absolute accuracy and attribution.

## Reproduction and scope

[Read-only telemetry availability](nvml-availability.json) and [hash-pinned binding requirements](../../../configs/telemetry-windows-py312-requirements.txt) identify nvidia-ml-py 13.615.71 in a separate Python 3.12.10 environment. Inspected binding wrappers load the system NVML DLL; source identity is retained. Original training, core, checker and planner environments remain unchanged. No power/clock/driver settings or unrelated processes were changed. Workers receive explicit minimal environments and owned scratch/profile paths; this remains trusted-code execution, not adversarial containment.

[Measurement source](../../../benchmarks/board_energy_control.py), [independent reconstruction](../../../benchmarks/derive_board_energy.py), [passive source](../../../benchmarks/passive_energy_control.py). Raw replay output is committed; data, installed packages and temporary logs are ignored. There is no claim that the original memory/time pilot itself measured energy or that these six repetitions add new learning seeds.
