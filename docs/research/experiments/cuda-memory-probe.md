# Physical RTX 3060 synthetic memory experiment

This is a measured feasibility/profile fixture on the host12 GB RTX 3060, not the proposed real-workload study or an Atlas planner evaluation. The first CPU-only specification is superseded on this narrow hardware-support point.

## Environment and reviewed execution

Python 3.12.10, official PyTorch 2.14.1+cu130 Windows wheel, CUDA runtime13.0, NVIDIA driver595.95, Ampere capability8.6. A separate ignored virtual environment leaves the CPU lock and CI unchanged. [Pinned hash-checked reproduction requirements](../../../configs/cuda-probe-windows-py312-requirements.txt) are specifically for Windows/Python 3.12; [environment record](cuda-probe-environment.json) records package versions and wheel source. No driver or power settings were changed.

[Original source](../../../benchmarks/cuda_memory_probe.py) and [raw results](cuda-memory-probe.json). Only reviewed original model code and generated tensors run. Eight owned child processes have180-second timeouts, no child spawning in workload code, and unique output directories. They load only the locally generated tensor reference with `weights_only=True`. This is process supervision for trusted code, not hostile-code isolation.

The model has 16,805,888 parameters: eight residual blocks, LayerNorm over 512 features,512→2048→512 GELU MLP and residual scale.1. Input is fixed generated float32 tensor16×256×512; objective is mean squared output. AdamW, one optimizer update per logical batch; no dropout, attention, data pipeline, scheduler or clipping. IEEE FP32 backend settings and deterministic algorithms are explicit. Smaller microbatch 4 accumulates four correctly normalized loss numerators.

Two fresh-process repetitions per plan; five warmup updates then ten synchronized measured updates, plus the separately captured first update. The baseline runs first to create the numerical reference; remaining order is seeded and recorded. Sixteen updates per process are repeated on the same synthetic input. This is unsuitable for any convergence claim.

## Observed results

MiB means2^20 bytes. These are PyTorch allocator metrics, not all memory reported by device telemetry. [Official CUDA semantics](https://docs.pytorch.org/docs/2.14/notes/cuda.html) distinguishes occupied tensor bytes, allocator reservation and caching. Resetting peak counters does not release the existing reservation.

| Microbatch / checkpoint | First-update allocated peak(MiB) | Steady-window allocated peak(MiB) | Reserved peak(MiB) | Per-process median update(ms), repetitions1/2 |
|---|---|---|---|---|
|16 / off |844.36 |972.58 |1018 |74.58 /74.47 |
|16 / on |344.45 |452.44 |486 |87.51 /87.97 |
|4 / off |380.28 |508.50 |514 |83.00 /83.91 |
|4 / on |338.45 |378.45 |398 |96.90 /97.90 |

The combined plan reduces this fixture's measured allocated peak by about 61.1% and is roughly31% slower by the average of process medians. This describes these two process repetitions only; there is no inferential performance confidence interval or benchmark-wide claim. Keeping persistent AdamW state in later forwards makes the later peak higher than the first-update peak. Measuring only the first optimizer update would understate the observed maximum.

## Numerical audit

Every state key, shape and dtype matches the locally generated full-batch reference after the first logical update. Parameters, gradients, Adam moments and step tensors are compared separately; mixing step counters into a relative optimizer norm would obscure moment differences. The initial pilot used that mixed norm, so the published run was repeated after separating them and ensuring unique trial directories. The pilot is retained locally as exploratory evidence, not counted as additional independent repetitions.

Checkpoint-only observations exactly match the reference in this fixture. Microbatching produces parameter max absolute error6.24e-8, gradient max4.73e-11(relative L2≈6.93e-7), first-moment relative L2≈6.94e-7 and second-moment relative L2≈1.14e-6. Step counters match; all compared tensors are finite. These are measured discrepancies, not a newly chosen scientific acceptance tolerance. Only the first update has differential state evidence; later updates are resource windows.

## Reproduction and limits

Create a separate Python 3.12 environment, install the platform-specific requirements with hash checking, and run the script with `--output artifacts/replayed-cuda-memory-probe.json`. Keep generated tensor references in ignored artifacts; do not commit them. This benchmark requires CUDA and is not run by CPU CI.

No reference failed the physical12 GB device budget. There is no demonstration of rescuing an OOM real experiment, expanded access, attention workload benefit, optimizer selection, full-training equivalence, cross-GPU portability or energy savings. Device background use and thermal conditions were observed but not controlled like a dedicated benchmarking host. Short fixed-input timing samples within a process are dependent. No 6–8/16/24 GB physical device was tested.
