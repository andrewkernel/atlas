# Verification systems — reuse and differentiation

Sources: [TrainCheck technical guide](https://orderlab.io/TrainCheck/technical-doc/), [Deep500 README](https://github.com/deep500/deep500), [KernelBench evaluation](https://github.com/ScalingIntelligence/KernelBench), [FPBench](https://github.com/FPBench/FPBench), and [TrainParity](trainparity.md). These are documented capabilities, not independently reproduced results.

TrainCheck contributes invariant inference/checking; Deep500 contributes correctness-oriented benchmarking; KernelBench joins functional checks with performance; FPBench distinguishes numerical-error measures. Atlas should borrow clear distinctions between intended computation, measured values and task outcomes. None of these observations establishes an unoccupied novelty gap.

The proposed direct experiment is to combine existing memory mechanisms with applicable existing checks and compare **decisions**, not merely whether Atlas can emit a report. Checkpoint buffers, RNG, loss denominators and update clocks form an initial case set, but its label/coverage quality requires independent review.

Track unsupported properties, non-triggered conditions, false positives, abstentions and execution errors. Do not count an unobserved invariant as satisfied. A passing operator/gradient comparison is a bounded observation; end-to-end training and scientific conclusion claims need different evidence.
