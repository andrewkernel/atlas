# Unused gradients and successful-update clocks

Original CPU float64 fixtures executed with PyTorch2.14.1+cpu; [source](../../../benchmarks/update_clock_probes.py), [raw evidence and source identity](update-clock-probes.json). These exercise documented behavior, not newly discovered framework defects.

## Absent versus zero gradient

Two scalar parameters are used in update1; only the shared parameter is used in update2. Both arms have the same second observed loss and same final shared parameter. With AdamW, `zero_grad(set_to_none=True)` leaves the unused parameter at1.370000 and its step at1. Zeroing its existing gradient tensor instead yields1.275594 and step2, because the optimizer still applies state/decay updates. Parameter difference is.094406.

This matches the distinction in the [official zero_grad API](https://docs.pytorch.org/docs/2.14/generated/torch.optim.Optimizer.zero_grad.html). It rules out treating the option as an unconditional harmless memory tweak. Record absence explicitly; a keyed gradient observation should distinguish missing, None, zero and nonfinite. A flattened list that drops absent gradients can misalign parameters.

## Attempted versus successful update

A CPU GradScaler fixture has three attempts: finite, deliberately infinite, finite. Both arms perform exactly two actual SGD updates, and the scaler correctly skips the middle one. Advancing StepLR and EMA on every attempt changes the learning rate before the final update. Correct clock: final parameter.85, LR.025, EMA.976. Wrong attempt clock: parameter.875, LR.0125, EMA.9704.

Counters are observed through an original optimizer subclass, not inferred from a None return value. Many optimizer steps return None even when executed, so return-value equality is not a success indicator. [Native AMP guidance](https://docs.pytorch.org/docs/2.14/notes/amp_examples.html) supplies the skip/effective-batch context. Different workloads may intentionally schedule per sample, attempt or successful update; the contract must state which clock is intended. The fixture's correct label refers to its successful-update contract.

No FP16/BF16 training, real overflow distribution, full-training outcome or generic event observer was tested. Deliberately nonfinite gradients are recorded as nonfinite events; they are not numerical-parity passes. The expanded CPU suite has19 passing tests.
