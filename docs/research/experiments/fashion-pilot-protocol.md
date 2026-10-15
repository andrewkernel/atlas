# Frozen exploratory Fashion-MNIST pilot protocol

This protocol is committed before data acquisition/training. It is a small exploratory implementation/variance pilot, **not a preregistered confirmatory equivalence test**. There is no claim that these settings reproduce a published scientific conclusion.

## Scope and safety review

Execute only the original reviewed [data preparation](../../../benchmarks/prepare_fashion_data.py) and [training fixture](../../../benchmarks/fashion_pilot.py) with the separately validated CUDA environment. No downloaded program, model checkpoint or remote code is imported. The official [Fashion-MNIST repository at fixed revision](https://github.com/zalandoresearch/fashion-mnist/tree/b2617bb6d3ffa2e429640350f613e3291e10b141) carries an [MIT notice](https://github.com/zalandoresearch/fashion-mnist/blob/b2617bb6d3ffa2e429640350f613e3291e10b141/LICENSE); retain it with local data. Record exact compressed/expanded byte hashes. All data/artifacts stay ignored by Git; commit small manifests/results only.

Only the60,000-example official training files are downloaded. Split by a fixed permutation seed9001 into50,000 training/10,000 validation; do not download or evaluate the canonical test set. Pixel normalization is float32/255 with no learned preprocessing or augmentation. Hash split IDs, per-epoch sample order and initialization. Use pilot seeds1101,1102,1103, with identical initialization and epoch order for each paired arm. These seeds are disjoint from any later confirmation.

## Fixed workload and arms

Original CNN: Conv1→32 valid3×3, ReLU; Conv32→64 valid3×3, ReLU,2×2 max-pool; flatten9216, Linear→10. Default PyTorch initialization with the recorded model seed. No dropout, BatchNorm, scheduler, clipping, EMA, AMP, weight decay or augmentation. SGD lr.01,momentum.9,foreach=False. IEEE FP32 and deterministic algorithms; cuDNN benchmark disabled. Physical host RTX3060, existing driver; no device settings changed.

ArmA: logical/full batch128, no checkpointing. ArmB: logical batch128, microbatch32, non-reentrant RNG-preserving checkpoints over the two explicitly pure convolution blocks. Each microbatch uses cross-entropy summed numerator divided by the **actual logical-batch size**, including the80-example epoch tail. Exactly one SGD step per logical batch. Five complete epochs:391 updates per epoch,1955 updates and250,000 sample exposures per run.

Six fresh child runs, three seeds×two arms; trial order shuffled with fixed ordering seed9051.300-second child timeout, one worker at a time. Every worker first computes the same first reference update from a fresh initial copy, retains only CPU reference observations, frees that reference model, then constructs its actual arm from the unchanged initialization. This verification preparation is timed separately and included in child wall time. No arbitrary process tree is created by workload code.

## Recorded evidence

- Source/config/environment identities, data bytes, initialization/split/order digests, all statuses and stderr.
- First-update parameter/gradient/momentum max absolute discrepancies; finite-state check. Report raw differences without selecting an acceptance tolerance after observing them. This check covers only the first update.
- First-update and whole-training-plus-validation allocated/reserved peaks, synchronized summed update windows, validation times, worker and parent-child wall time. Per-step synchronization is instrumentation overhead; this is not a low-overhead production throughput benchmark.
- At each epoch, validation summed cross-entropy/count and accuracy with exact correct/total counts; no test-set selection. No best-seed or best-epoch cherry-picking: final comparison uses epoch5 for all runs and preserves every epoch.
- Final parameter digest, successful-update/sample counts and failures. Keep failures in intended-run denominators; do not replace a timeout or divergence with a convenient accuracy.

## Interpretation and stop rules

Three pairs support debugging and descriptive paired differences only. Do not run an equivalence test, claim powered noninferiority, infer tail reliability or choose a scientific margin from these results. No Atlas policy, existing-checker comparison, OOM rescue or hardware-tier comparison is tested. The reference is expected to fit this device; a memory saving is not evidence of expanded access.

If data hashes/header checks, deterministic operation support, initialization/order matching or finite observations fail, preserve the error and investigate before attributing a difference to a transformation. Any source/protocol change is an amendment and must not be silently applied to earlier results. A difficult or negative result is retained rather than tuning the workload to obtain parity.
