# Real-data five-epoch CNN pilot — exploratory results

The [protocol and original code](fashion-pilot-protocol.md) were frozen in commit `c406bef697decfd98e9b39c93b5b07d9678ceefc` before data acquisition/training. [Raw results](fashion-pilot.json) started2026-10-15 at 05:48:30 UTC and identify the unchanged source. [Dataset manifest](fashion-data-manifest.json) records pinned provenance and compressed/expanded SHA256 values. Only official training files were acquired; canonical test files were not downloaded or evaluated. Data remain outside Git with the MIT notice.

## What ran

Six fresh-process runs: three pilot seeds 1101,1102,1103, paired full-batch 128/no checkpoint versus logical128/micro32 with both pure convolution blocks checkpointed. Five epochs on 50,000 training examples; fixed10,000 validation split. Each run completed1955 optimizer updates and250,000 sample exposures. The model has 110,986 parameters. All runs completed within the 300-second timeout with empty stderr. This is a deliberately small control that fits the physical device.

Initialization, sample-order and split digests match within every pair. The script independently computes a first unmodified update inside each worker, then rebuilds the actual arm from the same initial state. First-update reference-arm discrepancies are0. Candidate parameter max absolute discrepancies range3.73e-9–7.45e-9; gradient/momentum max4.28e-8–7.26e-8. These are raw finite numerical observations, not universal or scientific tolerances.

## Final validation outcomes

Every entry is the prespecified epoch 5 result; no best-epoch or best-seed selection.

|Pilot seed |Reference accuracy(%) |Combined accuracy(%) |Paired difference(percentage points) |Reference NLL |Combined NLL |
|---|---|---|---|---|---|
|1101 |89.22 |89.26 |+.04 |.306189 |.305860 |
|1102 |88.33 |88.12 |-.21 |.326548 |.328442 |
|1103 |88.45 |88.50 |+.05 |.323235 |.322900 |

The small first-update numerical differences coexist with later outcome differences. This is expected to be possible under FP32 trajectory changes; these runs do not establish a unique root cause or a distributional effect. All pairwise final parameter digests differ. Three pairs do not establish outcome equivalence, noninferiority, tail reliability, or preservation of a scientific method ranking. No margin or inferential test was selected after seeing the results.

## Resource and cost observations

All three reference runs peak at 166.79 MiB allocated/208 MiB reserved; all combined runs at 112.23 MiB allocated/140 MiB reserved for the whole training-plus-validation window. Measured allocated reduction is about 32.7%. First-update candidate peak is91.01 MiB, again lower than the later112.23 MiB maximum.

Mean summed synchronized update windows:8.56 seconds reference versus 14.24 seconds combined, about 66% higher. Mean worker wall time, including local data loading, reference preparation, comparisons and validation:10.39 versus 16.07 seconds. Summed six owned-child wall times are89.57 seconds, including interpreter/process overhead. This is an instrumented small workload; do not generalize these absolute timings to production throughput or large models.

Both arms fit easily on the 12 GB RTX 3060; **no real-workload OOM rescue or expanded-access benefit was shown**. Memory reduction at extra runtime is a measured tradeoff, not an energy saving or planner superiority result. Only the combined native strategy was tested against identity here; individual-transform resource controls are in the separate synthetic experiment.

## Audit and interpretation

The experiment executes original reviewed code with data only, not a downloaded research program. Epoch tail size80 uses its actual denominator and microbatches32+32+16. No dropout, BatchNorm, scheduler or AMP is present; the control deliberately avoids those incompatibilities. Final validation uses inference without checkpointing in both arms. Persistent gradients/optimizer state and evaluation can contribute to the whole-window peak.

The current recorded evidence checks data/config identities, first update and endpoint summaries. It does not compare every intermediate parameter, independently validate the entire executor, or constitute the primary Atlas-versus-existing-checker study. No independent annotators or external reviewers participated. These are three disjoint **pilot** seeds; future confirmation needs new seeds, justified margins, a frozen model of uncertainty and realistic power/cost analysis.

A later [fresh-process repeatability control](fashion-repeatability.md) replays the lowest frozen seed for both arms. Final parameter hashes and all recorded epoch validation values match each original arm exactly. These two replays strengthen the bounded reproducibility record but do not add independent seed pairs or establish between-arm equivalence.
