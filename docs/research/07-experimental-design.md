# Experimental design — proposed, not executed

The [experiment index](experiments/README.md) distinguishes executed CPU/statistical controls and a bounded synthetic CUDA fixture from the proposed studies. All real-workload selection, full-training, LLM and expert-labeling studies below are proposed. Freeze a protocol commit, manifest and analysis plan before confirmatory runs. Archive amendments with reasons; do not retroactively call exploratory decisions preregistered.

## Hypotheses and estimands

| ID | Falsifiable hypothesis | Estimand / failure condition |
|---|---|---|
| H1, primary | Contract-aware selection accepts fewer incompatible plans without reducing valid-feasible workload coverage compared with native mechanisms plus existing checks under an equal budget | Unsafe accepted / accepted, with counts; contract-satisfying feasible workloads / all declared workloads; resource-only policy is an additional ablation |
| H2 | Explicit state and update checks detect invalid composed plans missed by gradient-only or metric-only checks | Paired detection difference on held-out labeled cases, with abstention and false rejection reported |
| H3 | Profiling improves feasible-plan selection over fixed deterministic strategy ordering enough to cover its cost | Valid feasible completions and total search+verification+execution time; failure if added cost dominates |
| H4, optional | Local LLM ranking improves valid feasible coverage or time to a valid plan over deterministic ranking | Paired selection outcome at equal wall-time and trial budgets including inference; failure if no practical improvement |

H1 does not mean Atlas creates new optimization mechanisms. A precise superiority margin for policy selection must be frozen after a disjoint pilot; the initial study can report effect estimates and intervals without declaring universal superiority.

The primary decision criterion is observed safety/coverage dominance on the fixed held-out set, with uncertainty reported; no superiority conclusion is justified when the data are inconclusive. A separate utility gate requires at least one prespecified real workload that the unmodified reference cannot complete within the target budget but a contract-satisfying plan can. If there is no such workload, the project may still produce semantic findings but has not demonstrated expanded experimental access. These are decision criteria, not arbitrary final-accuracy equivalence margins.

## Two linked studies

**Study S: semantic decision benchmark.** Start with 24 designed cases: six fault/condition families with two positive and two incompatible/unknown variants per family. Families are loss denominator/tails, batch-coupled operations, RNG, checkpoint mutable state/hooks, optimizer/clipping clocks, and scheduler/EMA/unused-parameter behavior. Include single transformations and compositions; unknown custom effects should have an abstain label. These counts are a proposed starting design, not an executed corpus.

Reserve whole workload implementations and fault families for a held-out set; random splitting near-identical mutations is insufficient. At least two qualified humans independently label required properties and scope, blinded to tool recommendations. Resolve disagreements with recorded reasons, retain ambiguous cases, and report raw agreement and an agreement statistic with uncertainty. No experts have labeled cases in this session. Mutation detection is not real-world bug prevalence.

**Study R: useful resource feasibility.** Run a small set of real training workloads under the same logical protocol with a common finite candidate set. Evaluate complete-training outcomes only for preselected finalists and the matched reference, not for every searched plan. Keep candidate discovery and confirmatory outcomes separate to avoid choosing the apparent best seed/noisy validation result.

## Proposed workloads

Exact source/dataset/model revisions are in [asset snapshot](asset-snapshot.json) and [repository snapshot](competitors/repository-snapshot.json). Record file hashes and tokenizer/preprocessing hashes at download; a repository SHA alone does not identify data bytes.

| Workload | Fixed initial configuration proposal | Bottleneck / scientific purpose | Sources, downloads and rights |
|---|---|---|---|
| W1 small CNN | Fashion-MNIST; Conv(1,32,3)-ReLU-Conv(32,64,3)-ReLU-pool-linear; no normalization; FP32 SGD lr .01 momentum .9; logical batch 128; 5 fixed epochs | Cheap positive control for accumulation, not expected to stress VRAM | [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist), MIT; about 30 MB compressed, no weights. [PyTorch MNIST example](https://github.com/pytorch/examples) is reference style, BSD-3-Clause; adaptation is a new Atlas workload |
| W1b BatchNorm condition | Separate declared CNN architecture containing training BatchNorm; same data and budget | Deliberate incompatibility condition, not a replacement claimed equivalent to W1 | Same source/rights; BatchNorm changes are part of workload definition |
| W2 decoder training | WikiText-2 raw; train-only vocabulary, fixed tokenizer; causal pre-LN decoder; 6 layers, width 256, 4 heads, FFN 1024, context256; AdamW lr3e-4, betas(.9,.999), eps1e-8, decay.01; logical batch 32; 2,000 updates | Activation pressure, dropout and valid-token accounting | [Salesforce WikiText](https://huggingface.co/datasets/Salesforce/wikitext); [official word-language-model reference](https://github.com/pytorch/examples); text download in tens of MB for this subset, confirm exact bytes |
| W2-large stress variant | Independently declared 12-layer width 512 decoder, 8 heads, FFN 2048, context512; same objective and 2,000-update horizon; logical batch 32 | Different parameter/activation balance; scale chosen by disjoint pilot | Same dataset; not a claim to reproduce the smaller model |
| W3 fine-tuning | DistilBERT masked-language-model adaptation on WikiText-2; all parameters trainable; max length 128; fixed masks replayed; AdamW lr5e-5; logical batch 32; 1,000 updates | Pretrained fine-tuning and optimizer state, different masking objective | [DistilBERT](https://huggingface.co/distilbert/distilbert-base-uncased), Apache-2.0; roughly 0.3 GB FP32 weights plus tokenizer, exact manifest required; use safetensors and no remote code |

The WikiText card is internally inconsistent: metadata lists CC BY-SA3.0/GFDL while licensing prose links CC BY-SA4.0. Resolve provenance and applicable terms before downloading/redistributing benchmark inputs; do not assert a settled license from one field. Until resolved, W2/W3 are **conditional planned workloads**. [SST-2](https://huggingface.co/datasets/stanfordnlp/sst2) was considered but has unknown license metadata, so it is not the default fine-tuning dataset. Dataset/model availability has not been validated by downloading them here.

Freeze architecture/tokenization details in code, optimizer parameter groups, validation frequency, clipping policy, dropout, warmup and data-order manifest. Suggested initial policy: no clipping, no scheduler/warmup in the simplest positive control; separate condition cases add each. A token-weighted loss uses the total valid-target count, including partial windows. Do not alter context, epochs or trainable parameter set to make a strict candidate fit.

## Candidate search and equal budgets

For eligible registered workloads: microbatch divisors of logical batch {full, half, quarter, eighth}, crossed with checkpoint off/on declared regions: at most 8 strict plans. Identity is included. The mixed-precision extension is a separate contract and study, not a way to enlarge strict-mode coverage.

For a pilot, cap search at 8 candidate trials and 30 minutes per workload-policy-hardware cell, with at most 120 seconds per profile trial and explicitly measured short-check cost. These are engineering caps to be revised before confirmation based on a disjoint calibration workload. Both wall-time and trial limits apply; initialization/LLM/failed trials count. Finish confirmation of the selected plan separately, with a declared completion timeout. Exhaustive candidate profiling creates a descriptive oracle within this finite set; call it an observed finite-set oracle, not a global optimum.

## Baselines

| Label | Policy |
|---|---|
| A | Unmodified supplied PyTorch implementation, including any already-native efficient kernels |
| B | Documented manual configuration by a competent practitioner, same allowed transforms/contract; adaptation/tuning time recorded |
| C1 | Current native checkpoint selection/memory-budget option where compatible; exact API/version and unsupported cases recorded |
| C2 | DeepSpeed Autotuning or Rockmate on the subset they genuinely support; framework adaptation cost and methodology changes explicit |
| C3, essential | Native supported transforms plus applicable TrainCheck/TrainParity checks, without Atlas policy; separate compatible environment if required |
| D | Atlas deterministic planner with contract checks |
| E | D plus local-LLM ranking, equal resource/search budget |
| F | Same Atlas enumeration and profiling with fidelity filters disabled; unsafe candidates retained in evidence, not treated as scientifically valid |

Also compare a fixed native strategy ordering and random ordering with fixed seeds. Otherwise the comparison may only show that using any optimizer is better than doing nothing. If a framework cannot run in the same software stack, compare detection on shared traces/fixtures where valid and do not present its different-version runtime as a clean policy comparison.

## Hardware tiers and measurement

CPU-only is for correctness and integration. Physical6–8 GB,12 GB,16 GB and24 GB GPUs are proposed tiers. The host12 GB RTX3060 now has [bounded synthetic CUDA evidence](experiments/cuda-memory-probe.md), separate from this planned real-workload study. Other tiers require borrowed hardware or explicitly optional resources. An allocator cap or reserved tensor on12 GB hardware is an **emulated budget sensitivity study**, not an8 GB GPU benchmark: bandwidth, kernels, cache, compute and driver differ.

Run one workload at a time; record idle device use, temperature, power limits and background processes without changing user settings. Randomize policy order within workload/seed blocks. Record cold initialization separately; use5 warmup updates and 20 measured updates for initial profiling, extending only when timing precision warrants it. Capture at least one full initialization/first-optimizer-step peak and late/evaluation peaks. Synchronize CUDA around timing windows; keep instrumented diagnostic traces separate from low-overhead performance trials.

## Metrics and definitions

- Peak allocated and reserved device bytes, measured with reset peaks and explicit window. Optional device telemetry is separately named total device/process use; it is not interchangeable with allocator bytes.
- Host RSS for parent and child processes, sampling interval and shared-memory double-counting caveat; pinned memory and dataset caches noted.
- End-to-end seconds, training-only seconds, samples/sec or valid targets/sec, GPU-hours as summed elapsed allocated GPU time/3600 (not FLOPs), planning/verification overhead and failed-trial time.
- OOM failures / attempted runs; completions / intended runs; valid feasible completions / all prespecified workloads. Never discard failures from denominators. Report timeout/error types separately.
- Contract violation acceptance, false rejection, abstention coverage and execution errors versus labels. Classwise confusion matrix and risk-versus-coverage view; a checker that rejects everything must score poorly on coverage.
- Maximum and relative numerical errors, loss trajectory, update counts, final validation outcomes, paired differences and uncertainty. Trajectory summaries are descriptive; repeated steps are not independent experimental replicates.
- Optional joules: read [NVML cumulative energy](https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlDeviceQueries.html) only if supported, or integrate time-stamped board-power samples with sampling limitations. This measures board energy, not host/whole-system energy. External whole-system metering is a separate option. Unsupported readings are null, never zero; runtime/VRAM alone do not establish energy savings.

## Replication, inference and compute budget

Use3 disjoint pilot seeds solely to debug and estimate paired standard deviation. Plan10 fresh paired seeds per final workload/reference/selected-plan comparison if affordable; freeze the actual confirmatory count from power analysis before running. Ten is a planning target, not a guarantee of power. With only3–5 runs, emphasize intervals/exploration and allow inconclusive results.

For paired outcome d, symmetric equivalence bounds ±Delta, true mean difference zero and known SD, joint TOST power is max(0, 2*Phi(Delta*sqrt(n)/sigma_d - z_(1-alpha)) - 1). An 80%-power approximation therefore uses n ≈ ((z_.95 + z_.90)*sigma_d/Delta)^2, **not z_.80**. Both one-sided tests must reject. The earlier draft underestimated this requirement and is explicitly superseded by this correction.

With illustrative Delta=.5 percentage points and sigma_d=1.0 point, the corrected known-SD approximation rounds up to35 pairs. Under independent normal differences with estimated SD, direct integration and 200,000 synthetic repetitions per condition give a minimum of36 pairs for80% power;25 pairs give about56.7% and10 pairs about6.5%. See the [calibration and derivation](experiments/equivalence-calibration.md), [raw evidence](experiments/equivalence-calibration.json), and primary [TOSTER power documentation](https://aaroncaldwell.us/TOSTERpkg/reference/power_t_TOST.html). These assumptions and numbers are not measured ML variance or a justified domain margin. A three-seed pilot estimates SD very imprecisely; use plausible SD sensitivity and a conservative design, not a plug-in estimate alone.

The margin is illustrative; domain justification, not observed variance or convenience, determines its acceptability. Use paired TOST only when independence/normality assumptions for differences are defensible; otherwise preregister a suitable interval approach and validate its coverage. Report failure to demonstrate equivalence as inconclusive unless a violation is established. Ten confirmatory seeds remain only an affordability target; if power is inadequate, narrow the claim or report exploratory intervals instead of calling it a conclusive equivalence study.

Report unstandardized effects and 95% intervals, plus the 90% interval for a single alpha.05 TOST. Adjust or hierarchically control multiple confirmatory claims, e.g. a primary comparison then Holm-adjusted secondary comparisons; bootstrap workload-level summaries only when the sampling interpretation is defensible. A few fixed workloads do not justify population-wide benchmark generalization. Sample-level resampling cannot replace independent seed/run uncertainty.

For classification proportions, give exact or Wilson intervals and denominators. Use paired tests such as McNemar for paired binary detections only when the case independence assumption is reasonable; correlated mutations require workload/fault-family grouping and cautious interpretation. Do not pool thousands of tensor elements as independent samples of tool reliability.

Compute inventory: with 5 reference workload configurations,10 paired seeds and two final arms, there are100 full runs; a20-minute run would imply about 33 GPU-hours **before** search, pilots, failures or extra baselines. This is a planning arithmetic example, not a runtime prediction. Estimate real per-run cost in the pilot and reduce workload count/claims before confirmation if the budget is insufficient.

## Ablations and threats

Remove precondition rules, state checks, gradient checks, profiling and LLM ranking one at a time. Test each transform alone and their combination. Hold out code and fault families. Preserve all candidate records to expose survivor bias and selection on noisy measurements.

Threats: hand-selected easy faults; incomplete adapters; tolerance tuning; baseline version mismatch; deterministic modes atypical of normal training; missing full-scale reference; late instability; hardware-specific kernels; dataset license uncertainty; investigator-generated labels; and annotation effort excluding under-resourced users. Institutional-access or energy conclusions require their own measurements and should not be inferred from this study.
