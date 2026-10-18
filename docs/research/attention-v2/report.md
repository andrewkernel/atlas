# Atlas Attention Research v2

Protocol: `atlas-attention-v2-93518-a1`. Recorded 51/51 planned cases. Execution: {'completed': 51}.

**These benchmarks are not 100% accurate or unbiased.** The records support bounded numerical and performance observations on this machine. Passing execution and numerical eligibility are separate. Original failures and the interrupted first protocol attempt are preserved.

## Matched model results

Native generation time divided by candidate time; values below 1 mean the candidate is slower. Brackets show a descriptive 95% paired process-block bootstrap interval. Three blocks give limited uncertainty resolution. Failed numerical candidates are blocked even when timing is available. The historical method ID `hybrid50` uses the explicitly configured fixed 256-token GPU window in all model cases; it does not mean 50% GPU placement there. Raw labels are preserved, and displayed model labels use the actual window. Synthetic layer fractions remain fractional.

| Dtype | Context | Method | Generation ratio [95% CI] | Eligible blocks |
| --- | ---: | --- | --- | ---: |
| float32 | 512 | resident | 0.942 [0.925, 1.071] | 3/3 |
| float32 | 512 | gpu_dense | 0.781 [0.741, 0.834] | 3/3 |
| float32 | 512 | hybrid50 | 0.499 [0.488, 0.585] | 3/3 |
| float32 | 2048 | resident | 1.009 [0.912, 1.019] | 3/3 |
| float32 | 2048 | gpu_dense | 0.970 [0.910, 0.994] | 3/3 |
| float32 | 2048 | hybrid50 | 0.515 [0.492, 0.531] | 3/3 |
| float32 | 8192 | resident | 1.008 [0.987, 1.016] | 3/3 |
| float32 | 8192 | gpu_dense | 1.043 [1.022, 1.043] | 3/3 |
| float32 | 8192 | hybrid50 | 0.541 [0.535, 0.542] | 3/3 |
| float16 | 512 | resident | 0.989 [0.932, 1.021] | 3/3 |
| float16 | 512 | gpu_dense | 0.702 [0.646, 0.718] | 0/3 |
| float16 | 512 | hybrid50 | 0.442 [0.419, 0.475] | 0/3 |
| float16 | 2048 | resident | 1.008 [0.998, 1.042] | 3/3 |
| float16 | 2048 | gpu_dense | 0.776 [0.755, 0.861] | 0/3 |
| float16 | 2048 | hybrid50 | 0.290 [0.272, 0.308] | 0/3 |
| float16 | 8192 | resident | 1.043 [1.032, 1.060] | 3/3 |
| float16 | 8192 | gpu_dense | 0.812 [0.811, 0.822] | 0/3 |
| float16 | 8192 | hybrid50 | 0.228 [0.225, 0.230] | 0/3 |
| bfloat16 | 512 | resident | 1.055 [0.988, 1.068] | 3/3 |
| bfloat16 | 512 | gpu_dense | 0.724 [0.673, 0.724] | 0/3 |
| bfloat16 | 512 | hybrid50 | 0.512 [0.461, 0.514] | 0/3 |
| bfloat16 | 2048 | resident | 0.971 [0.959, 0.996] | 3/3 |
| bfloat16 | 2048 | gpu_dense | 0.772 [0.737, 0.794] | 0/3 |
| bfloat16 | 2048 | hybrid50 | 0.304 [0.295, 0.312] | 0/3 |
| bfloat16 | 8192 | resident | 1.060 [1.050, 1.062] | 3/3 |
| bfloat16 | 8192 | gpu_dense | 0.826 [0.823, 0.836] | 0/3 |
| bfloat16 | 8192 | hybrid50 | 0.234 [0.229, 0.234] | 0/3 |

## Baselines and precision

Native SDPA dispatch was observed as: aten::_efficient_attention_forward, aten::_scaled_dot_product_efficient_attention, aten::scaled_dot_product_attention. Flash attention is unavailable in this installed build; no Flash performance baseline is claimed. Exact build flags and the external shared-session driver observation are in `backend-controls.json`.

FP32 retains atol=rtol=1e-4. New FP16 (atol=0.02, rtol=0.02) and BF16 (atol=0.15, rtol=0.05) relations were declared before their confirmation cases. They compare with precision-matched native logits, not FP32 equivalence. Finite real-token NLL/softmax diagnostics supplement rather than replace these literal gates. Native warmup and reference replay precede timing; custom first-call costs remain included. This does not measure universal steady-state performance.

## Real memory pressure

Full generation includes prefill, selection, decode/cache movement and synchronization. No allocator quota is imposed. Windows WDDM can spill into system memory, so allocator and board observations do not prove physical residency or a hard hardware frontier. Full-context numerical replay is unknown and therefore these rows are not correctness-eligible.

| Context | Block | Method | Outcome | Generation ms | Allocator peak MiB | Board observed MiB | Host RSS MiB |
| ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |
| 32768 | 0 | native | completed | 46131.3 | 11146.9 | unknown | 8210.9 |
| 32768 | 0 | hybrid50 | completed | 50449.1 | 4034.9 | unknown | 11661.6 |
| 40000 | 0 | native | completed | 94633.4 | 13104.2 | unknown | 7118.8 |
| 40000 | 0 | hybrid50 | completed | 72034.9 | 4410.2 | 6552.0 | 18769.7 |
| 32768 | 1 | native | completed | 43048.0 | 11146.9 | 11667.0 | 8122.9 |
| 32768 | 1 | hybrid50 | completed | 50984.4 | 4034.9 | 5915.0 | 11662.0 |
| 40000 | 1 | native | completed | 90724.0 | 13104.2 | 11768.0 | 7230.7 |
| 40000 | 1 | hybrid50 | completed | 71725.4 | 4410.2 | 6564.0 | 18885.1 |
| 32768 | 2 | native | completed | 43772.3 | 11146.9 | 11760.0 | 8187.3 |
| 32768 | 2 | hybrid50 | completed | 50527.7 | 4034.9 | 6050.0 | 11662.0 |
| 40000 | 2 | native | completed | 90790.6 | 13104.2 | 11796.0 | 7269.6 |
| 40000 | 2 | hybrid50 | completed | 72104.6 | 4410.2 | 6773.0 | 18887.6 |

## Correctness diagnosis

The legacy relation remains atol=rtol=1e-4. Captured local attention agrees with the independent NumPy float64 oracle where reported, while some full-model logits differ from native. This bounds the diagnosis to arithmetic and propagation differences; it does not prove every operation correct. Shift-invariant KL and centered-logit diagnostics are reported alongside the literal gate, never as a replacement.

[
  {
    "plan": "resident",
    "failed_steps": [],
    "local_oracle_passed": true,
    "max_kl": 0.0,
    "max_centered_logit_error": 0.0,
    "earliest_checked_layer_failure": null
  },
  {
    "plan": "gpu_dense",
    "failed_steps": [
      47
    ],
    "local_oracle_passed": true,
    "max_kl": 6.307987894235988e-11,
    "max_centered_logit_error": 0.00010872885343360394,
    "earliest_checked_layer_failure": null
  },
  {
    "plan": "hybrid50",
    "failed_steps": [
      47
    ],
    "local_oracle_passed": true,
    "max_kl": 6.283780395745361e-11,
    "max_centered_logit_error": 0.00021159732628461825,
    "earliest_checked_layer_failure": null
  }
]

## Cache and adaptation

The ring has O(1) append/eviction and stores its GPU window once. Its measured complete-step ablation still includes attention, CPU copies and fusion; O(1) append alone does not guarantee a faster full step. Calibration binds implementation, software, GPU, precision, shape and timing scope, abstains outside support, and is compared with a budget rule and calibration-selected fixed plan. The held-out oracle is retrospective only. No net adaptive advantage is claimed.

## Remaining limits

- Small, non-random public corpus selection, overlapping windows and short continuations
- One modern model family, one GPU and shared desktop load
- Method order only partially balanced with three blocks and four arms; cache ablation uses fixed shift-then-ring order
- Native warmup/reference replay precede timing; custom decoder first-call costs remain included. These are registered generation windows, not all-method steady-state timings.
- Low-precision tolerances are finite relations, not universal equivalence
- In-worker NVML unavailable in primary study; external shared-board trace starts later. Earlier absent measurements remain unknown. Observer placement is not randomized.
- No actual CXL hardware, independent human replication, energy data or novelty proof

The source-bound reproduction kit and arithmetic validation are self-produced evidence, not external independent reproduction. See `reproduce.md`, the frozen protocol, exact raw run hashes in `execution-amendment-1.json`, and `summary.json` for every metric.
