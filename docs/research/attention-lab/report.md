# Atlas Attention Lab: measured engineering study

The end-to-end prototype runs persistent exact dense CPU/GPU attention, calibrated selection and pretrained GPT-2 inference behind a local interactive dashboard. This is an exploratory engineering study on one shared Windows PC, not a novel CXL system or a replication of the optimized sparse HGCA/BEYOND runtime.

## What the evidence says

The selected package contains 3 unrestricted full sweeps, 1 constrained-selection sweep and 2 pretrained-model comparisons. Each unrestricted sweep uses 30 samples per plan and disjoint calibration/held-out lengths. All numbers below are reconstructed from saved synchronized wall-time samples.

Direct GPU dense attention is an essential comparator. Initial SDPA-only pilots made hybrid execution appear more advantageous; these stronger-baseline records supersede that interpretation. Per-layer improvements do not establish end-to-end model improvement.

## Held-out attention latency

Median of per-run medians in milliseconds, across unrestricted runs. These are descriptive aggregates, not confidence intervals.

| Context | Native SDPA GPU | Direct dense GPU | CPU dense | Transfer to GPU | Hybrid 75% GPU |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1,024 | 0.244 | 0.102 | 0.199 | 0.485 | 0.450 |
| 4,096 | 0.774 | 0.142 | 0.462 | 1.735 | 0.514 |
| 16,384 | 3.312 | 0.388 | 2.454 | 6.758 | 0.901 |

The native float32 SDPA probe recorded the following operations: `aten::scaled_dot_product_attention`, `aten::_scaled_dot_product_efficient_attention`, `aten::_efficient_attention_forward`. This does not compare all low-precision fused kernels.

## Adaptive policy

| Run | KV selection budget (MiB) | Held-out selections | Regret versus eligible measured optimum | Calibration cost |
| --- | ---: | --- | --- | ---: |
| benchmark-8de4c6c31991 | 2048 | 1024: gpu_dense, 4096: gpu_dense, 16384: gpu_dense | 0.0%, 0.0%, 0.0% | 5.93 s |
| benchmark-bf077396b657 | 2048 | 1024: gpu_dense, 4096: gpu_dense, 16384: gpu_dense | 0.0%, 0.0%, 0.0% | 5.84 s |
| benchmark-5791670342ea | 2048 | 1024: gpu_dense, 4096: gpu_dense, 16384: gpu_dense | 0.0%, 0.0%, 0.0% | 5.79 s |
| benchmark-91d687643c2f | 48 | 1024: gpu_dense, 4096: gpu_dense, 16384: hybrid50 | 0.0%, 0.0%, 0.0% | 5.86 s |

Selection uses calibration rows only. The fixed comparator is selected on calibration and must fit the same maximum held-out KV budget. The budget applies to policy selection; every plan is benchmarked. It is not a process memory cap. Calibration overhead is real cost, separate from selected-call latency. Matching a fixed plan is not an adaptive-policy advantage.

## Full pretrained-model comparisons

Pinned openai-community/gpt2 safetensors; float32; native SDPA reference. History preparation is separate; timed method order is randomized. The same native greedy token history is teacher-forced into every arm. Numerical bounds are elementwise `atol=1e-4, rtol=1e-4`. GPU windows for hybrid plans are fixed from the initial prompt.

### model-dc63e61cb94f — 32 prompt tokens, 48 decode steps

| Plan | Median forward ms | Decode tok/s | Logit vectors within bounds | Violating elements | Greedy agreement |
| --- | ---: | ---: | ---: | ---: | ---: |
| Hybrid · 75% GPU | 13.569 | 67.2 | 48/49 | 32,788 | 100% |
| Transfer to GPU | 7.421 | 129.2 | 49/49 | 0 | 100% |
| GPU dense reference | 7.208 | 135.8 | 48/49 | 35,217 | 100% |
| Atlas adaptive | 7.354 | 134.4 | 48/49 | 35,217 | 100% |
| Native model SDPA | 6.429 | 146.1 | 49/49 | 0 | 100% |
| Hybrid · 50% GPU | 14.367 | 69.1 | 48/49 | 30,772 | 100% |
| CPU attention | 9.008 | 110.8 | 48/49 | 33,782 | 100% |
| Hybrid · 25% GPU | 13.169 | 72.0 | 48/49 | 41,913 | 100% |
| Native GPU | 7.149 | 137.0 | 49/49 | 0 | 100% |

Model loading took 0.56 s; native reference preparation took 0.33 s. These are excluded from decode throughput, as are numerical validation, greedy selection and token-input preparation. Prefill, setup, raw timings and source identities remain in the JSON record.

### model-566cda7ff212 — 249 prompt tokens, 16 decode steps

| Plan | Median forward ms | Decode tok/s | Logit vectors within bounds | Violating elements | Greedy agreement |
| --- | ---: | ---: | ---: | ---: | ---: |
| Hybrid · 75% GPU | 13.625 | 58.4 | 16/17 | 4,054 | 100% |
| Transfer to GPU | 9.237 | 106.6 | 17/17 | 0 | 100% |
| GPU dense reference | 7.482 | 132.6 | 16/17 | 854 | 100% |
| Atlas adaptive | 6.331 | 158.2 | 16/17 | 854 | 100% |
| Native model SDPA | 6.436 | 156.1 | 17/17 | 0 | 100% |
| Hybrid · 50% GPU | 14.872 | 66.8 | 17/17 | 0 | 100% |
| CPU attention | 9.206 | 108.5 | 17/17 | 0 | 100% |
| Hybrid · 25% GPU | 18.218 | 54.6 | 16/17 | 696 | 100% |
| Native GPU | 7.094 | 136.3 | 17/17 | 0 | 100% |

Model loading took 0.56 s; native reference preparation took 0.18 s. These are excluded from decode throughput, as are numerical validation, greedy selection and token-input preparation. Prefill, setup, raw timings and source identities remain in the JSON record.

## Interpretation and next steps

- The system executes the actual workload and exposes raw evidence, including negative results. It does not establish that hybrid attention generally accelerates consumer LLM inference.
- Some methods produce matching greedy tokens while failing the stricter logit predicate on a small number of vectors. Those literal failures remain failures; no tolerance was relaxed after seeing results. Many vocabulary elements can violate a bound in one vector. These records do not establish downstream quality equivalence.
- A policy calibrated on independent synthetic attention can fail a full-model numerical contract or mispredict complete model costs. Calibration results do not certify later inference.
- Real CXL/DDR placement, sparse attention, long-context quality, low-precision kernel comparisons, multiple devices and useful OOM rescue remain future studies.
- A useful discussion with Jia Rao would focus on the observed kernel/transfer crossover, the gap between layer and whole-model behavior, and whether application hints merit a separate real-CXL experiment. No advisor collaboration or endorsement is implied.

## Provenance

Raw selected records are under `runs/`; `manifest.json` binds their bytes. Recent local run directories additionally retain source snapshots. Environment hashes, data hashes, model revision, package versions, all timing samples, preparation costs and failed predicates are retained. Shared desktop interference and Windows/WDDM scheduling limit generalization. The individual runs, not this report prose, are authoritative.

- [benchmark-8de4c6c31991](runs/benchmark-8de4c6c31991.json) · SHA-256 `61318a074a48e6d87e05f9423117611726eec67542182e5abc69a9c3aa05e4a0`
- [benchmark-bf077396b657](runs/benchmark-bf077396b657.json) · SHA-256 `8e54b7d8e5b8126d5ffae0e06211b1341a3a8a44913b17175f7be46805658a91`
- [benchmark-5791670342ea](runs/benchmark-5791670342ea.json) · SHA-256 `004a06fe819fbf90edf30b1b6bae9ecb84b71371f9cbf12854fdcf59607ee725`
- [benchmark-91d687643c2f](runs/benchmark-91d687643c2f.json) · SHA-256 `e99a5ea10c2c8de58951a386304f91d5d0036590a24fdbabb55184632223dafa`
- [model-dc63e61cb94f](runs/model-dc63e61cb94f.json) · SHA-256 `b54fe811da741601fac8e77fd09cabce1b08255b3ee17ab68941283de272008f`
- [model-566cda7ff212](runs/model-566cda7ff212.json) · SHA-256 `716f1f4028fa4d687b4b85f5e07d08d505fb199a749ae0b21f8babc42cc8df83`
