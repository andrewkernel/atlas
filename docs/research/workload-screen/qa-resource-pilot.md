# BERT-base / real QA resource segment and measurement amendment

All sixteen intended fresh-process trials complete eight logical updates each: 128 updates total, about 408.63 seconds summed launcher-to-exit workflow time. This is a 256-feature segment of the frozen 518-feature training subset, not full fine-tuning or QA evaluation. [Protocol](qa-resource-protocol.md), [raw record](qa-resource-pilot.json), [independently derived record](qa-resource-derived.json), [derivation](../../../benchmarks/derive_qa_resource.py).

## Native tradeoff on the observed desktop

Later allocated peaks are identical across the two repetitions; each time entry below is the median of five synchronized measured updates in one process. GiB means 2^30 bytes. These repeated seeds are not independent scientific outcomes or a population inference.

| Attention request | Microbatch | Per-layer checkpoint | Later allocated GiB | Update seconds, repetitions 1 / 2 |
|---|---:|---|---:|---:|
| eager | 32 | off | 14.1815 | 4.4804 / 3.4743 |
| eager | 32 | on | 2.9622 | 1.8741 / 1.8821 |
| eager | 4 | off | 3.3772 | 1.5013 / 1.5030 |
| eager | 4 | on | 1.9241 | 1.9816 / 2.0010 |
| SDPA | 32 | off | 8.7037 | 1.3370 / 1.3501 |
| SDPA | 32 | on | 2.4926 | 1.7916 / 1.7965 |
| SDPA | 4 | off | 2.6505 | 1.4525 / 1.4641 |
| SDPA | 4 | on | 1.8647 | 1.9033 / 1.9183 |

The eager full-batch run allocates 15,227,238,912 bytes against CUDA-reported device total 12,603,752,448 bytes. Completion therefore cannot establish that its allocations fit exclusively in physical VRAM. [NVIDIA documents](https://nvidia.custhelp.com/app/answers/detail/a_id/5490) driver support for falling back to system memory with a possible speed penalty. That mechanism is consistent with the observation; the local per-application driver policy and actual residency/migration were not measured. Do not assert a causal explanation for the timing difference, infer another physical GPU's feasibility, or relabel this as an OOM rescue. Driver settings were not changed. Allocated and reserved bytes remain separate fields in the raw data.

Native SDPA already removes much of the eager allocation/time cost. Its Python option does not identify the selected CUDA kernel; dispatch needs a separate diagnostic. A memory-mechanism comparison that omits this competent native baseline would overstate Atlas's opportunity. Low allocation alone does not establish complete-experiment access or equivalent training.

## Numerical observations and reference availability

All trials load the same 197 encoder tensors, apply the same 50 legacy LayerNorm key renames, initialize only the QA head, and record identical initial model, optimizer-group, input-manifest and scientific-start RNG identities. Every completed loss and preclip norm is finite. The first local builtin loss agrees with its explicit endpoint log-softmax accounting within the recorded floating-point difference.

Two full-batch repetitions have identical first reported losses but different first state, gradient, optimizer and final state hashes. The four microbatch-4 configurations repeat those hashes exactly within each configuration. Different hashes are not distances: no tolerance decision follows, and matching hashes on these finite segments do not certify other inputs or full training. Deterministic-algorithm mode was false. Dropout remains .1; first losses differ across microbatch shape and attention request. A differential checker must measure reference self-consistency before calling these differences transformation failures. A strict pathwise relation and a distributional relation require different evidence.

## Host-monitor defect, retained rather than overwritten

The original raw fields `host_sampled_peak_rss_bytes` and `host_os_lifetime_peak_working_set_bytes` measure only the environment launcher, around 5 MB. They are **invalid as training-worker host-memory measurements**. The separate [ownership control](worker-ownership-protocol.md) verifies that the actual interpreter is a descendant: with 64 MiB touched memory its working set is about 87 MB while the launcher remains about 5 MB. Natural exit and explicit cancellation of the verified interpreter leave no retained owned process alive in that bounded diagnostic.

Consequently the pilot's RSS emergency threshold did not guard the actual interpreter, and terminating only its launcher would not be a validated tree cancellation method. No emergency stop was triggered, all sixteen actual workers wrote completed results, and a post-run process inspection found no matching training worker. System available-memory sampling is still a host-wide observation; it is not attributable process memory. Preserve the frozen supervisor and raw fields as historical evidence of the defect; the derived record explicitly nulls worker RSS and marks guard applicability invalid. Do not launch more training through that supervisor until ownership is repaired and tested.

For architecture, require an owned process-tree identity and a worker rendezvous before scientific work, plus verified cleanup. On Windows, design/test a [job-object boundary](https://learn.microsoft.com/en-us/windows/win32/procthread/job-objects) and handle assignment/launch failures explicitly. Process enumeration alone is not a race-free hostile-code boundary; none is claimed here.

## Decision

This is a more meaningful native resource profile than the original synthetic model, and it strengthens the baseline. It still fails the motivating full useful-experiment rescue gate: the eager reference completed through the observed memory behavior, SDPA is a strong existing mechanism, no full training/evaluation ran, and no independently adjudicated selection study exists. Continue with objective/denominator controls, reference-noise diagnosis and precise comparison design before building an application or claiming a new optimizer.
