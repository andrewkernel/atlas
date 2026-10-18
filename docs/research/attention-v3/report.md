# Atlas: long-context memory tradeoffs with inspectable evidence

**Atlas is a local experiment workspace that asks whether moving part of
an LLM attention cache to CPU memory saves GPU allocation while preserving
a declared numerical relation, and how much generation time that costs.**

This report covers 18/18 registered cases (complete collection).
Execution completion is separate from numerical eligibility. Failed or
unavailable arms stay in the denominator and block the affected comparison.
The system is a dense CPU/GPU baseline, not a new sparse-attention algorithm
or a CXL/DSA implementation. No external reproduction is established.

## What was actually tested

- Pinned Qwen3-0.6B, FP32, one unpadded sequence on an RTX 3060 12 GB
  and Ryzen 5 7600X with 32 GB RAM, Windows/WDDM shared desktop.
- 32,768 and 40,000 token prompts. Six cases revisit three overlapping
  WikiText inputs from v2; twelve use six separate PG-19 validation books.
- Native model execution, a persistent native-SDPA cache and a dense
  hybrid cache keeping 256 recent tokens on GPU and older entries on CPU.
- Separate unchanged native reference; all 32 full-vocabulary predictions
  checked on the same real continuation, with original atol=rtol=1e-4.
  Both controls must pass before any candidate is eligible.
- Six method-order permutations in the book cohort. Identical short
  warmups. One free 16-token generation window per arm; prefill, selection,
  cache updates, transfers and synchronized forwards included.
- Full arrays, token identities, source snapshots, raw timings, memory
  observations and logs retained. A separate NumPy verifier reconstructs
  decisions and arithmetic; it imports no runtime comparator.

## Results and limits of the claim

| Input cohort | Context | Qualified pairs / planned | Native time / hybrid time, median | Allocator peak reduction, median | Meaning |
| --- | ---: | ---: | ---: | ---: | --- |
| v2-replay | 32,768 | 3/3 | 0.525 | 63.8% | Hybrid took longer |
| v2-replay | 40,000 | 1/3 | 0.737 | 66.3% | Hybrid took longer |
| pg19-books | 32,768 | 4/6 | 0.542 | 63.8% | Hybrid took longer |
| pg19-books | 40,000 | 2/6 | 0.749 | 66.3% | Hybrid took longer |

These are descriptive paired observations, not population estimates.
A median from qualifying pairs does not repair failure on other inputs.
Allocator reductions apply to measured PyTorch peaks, not exclusive
physical VRAM residency or a hard memory quota. An OOM rescue requires
a completed, numerically qualified candidate and a failed native arm;
the predeclared two-control gate does not establish such a rescue.

4 completed arms failed the literal numerical contract; 7 arms have unavailable numerical outcomes.
These are arm counts, not a scientific-validity score. Small distribution
differences or matching greedy tokens do not override failed bounds.

| Case | Arm | Outcome | Violating elements |
| --- | --- | --- | ---: |
| long-005 | hybrid50 | Outside bounds | 87 |
| long-008 | hybrid50 | Outside bounds | 507 |
| long-013 | hybrid50 | Outside bounds | 47 |
| long-014 | hybrid50 | Outside bounds | 50 |
| long-003 | resident | oom | — |
| long-009 | resident | oom | — |
| long-013 | native | oom | — |
| long-015 | native | oom | — |
| long-015 | resident | oom | — |
| long-017 | resident | oom | — |
| long-017 | native | oom | — |

## What another researcher should try to falsify

1. Verify the shipped bytes and reconstruct every numerical check from
   the full arrays before trusting the summary.
2. Run the fixed registry under a fresh replication ID on another machine.
   Preserve native-control failures, OOMs, timeouts and all attempts.
3. Test whether the memory/time tradeoff persists on other models,
   precisions, documents, continuation lengths and separately scheduled sessions.
4. Investigate whether attention arithmetic or framework propagation
   explains literal failures. Any implementation repair needs a new
   source-bound protocol and fresh cases; do not relax old bounds.
5. Compare with competent native/offload/sparse baselines before claiming
   a policy or algorithm contribution. Actual CXL needs actual hardware.

## Remaining bias and uncertainty

One shared desktop session, one modern model family and FP32 do not
establish generalization. Long book prefixes are selected by length and
historical writing has corpus bias. These public texts may be in model
training data. Contexts within a book are nested; prior WikiText windows
overlap and were already inspected. Native reference always precedes
candidate arms. Short warmups do not establish long steady-state timing.
Arms share one loaded model and CUDA process. Device/host allocator
history may affect later arms, including OOM outcomes; clearing the
device cache is not a proof that all host/framework state was reset.
Balanced arm order does not establish per-arm process isolation. These
OOMs therefore do not prove that an arm cannot run in a fresh process.
Recorded OOM exception prefixes also contain implausibly large per-process
memory numbers. Those vendor/framework diagnostics are preserved as
text, not accepted as measured memory. NVIDIA documents per-process
usedGpuMemory as unavailable under WDDM; an unavailable diagnostic is
consistent with these values, but the exact formatting cause was not
independently traced. See the [NVML reference](https://docs.nvidia.com/deploy/nvml-api/latest/api/structnvmlProcessInfo__t.html).
Desktop/UI work and read-only analyses occurred during the shared
study session. Full-array reconstruction was scheduled in native
reference/validation phases, outside timed generation windows. There
is no exclusive resource reservation or calibrated power measurement.
The 32-position relation is not broad language quality, truth, training
fidelity or scientific-conclusion preservation. No zero-bias or universal
correctness claim is justified. Prior FP16/BF16 failures remain intact.

## Reproduce and inspect

Follow [reproduce.md](reproduce.md). The frozen [protocol](protocol.json)
binds sources, assets, cases, bounds and analysis. [execution.json](execution.json)
retains outcomes; [summary.json](summary.json) is independently reconstructed.
Full arrays stay outside Git and are included in the local researcher ZIP.
Byte verification cannot prove author honesty or real device execution;
fresh independent execution and human review remain separate.

Protocol SHA-256: `9db4132e3d09a471f9aa2a2d79103d797a1f42dcde0429c563e4e835942f0cea`.

## Isolated-process follow-up (post-hoc diagnostic)

Book 2383 was selected after its v3 numerical failure and control OOMs. Each arm/context now uses a fresh process, the same short warmup, and generation before reference preparation. This is a diagnostic of the known failure pair, not an unseen confirmation cohort.

| Context | Arm | Outcome | Literal relation | Generation (s) |
| ---: | --- | --- | --- | ---: |
| 32,768 | native | completed | pass | 50.530 |
| 40,000 | native | completed | pass | 97.773 |
| 32,768 | resident | completed | pass | 56.543 |
| 40,000 | resident | completed | pass | 74.328 |
| 32,768 | hybrid50 | completed | fail | 89.338 |
| 40,000 | hybrid50 | completed | pass | 117.696 |

Pinned host allocator observations after generation cleanup:

| Context | Arm | Allocator-owned (GiB) | Active (GiB) |
| ---: | --- | ---: | ---: |
| 32,768 | native | 0.000000 | 0.000000 |
| 40,000 | native | 0.000000 | 0.000000 |
| 32,768 | resident | 0.000000 | 0.000000 |
| 40,000 | resident | 0.000000 | 0.000000 |
| 32,768 | hybrid50 | 7.000000 | 0.000000 |
| 40,000 | hybrid50 | 14.000000 | 0.000000 |

These are runtime allocator counters, not physical-RAM or exclusive-residency measurements. The [PyTorch 2.14 API](https://docs.pytorch.org/docs/2.14/generated/torch.cuda.memory.host_memory_stats.html) distinguishes allocator-owned blocks from active requests and describes block-size rounding. Missing observations remain unknown. Retained inactive blocks would support a process-history concern, without proving which original OOM they caused.

All original v3 failures remain unchanged. Disappearing OOMs would be a counterexample to a hard feasibility interpretation, not proof of a particular allocator cause. Host allocator observations are retained, not inferred from RSS. Numerical failures still block eligibility. See [diagnostic evidence](../attention-isolation/summary.json) and its separately frozen protocol.
