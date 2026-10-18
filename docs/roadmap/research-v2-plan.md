# Research v2 implementation and acceptance plan

Owner requested all eight audit findings be addressed end to end. This is the authoritative work plan; unchecked gates are unfinished, not waived. Historical v1 records remain immutable. Development diagnoses use legacy prompts or a separate corpus training split. A new protocol and exact assets must be frozen before inspecting new evaluation results.

## Requirements and proof

| Audit finding | Implementation | Required evidence |
| --- | --- | --- |
| Correctness | Layer/token diagnosis, independent numerical oracle, explicit candidate eligibility and safe abstention/fallback | Preserve v1 failures; new diagnostic report; fault controls; rejected candidates never promoted by speed |
| Representative workloads | Builtin GPT-2 and Qwen3 adapters; pinned WikiText public corpus; longer native-context inference | Exact revisions, split/offset/input hashes; measured multi-context model cases and language-model NLL, plus separately labeled synthetic cases |
| Statistical design | Frozen protocol, independent process blocks, changing balanced method orders, paired estimates/cluster bootstrap confidence intervals | Run-level replication, all failures included, raw samples; fresh evaluation cases never used for tuning; no token-level pseudo-replication |
| Practical baselines | Matched FP32, FP16 and BF16 where supported; native SDPA backend identity and direct dense reference | Actual backend diagnostics, unsupported statuses, dtype-specific predeclared contracts and fair timing scopes |
| Independent oracle | Separately written NumPy float64 head-wise attention with GQA and stable normalization | Adversarial, uneven-partition, ring-wrap and corruption controls; real captured Q/K/V comparisons where feasible |
| Cache overhead | O(1) ring append, no window-sized clone/shift, bounded persistent cache | Wraparound and capacity controls; shift-versus-ring ablation includes append, eviction, attention and transfer costs |
| Adaptation | Domain/identity-bound calibration, out-of-domain abstention, budget-aware rule baseline and best fixed comparator | Calibration-only decisions, reported prediction error/regret/total cost; held-out oracle never informs selection |
| Real memory pressure | Near-capacity Qwen3-0.6B FP32 32k/40k study after larger asset exceeded disk capacity, native vs offloaded cache, device/host observations | Real measured prefill/decode/complete generation windows, all outcomes retained; WDDM spill and soft observations never called a hard VRAM quota |
| README/dashboard | README architecture diagram, research controls, validity badges, confidence intervals, protocol/lineage and study views | Browser-to-worker-to-record-to-render verification and inspected desktop/narrow layouts |

## Architecture review before v2 implementation

1. Keep v1 raw records and report. Version new contracts and schemas; do not retroactively loosen tolerances.
2. Ring cache stores recent GPU entries once. Wraparound is represented as two segments and stable LSE fusion; a mirrored buffer would double memory and is excluded.
3. CPU math promotes low-precision cached values to float32 explicitly; conversion belongs inside execution timing. Query heads and KV heads are distinct; GQA must be checked against an independent oracle.
4. Qwen3 adapter uses the installed reviewed Transformers projection, Q/K normalization, rotary embeddings and native causal prefill. No custom remote model code. Request final-token logits only during long prefill to avoid manufacturing a vocabulary-logit OOM.
5. Numerical diagnostics, reference preparation and telemetry sampling are separately costed. Full generation timings include token selection, transfers, cache maintenance and all forward calls, with numerical validation in a separate replay.
6. Completed execution and evidence validity are different fields. Unknown evidence cannot make a candidate eligible. A failed synthetic or model relation is retained and blocks recommendation under that relation.
7. Calibrations bind device, software, dtype, heads, KV heads, dimension, batch, thread count, implementation and execution scope. No extrapolation from a layer benchmark to a full model without an explicit out-of-domain label.
8. Full-process independent repeats are the uncertainty unit. Adjacent tokens or repeated kernels are not independent research replications. Protocol changes after inspecting evaluation require an amendment and fresh future cases.
9. No actual CXL hardware is present. CPU/GPU results and a reproduction kit can be implemented locally; real CXL claims and human external review remain explicitly unverified.

The architecture review is a self-review, not external endorsement. A useful null/negative result remains a valid research deliverable; performance superiority and novelty are not acceptance criteria that may be fabricated.

## Work ledger

- [x] Numerical diagnosis and independent oracle
- [x] Ring/GQA/precision runtime and meaningful controls
- [x] Pinned corpus and model assets
- [x] Frozen protocol and provenance gate
- [x] Repeated full-model and precision studies
- [x] Real memory-pressure study
- [x] Statistical analysis and baseline ablations
- [x] Dashboard/README/reproduction kit and full verification

## Acceptance closeout

The bounded implementation gates above are satisfied by the 51-case amended study, 579 raw rows and exact-hash/arithmetic reconstruction. Numerical outcomes are 531 pass, 36 fail and 12 unknown; completed execution never overrides eligibility. See `docs/research/attention-v2/assessment.md` for findings and research gates beyond this artifact's scope.

Controls: 57 attention/API/integrity tests passed in the CUDA environment; the locked CPU suite passed 89 with five explicit optional/CUDA skips. Browser verification launched case-012 through the actual API/owned worker, observed six passing rows, filtered/inspected raw samples and downloaded the complete 51-case summary. Desktop and 390px responsive layouts were inspected; the latter has contained horizontal table scrolling and no page overflow. No browser console errors were observed. Screenshots are local under `artifacts/attention-v2-ui/`.

The CLI repetition path was also exercised with `--case-id case-012 --replication-id reproduction-smoke-01`: a new owned process produced six passing rows and its separate partial analysis correctly reports 1/51 cases. Re-running the already-complete main ledger returns without changing it. This is a self-reproduction smoke check, not external independent reproduction.

Corrections are explicit: original device-resolution attempt retained; execution-KV accounting declared in `selector-amendment.json` before inspecting selector outcomes; early unavailable NVML observations remain unknown and a timestamped external shared-board trace supplements later cases. The worker environment fix was verified by the separate browser repeat. Full-pressure numerical validation, human external reproduction and actual CXL evidence are not passed gates.
