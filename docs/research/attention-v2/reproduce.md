# Reproduce Atlas Attention Research v2

This kit is self-produced evidence. A second human has not independently reproduced it. Read the frozen protocol, its amendment and all failed/unknown outcomes before interpreting comparisons.

## Environment and assets

Use Windows, Python 3.12, an official CUDA driver and a separate environment. Exact versions and wheel hashes are in the v2 requirements. The historical CPU lock is separate. The measured machine has 32GB RAM and an RTX3060 12GB under shared Windows WDDM. Different hardware may differ substantially.

```powershell
uv venv .tools/attention-env --python 3.12
uv pip install --python .tools/attention-env/Scripts/python.exe --require-hashes -r configs/attention-v2-windows-py312-requirements.txt
.tools/attention-env/Scripts/python.exe lab.py prepare-model
.tools/attention-env/Scripts/python.exe lab.py prepare-research --model qwen3-0.6b
.tools/attention-env/Scripts/python.exe lab.py prepare-research --model corpus
```

Preparation pins builtin Qwen3-0.6B at `c1899de289a04d12100db370d81485cdf75e47ca` and Salesforce/WikiText at `b08601e04326c79dfdd32d625aee71d232d685c3`, with byte manifests. Inference is offline, safetensors-only and refuses remote model code. Qwen3 is Apache-2.0; WikiText attribution/license is recorded in its manifest. Corpus text and weights are excluded from Git. Qwen3-4B preparation failed for lack of disk space and supplies no measured evidence.

## Controls and fresh repeats

```powershell
.tools/attention-env/Scripts/python.exe -m pytest tests/test_attention_runtime.py tests/test_attention_model.py tests/test_attention_v2.py tests/test_attention_selector.py tests/test_attention_api.py tests/test_attention_study_api.py tests/test_attention_report_integrity.py -q
.tools/attention-env/Scripts/python.exe benchmarks/attention_v2_study.py analyze
.tools/attention-env/Scripts/python.exe benchmarks/attention_v2_study.py run --replication-id independent-01
.tools/attention-env/Scripts/python.exe benchmarks/attention_v2_study.py analyze --replication-id independent-01
```

The first analysis verifies stored record hashes and recomputes arithmetic. Fresh repeats use separate owned processes and write to `artifacts/attention-replications/independent-01/`, preserving original evidence. Resuming an ID skips recorded cases, including failures. A new ID creates new evidence. For one bounded case add `--case-id case-012 --replication-id cache-control-01`. Each study worker has a 420-second wall limit and owned-process cleanup. All outcomes remain in the ledger. The API accepts registered configurations only.

Execution checks the frozen core source hashes. Preserve this revision when modifying execution code; record an amendment and fresh cases instead of silently retaining a confirmatory label. Workers snapshot all attention-module and CLI bytes, including supplementary helpers, with software/hardware identities. `protocol.json` and `execution.json` retain the first failed device-name attempt; the next model run was interrupted before analysis and remains under its recorded ID.

## Interpretation

- Model comparisons match precision/context and include full-generation timing, separate validation replay, literal logit bounds and bounded real-token NLL. Matched native FP16/BF16 agreement does not prove FP32 equivalence.
- Three paired process blocks vary contiguous text offsets and cyclic arm order. Tokens inside a process are correlated; bootstrap intervals are descriptive with small n.
- Shift/ring ablations include complete append, eviction, transfers and attention. Setup is separately costed. O(1) eviction alone does not guarantee a faster full step.
- Selection compares calibration-only predictions, a budget rule and calibration-selected fixed plan. The held-out oracle is retrospective. Calibration cost and prediction errors remain visible. The soft GPU-KV budget is not a whole-process or exclusive residency guarantee.
- Pressure cases use actual 32k/40k FP32 model contexts, allocator/RSS/board observations and complete generation. Full-context numerical replay is unknown and blocks correctness recommendation. WDDM may spill into host memory; infer no OOM rescue without actual paired outcomes.

The separate NumPy float64 oracle covers GQA, extreme scores, uneven partitions, wraparound and fault controls. It strengthens finite checks without proving universal correctness. No actual CXL/DSA measurement, calibrated energy result, broad downstream quality evaluation, human external reproduction or established novelty is claimed.
