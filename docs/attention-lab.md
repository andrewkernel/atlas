# Atlas Attention Lab

An executable research workspace for exact dense attention on a consumer CPU and GPU. Inspect saved measurements, run fresh experiments, compare a pretrained GPT-2 model, and export the evidence. This owner-authorized implementation track supplements the earlier training fidelity studies; it does not establish a CXL runtime, sparse HGCA/BEYOND reproduction, or research novelty.

## Open the application

On the configured Windows machine, run `./Start-Atlas.ps1` from the Atlas repository. It starts a hidden local backend and opens http://127.0.0.1:8765/. Re-running reuses a responding server. Use `-NoBrowser` to start without opening the browser. `./Stop-Atlas.ps1` verifies the listener belongs to this checkout before stopping it; saved runs remain. It is a local application: a remotely hosted static page cannot run this GPU.

The dashboard supports new sweeps, CPU-thread/sample/selection-budget controls, live progress and cancellation, run history, chart-series toggles and logarithmic scale, phase/plan filters, sorting, raw-sample inspection, evidence/lineage export, a custom-prompt model comparison, architecture notes, all five original papers, and the earlier Atlas dashboard.

## Install a separate, pinned environment

Python 3.12 and an NVIDIA CUDA-compatible driver are required for the measured profile. The existing CPU `uv.lock` and earlier worker environments remain separate. The Windows requirements include exact package versions, hashes, and the official CUDA wheel URL. This profile was run with Python 3.12.10, PyTorch 2.14.1+cu130, Transformers 5.19.0, a Ryzen 5 7600X and an RTX 3060 12 GB under Windows/WDDM.

```powershell
.venv/Scripts/uv.exe venv .tools/attention-env --python 3.12
.venv/Scripts/uv.exe pip install --python .tools/attention-env/Scripts/python.exe --require-hashes -r configs/attention-windows-py312-requirements.txt
.tools/attention-env/Scripts/python.exe lab.py prepare-model
./Start-Atlas.ps1
```

Use an existing official uv installation if the repository-local uv executable is absent. Asset preparation downloads only allowlisted model configuration/tokenizer/safetensors files at an exact resolved revision and saves a byte manifest. Inference is offline, uses builtin GPT-2 code and refuses remote model code and pickle checkpoints. Python packages are official registry wheels; no upstream research repositories are executed.

## CLI and evidence

```powershell
.tools/attention-env/Scripts/python.exe lab.py benchmark --profile full --repeats 30 --seed 2026
.tools/attention-env/Scripts/python.exe lab.py benchmark --profile full --repeats 30 --kv-budget-mib 48
.tools/attention-env/Scripts/python.exe lab.py model --new-tokens 48
.tools/attention-env/Scripts/python.exe lab.py serve
.tools/attention-env/Scripts/python.exe -m pytest tests/test_attention_runtime.py tests/test_attention_model.py tests/test_attention_api.py -q
```

Each run lives in `artifacts/attention-lab/<id>/result.json`. Recent runs also retain the actual worker source bytes under `source/`. Model assets are under `weights/gpt2-attention-lab/`. Neither directory is tracked. Small, selected sanitized records and the report live under `docs/research/attention-lab/`. Earlier pilot results remain in the local run history, with their original source identities and smaller comparator sets.

API: `GET /api/runs`, `GET /api/runs/<id>`, `GET /api/runs/<id>/export?format=json|csv`. Mutations require the per-session token from `/api/session` and a matching origin. `POST /api/runs` accepts only bounded benchmark/model configurations; `POST /api/runs/<id>/cancel` stops only the owned run. No arbitrary scripts, interpreters or file paths are accepted.

## Research contract

- Seven fixed plans: native GPU SDPA, direct dense GPU reference, transfer-to-GPU SDPA, CPU dense attention, and three dense CPU-GPU splits. All use identical entries and float32 inputs. Hybrid is stable log-sum-exp fusion, without sparse pruning.
- Full synthetic sweep: 512/2,048/8,192/32,768 calibration lengths; 1,024/4,096/16,384 held out. B=1, H=12, D=64, one decode query. Inputs are synthetic and seeded. This is not a long-context model-quality experiment.
- Three warmups and 5–30 synchronized wall-time samples per plan. Method order is randomized per length. Inputs and cache setup are outside layer timing and separately recorded. CPU-query transfer, attention, return transfer and fusion are inside. Calibration cost is recorded separately from selected execution latency.
- Policy linearly interpolates validated calibration medians. The held-out oracle is diagnostic only. Fixed comparator is selected using calibration, eligible under the same maximum held-out KV budget. A selection budget is not a whole-process memory cap; every plan is still measured.
- GPT-2 uses its real pretrained weights, stock positional limit, one unpadded sequence and persistent caches. Native reference history is prepared separately, then timed method order is randomized. Every full-vocabulary logit vector, including prefill and the final decode step, is compared on the same teacher-forced history. Token-forward timings include cache append/eviction/transfers, but exclude validation, greedy selection and token-input preparation. These timings are not complete user-perceived text-generation latency.
- Float64 independent dense synthetic reference; native SDPA model reference. Predeclared `atol=1e-4, rtol=1e-4`, elementwise. Max absolute error may exceed atol while the combined bound still passes. Greedy agreement is a diagnostic, not downstream quality equivalence.

The full architecture and pre-implementation review are in [hybrid-attention-lab.md](architecture/hybrid-attention-lab.md). Shared desktop activity, driver scheduling, allocator history, small repeats and workload choice limit inference. Allocator peaks do not prove exclusive physical VRAM residency. The GPU direct implementation is a necessary comparator: early SDPA-only comparisons overstated the case for hybrid execution.

## Relationship to the papers

[HGCA](https://arxiv.org/abs/2507.03153) and [BEYOND](https://huilucs.github.io/pubs/hpdc2026.pdf) motivate computation across CPU and GPU and stable partition fusion. Atlas's dense Python implementation is independently authored, narrower, and not their optimized sparse implementation. [NOMAD](https://www.usenix.org/system/files/osdi24-xiang.pdf), [MIKU](https://arxiv.org/abs/2503.17864v2), and [DSA-2LM](https://ranger.uta.edu/~jrao/papers/atc25.pdf) motivate future migration/interference studies. No CXL hardware, DSA execution, advisor endorsement, or publication novelty is claimed.
