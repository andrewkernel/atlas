# Local LLM evaluation and authority boundary

**Decision:** no LLM in the required MVP. Evaluate one only after the deterministic planner works. There are no inference results from this session and no basis to claim a model understands unfamiliar ML code reliably.

## Candidate models

Cards are first-party descriptions, not independent Atlas evaluations. Quantization choices below are proposed GGUF/Ollama test conditions, not downloaded or measured configurations. Pin the exact Ollama manifest/blob digest, tokenizer/template and quantizer provenance before use; a model tag is mutable. First-party model revisions are in [asset snapshot](asset-snapshot.json).

| Model | Verified card properties | Proposed test / analytical weight floor | Judgment |
|---|---|---|---|
| [Qwen2.5-Coder-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct) | 7.61B parameters; Apache-2.0; long-context instructions describe extension | Q4_K_M or Q5_K_M; ideal raw4-bit payload about 3.81 GB, excluding metadata/KV/workspaces | Coding-oriented comparator; target8–12 GB or CPU offload; actual fit unverified |
| [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) | 4B; Apache-2.0; native32,768 context; reasoning/non-reasoning modes | Q4_K_M; ideal raw4-bit payload2.0 GB | First small-model candidate; cap context at 4,096–8,192 for initial study; evaluate mode and thinking budget separately |
| [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) |Apache-2.0; hybrid attention; multimodal bundle |Current small-family comparison; metadata export about 3.324 GB total listed payload |Unmeasured; [capacity audit](local-model-capacity.md) distinguishes text label, bundle, cache and runtime requirements |
| [Phi-4-mini-instruct](https://huggingface.co/microsoft/Phi-4-mini-instruct) | 3.8B; MIT;128K advertised context; function-calling format | Q4/Q5; ideal raw4-bit payload1.9 GB | Alternative family; tool-call support must be validated through selected runtime/template |
| [Qwen2.5-Coder-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct) | 3.09B;32,768 context; **Qwen Research license**, not Apache-2.0 | Exclude as default pending terms review | Important licensing counterexample: do not infer a family-wide license from the 7B model |

These are a small comparison set chosen for manageable size and inspectable cards, not a claim to identify the best models available in 2026. No vendor benchmark ranking is used to predict Atlas task performance. Four-bit parameter arithmetic is a lower-bound planning calculation; mixed-bit storage, KV cache, context, concurrency, runtime buffers and partial CPU placement determine actual memory. No tokens/sec or latency estimate is asserted.

The continuation [config/registry snapshot](local-model-metadata.json) adds exact component identities and cache arithmetic for current candidates. Native context length, config position limits, disk payload and resident memory are different quantities. No LLM weights or inference were added during this audit.

## Allowed tasks

The model may explain measured bottlenecks using evidence IDs, identify candidate code locations, rank **registered** plan IDs, propose a precondition checklist for review, or summarize a deterministic check failure. It must mark hypotheses as hypotheses. Explanation quality can be useful without improving selection, but should be evaluated separately.

Defer generated patches until a separate research arm: patch syntax, changed paths, forbidden operations, adapter integration and independent semantic checks all need validation. A model cannot certify its own output, change scientific margins, invent measurements, install packages, invoke a shell or grant execution approval. Structured JSON improves parseability, not truth. [Ollama structured outputs](https://docs.ollama.com/capabilities/structured-outputs) can constrain a response schema; deterministic validation still decides whether a registered plan is eligible.

## Interface and resource scheduling

Define `ModelBackend.propose(context, allowed_plan_ids, budget) -> ProposalSet`. An Ollama adapter is replaceable by a stub or another local backend. Inputs contain sanitized summaries and explicitly allowed source excerpts; repository comments are untrusted data, not instructions. Outputs contain plan IDs, evidence references and explanation, not executable tool calls.

Serialize inference and training on the same GPU. Request unload and verify memory release before profiling; [Ollama's keep_alive/stop controls](https://docs.ollama.com/faq) provide lifecycle mechanisms, not proof that all device memory is immediately gone. A concurrent local model can consume precisely the VRAM Atlas is meant to free. CPU inference is allowed but its latency, RAM and energy costs count. No proprietary API is required.

## Experiment

Freeze20 held-out planning cases drawn from separate workload/fault implementations, with equal information for the rule and model arms. Use the same candidate registry for the ranking experiment: here an LLM cannot improve discovery coverage beyond the enumerated set, only ordering/explanation. A separate discovery study would need new candidate mechanisms, independent implementation/review cost and its own protocol.

Compare deterministic enumeration, profile-guided ranking, random ordering and each model. Evaluate valid-plan precision/recall within the registry, time to first justified feasible plan, total overhead, malformed/unsupported suggestions, correct abstention, grounded explanation accuracy and misleading correctness claims. Use 3 prespecified generation seeds/repetitions per case as a starting variability study, not20×3 independent workload samples. Include parse failures and timeouts. Keep prompts/templates, digest, quantization, context limit, generation seed, temperature/top-p/top-k, reasoning mode, max output and elapsed/load times in the run record.

For reproducibility, archive exact outputs; fixed seeds and temperature do not guarantee identical inference across runtime/hardware versions. Qwen3's card discourages greedy decoding in thinking mode, so “temperature zero for every model” is not an evidence-based fairness policy. Prespecify a per-family recommended-setting arm and a common capped-budget arm if both are affordable.

Adoption gate: retain model assistance only if a prespecified practically meaningful gain survives held-out comparison after inference costs. If it merely restates deterministic findings, keep it as an optional explanation feature or omit it. Tool-calling claims require schema/argument validation tests; Atlas does not need the model to call tools in the initial design.
