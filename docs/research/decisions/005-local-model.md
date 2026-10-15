# ADR 005 — Local model assistance is optional and evidence-gated

**Status:** accepted.

**Context:** a finite native plan registry may need no language model. Model memory and latency compete with local training resources; generated explanations can be wrong.

**Alternatives:** mandatory local coding agent; proprietary remote API; deterministic planner only forever; replaceable optional Ollama ranking/explanation backend.

**Evidence:** [KernelBench](https://arxiv.org/abs/2502.10517) motivates executable correctness/performance evaluation. [Ollama lifecycle controls](https://docs.ollama.com/faq) allow unloading but do not establish net benefit. Model cards differ in licenses; the 3B and 7B Qwen2.5-Coder variants must not be assumed identical in terms.

**Decision:** omit LLM from required MVP. Later evaluate registered-plan ranking/explanation with sanitized context, schema validation, no execution/waiver authority and serialized inference/training.

**Tradeoffs:** less visible AI branding; simpler causal evaluation and lower resource demand. A model could still help navigation or explanation even if it does not improve selection.

**Revisit when:** held-out equal-budget evaluation shows a practically meaningful gain after inference/review costs. Otherwise omit or keep optional explanation only.
