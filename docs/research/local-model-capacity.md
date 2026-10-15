# Local-model capacity audit — metadata and arithmetic only

The original comparison set was deliberately small, but continued research should include a current small-model family. This note adds [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) as an **unmeasured candidate**, not a recommended winner. No model/projector weights were downloaded, no inference ran, and the decision to omit a mandatory LLM remains unchanged.

## First-party identities

[Read-only snapshot source](../../benchmarks/snapshot_local_model_metadata.py) captures exact Hugging Face revisions/config hashes and the Ollama registry manifest/config in [metadata JSON](local-model-metadata.json). The original asset snapshot remains historical. Retrieved2026-10-15.

- Qwen3.5-4B metadata declares Apache-2.0. Its config has 32 text layers, eight full-attention and24 linear-attention layers. Full-attention KV heads4, head dimension256. The advertised4B text label is not total bundle storage: the safetensors metadata counts about 4.66B tensors across included components.
- Qwen3-4B has 36 attention layers, eight KV heads and head dimension128. Its card describes32,768 native context while the config's40,960 position value is explained separately in that card. Record prompt/output limits and runtime settings rather than treating every context field as interchangeable.
- Qwen2.5-Coder-7B-Instruct has 28 layers, four KV heads and head dimension128 derived from 3584/28 in its config.

The inspected Ollama `qwen3.5:4b` export lists a2,648,593,216-byte model layer and675,569,120-byte projector layer; total listed payload about 3.324 GB. Config declares Q4_K_M, renderer/parser `qwen3.5`, and minimum runtime0.30.0. These are registry metadata, not observed resident memory or proof that the installed runtime executes it correctly. A raw-response SHA256 is a captured-byte fingerprint; do not confuse it with the website's displayed tag identifier. Component digests are preserved.

## Cache arithmetic

For ordinary full attention with one sequence, the unquantized two-byte K/V payload is `2 * full_attention_layers * KV_heads * head_dimension * context_tokens * 2 bytes`. Multiply by simultaneous sequences where applicable. This formula excludes weights, allocation padding, workspaces, logits, activations, offload transfers and backend overhead. For hybrid models it also excludes recurrent/linear-attention state; it is therefore only a component estimate.

| Model/config | Full-attention KV bytes per token |4,096 tokens(MiB) |8,192 tokens(MiB) |32,768 tokens(MiB) |
|---|---|---|---|---|
|Qwen3-4B |147,456 |576 |1152 |4608 |
|Qwen2.5-Coder-7B-Instruct |57,344 |224 |448 |1792 |
|Qwen3.5-4B, full-attention part only |32,768 |128 |256 |1024 |

At the Qwen3.5 config's262,144-token native limit, the full-attention component alone is8192 MiB under these assumptions. A long advertised context therefore does not establish that it fits with weights and other state on this12 GB GPU. Hybrid state and the exact runtime implementation must be measured. A quantized weight export does not imply a quantized cache; the [Ollama FAQ](https://docs.ollama.com/faq) separately documents the default f16 cache, cache-quantization choices and concurrency's memory scaling.

## Study consequence

Use an explicit4,096/8,192 context cap and concurrency1 in an initial optional comparison. Pin all model/projector/template/config digests, selected runtime/backend, cache dtype, reasoning/output budget and actual offload placement. Record cold load, prefill, generation, resident RAM/VRAM, unload verification and task quality separately. Do not infer tokens/sec, reliable code understanding or tool correctness from model size, family marketing or this arithmetic. A newer candidate can still lose to deterministic ranking or an older text-only model on the Atlas task.
