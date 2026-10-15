"""Explicit tiny builtin-library compatibility check, not a BERT-base profile."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform


def main():
    from qa_native_helpers import ROOT, validate_runtime
    identities = validate_runtime()
    import torch
    import transformers
    from transformers import BertConfig, BertForQuestionAnswering
    torch.set_num_threads(1)
    rows = []
    for attention in ("eager", "sdpa"):
        torch.manual_seed(2490)
        config = BertConfig(vocab_size=64, hidden_size=16, intermediate_size=32,
                            num_hidden_layers=1, num_attention_heads=2,
                            hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1)
        config._attn_implementation = attention
        config.use_cache = False
        model = BertForQuestionAnswering(config).train()
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False, "preserve_rng_state": True})
        ids = torch.tensor([[1, 2, 3, 4, 0, 0], [4, 3, 2, 1, 5, 6]])
        output = model(input_ids=ids, attention_mask=(ids != 0).long(),
                       start_positions=torch.tensor([1, 2]), end_positions=torch.tensor([2, 3]))
        output.loss.backward()
        rows.append({"attention": attention, "loss": output.loss.item(),
                     "start_shape": list(output.start_logits.shape),
                     "all_gradients_finite": all(p.grad is None or bool(torch.isfinite(p.grad).all())
                                                 for p in model.parameters())})
    result = {"captured_utc": datetime.now(timezone.utc).isoformat(),
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "helper_source_sha256": hashlib.sha256((ROOT / "benchmarks/qa_native_helpers.py").read_bytes()).hexdigest(),
              "python": platform.python_version(), "torch": torch.__version__,
              "transformers": transformers.__version__, "runtime_identities": identities,
              "trials": rows,
              "scope": "Tiny one-layer random CPU library smoke only; no pretrained BERT-base training, CUDA fit, OOM, QA quality or full equivalence result."}
    target = ROOT / "docs/research/workload-screen/qa-compatibility-smoke.json"
    target.write_bytes((json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
