"""Trusted fixed BERT QA research helpers, not an arbitrary model loader."""
import hashlib
import importlib.metadata
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "docs/research/workload-screen"


def sha_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def validate_runtime():
    environment = json.loads((DIRECTORY / "qa-environment.json").read_text(encoding="utf-8"))
    expected = {row["name"].lower().replace("_", "-"): row["version"]
                for row in environment["packages"]}
    for name in ("torch", "transformers", "numpy", "safetensors", "tokenizers"):
        assert importlib.metadata.version(name) == expected[name], name
    review = json.loads((DIRECTORY / "qa-execution-review.json").read_text(encoding="utf-8"))
    site = ROOT / ".tools/qa-env/Lib/site-packages"
    for entry in review["files"]:
        assert sha_file(site / entry["path"]) == entry["sha256"], entry["path"]
    assets = json.loads((DIRECTORY / "qa-asset-manifest.json").read_text(encoding="utf-8"))
    for name in ("model.safetensors", "config.json"):
        path = ROOT / "weights/bert-base-uncased-pinned" / name
        assert sha_file(path) == assets["files"]["model/" + name]["sha256"], name
    return {"review_sha256": sha_file(DIRECTORY / "qa-execution-review.json"),
            "environment_canonical_sha256": canonical(environment),
            "asset_manifest_canonical_sha256": canonical(assets)}


def load_features():
    import numpy as np
    import torch
    manifest = json.loads((DIRECTORY / "qa-feature-manifest.json").read_text(encoding="utf-8"))
    path = ROOT / "data/squad/qa-resource-features.npz"
    assert sha_file(path) == manifest["npz_sha256"]
    tensors = {}
    with np.load(path, allow_pickle=False) as archive:
        assert set(archive.files) == set(manifest["arrays"])
        for name, recorded in manifest["arrays"].items():
            array = archive[name]
            assert array.dtype == np.dtype("<i8") and list(array.shape) == recorded["shape"]
            assert hashlib.sha256(array.tobytes(order="C")).hexdigest() == recorded["sha256_c_order_little_endian"]
            tensors[name] = torch.from_numpy(array)
    assert tensors["input_ids"].shape == (manifest["features"], 384)
    assert 0 <= tensors["input_ids"].min() and tensors["input_ids"].max() < 30522
    assert set(tensors["attention_mask"].unique().tolist()) <= {0, 1}
    assert set(tensors["token_type_ids"].unique().tolist()) <= {0, 1}
    assert ((0 <= tensors["start_positions"]) &
            (tensors["start_positions"] <= tensors["end_positions"]) &
            (tensors["end_positions"] < 384)).all()
    return tensors, manifest


def tensor_digest(items):
    digest = hashlib.sha256()
    for name, value in items:
        array = value.detach().cpu().contiguous().numpy()
        header = json.dumps({"name": name, "shape": list(array.shape),
                             "dtype": str(array.dtype)}, sort_keys=True).encode()
        digest.update(header + b"\0" + array.tobytes(order="C"))
    return digest.hexdigest()


def model_state_digest(model):
    return tensor_digest([*(('parameter/' + n, p) for n, p in model.named_parameters()),
                          *(('buffer/' + n, b) for n, b in model.named_buffers())])


def build_model(seed, attention, checkpointed, device):
    import torch
    from safetensors import safe_open
    from transformers import BertConfig, BertForQuestionAnswering
    if attention not in ("eager", "sdpa") or device not in ("cpu", "cuda"):
        raise ValueError("Only fixed builtin attention/device choices are supported")
    torch.manual_seed(seed)
    raw = json.loads((ROOT / "weights/bert-base-uncased-pinned/config.json").read_text(encoding="utf-8"))
    expected = {"hidden_size": 768, "intermediate_size": 3072, "num_hidden_layers": 12,
                "num_attention_heads": 12, "vocab_size": 30522, "max_position_embeddings": 512,
                "model_type": "bert", "hidden_dropout_prob": 0.1,
                "attention_probs_dropout_prob": 0.1}
    assert all(raw[key] == value for key, value in expected.items())
    assert "auto_map" not in raw
    config = BertConfig(**raw)
    config._attn_implementation = attention
    config.use_cache = False
    config.num_labels = 2
    model = BertForQuestionAnswering(config)
    current = model.state_dict()
    weights_path = ROOT / "weights/bert-base-uncased-pinned/model.safetensors"
    with safe_open(weights_path, framework="pt", device="cpu") as archive:
        keys = set(archive.keys())
        mapping = {}
        for original in keys:
            name = original
            if ".LayerNorm." in name:
                if name.endswith(".gamma"):
                    name = name[:-6] + ".weight"
                elif name.endswith(".beta"):
                    name = name[:-5] + ".bias"
            assert name not in mapping, "Checkpoint key normalization collision"
            mapping[name] = original
        supplied = {name: archive.get_tensor(mapping[name]) for name in current if name in mapping}
        unused = sorted(original for name, original in mapping.items() if name not in current)
        loaded_key_renames = sum(name != mapping[name] for name in supplied)
        assert all(name.startswith(("cls.", "bert.pooler.")) for name in unused)
        result = model.load_state_dict(supplied, strict=False)
    assert sorted(result.missing_keys) == ["qa_outputs.bias", "qa_outputs.weight"]
    assert not result.unexpected_keys
    del supplied, current
    if checkpointed:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False, "preserve_rng_state": True})
    assert len(model.bert.encoder.layer) == 12
    assert all(bool(layer.gradient_checkpointing) == checkpointed for layer in model.bert.encoder.layer)
    assert all(p.dtype == torch.float32 for p in model.parameters())
    initial = model_state_digest(model)
    model.to(device).train()
    return model, {"initial_state_digest": initial,
                   "parameter_count": sum(p.numel() for p in model.parameters()),
                   "loaded_parameter_tensors": len(keys) - len(unused),
                   "loaded_legacy_layernorm_key_renames": loaded_key_renames,
                   "unused_pretraining_keys": unused,
                   "initialized_qa_head_keys": sorted(result.missing_keys),
                   "attention_requested": attention, "checkpointed_layers": 12 if checkpointed else 0}


def build_optimizer(model):
    import torch
    decay, no_decay = [], []
    group_names = {"decay": [], "no_decay": []}
    for name, parameter in model.named_parameters():
        skip = name.endswith(".bias") or "LayerNorm.weight" in name
        (no_decay if skip else decay).append(parameter)
        group_names["no_decay" if skip else "decay"].append(name)
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.01}, {"params": no_decay, "weight_decay": 0.0}],
        lr=3e-5, betas=(0.9, 0.999), eps=1e-8, foreach=False, fused=False)
    return optimizer, canonical(group_names)
