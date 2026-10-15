"""Original, offline CPU counterexamples; not an Atlas application or training benchmark.

No external models, datasets, checkpoints, subprocesses, or network access are used.
Each comparison starts from a deep copy of the same small model.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


def gradient_vector(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.grad.detach().reshape(-1) for p in model.parameters() if p.grad is not None])


def accumulation_probe(n: int = 7, split: int = 3, batch_norm: bool = False) -> dict:
    """Compare full-batch, sample-weighted, and mean-of-means gradients."""
    if not 0 < split < n:
        raise ValueError("split must leave two nonempty microbatches")
    torch.manual_seed(17)
    layers: list[nn.Module] = [nn.Linear(4, 4, bias=False)]
    if batch_norm:
        layers.append(nn.BatchNorm1d(4))
    layers.extend([nn.Tanh(), nn.Linear(4, 1)])
    original = nn.Sequential(*layers).double()
    x = torch.randn(n, 4, dtype=torch.float64)
    # Shift one partition so batch-dependent operations have a visible counterexample.
    x[split:] += 3.0
    y = torch.randn(n, 1, dtype=torch.float64)
    gradients = {}
    for name in ("full", "weighted", "mean_of_means"):
        model = copy.deepcopy(original).train()
        chunks = [(x, y)] if name == "full" else [(x[:split], y[:split]), (x[split:], y[split:])]
        for xi, yi in chunks:
            loss_sum = (model(xi) - yi).square().sum()
            denominator = n if name in ("full", "weighted") else len(chunks) * len(xi)
            (loss_sum / denominator).backward()
        gradients[name] = gradient_vector(model)
    return {
        "n": n,
        "split": split,
        "batch_norm": batch_norm,
        "weighted_gradient_max_abs_error": (gradients["full"] - gradients["weighted"]).abs().max().item(),
        "mean_of_means_gradient_max_abs_error": (gradients["full"] - gradients["mean_of_means"]).abs().max().item(),
    }


def checkpoint_dropout_probe(preserve_rng_state: bool) -> dict:
    torch.manual_seed(23)
    original = nn.Sequential(nn.Linear(8, 16), nn.Dropout(0.5), nn.Tanh(), nn.Linear(16, 2)).double()
    x = torch.randn(12, 8, dtype=torch.float64)
    outputs, gradients = [], []
    for use_checkpoint in (False, True):
        model = copy.deepcopy(original).train()
        torch.manual_seed(41)
        out = checkpoint(model, x, use_reentrant=False, preserve_rng_state=preserve_rng_state) if use_checkpoint else model(x)
        out.square().mean().backward()
        outputs.append(out.detach())
        gradients.append(gradient_vector(model))
    return {
        "preserve_rng_state": preserve_rng_state,
        "forward_max_abs_error": (outputs[0] - outputs[1]).abs().max().item(),
        "gradient_max_abs_error": (gradients[0] - gradients[1]).abs().max().item(),
    }


def checkpoint_buffer_probe() -> dict:
    torch.manual_seed(31)
    original = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4), nn.Tanh(), nn.Linear(4, 1)).double()
    x = torch.randn(8, 4, dtype=torch.float64)
    buffers, gradients, eval_outputs = [], [], []
    for use_checkpoint in (False, True):
        model = copy.deepcopy(original).train()
        out = checkpoint(model, x, use_reentrant=False, preserve_rng_state=True) if use_checkpoint else model(x)
        out.square().mean().backward()
        buffers.append({name: value.detach().clone() for name, value in model.named_buffers()})
        gradients.append(gradient_vector(model))
        with torch.no_grad():
            eval_outputs.append(model.eval()(x))
    return {
        "gradient_max_abs_error": (gradients[0] - gradients[1]).abs().max().item(),
        "running_mean_max_abs_error": (buffers[0]["1.running_mean"] - buffers[1]["1.running_mean"]).abs().max().item(),
        "baseline_batches_tracked": buffers[0]["1.num_batches_tracked"].item(),
        "checkpoint_batches_tracked": buffers[1]["1.num_batches_tracked"].item(),
        "evaluation_output_max_abs_error": (eval_outputs[0] - eval_outputs[1]).abs().max().item(),
    }


def precision_probe() -> dict:
    # Rounding-order counterexample: a property of the arithmetic, not a training result.
    a = torch.tensor(1e8, dtype=torch.float32)
    b = torch.tensor(-1e8, dtype=torch.float32)
    c = torch.tensor(1.0, dtype=torch.float32)
    return {"left_associated": ((a + b) + c).item(), "right_associated": (a + (b + c)).item()}


def run() -> dict:
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    return {
        "schema_version": "1.0",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "CPU float64 semantic probes and one float32 arithmetic counterexample; no training-outcome or GPU performance claim",
        "environment": {"python": sys.version, "torch": torch.__version__, "platform": platform.platform(), "device": "cpu", "threads": 1},
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "seed_policy": {"accumulation": 17, "checkpoint_dropout_model": 23, "checkpoint_dropout_execution": 41, "checkpoint_buffers": 31},
        "accumulation_unequal": accumulation_probe(),
        "accumulation_equal_control": accumulation_probe(n=8, split=4),
        "accumulation_batchnorm": accumulation_probe(n=8, split=4, batch_norm=True),
        "checkpoint_dropout_rng_preserved": checkpoint_dropout_probe(True),
        "checkpoint_dropout_rng_not_preserved": checkpoint_dropout_probe(False),
        "checkpoint_batchnorm_buffers": checkpoint_buffer_probe(),
        "float32_reduction_order": precision_probe(),
        "elapsed_seconds": time.perf_counter() - started,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))

