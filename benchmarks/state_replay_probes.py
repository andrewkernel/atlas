"""Original CPU controls for recomputation state and partitioned stochasticity.

The buffer context is a fixture for reviewed BatchNorm, not a generic safe wrapper.
No external repository code, datasets, models or network calls are executed.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import sys
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


def gradients(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.grad.detach().flatten() for p in model.parameters()])


@contextmanager
def restore_registered_buffers(module: nn.Module):
    """Restore existing buffer values after this fixture's recomputation.

    Does not cover Python attributes, I/O, buffer replacement, hooks, aliasing,
    state used by the backward graph, custom RNGs or distributed effects.
    """
    saved = [(buffer, buffer.detach().clone()) for buffer in module.buffers()]
    try:
        yield
    finally:
        with torch.no_grad():
            for buffer, value in saved:
                buffer.copy_(value)


def batchnorm_recompute_control() -> dict:
    torch.manual_seed(771)
    original = nn.Sequential(nn.Linear(4, 8), nn.BatchNorm1d(8), nn.Tanh(), nn.Linear(8, 1)).double()
    x = torch.randn(12, 4, dtype=torch.float64) + 2
    observations = {}
    for mode in ('reference', 'native_checkpoint', 'buffer_restore_checkpoint'):
        model = copy.deepcopy(original).train()
        if mode == 'reference':
            output = model(x)
        else:
            options = {}
            if mode == 'buffer_restore_checkpoint':
                options['context_fn'] = lambda: (nullcontext(), restore_registered_buffers(model))
            output = checkpoint(model, x, use_reentrant=False, preserve_rng_state=True, **options)
        output.square().mean().backward()
        grad = gradients(model)
        buffer = {name: value.clone() for name, value in model.named_buffers()}
        with torch.no_grad():
            evaluation = model.eval()(x)
        observations[mode] = {'gradient': grad, 'buffers': buffer, 'evaluation': evaluation}
    reference = observations['reference']
    return {mode: {'gradient_max_abs_error': (record['gradient']-reference['gradient']).abs().max().item(),
                   'buffer_max_abs_error': max((record['buffers'][key]-value).abs().max().item() for key, value in reference['buffers'].items()),
                   'batches_tracked': int(record['buffers']['1.num_batches_tracked']),
                   'evaluation_max_abs_error': (record['evaluation']-reference['evaluation']).abs().max().item()}
            for mode, record in observations.items()}


def stochastic_partition_control(dropout_layers: int = 1) -> dict:
    if dropout_layers not in (1, 2):
        raise ValueError('This fixture supports one or two dropout layers')
    torch.manual_seed(772)
    layers = [nn.Linear(5, 16), nn.Dropout(.5), nn.Tanh()]
    if dropout_layers == 2:
        layers += [nn.Linear(16, 16), nn.Dropout(.5), nn.Tanh()]
    original = nn.Sequential(*layers, nn.Linear(16, 1)).double()
    x, y = torch.randn(8, 5, dtype=torch.float64), torch.randn(8, 1, dtype=torch.float64)
    records = {}
    for partition in ('full', '4+4'):
        model = copy.deepcopy(original).train()
        torch.manual_seed(773)
        outputs = []
        chunks = [(x, y)] if partition == 'full' else [(x[:4], y[:4]), (x[4:], y[4:])]
        for xi, yi in chunks:
            output = model(xi)
            outputs.append(output.detach())
            ((output-yi).square().sum()/len(x)).backward()
        records[partition] = {'gradient': gradients(model), 'output': torch.cat(outputs), 'rng': torch.get_rng_state().clone()}
    return {'dropout_layers': dropout_layers,
            'gradient_max_abs_error': (records['full']['gradient']-records['4+4']['gradient']).abs().max().item(),
            'forward_max_abs_error': (records['full']['output']-records['4+4']['output']).abs().max().item(),
            'final_rng_states_equal': torch.equal(records['full']['rng'], records['4+4']['rng']),
            'limitation': 'One CPU dropout implementation/shape. Different masks can preserve a distribution while failing a pathwise contract.'}


def run() -> dict:
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    return {'schema_version': '1.0', 'started_utc': datetime.now(timezone.utc).isoformat(),
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'environment': {'python': sys.version, 'torch': torch.__version__, 'platform': platform.platform(), 'device': 'cpu'},
            'seeds': [771, 772, 773], 'batchnorm_recompute': batchnorm_recompute_control(),
            'stochastic_partition': stochastic_partition_control(),
            'stochastic_interleaving': stochastic_partition_control(dropout_layers=2),
            'limitation': 'Original fixtures only; no FairScale execution, generic buffer transaction guarantee, GPU or full training.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2, allow_nan=False))
