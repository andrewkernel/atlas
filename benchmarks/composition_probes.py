"""Offline two-update composition controls, independent of a production Atlas runner."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


def composition_probe(fault: str | None = None) -> dict:
    if fault not in (None, 'clip_per_microbatch', 'step_per_microbatch'):
        raise ValueError('Unknown controlled fault')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(61)
    original = nn.Sequential(nn.Linear(4, 8), nn.Tanh(), nn.Linear(8, 1)).double()
    batches = [(torch.randn(7, 4, dtype=torch.float64), torch.randn(7, 1, dtype=torch.float64)) for _ in range(2)]
    final_parameters, counters = [], []
    for candidate in (False, True):
        model = copy.deepcopy(original).train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01, betas=(.9, .99), eps=1e-8, weight_decay=.01, foreach=False)
        successful_updates = 0
        for x, y in batches:
            optimizer.zero_grad(set_to_none=True)
            chunks = [(x[:3], y[:3]), (x[3:], y[3:])] if candidate else [(x, y)]
            for xi, yi in chunks:
                output = checkpoint(model, xi, use_reentrant=False, preserve_rng_state=True) if candidate else model(xi)
                ((output - yi).square().sum() / len(x)).backward()
                if candidate and fault in ('clip_per_microbatch', 'step_per_microbatch'):
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=.25)
                if candidate and fault == 'step_per_microbatch':
                    optimizer.step()
                    successful_updates += 1
                    optimizer.zero_grad(set_to_none=True)
            if not (candidate and fault == 'step_per_microbatch'):
                if not (candidate and fault == 'clip_per_microbatch'):
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=.25)
                optimizer.step()
                successful_updates += 1
        final_parameters.append(torch.cat([p.detach().reshape(-1) for p in model.parameters()]))
        counters.append({'successful_updates': successful_updates, 'adam_state_steps': sorted({int(state['step'].item()) for state in optimizer.state.values()})})
    return {
        'fault': fault,
        'scope': 'two controlled CPU float64 AdamW updates; candidate combines 3+4 accumulation and pure-region checkpointing',
        'parameter_max_abs_error': (final_parameters[0] - final_parameters[1]).abs().max().item(),
        'reference': counters[0],
        'candidate': counters[1],
    }


def run() -> dict:
    return {
        'schema_version': '1.0',
        'started_utc': datetime.now(timezone.utc).isoformat(),
        'environment': {'python': sys.version, 'torch': torch.__version__, 'platform': platform.platform(), 'device': 'cpu'},
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'seed': 61,
        'correct_composition': composition_probe(),
        'clip_timing_fault': composition_probe('clip_per_microbatch'),
        'optimizer_timing_fault': composition_probe('step_per_microbatch'),
        'limitation': 'No planner, external dataset, full training, GPU measurements or scientific outcome equivalence tested.',
    }


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=run()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2,allow_nan=False))
