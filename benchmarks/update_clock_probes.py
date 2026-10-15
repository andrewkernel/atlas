"""Original CPU fixtures for unused gradients and successful-update clocks."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import nn


def unused_parameter_probe() -> dict:
    records = {}
    for set_to_none in (True, False):
        shared = nn.Parameter(torch.tensor(.5, dtype=torch.float64))
        conditional = nn.Parameter(torch.tensor(1.5, dtype=torch.float64))
        optimizer = torch.optim.AdamW([shared, conditional], lr=.1, weight_decay=.2, foreach=False)
        optimizer.zero_grad(set_to_none=set_to_none)
        (shared+conditional).square().backward()
        optimizer.step()
        after_first = conditional.item()
        optimizer.zero_grad(set_to_none=set_to_none)
        loss = shared.square()
        loss.backward()
        unused_grad_is_none = conditional.grad is None
        optimizer.step()
        records[str(set_to_none)] = {'second_loss': loss.item(), 'conditional_after_first': after_first,
                                    'conditional_after_second': conditional.item(), 'shared_after_second': shared.item(),
                                    'unused_gradient_is_none': unused_grad_is_none,
                                    'conditional_optimizer_steps': int(optimizer.state[conditional]['step'].item())}
    return {'set_to_none': records['True'], 'zero_tensor': records['False'],
            'conditional_parameter_abs_difference': abs(records['True']['conditional_after_second']-records['False']['conditional_after_second'])}


def skipped_step_probe() -> dict:
    class CountedSGD(torch.optim.SGD):
        def __init__(self, params):
            super().__init__(params, lr=.1)
            self.successful_updates = 0

        def step(self, closure=None):
            returned = super().step(closure)
            self.successful_updates += 1
            return returned

    results = {}
    for wrong_clock in (False, True):
        parameter = nn.Parameter(torch.tensor(1., dtype=torch.float64))
        optimizer = CountedSGD([parameter])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=.5)
        scaler = torch.amp.GradScaler('cpu', init_scale=16., growth_interval=100)
        ema = torch.tensor(1., dtype=torch.float64)
        events = []
        for factor in (1., float('inf'), 1.):
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(parameter*factor).backward()
            gradient_finite = bool(torch.isfinite(parameter.grad))
            previous = optimizer.successful_updates
            scaler.step(optimizer)
            scaler.update()
            did_update = optimizer.successful_updates != previous
            if did_update or wrong_clock:
                scheduler.step()
                ema.mul_(.9).add_(parameter.detach(), alpha=.1)
            events.append({'gradient_finite': gradient_finite, 'optimizer_updated': did_update,
                           'successful_updates': optimizer.successful_updates, 'parameter': parameter.item(),
                           'lr_after_scheduler': optimizer.param_groups[0]['lr'], 'ema': ema.item(), 'scale': scaler.get_scale()})
        results['attempt_clock' if wrong_clock else 'successful_update_clock'] = events
    return {'events': results, 'final_parameter_abs_difference': abs(results['attempt_clock'][-1]['parameter']-results['successful_update_clock'][-1]['parameter']),
            'scope': 'CPU GradScaler with an injected infinite gradient; no FP16/BF16 training or GPU arithmetic study'}


def run() -> dict:
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    return {'schema_version': '1.0', 'started_utc': datetime.now(timezone.utc).isoformat(),
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'environment': {'python': sys.version, 'torch': torch.__version__, 'platform': platform.platform(), 'device': 'cpu'},
            'unused_parameter': unused_parameter_probe(), 'skipped_optimizer_step': skipped_step_probe(),
            'limitation': 'Known API-semantics fixtures, not novel failures, full training or a generic verifier.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2, allow_nan=False))
