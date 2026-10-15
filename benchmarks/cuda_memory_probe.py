"""Reviewed synthetic CUDA research fixture; not an Atlas runner or efficacy study.

Each plan executes in an owned child with a wall timeout. Only original model
code, generated tensors and locally generated tensor-only reference files run.
No dataset, downloaded model or third-party research repository is imported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

CONFIG = {'logical_batch': 16, 'tokens': 256, 'width': 512, 'depth': 8,
          'warmup_updates': 5, 'measured_updates': 10, 'model_seed': 884, 'data_seed': 885}


def child_trial(microbatch: int, checkpointed: bool, reference: Path, write_reference: bool) -> dict:
    import torch
    from torch import nn
    from torch.utils.checkpoint import checkpoint

    if not torch.cuda.is_available():
        return {'status': 'unavailable', 'reason': 'CUDA not available'}
    if microbatch <= 0 or CONFIG['logical_batch'] % microbatch:
        raise ValueError('Microbatch must divide the logical batch')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.fp32_precision = 'ieee'
    torch.backends.cuda.matmul.fp32_precision = 'ieee'
    torch.backends.cudnn.fp32_precision = 'ieee'
    torch.manual_seed(CONFIG['model_seed'])

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            d = CONFIG['width']
            self.norm = nn.LayerNorm(d)
            self.up = nn.Linear(d, 4*d)
            self.down = nn.Linear(4*d, d)

        def forward(self, x):
            return x + .1*self.down(torch.nn.functional.gelu(self.up(self.norm(x))))

    start = time.perf_counter()
    model = nn.Sequential(*(Block() for _ in range(CONFIG['depth']))).to('cuda').train()
    generator = torch.Generator(device='cpu').manual_seed(CONFIG['data_seed'])
    data = torch.randn(CONFIG['logical_batch'], CONFIG['tokens'], CONFIG['width'], generator=generator).to('cuda')
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(.9, .999), eps=1e-8, weight_decay=.01, foreach=False, fused=False)
    torch.cuda.synchronize()
    preparation_seconds = time.perf_counter()-start
    initial_allocated = torch.cuda.memory_allocated()

    def update():
        optimizer.zero_grad(set_to_none=True)
        for chunk in data.split(microbatch):
            value = chunk
            for block in model:
                value = checkpoint(block, value, use_reentrant=False, preserve_rng_state=True) if checkpointed else block(value)
            (value.square().sum()/data.numel()).backward()
        optimizer.step()

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    first_start = time.perf_counter()
    update()
    torch.cuda.synchronize()
    first_seconds = time.perf_counter()-first_start
    first_peak = {'allocated_bytes': torch.cuda.max_memory_allocated(), 'reserved_bytes': torch.cuda.max_memory_reserved()}
    first = {}
    for name, parameter in model.named_parameters():
        first[f'parameter/{name}'] = parameter.detach().cpu().clone()
        first[f'gradient/{name}'] = parameter.grad.detach().cpu().clone()
        for key, value in optimizer.state[parameter].items():
            if torch.is_tensor(value):
                first[f'optimizer/{name}/{key}'] = value.detach().cpu().clone()
    for name, value in model.named_buffers():
        first[f'buffer/{name}'] = value.detach().cpu().clone()
    if write_reference:
        reference.parent.mkdir(parents=True, exist_ok=True)
        torch.save(first, reference)
    baseline = torch.load(reference, map_location='cpu', weights_only=True)
    if baseline.keys() != first.keys():
        raise ValueError('Reference and candidate state keys differ')
    errors = {}
    for name, value in first.items():
        expected = baseline[name]
        if value.shape != expected.shape or value.dtype != expected.dtype:
            raise ValueError('State shape/dtype mismatch')
        difference = value.double()-expected.double()
        group = 'optimizer/'+name.rsplit('/', 1)[1] if name.startswith('optimizer/') else name.split('/')[0]
        record = errors.setdefault(group, {'max_abs_error': 0., 'squared_error': 0., 'reference_squared_norm': 0., 'all_finite': True, 'tensors': 0})
        record['max_abs_error'] = max(record['max_abs_error'], difference.abs().max().item())
        record['squared_error'] += difference.square().sum().item()
        record['reference_squared_norm'] += expected.double().square().sum().item()
        record['all_finite'] &= bool(torch.isfinite(value).all() and torch.isfinite(expected).all())
        record['tensors'] += 1
    for record in errors.values():
        record['relative_l2_error'] = (record['squared_error']/record['reference_squared_norm'])**.5 if record['reference_squared_norm'] else None
    del first, baseline
    for _ in range(CONFIG['warmup_updates']):
        update()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    durations = []
    for _ in range(CONFIG['measured_updates']):
        torch.cuda.synchronize()
        began = time.perf_counter()
        update()
        torch.cuda.synchronize()
        durations.append(time.perf_counter()-began)
    properties = torch.cuda.get_device_properties(0)
    return {'status': 'completed', 'microbatch': microbatch, 'checkpointed': checkpointed,
            'environment': {'torch': torch.__version__, 'cuda_runtime': torch.version.cuda, 'python': sys.version,
                            'platform': platform.platform(), 'gpu': properties.name, 'device_total_bytes': properties.total_memory,
                            'capability': list(torch.cuda.get_device_capability(0)), 'allocator': torch.cuda.memory.get_allocator_backend(),
                            'matmul_fp32_precision': torch.backends.cuda.matmul.fp32_precision, 'deterministic_algorithms': torch.are_deterministic_algorithms_enabled()},
            'parameter_count': sum(p.numel() for p in model.parameters()), 'dtype': 'float32',
            'model_input_preparation_seconds': preparation_seconds, 'initial_allocated_bytes': initial_allocated,
            'first_update_seconds': first_seconds, 'first_update_peak': first_peak,
            'measured_update_seconds': durations, 'median_update_seconds': statistics.median(durations),
            'measured_peak': {'allocated_bytes': torch.cuda.max_memory_allocated(), 'reserved_bytes': torch.cuda.max_memory_reserved()},
            'first_update_comparison': errors, 'completed_logical_updates': 1+CONFIG['warmup_updates']+CONFIG['measured_updates']}


def run(output: Path, artifact_root: Path) -> dict:
    # A unique owned directory prevents stale child evidence/reference reuse.
    artifact_root = artifact_root/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    artifact_root.mkdir(parents=True, exist_ok=True)
    reference = artifact_root/'first-update-reference.pt'
    trials = []
    plans = [(16, False), (4, False), (16, True), (4, True)]
    order = []
    for repetition in (1, 2):
        ordered = plans.copy()
        random.Random(886+repetition).shuffle(ordered)
        if repetition == 1:
            ordered.remove((16, False))
            ordered.insert(0, (16, False))
        order += [(repetition, micro, check) for micro, check in ordered]
    for number, (repetition, micro, check) in enumerate(order):
        destination = artifact_root/f'trial-{number}.json'
        command = [sys.executable, str(Path(__file__).resolve()), '--child', '--microbatch', str(micro),
                   '--reference', str(reference.resolve()), '--output', str(destination.resolve())]
        if check:
            command.append('--checkpointed')
        if number == 0:
            command.append('--write-reference')
        environment = os.environ.copy()
        environment['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        began = time.perf_counter()
        try:
            process = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=180, check=False)
            record = json.loads(destination.read_text(encoding='utf-8')) if destination.exists() else {'status': 'error', 'reason': 'No child result'}
            record.update({'returncode': process.returncode, 'stderr': process.stderr[-4000:]})
        except subprocess.TimeoutExpired:
            record = {'status': 'timeout', 'timeout_seconds': 180}
        record.update({'repetition': repetition, 'microbatch': micro, 'checkpointed': check, 'owned_child_wall_seconds': time.perf_counter()-began})
        trials.append(record)
        output.parent.mkdir(parents=True, exist_ok=True)
        partial = {'schema_version': '1.0', 'started_utc': STARTED, 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   'config': CONFIG, 'trials': trials, 'limitations': 'Synthetic separable residual MLP, repeated fixed input, two process repetitions per plan. No dataset, attention, OOM rescue, planner, full training, energy or scientific-equivalence result. Timing samples within a process are dependent.'}
        output.write_text(json.dumps(partial, indent=2, allow_nan=False)+'\n', encoding='utf-8')
        print(json.dumps({'trial': number, 'repetition': repetition, 'microbatch': micro, 'checkpointed': check, 'status': record['status'],
                          'peak': record.get('measured_peak'), 'median_seconds': record.get('median_update_seconds')}, allow_nan=False), flush=True)
    return partial


STARTED = datetime.now(timezone.utc).isoformat()
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--artifact-root', type=Path, default=Path('artifacts/cuda-memory-probe'))
    parser.add_argument('--child', action='store_true')
    parser.add_argument('--microbatch', type=int, default=16)
    parser.add_argument('--checkpointed', action='store_true')
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--write-reference', action='store_true')
    args = parser.parse_args()
    if args.child:
        try:
            record = child_trial(args.microbatch, args.checkpointed, args.reference, args.write_reference)
        except Exception as error:
            record = {'status': 'error', 'error_type': type(error).__name__, 'reason': str(error)[:2000]}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(record, indent=2, allow_nan=False)+'\n', encoding='utf-8')
        raise SystemExit(0 if record['status'] == 'completed' else 1)
    run(args.output, args.artifact_root)
