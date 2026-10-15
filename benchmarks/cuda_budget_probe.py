"""Allocator-budget sensitivity for the reviewed synthetic CUDA fixture.

An allocator limit on an RTX3060 does not emulate a physical smaller GPU.
Only explicitly supplied, locally generated tensor references are accepted here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def child(budget_mib: int, microbatch: int, checkpointed: bool, reference: Path) -> dict:
    import torch
    from cuda_memory_probe import child_trial

    if not torch.cuda.is_available():
        return {'status': 'unavailable'}
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(budget_mib*2**20/total)
    completed_updates = 0
    original_step = torch.optim.AdamW.step

    def counted_step(optimizer, *args, **kwargs):
        nonlocal completed_updates
        returned = original_step(optimizer, *args, **kwargs)
        completed_updates += 1
        return returned

    torch.optim.AdamW.step = counted_step
    try:
        result = child_trial(microbatch, checkpointed, reference, write_reference=False)
    except torch.cuda.OutOfMemoryError as error:
        message = str(error)
        result = {'status': 'allocator_oom', 'driver_oom_message_excerpt': message.split('Process ')[0][:500],
                  'invalid_process_memory_diagnostic': '17179869184.00 GiB' in message,
                  'allocated_peak_bytes_before_failure': torch.cuda.max_memory_allocated(),
                  'reserved_peak_bytes_before_failure': torch.cuda.max_memory_reserved()}
    finally:
        torch.optim.AdamW.step = original_step
    result['completed_optimizer_updates'] = completed_updates
    return result


def run(reference: Path, output: Path, artifact_root: Path) -> None:
    reference = reference.resolve(strict=True)
    artifact_root = artifact_root/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    artifact_root.mkdir(parents=True)
    trials = []
    for budget, micro, check in ((450,16,False), (450,16,True), (450,4,False), (450,4,True)):
        destination = artifact_root/f'{len(trials)}.json'
        command = [sys.executable, str(Path(__file__).resolve()), '--child', '--budget-mib', str(budget),
                   '--microbatch', str(micro), '--reference', str(reference), '--output', str(destination.resolve())]
        if check:
            command.append('--checkpointed')
        environment = os.environ.copy()
        environment['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        try:
            process = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=180)
            result = json.loads(destination.read_text(encoding='utf-8')) if destination.exists() else {'status': 'error', 'reason': 'No child result'}
            result.update({'returncode': process.returncode, 'stderr': process.stderr[-4000:]})
        except subprocess.TimeoutExpired:
            result = {'status': 'timeout'}
        result.update({'allocator_budget_mib': budget, 'microbatch': micro, 'checkpointed': check})
        trials.append(result)
        record = {'schema_version': '1.0', 'started_utc': STARTED,
                  'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  'fixture_source_sha256': hashlib.sha256(Path(__file__).with_name('cuda_memory_probe.py').read_bytes()).hexdigest(),
                  'reference_sha256': hashlib.sha256(reference.read_bytes()).hexdigest(),
                  'trials': trials,
                  'limitation': 'One450 MiB PyTorch allocator cap on a physical12 GB RTX3060; not a450 MiB device, hard total-GPU quota, physical8 GB benchmark, real-workload rescue or primary selection evaluation.'}
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(record, indent=2, allow_nan=False)+'\n', encoding='utf-8')
        print(json.dumps({'budget_mib': budget, 'microbatch': micro, 'checkpointed': check, 'status': result['status']}), flush=True)


STARTED = datetime.now(timezone.utc).isoformat()
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--artifact-root', type=Path, default=Path('artifacts/cuda-budget-probe'))
    parser.add_argument('--child', action='store_true')
    parser.add_argument('--budget-mib', type=int, default=450)
    parser.add_argument('--microbatch', type=int, default=16)
    parser.add_argument('--checkpointed', action='store_true')
    args = parser.parse_args()
    if args.child:
        try:
            result = child(args.budget_mib, args.microbatch, args.checkpointed, args.reference)
        except Exception as error:
            result = {'status': 'error', 'error_type': type(error).__name__, 'reason': str(error)[:2000]}
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
        raise SystemExit(0 if result['status'] in ('completed', 'allocator_oom') else 1)
    run(args.reference, args.output, args.artifact_root)
