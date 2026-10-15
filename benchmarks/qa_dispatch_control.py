"""Observe actual SDPA operator/device events; no performance or equivalence claim."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

from board_energy_control import ROOT
from worker_ownership_control import save
from qa_owned_worker import await_owned_ack, run_worker


def worker(directory, nonce, microbatch):
    await_owned_ack(directory, nonce)
    result = {'status': 'running', 'stage': 'validation', 'microbatch': microbatch,
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    save(directory / 'worker.json', result)
    started = time.perf_counter()
    try:
        from qa_native_helpers import validate_runtime, load_features, build_model, sha_file, canonical
        result['runtime_identities'] = validate_runtime()
        review = json.loads((ROOT / 'docs/research/workload-screen/qa-profiler-review.json').read_text(encoding='utf-8'))
        for entry in review['files']:
            assert sha_file(ROOT / '.tools/qa-env/Lib/site-packages' / entry['path']) == entry['sha256']
        import torch
        from torch.profiler import profile, ProfilerActivity, supported_activities
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(False)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision('highest')
        arrays, manifest = load_features()
        model, preparation = build_model(2501, 'sdpa', False, 'cuda')
        supported = supported_activities()
        activities = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if ProfilerActivity.CUDA in supported else [])
        result.update({'model_preparation': preparation, 'feature_manifest_canonical_sha256': canonical(manifest),
                       'feature_rows': [0, 32], 'supported_activities': sorted(str(x) for x in supported),
                       'requested_activities': [str(x) for x in activities],
                       'enabled_backends': {name: getattr(torch.backends.cuda, name + '_sdp_enabled')()
                                            for name in ('flash', 'mem_efficient', 'math', 'cudnn')},
                       'torch': torch.__version__, 'cuda': torch.version.cuda,
                       'dropout_probabilities': sorted({m.p for m in model.modules() if isinstance(m, torch.nn.Dropout)}),
                       'tf32_matmul': torch.backends.cuda.matmul.allow_tf32,
                       'tf32_cudnn': torch.backends.cudnn.allow_tf32, 'deterministic_algorithms': torch.are_deterministic_algorithms_enabled()})
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        result['stage'] = 'profile_forward_backward'
        save(directory / 'worker.json', result)
        loss_value = 0.
        with profile(activities=activities, record_shapes=False, profile_memory=False, with_stack=False) as observed:
            torch.manual_seed(2503)
            for begin in range(0, 32, microbatch):
                batch = {name: value[begin:begin + microbatch].to('cuda') for name, value in arrays.items()}
                output = model(**batch)
                if not bool(torch.isfinite(output.loss)):
                    raise ValueError('Nonfinite QA loss')
                contribution = output.loss * (microbatch / 32)
                loss_value += float(contribution.detach())
                contribution.backward()
                del output, contribution, batch
            torch.cuda.synchronize()
        events = observed.events()
        cpu = Counter(event.name for event in events if 'CPU' in str(event.device_type))
        cuda = Counter(event.name for event in events if 'CUDA' in str(event.device_type))
        attention = {name: count for name, count in sorted(cpu.items())
                     if any(token in name.lower() for token in ('attention', 'softmax', 'dropout'))}
        result.update({'status': 'completed', 'stage': 'finished', 'logical_forward_backward_batches': 1,
                       'successful_optimizer_updates': 0, 'loss': loss_value,
                       'all_present_gradients_finite': all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None),
                       'present_gradient_parameter_tensors': sum(p.grad is not None for p in model.parameters()),
                       'event_device_type_counts': dict(Counter(str(e.device_type) for e in events)),
                       'selected_cpu_operator_counts': attention, 'cuda_event_count': sum(cuda.values()),
                       'cuda_unique_event_names': len(cuda), 'cuda_event_names_truncated': len(cuda) > 512,
                       'cuda_event_names_and_counts': dict(sorted(cuda.items())[:512]),
                       'device_kernel_capture': 'observed_device_events' if cuda else 'unavailable_no_cuda_events',
                       'profiled_forward_backward_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                       'profiled_forward_backward_peak_reserved_bytes': torch.cuda.max_memory_reserved()})
    except Exception as error:
        result.update({'status': 'error', 'error_type': type(error).__name__,
                       'message': str(error).replace(str(ROOT), '<atlas>')[:3000]})
    result['worker_wall_seconds'] = time.perf_counter() - started
    save(directory / 'worker.json', result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', type=Path)
    parser.add_argument('--nonce')
    parser.add_argument('--microbatch', type=int, choices=[32, 4])
    args = parser.parse_args()
    if args.worker:
        worker(args.worker, args.nonce, args.microbatch)
        return
    directory = ROOT / 'artifacts/qa-dispatch-control' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    result = {'captured_utc': datetime.now(timezone.utc).isoformat(), 'source_sha256': digest(Path(__file__)),
              'profiler_review_sha256': digest(ROOT / 'docs/research/workload-screen/qa-profiler-review.json'),
              'helper_sha256': {name: digest(ROOT / 'benchmarks' / name) for name in (
                  'qa_owned_worker.py', 'windows_job_scope.py', 'qa_native_helpers.py', 'worker_ownership_control.py', 'board_energy_control.py')},
              'intended_microbatches': [32, 4], 'trials': [],
              'scope': 'Instrumented SDPA forward/backward dispatch only; no optimizer update, fair timing, complete-update memory, fidelity or quality conclusion.'}
    for microbatch in result['intended_microbatches']:
        row = run_worker(Path(__file__).name, ['--microbatch', str(microbatch)], directory / str(microbatch))
        row['microbatch'] = microbatch
        result['trials'].append(row)
        save(ROOT / 'docs/research/workload-screen/qa-dispatch-control.json', result)
        print(json.dumps({'microbatch': microbatch, 'status': row['status'],
                          'kernels': row.get('worker_result', {}).get('device_kernel_capture'),
                          'operators': row.get('worker_result', {}).get('selected_cpu_operator_counts')}), flush=True)


if __name__ == '__main__':
    main()
