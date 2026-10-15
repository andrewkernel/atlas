"""Reduced-batch real BERT state capture using an existing reviewed comparator."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
from pathlib import Path
import time

from board_energy_control import ROOT
from worker_ownership_control import save
from qa_owned_worker import await_owned_ack, run_worker

ARMS = [('reference', 3, False), ('repeat', 3, False), ('checkpoint', 3, True),
        ('microbatch', 2, False), ('microbatch_checkpoint', 2, True)]


def tensor_metrics(reference, candidate):
    import numpy as np
    from trainparity.state import FrozenMapping, FrozenSequence, FrozenTensor

    def flatten(value, path=()):
        if isinstance(value, FrozenTensor):
            yield path, value
        elif isinstance(value, FrozenMapping):
            for key, child in value.entries:
                yield from flatten(child, (*path, key))
        elif isinstance(value, FrozenSequence):
            for index, child in enumerate(value.items):
                yield from flatten(child, (*path, index))

    left, right = dict(flatten(reference.state)), dict(flatten(candidate.state))
    assert left.keys() == right.keys()
    totals = {}
    for path, first in left.items():
        second = right[path]
        assert (first.shape, first.dtype, first.device) == (second.shape, second.dtype, second.device)
        group = path[0]
        if group == 'optimizer':
            group += '.' + (str(path[-1]) if path[-1] in ('step', 'exp_avg', 'exp_avg_sq') else 'other')
        row = totals.setdefault(group, {'tensors': 0, 'byte_equal_tensors': 0, 'max_abs_error': 0.,
                                        'difference_square_sum': 0., 'reference_square_sum': 0., 'finite': True})
        row['tensors'] += 1
        row['byte_equal_tensors'] += first.data == second.data
        dtype = np.dtype(first.dtype.removeprefix('torch.'))
        a, b = np.frombuffer(first.data, dtype=dtype), np.frombuffer(second.data, dtype=dtype)
        for offset in range(0, len(a), 1_048_576):
            x = a[offset:offset + 1_048_576].astype(np.float64)
            y = b[offset:offset + 1_048_576].astype(np.float64)
            row['finite'] = row['finite'] and bool(np.isfinite(x).all() and np.isfinite(y).all())
            difference = x - y
            if len(difference):
                row['max_abs_error'] = max(row['max_abs_error'], float(np.abs(difference).max()))
                row['difference_square_sum'] += float(np.dot(difference, difference))
                row['reference_square_sum'] += float(np.dot(x, x))
    for row in totals.values():
        numerator, denominator = row['difference_square_sum'], row['reference_square_sum']
        row['relative_l2_error'] = math.sqrt(numerator / denominator) if denominator else 0. if not numerator else None
    return totals


def worker(directory, nonce, dropout):
    await_owned_ack(directory, nonce)
    result = {'status': 'running', 'stage': 'validation', 'dropout': dropout, 'arms': [],
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    save(directory / 'worker.json', result)
    started = time.perf_counter()
    try:
        from qa_native_helpers import validate_runtime, load_features, build_model, build_optimizer, canonical
        from qa_native_helpers import sha_file
        result['runtime_identities'] = validate_runtime()
        review = json.loads((ROOT / 'docs/research/competitors/trainparity-execution-review.json').read_text(encoding='utf-8'))
        for entry in review['files']:
            assert sha_file(ROOT / '.tools/trainparity-reviewed' / entry['path']) == entry['sha256']
        import torch
        from trainparity.snapshot import capture_snapshot
        from trainparity.api import ToleranceComparison
        from planner_reuse_control import finite_json
        assert torch.cuda.is_available()
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(False)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision('highest')
        arrays, manifest = load_features()
        arrays = {name: values[:3] for name, values in arrays.items()}
        result.update({'feature_manifest_canonical_sha256': canonical(manifest), 'feature_rows': [0, 3],
                       'comparison': {'rtol': 1e-5, 'atol': 1e-7, 'equal_nan': False, 'rng_in_snapshot': False},
                       'checker_revision': review['revision'], 'torch': torch.__version__, 'cuda': torch.version.cuda,
                       'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
                       'tf32_matmul': torch.backends.cuda.matmul.allow_tf32, 'tf32_cudnn': torch.backends.cudnn.allow_tf32})
        policy = ToleranceComparison(rtol=1e-5, atol=1e-7, equal_nan=False)
        reference = None
        for name, micro, checkpointed in ARMS:
            result['stage'] = 'prepare_' + name
            save(directory / 'worker.json', result)
            model, preparation = build_model(2501, 'eager', checkpointed, 'cuda')
            optimizer, groups = build_optimizer(model)
            modules = [module for module in model.modules() if isinstance(module, torch.nn.Dropout)]
            for module in modules:
                module.p = dropout
            torch.manual_seed(2503)
            initial_rng = {'cpu': hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
                           'cuda': hashlib.sha256(torch.cuda.get_rng_state().cpu().numpy().tobytes()).hexdigest()}
            row = {'name': name, 'microbatch': micro, 'checkpointed': checkpointed,
                   'model_preparation': preparation, 'optimizer_group_names_digest': groups,
                   'effective_dropout_probabilities': sorted({module.p for module in modules}),
                   'dropout_module_count': len(modules), 'scientific_start_rng': initial_rng}
            result['stage'] = 'update_' + name
            save(directory / 'worker.json', result)
            optimizer.zero_grad(set_to_none=True)
            loss_value = 0.
            for begin in range(0, 3, micro):
                stop = min(begin + micro, 3)
                batch = {key: value[begin:stop].to('cuda') for key, value in arrays.items()}
                output = model(**batch)
                if not bool(torch.isfinite(output.loss)):
                    raise ValueError('Nonfinite QA loss')
                loss = output.loss * ((stop - begin) / 3)
                loss_value += float(loss.detach())
                loss.backward()
                del output, loss, batch
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True, foreach=False)
            optimizer.step()
            torch.cuda.synchronize()
            row.update({'successful_updates': 1, 'loss': loss_value, 'preclip_norm': float(norm),
                        'scientific_end_rng': {'cpu': hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
                                               'cuda': hashlib.sha256(torch.cuda.get_rng_state().cpu().numpy().tobytes()).hexdigest()}})
            result['stage'] = 'capture_' + name
            save(directory / 'worker.json', result)
            before = time.perf_counter()
            captured = capture_snapshot(model, step=1, optimizer=optimizer, capture_rng=False)
            row.update({'capture_outcome': captured.outcome.value, 'capture_seconds': time.perf_counter() - before})
            if captured.snapshot is None:
                row['capture_issue'] = asdict(captured.issue) if captured.issue else None
                result['arms'].append(row)
                raise RuntimeError('Existing capture did not produce complete snapshot')
            snapshot = captured.snapshot
            del captured
            if name == 'reference':
                reference = snapshot
            else:
                comparison_nonfinite = []
                row['comparison_to_reference'] = finite_json(asdict(policy.compare(reference, snapshot)), invalid=comparison_nonfinite)
                row['comparison_nonfinite_fields'] = comparison_nonfinite
                row['tensor_metrics'] = tensor_metrics(reference, snapshot)
            result['arms'].append(row)
            save(directory / 'worker.json', result)
            if name == ARMS[-1][0]:
                reference = None
                gc.collect()
                result['stage'] = 'negative_capture'
                save(directory / 'worker.json', result)
                with torch.no_grad():
                    model.qa_outputs.weight[0, 0].add_(.01)
                negative = capture_snapshot(model, step=1, optimizer=optimizer, capture_rng=False)
                result['negative_capture_outcome'] = negative.outcome.value
                result['negative_comparison_to_unperturbed_candidate'] = (asdict(policy.compare(snapshot, negative.snapshot))
                                                                          if negative.snapshot is not None else None)
                del negative
            del model, optimizer, snapshot
            gc.collect()
            torch.cuda.empty_cache()
        result.update({'status': 'completed', 'stage': 'finished'})
    except Exception as error:
        result.update({'status': 'error', 'error_type': type(error).__name__,
                       'message': str(error).replace(str(ROOT), '<atlas>')[:3000]})
    result['worker_wall_seconds'] = time.perf_counter() - started
    from planner_reuse_control import finite_json
    invalid = []
    result = finite_json(result, invalid=invalid)
    result['nonfinite_numeric_fields'] = invalid
    save(directory / 'worker.json', result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', type=Path)
    parser.add_argument('--nonce')
    parser.add_argument('--dropout', type=float, choices=[0., .1])
    args = parser.parse_args()
    if args.worker:
        worker(args.worker, args.nonce, args.dropout)
        return
    directory = ROOT / 'artifacts/qa-numerical-control' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    result = {'captured_utc': datetime.now(timezone.utc).isoformat(), 'source_sha256': digest(Path(__file__)),
              'helper_sha256': {name: digest(ROOT / 'benchmarks' / name) for name in (
                  'qa_owned_worker.py', 'windows_job_scope.py', 'qa_native_helpers.py',
                  'worker_ownership_control.py', 'board_energy_control.py', 'planner_reuse_control.py')},
              'checker_review_sha256': digest(ROOT / 'docs/research/competitors/trainparity-execution-review.json'),
              'scope': 'Reduced three-feature one-update numerical/capture control. No full-batch oracle, scientific outcome equivalence or policy advantage.',
              'intended_dropout_workers': [.1, 0.], 'trials': []}
    for index, dropout in enumerate(result['intended_dropout_workers']):
        trial = run_worker(Path(__file__).name, ['--dropout', str(dropout)], directory / str(index))
        trial['dropout'] = dropout
        result['trials'].append(trial)
        save(ROOT / 'docs/research/workload-screen/qa-numerical-control.json', result)
        print(json.dumps({'dropout': dropout, 'status': trial['status'],
                          'arms': [(arm['name'], arm.get('comparison_to_reference', {}).get('outcome'))
                                   for arm in trial.get('worker_result', {}).get('arms', [])]}), flush=True)


if __name__ == '__main__':
    main()
