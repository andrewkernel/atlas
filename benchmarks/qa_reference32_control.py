"""Fresh-process logical-32 selected-state reference-noise measurements."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

from board_energy_control import ROOT
from worker_ownership_control import save
from qa_reference_supervisor import await_owned_ack, run_worker

SELECTED = ['bert.embeddings.position_embeddings.weight', 'bert.encoder.layer.0.attention.self.query.weight',
            'qa_outputs.weight', 'qa_outputs.bias']
ORDER = [(.1, 0), (0., 0), (0., 1), (.1, 1)]


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def worker(directory, nonce, dropout):
    await_owned_ack(directory, nonce)
    result = {'status': 'running', 'stage': 'validation', 'dropout': dropout,
              'source_sha256': sha(Path(__file__))}
    save(directory / 'worker.json', result)
    started = time.perf_counter()
    try:
        from qa_native_helpers import validate_runtime, load_features, build_model, build_optimizer, canonical, model_state_digest, tensor_digest
        result['runtime_identities'] = validate_runtime()
        import numpy as np
        import torch
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(False)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision('highest')
        arrays, manifest = load_features()
        model, preparation = build_model(2501, 'sdpa', False, 'cuda')
        optimizer, groups = build_optimizer(model)
        modules = [m for m in model.modules() if isinstance(m, torch.nn.Dropout)]
        assert len(modules) == 37
        for module in modules:
            module.p = dropout
        result.update({'model_preparation': preparation, 'optimizer_group_names_digest': groups,
                       'feature_manifest_canonical_sha256': canonical(manifest), 'feature_rows': [0, 32],
                       'torch': torch.__version__, 'cuda': torch.version.cuda,
                       'effective_dropout_probabilities': sorted({m.p for m in modules}),
                       'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
                       'tf32_matmul': torch.backends.cuda.matmul.allow_tf32, 'tf32_cudnn': torch.backends.cudnn.allow_tf32})
        def rng():
            return {'cpu': hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
                    'cuda': hashlib.sha256(torch.cuda.get_rng_state().cpu().numpy().tobytes()).hexdigest()}
        torch.manual_seed(2503)
        result['scientific_start_rng'] = rng()
        result['stage'] = 'update'
        save(directory / 'worker.json', result)
        optimizer.zero_grad(set_to_none=True)
        batch = {name: value[:32].to('cuda') for name, value in arrays.items()}
        output = model(**batch)
        if not bool(torch.isfinite(output.loss)):
            raise ValueError('Nonfinite reference loss')
        result['loss'] = float(output.loss.detach())
        output.loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True, foreach=False)
        optimizer.step()
        torch.cuda.synchronize()
        result.update({'successful_updates': 1, 'preclip_norm': float(norm), 'scientific_end_rng': rng()})
        del output, batch
        result['stage'] = 'capture'
        save(directory / 'worker.json', result)
        result['post_model_state_digest'] = model_state_digest(model)
        result['post_clip_gradient_digest'] = tensor_digest((n, p.grad) for n, p in model.named_parameters() if p.grad is not None)
        result['all_parameters_finite'] = all(bool(torch.isfinite(p).all()) for p in model.parameters())
        result['all_optimizer_steps_one'] = all(float(optimizer.state[p]['step']) == 1. for p in model.parameters())
        parameters = dict(model.named_parameters())
        exported = {}
        for name in SELECTED:
            parameter = parameters[name]
            for phase, value in [('parameter', parameter), ('gradient', parameter.grad),
                                 ('exp_avg', optimizer.state[parameter]['exp_avg']), ('exp_avg_sq', optimizer.state[parameter]['exp_avg_sq'])]:
                exported[phase + '__' + name] = value.detach().cpu().contiguous().numpy()
        assert sum(value.nbytes for value in exported.values()) < 32 * 1024**2
        assert all(value.dtype == np.float32 and bool(np.isfinite(value).all()) for value in exported.values())
        target = directory / 'selected-state.npz'
        np.savez(target, **exported)
        result['selected_arrays'] = {name: {'shape': list(value.shape), 'dtype': str(value.dtype),
                                           'sha256': hashlib.sha256(value.tobytes()).hexdigest()}
                                     for name, value in exported.items()}
        result['selected_state_file'] = {'bytes': target.stat().st_size, 'sha256': sha(target)}
        result.update({'status': 'completed', 'stage': 'finished'})
    except Exception as error:
        result.update({'status': 'error', 'error_type': type(error).__name__, 'message': str(error).replace(str(ROOT), '<atlas>')[:3000]})
    result['worker_wall_seconds'] = time.perf_counter() - started
    save(directory / 'worker.json', result)


def load_selected(target, worker):
    import numpy as np
    assert sha(target) == worker['selected_state_file']['sha256']
    with np.load(target, allow_pickle=False) as archive:
        assert set(archive.files) == set(worker['selected_arrays'])
        values = {name: archive[name] for name in archive.files}
    for name, value in values.items():
        expected = worker['selected_arrays'][name]
        assert str(value.dtype) == expected['dtype'] == 'float32' and list(value.shape) == expected['shape']
        assert hashlib.sha256(value.tobytes()).hexdigest() == expected['sha256']
    return values


def compare_arrays(reference, candidate):
    import numpy as np
    rows = {}
    assert reference.keys() == candidate.keys()
    for name, first in reference.items():
        second = candidate[name]
        a, b = first.astype(np.float64), second.astype(np.float64)
        difference = np.abs(a - b)
        index = int(np.argmax(difference))
        squared_difference = float(np.square(a - b).sum())
        squared_reference = float(np.square(a).sum())
        rows[name] = {'elements': first.size, 'finite': bool(np.isfinite(a).all() and np.isfinite(b).all()),
                      'bitwise_equal': first.tobytes() == second.tobytes(), 'numerically_exact_equal': bool(np.array_equal(first, second)),
                      'maximum_absolute_difference': float(difference.flat[index]), 'maximum_flat_index': index,
                      'reference_at_maximum': float(a.flat[index]), 'candidate_at_maximum': float(b.flat[index]),
                      'diagnostic_exceeded_elements': int((difference > 1e-7 + 1e-5 * np.abs(b)).sum()),
                      'difference_square_sum': squared_difference, 'reference_square_sum': squared_reference,
                      'relative_l2_error': (squared_difference / squared_reference)**.5 if squared_reference else None,
                      'relative_l2_reason': None if squared_reference else 'zero_reference_norm'}
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', type=Path)
    parser.add_argument('--nonce')
    parser.add_argument('--dropout', type=float, choices=[.1, 0.])
    args = parser.parse_args()
    if args.worker:
        worker(args.worker, args.nonce, args.dropout)
        return
    original = (ROOT / 'benchmarks/qa_owned_worker.py').read_text(encoding='utf-8')
    expected = original.replace("ALLOWED = {'qa_numerical_control.py', 'qa_dispatch_control.py'}", "ALLOWED = {'qa_reference32_control.py'}")
    assert (ROOT / 'benchmarks/qa_reference_supervisor.py').read_text(encoding='utf-8') == expected
    directory = ROOT / 'artifacts/qa-reference32' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    result = {'captured_utc': datetime.now(timezone.utc).isoformat(), 'source_sha256': sha(Path(__file__)),
              'helper_sha256': {name: sha(ROOT / 'benchmarks' / name) for name in (
                  'qa_reference_supervisor.py', 'qa_owned_worker.py', 'windows_job_scope.py', 'qa_native_helpers.py',
                  'worker_ownership_control.py', 'board_energy_control.py')},
              'order': ORDER, 'selected_parameters': SELECTED, 'trials': [], 'pairs': [],
              'diagnostic_predicate': {'arithmetic': 'float64', 'atol': 1e-7, 'rtol': 1e-5, 'relative_scale': 'candidate'},
              'scope': 'Fresh-process first-update noise; distances cover only four selected named parameters/gradients/moments. Whole hashes are identities, not distances or scientific verdicts.'}
    for number, (dropout, repetition) in enumerate(ORDER):
        scratch = directory / str(number)
        trial = run_worker(Path(__file__).name, ['--dropout', str(dropout)], scratch)
        trial.update({'dropout': dropout, 'repetition': repetition,
                      'selected_state_local_artifact': str((scratch / 'selected-state.npz').relative_to(ROOT)).replace('\\', '/')})
        result['trials'].append(trial)
        save(ROOT / 'docs/research/workload-screen/qa-reference32-control.json', result)
        print(json.dumps({'trial': number, 'dropout': dropout, 'repetition': repetition, 'status': trial['status']}), flush=True)
    import numpy as np
    result['controller_numpy'] = np.__version__
    for dropout in (.1, 0.):
        pair = [t for t in result['trials'] if t['dropout'] == dropout]
        observed = {'dropout': dropout, 'status': 'unavailable_pair'}
        if all(t['status'] == 'completed' for t in pair):
            first, second = (t['worker_result'] for t in pair)
            assert first['model_preparation']['initial_state_digest'] == second['model_preparation']['initial_state_digest']
            assert first['scientific_start_rng'] == second['scientific_start_rng']
            values = [load_selected(ROOT / t['selected_state_local_artifact'], t['worker_result']) for t in pair]
            observed.update({'status': 'measured', 'whole_model_digest_equal': first['post_model_state_digest'] == second['post_model_state_digest'],
                             'whole_gradient_digest_equal': first['post_clip_gradient_digest'] == second['post_clip_gradient_digest'],
                             'scientific_end_rng_equal': first['scientific_end_rng'] == second['scientific_end_rng'],
                             'selected_tensor_differences': compare_arrays(*values)})
        result['pairs'].append(observed)
        save(ROOT / 'docs/research/workload-screen/qa-reference32-control.json', result)
        print(json.dumps({'dropout': dropout, 'pair_status': observed['status'], 'whole_model_digest_equal': observed.get('whole_model_digest_equal'),
                          'exceeded_by_field': {k: v['diagnostic_exceeded_elements'] for k, v in observed.get('selected_tensor_differences', {}).items()}}), flush=True)


if __name__ == '__main__':
    main()
