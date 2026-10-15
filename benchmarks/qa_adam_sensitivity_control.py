"""Selected-component optimizer replay, not a new equivalence predicate."""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
from pathlib import Path
import struct
import time

from board_energy_control import ROOT
from worker_ownership_control import save
from qa_sensitivity_supervisor import await_owned_ack, run_worker


def ulp_distance(first, second):
    if not math.isfinite(first) or not math.isfinite(second):
        raise ValueError('ULP distance requires finite float32 values')
    if first == second:
        return 0
    def ordered(value):
        bits, = struct.unpack('>I', struct.pack('>f', value))
        return (~bits & 0xFFFFFFFF) if bits & 0x80000000 else bits | 0x80000000
    return abs(ordered(first) - ordered(second))


def worker(directory, nonce):
    await_owned_ack(directory, nonce)
    result = {'status': 'running', 'stage': 'validation', 'executions': [],
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    save(directory / 'worker.json', result)
    started = time.perf_counter()
    try:
        from qa_native_helpers import validate_runtime, load_features, build_model, build_optimizer, canonical, sha_file
        result['runtime_identities'] = validate_runtime()
        review = json.loads((ROOT / 'docs/research/workload-screen/qa-optimizer-review.json').read_text(encoding='utf-8'))
        for entry in review['files']:
            assert sha_file(ROOT / '.tools/qa-env/Lib/site-packages' / entry['path']) == entry['sha256']
        import torch
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(False)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision('highest')
        arrays, manifest = load_features()
        result['feature_manifest_canonical_sha256'] = canonical(manifest)
        result.update({'torch': torch.__version__, 'cuda': torch.version.cuda, 'feature_rows': [0, 3],
                       'tf32_matmul': torch.backends.cuda.matmul.allow_tf32, 'tf32_cudnn': torch.backends.cudnn.allow_tf32,
                       'deterministic_algorithms': torch.are_deterministic_algorithms_enabled()})
        states = {}
        for name, micro in (('reference', 3), ('microbatch', 2)):
            result['stage'] = 'prepare_' + name
            save(directory / 'worker.json', result)
            model, preparation = build_model(2501, 'eager', False, 'cuda')
            optimizer, groups = build_optimizer(model)
            modules = [m for m in model.modules() if isinstance(m, torch.nn.Dropout)]
            assert len(modules) == 37
            for module in modules:
                module.p = 0.
            state = {'initial': {n: p.detach().cpu() for n, p in model.named_parameters()} if name == 'reference' else {}}
            group_decay = {id(p): group['weight_decay'] for group in optimizer.param_groups for p in group['params']}
            state['decays'] = {n: group_decay[id(p)] for n, p in model.named_parameters()}
            torch.manual_seed(2503)
            initial_rng = {'cpu': hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
                           'cuda': hashlib.sha256(torch.cuda.get_rng_state().cpu().numpy().tobytes()).hexdigest()}
            optimizer.zero_grad(set_to_none=True)
            result['stage'] = 'update_' + name
            save(directory / 'worker.json', result)
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
            state['gradient'] = {n: p.grad.detach().cpu() for n, p in model.named_parameters()}
            optimizer.step()
            torch.cuda.synchronize()
            state['post'] = {n: p.detach().cpu() for n, p in model.named_parameters()}
            for key in ('exp_avg', 'exp_avg_sq'):
                state[key] = {n: optimizer.state[p][key].detach().cpu() for n, p in model.named_parameters()}
            steps = {float(optimizer.state[p]['step']) for p in model.parameters()}
            assert steps == {1.}
            states[name] = state
            result['executions'].append({'name': name, 'microbatch': micro, 'successful_updates': 1,
                                         'loss': loss_value, 'preclip_norm': float(norm), 'model_preparation': preparation,
                                         'optimizer_group_names_digest': groups, 'scientific_start_rng': initial_rng,
                                         'effective_dropout_probabilities': [0.], 'dropout_module_count': len(modules),
                                         'all_optimizer_steps_one': True})
            save(directory / 'worker.json', result)
            del model, optimizer
            gc.collect()
            torch.cuda.empty_cache()
        reference, candidate = states['reference'], states['microbatch']
        assert reference['decays'] == candidate['decays']
        assert result['executions'][0]['model_preparation']['initial_state_digest'] == result['executions'][1]['model_preparation']['initial_state_digest']
        pool = []
        for name, value in reference['post'].items():
            difference = (value - candidate['post'][name]).abs().flatten()
            if not bool(difference.max() > 0):
                continue
            top = torch.topk(difference, min(3, difference.numel())).values.tolist()
            chosen = []
            for magnitude in sorted(set(top), reverse=True):
                positions = (difference == magnitude).nonzero().flatten()[:3 - len(chosen)].tolist()
                chosen.extend(positions)
            for index in chosen:
                pool.append((float(difference[index]), name, index))
        pool.sort(key=lambda value: (-value[0], value[1], value[2]))
        assert len(pool) >= 10, 'Fewer than ten differing components; record unsupported selection rather than inventing them'
        selected = []
        for difference, name, index in pool[:10]:
            row = {'parameter': name, 'flat_index': index, 'shape': list(reference['post'][name].shape),
                   'actual_absolute_parameter_difference': difference, 'weight_decay': reference['decays'][name],
                   'initial_value': float(reference['initial'][name].flatten()[index])}
            for label, state in (('reference', reference), ('candidate', candidate)):
                row[label] = {key: float(state[key][name].flatten()[index]) for key in ('gradient', 'post', 'exp_avg', 'exp_avg_sq')}
                g, value = row[label]['gradient'], row['initial_value']
                row[label]['ideal_first_step_adamw_value'] = value * (1 - 3e-5 * row['weight_decay']) - 3e-5 * g / (abs(g) + 1e-8)
            selected.append(row)
        result['selected_components'] = selected
        result['stage'] = 'selected_component_replay'
        save(directory / 'worker.json', result)
        del states, reference, candidate, state
        gc.collect()

        def replay(kind, eps, gradient_side):
            values = [None] * len(selected)
            for decay in sorted({row['weight_decay'] for row in selected}):
                indices = [i for i, row in enumerate(selected) if row['weight_decay'] == decay]
                parameter = torch.nn.Parameter(torch.tensor([selected[i]['initial_value'] for i in indices], dtype=torch.float32, device='cuda'))
                if kind == 'adamw':
                    optimizer = torch.optim.AdamW([parameter], lr=3e-5, betas=(.9, .999), eps=eps,
                                                  weight_decay=decay, foreach=False, fused=False)
                else:
                    optimizer = torch.optim.SGD([parameter], lr=3e-5, weight_decay=decay, momentum=0., foreach=False, fused=False)
                parameter.grad = torch.tensor([selected[i][gradient_side]['gradient'] for i in indices], dtype=torch.float32, device='cuda')
                optimizer.step()
                torch.cuda.synchronize()
                for offset, index in enumerate(indices):
                    values[index] = {'post': float(parameter[offset].detach())}
                    if kind == 'adamw':
                        for key in ('exp_avg', 'exp_avg_sq'):
                            values[index][key] = float(optimizer.state[parameter][key][offset])
            return values

        replays = []
        for kind, eps in (('adamw', 1e-8), ('adamw', 1e-6), ('adamw', 1e-4), ('sgd', None)):
            first, second = replay(kind, eps, 'reference'), replay(kind, eps, 'candidate')
            equal = replay(kind, eps, 'reference')
            row = {'optimizer': kind, 'eps': eps, 'reference': first, 'candidate': second,
                   'equal_gradient_positive_exact': first == equal,
                   'max_replayed_pair_difference': max(abs(a['post'] - b['post']) for a, b in zip(first, second, strict=True))}
            if kind == 'adamw' and eps == 1e-8:
                checks = []
                for observed, a, b in zip(selected, first, second, strict=True):
                    checks.append({side: {key: ulp_distance(observed[side][key], replayed[key]) for key in replayed}
                                   for side, replayed in (('reference', a), ('candidate', b))})
                row['actual_full_model_replay_ulp_distances'] = checks
                row['all_selected_values_within_two_ulps'] = all(distance <= 2 for check in checks for side in check.values() for distance in side.values())
            replays.append(row)
        result.update({'status': 'completed', 'stage': 'finished', 'replays': replays,
                       'selection_scope': 'Ten post-hoc diagnostic components selected by the frozen largest-difference rule; not independent outcomes or a full-model causal proof.'})
    except Exception as error:
        result.update({'status': 'error', 'error_type': type(error).__name__, 'message': str(error).replace(str(ROOT), '<atlas>')[:3000]})
    result['worker_wall_seconds'] = time.perf_counter() - started
    save(directory / 'worker.json', result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', type=Path)
    parser.add_argument('--nonce')
    args = parser.parse_args()
    if args.worker:
        worker(args.worker, args.nonce)
        return
    original = (ROOT / 'benchmarks/qa_owned_worker.py').read_text(encoding='utf-8')
    expected = original.replace("ALLOWED = {'qa_numerical_control.py', 'qa_dispatch_control.py'}", "ALLOWED = {'qa_adam_sensitivity_control.py'}")
    assert (ROOT / 'benchmarks/qa_sensitivity_supervisor.py').read_text(encoding='utf-8') == expected
    directory = ROOT / 'artifacts/qa-adam-sensitivity' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    result = {'captured_utc': datetime.now(timezone.utc).isoformat(), 'source_sha256': digest(Path(__file__)),
              'helper_sha256': {name: digest(ROOT / 'benchmarks' / name) for name in (
                  'qa_sensitivity_supervisor.py', 'qa_owned_worker.py', 'windows_job_scope.py', 'qa_native_helpers.py',
                  'worker_ownership_control.py', 'board_energy_control.py')},
              'optimizer_review_sha256': digest(ROOT / 'docs/research/workload-screen/qa-optimizer-review.json'),
              'supervisor_difference': 'Only fixed worker registry differs from preserved original; equality checked before launch.',
              'scope': 'Selected-component adaptive-update diagnostic; altered eps/SGD are counterfactual methods, not scientific-preserving repairs.'}
    result['trial'] = run_worker(Path(__file__).name, [], directory)
    save(ROOT / 'docs/research/workload-screen/qa-adam-sensitivity-control.json', result)
    print(json.dumps({'status': result['trial']['status'], 'stage': result['trial'].get('worker_result', {}).get('stage'),
                      'replay': [(r['optimizer'], r['eps'], r['max_replayed_pair_difference'], r.get('all_selected_values_within_two_ulps'))
                                 for r in result['trial'].get('worker_result', {}).get('replays', [])]}))


if __name__ == '__main__':
    main()
