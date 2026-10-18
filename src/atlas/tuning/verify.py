"""Independent NumPy evidence and decision reconstruction; no PyTorch imports."""
from __future__ import annotations

import math
import statistics
from pathlib import Path

import numpy as np

from .common import ROOT, artifact_path, digest, file_digest, load_array, read_json
from .registry import calibration_cases, confirmation_cases, reference_cases, verify_protocol


def compare(actual, reference, atol=1e-4, rtol=1e-4):
    if actual.shape != reference.shape:
        raise ValueError('Full-vocabulary array shape differs')
    results = []
    for candidate, native in zip(actual, reference, strict=True):
        a, b = candidate.astype(np.float64), native.astype(np.float64)
        finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
        difference = np.abs(a-b)
        bound = atol + rtol*np.abs(b)
        results.append({'passed': finite and bool(np.all(difference <= bound)), 'finite': finite,
                        'violating_elements': int(np.count_nonzero(difference > bound)) if finite else None,
                        'max_abs': float(difference.max()) if finite else None,
                        'max_normalized_error': float((difference/bound).max()) if finite else None})
    return results


def check_saved_checks(actual, saved):
    if len(actual) != len(saved):
        raise ValueError('Incomplete numerical positions')
    for measured, recorded in zip(actual, saved, strict=True):
        for key in ('passed', 'finite', 'violating_elements'):
            if measured[key] != recorded[key]:
                raise ValueError('Stored numerical decision disagrees with arrays')
        for key in ('max_abs', 'max_normalized_error'):
            a, b = measured[key], recorded.get(key)
            if (a is None) != (b is None) or (a is not None and not math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-12)):
                raise ValueError('Stored numerical magnitude disagrees with arrays')


def check_measurement(value, steps, vocab):
    numbers = value['samples_ms'] + [value[k] for k in
        ('generation_ms', 'prefill_ms', 'first_token_ms', 'allocated_peak_mib', 'reserved_peak_mib')]
    if not all(type(x) in (int, float) and math.isfinite(x) and x > 0 for x in numbers):
        raise ValueError('Invalid timing/allocation observation')
    if len(value['samples_ms']) != steps-1 or len(value['generated_token_ids']) != steps:
        raise ValueError('Incomplete fixed generation window')
    if any(type(i) is not int or not 0 <= i < vocab for i in value['generated_token_ids']):
        raise ValueError('Invalid generated token identity')
    if value['generation_ms'] + 1e-5 < value['prefill_ms'] + sum(value['samples_ms']):
        raise ValueError('Generation timing omits recorded work')
    if not value['prefill_ms'] <= value['first_token_ms'] <= value['generation_ms']:
        raise ValueError('First-token timing is inconsistent')
    if value['reserved_peak_mib'] + 1e-6 < value['allocated_peak_mib']:
        raise ValueError('Allocator counters are inconsistent')


def inspect(folder, root=ROOT):
    protocol = verify_protocol(folder, root)
    ledger = read_json(folder / 'execution.json')
    if ledger['protocol_sha256'] != file_digest(folder / 'protocol.json'):
        raise ValueError('Protocol/execution identity mismatch')
    for name, sha in protocol['sources'].items():
        if file_digest(artifact_path(folder / 'source', name)) != sha:
            raise ValueError('Executed source snapshot changed')
    nominee_path = folder / 'nomination.json'
    nomination = read_json(nominee_path) if nominee_path.exists() else None
    if nomination and ledger.get('nomination_sha256') != file_digest(nominee_path):
        raise ValueError('Frozen nomination identity differs')
    expected = reference_cases('cal') + calibration_cases(protocol)
    if nomination and nomination['profile'] is not None:
        if nomination['profile'] not in {p['id'] for p in protocol['profiles']}:
            raise ValueError('Unregistered nominee')
        expected += reference_cases('hold') + confirmation_cases(protocol, nomination['profile'])
    registered = {c['id']: c for c in expected}
    if [e['id'] for e in ledger['cases']] != [c['id'] for c in expected][:len(ledger['cases'])]:
        raise ValueError('Execution order differs from the predeclared sequence')
    records, seen, processes = {}, set(), set()
    for entry in ledger['cases']:
        ident = entry['id']
        if ident in seen or ident not in registered:
            raise ValueError('Duplicate/unregistered execution case')
        seen.add(ident)
        if entry['status'] in ('queued', 'running'):
            continue
        path = folder / 'runs' / ident / 'result.json'
        if file_digest(path) != entry['sha256']:
            raise ValueError('Raw result changed')
        raw = read_json(path)
        if raw['config'] != registered[ident] or raw['status'] != entry['status']:
            raise ValueError('Raw case/status identity differs')
        if raw.get('protocol_sha256') != ledger['protocol_sha256']:
            # Startup failures have no numerical authority but remain real attempts.
            if raw['status'] == 'completed':
                raise ValueError('Completed case lacks protocol identity')
            records[ident] = (entry, raw)
            continue
        if raw['environment']['source_sha256'] != protocol['sources']:
            raise ValueError('Executed source identities differ')
        if raw['environment']['identity'] != protocol['identity']:
            raise ValueError('Actual machine/package identity differs')
        if raw.get('id') != ident or raw.get('campaign') != folder.name:
            raise ValueError('Raw campaign identity differs')
        if raw['status'] != 'completed' and 'configured' not in raw:
            records[ident] = (entry, raw)
            continue
        if raw['configured'] != {'dtype': protocol['settings']['dtype'], 'threads': registered[ident]['profile']['threads'],
                                 'interop_threads': 1, 'tf32': False, 'matmul_precision': 'highest'}:
            raise ValueError('Actual runtime configuration differs')
        process = raw['execution_process']
        identity = (process['pid'], process['created_at'])
        owned = {(p['pid'], p['created_at']) for p in entry['ownership']['observed_processes']}
        if identity not in owned or identity in processes or not entry['ownership']['closed']:
            raise ValueError('Fresh owned process evidence is incomplete')
        processes.add(identity)
        for item in raw.get('artifacts', {}).values():
            load_array(folder, item)
        records[ident] = (entry, raw)
    array_cache, metric_cache = {}, {}

    def array(item):
        sha = item['sha256']
        if sha not in array_cache:
            value = load_array(folder, item)
            if value.shape != (32, protocol['vocabulary_size']) or value.dtype != np.float32:
                raise ValueError('Full 32-position vocabulary output required')
            array_cache[sha] = value
        return array_cache[sha]

    results = []
    for ident in [c['id'] for c in expected if c['id'] in records]:
        entry, raw = records[ident]
        case = registered[ident]
        result = {'config': case, 'status': raw['status'], 'passed': None, 'generation_ms': None,
                  'peak_allocated_mib': None, 'error': raw.get('error')}
        if raw['status'] == 'completed':
            source_input = next(i for i in protocol['inputs'] if i['id'] == case['input'])
            if raw['input_artifact'] != source_input['artifact']:
                raise ValueError('Actual input identity differs')
            if case['stage'] == 'reference':
                reference = array(raw['artifacts']['reference_logits'])
                if not np.isfinite(reference).all():
                    raise ValueError('Completed native reference is nonfinite')
                result['passed'] = True
            else:
                reference_id = 'ref-' + case['input']
                if reference_id not in records or records[reference_id][1]['status'] != 'completed':
                    raise ValueError('Completed candidate lacks its native reference')
                reference_raw = records[reference_id][1]
                reference_item = reference_raw['artifacts']['reference_logits']
                if raw['reference_sha256'] != records[reference_id][0]['sha256']:
                    raise ValueError('Reference record identity mismatch')
                actual_item = raw['artifacts']['candidate_logits']
                key = (actual_item['sha256'], reference_item['sha256'])
                if key not in metric_cache:
                    metric_cache[key] = compare(array(actual_item), array(reference_item))
                checks = metric_cache[key]
                check_saved_checks(checks, raw['logit_checks'])
                passed = all(c['passed'] for c in checks)
                if raw['correctness']['passed'] is not passed:
                    raise ValueError('Saved pass flag lies about the full arrays')
                for field in ('warmup', 'measurement'):
                    check_measurement(raw[field], protocol['settings']['generation_tokens'], protocol['vocabulary_size'])
                phases = raw['phase_times']
                if not phases['warm_end'] <= phases['measure_start'] <= phases['measure_end'] <= phases['validation_start']:
                    raise ValueError('Validation overlaps scored generation')
                source_input = next(i for i in protocol['inputs'] if i['id'] == case['input'])
                if raw['input_artifact'] != source_input['artifact']:
                    raise ValueError('Actual input identity differs')
                ids = load_array(folder, source_input['artifact'])
                logits = array(actual_item).astype(np.float64)
                maximum = logits.max(axis=1)
                losses = np.log(np.exp(logits-maximum[:, None]).sum(axis=1)) + maximum
                losses -= logits[np.arange(32), ids[protocol['settings']['context']:]]
                loss = float(losses.mean()) if np.isfinite(losses).all() else None
                if ((loss is None) != (raw['mean_nll'] is None) or
                    (loss is not None and not math.isclose(loss, raw['mean_nll'], rel_tol=1e-10, abs_tol=1e-12))):
                    raise ValueError('NLL arithmetic differs from full arrays')
                result.update(passed=passed, violating_elements=sum(c['violating_elements'] or 0 for c in checks),
                              generation_ms=raw['measurement']['generation_ms'],
                              peak_allocated_mib=max(raw['warmup']['allocated_peak_mib'], raw['measurement']['allocated_peak_mib']))
        results.append(result)
    return protocol, ledger, results, registered


def calibration_summary(protocol, cases):
    """Independent reconstruction; intentionally does not import the selector."""
    rows = {r['config']['id']: r for r in cases if r['config']['stage'] == 'calibration'}
    if set(rows) != {c['id'] for c in calibration_cases(protocol)}:
        raise ValueError('Calibration collection is incomplete')
    controls = all(rows[f'cal-{n}-{name}']['passed'] is True and
                   rows[f'cal-{n}-{name}']['status'] == 'completed'
                   for n in range(2) for name in ('native-t4', 'resident-t4'))
    table = []
    budget = protocol['settings']['budget_mib']
    for p in protocol['profiles']:
        pair = [rows[f'cal-{n}-{p["id"]}'] for n in range(2)]
        numerical = controls and all(r['passed'] is True and r['status'] == 'completed' for r in pair)
        times = [r['generation_ms'] for r in pair]
        peaks = [r['peak_allocated_mib'] for r in pair]
        score = math.sqrt(times[0]*times[1]) if all(t is not None for t in times) else None
        peak = max(peaks) if all(v is not None for v in peaks) else None
        table.append({'profile': p, 'numerical': numerical,
                      'eligible': numerical and score is not None and peak is not None and (budget is None or peak <= budget),
                      'generation_geomean_ms': score, 'peak_allocated_mib': peak,
                      'passed_windows': sum(r['passed'] is True for r in pair), 'planned_windows': 2})
    eligible = [r for r in table if r['eligible']]
    best = sorted(eligible, key=lambda r: (r['generation_geomean_ms'],
                   r['profile']['id'] != protocol['baseline'], r['profile']['id']))
    return table, best[0]['profile']['id'] if best else None


def decide(protocol, cases, table, nominated):
    base = protocol['baseline']
    outcome = {'outcome': 'no_eligible_profile', 'profile': None, 'is_change': False,
               'reason': 'No eligible calibration setting', 'pair_ratios': [], 'median_saved_ms': None}
    if nominated is None:
        return outcome
    planned = confirmation_cases(protocol, nominated)
    rows = {r['config']['id']: r for r in cases if r['config']['stage'] in ('guard', 'confirmation')}
    if set(rows) != {c['id'] for c in planned}:
        raise ValueError('Confirmation collection is incomplete')
    good = lambda r: r['status'] == 'completed' and r['passed'] is True
    if not all(good(r) for r in rows.values() if r['config']['stage'] == 'guard'):
        return {**outcome, 'outcome': 'inconclusive', 'reason': 'Confirmation numerical controls failed or unavailable'}
    pairs, nominated_rows, baseline_rows = [], [], []
    for n in range(2):
        for rep in range(3):
            current = rows[f'confirm-{n}-{rep}-{base}']
            chosen = rows[f'confirm-{n}-{rep}-{nominated}']
            nominated_rows.append(chosen); baseline_rows.append(current)
            if good(current) and good(chosen):
                pairs.append({'input': f'hold-{n}', 'repetition': rep,
                              'baseline_ms': current['generation_ms'], 'candidate_ms': chosen['generation_ms'],
                              'ratio': current['generation_ms']/chosen['generation_ms']})
    budget = protocol['settings']['budget_mib']
    fit = lambda rs: all(good(r) and (budget is None or r['peak_allocated_mib'] <= budget) for r in rs)
    base_cal = next(r for r in table if r['profile']['id'] == base)
    candidate_good = fit(nominated_rows)
    baseline_good = base_cal['eligible'] and fit(baseline_rows)
    outcome['pair_ratios'] = pairs
    if candidate_good and budget is not None and not base_cal['eligible']:
        return {**outcome, 'outcome': 'confirmed_memory_fit', 'profile': nominated,
                'is_change': nominated != base,
                'reason': 'Candidate confirmed within allocation target; current was not qualified under that target'}
    if candidate_good and baseline_good and nominated != base and len(pairs) == 6 and all(p['ratio'] > 1.05 for p in pairs):
        return {**outcome, 'outcome': 'confirmed_improvement', 'profile': nominated, 'is_change': True,
                'median_saved_ms': statistics.median(p['baseline_ms']-p['candidate_ms'] for p in pairs),
                'reason': 'All six fresh paired generation ratios exceed 1.05 and all numerical/budget controls pass'}
    if baseline_good:
        return {**outcome, 'outcome': 'keep_current', 'profile': base,
                'reason': 'No confirmed change improves this objective; current remains qualified'}
    return {**outcome, 'outcome': 'inconclusive', 'reason': 'Confirmation did not establish a qualified profile'}


def build(folder, root=ROOT, calibration_only=False):
    protocol, ledger, cases, registered = inspect(folder, root)
    table, nominated = calibration_summary(protocol, cases)
    result = {'protocol_sha256': file_digest(folder / 'protocol.json'), 'cases': cases,
              'calibration': table, 'nominated': nominated,
              'basis_sha256': digest([r for r in cases if r['config']['stage'] == 'calibration']),
              'evidence_sha256': digest({'protocol': file_digest(folder / 'protocol.json'),
                                       'runs': {e['id']: e.get('sha256') for e in ledger['cases']}})}
    if calibration_only:
        return result
    saved = read_json(folder / 'nomination.json')
    if saved['profile'] != nominated or saved['basis_sha256'] != result['basis_sha256']:
        raise ValueError('Nomination was not frozen from calibration alone')
    result['decision'] = decide(protocol, cases, table, nominated)
    result['scope'] = 'Finite same-machine observations; not global optimality, population confidence or universal output quality.'
    return result
