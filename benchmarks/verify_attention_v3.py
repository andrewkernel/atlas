"""Read-only, NumPy/stdlib verification independent of runtime comparisons.

This verifies artifact consistency and numerical arithmetic, not that an author
actually ran hardware. An independent fresh replication is a separate activity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from atlas.attention.portable import artifact_path, file_digest, read_json, verify_files, write_json


def compare_arrays(actual, reference, atol, rtol):
    if actual.shape != reference.shape:
        raise ValueError('Logit array shape mismatch')
    checks = []
    for a, r in zip(actual, reference, strict=True):
        a, r = a.astype(np.float64), r.astype(np.float64)
        finite = bool(np.isfinite(a).all() and np.isfinite(r).all())
        delta = np.abs(a-r)
        bound = atol + rtol*np.abs(r)
        checks.append({'passed':finite and bool(np.all(delta<=bound)), 'finite':finite,
                       'violating_elements':int(np.count_nonzero(delta>bound)) if finite else None,
                       'max_abs':float(np.max(delta)) if finite else None,
                       'max_normalized_error':float(np.max(delta/np.maximum(bound, 1e-30))) if finite else None})
    return checks


def checked_array(directory, record, name):
    item = record['artifacts'][name]
    path = artifact_path(directory, item['path'])
    if file_digest(path) != item['sha256']:
        raise ValueError('Array hash mismatch: '+name)
    array = np.load(path, allow_pickle=False)
    if list(array.shape) != item['shape'] or str(array.dtype) != item['dtype']:
        raise ValueError('Array descriptor mismatch: '+name)
    return array


def nll(values, target):
    values = values.astype(np.float64)
    shifted = values-values.max()
    return float(np.log(np.exp(shifted).sum())-shifted[target])


def verify_record(directory, record, case, protocol):
    if record['config'] != case:
        raise ValueError('Run config identity mismatch')
    # Missing reference/arms are allowed only as explicitly unsuccessful outcomes.
    result = {'case_id':case['case_id'], 'cohort':case['cohort'], 'block':case['block'],
              'context':case['context'], 'book_id':case['book_id'], 'status':record['status'], 'arms':[]}
    if record['status'] not in ('completed','failed','cancelled','timed_out'):
        raise ValueError('Nonterminal evidence cannot be analyzed')
    if record.get('protocol_sha256') != protocol['_sha256']:
        if record['status']=='completed':raise ValueError('Completed run protocol identity mismatch')
        result['identity_verified']=False
        result['error']=record.get('error','Failure before protocol verification')
        return result
    for relative, expected in protocol.get('sources',{}).items():
        if relative=='lab.py' or relative.startswith('src/atlas/attention/'):
            if record['environment']['source_sha256'].get(relative)!=expected:
                raise ValueError('Executed source identity mismatch: '+relative)
            snapshot=artifact_path(directory,'source/'+relative)
            if not snapshot.is_file() or file_digest(snapshot)!=expected:
                raise ValueError('Executed source snapshot mismatch: '+relative)
    for name, item in record.get('artifacts', {}).items():
        path = artifact_path(directory, item['path'])
        if not path.is_file() or file_digest(path) != item['sha256']:
            raise ValueError('Missing or changed run artifact: '+name)
    if 'reference_logits' not in record.get('artifacts', {}):
        if record['status']=='completed':
            raise ValueError('Completed run lacks native reference')
        return result
    ids = checked_array(directory, record, 'input_ids')
    if ids.shape != (case['context']+case['validation_tokens'],) or ids.dtype.kind!='i':
        raise ValueError('Input array shape/type mismatch')
    token_hash = hashlib.sha256(ids.astype('<i8').tobytes()).hexdigest()
    if token_hash != case['input_token_sha256']:
        raise ValueError('Input token hash mismatch')
    reference = checked_array(directory, record, 'reference_logits')
    expected = (case['validation_tokens'], protocol['vocabulary_size'])
    if reference.shape != expected or reference.dtype != np.float32:
        raise ValueError('Reference shape/type differs from frozen contract')
    if not np.isfinite(reference).all():
        raise ValueError('Native reference is nonfinite; no comparison can pass')
    rows = record.get('rows', [])
    if len({r['plan'] for r in rows}) != len(rows) or any(r['plan'] not in case['methods'] for r in rows):
        raise ValueError('Duplicate or unexpected arm')
    if record['status']=='completed' and [r['plan'] for r in rows] != case['methods']:
        raise ValueError('Completed run lacks registered arms/order')
    for row in rows:
        arm = {'plan':row['plan'], 'status':row['status'], 'passed':None, 'eligible':False,
               'generation_ms':row.get('generation_ms'), 'allocated_peak_mib':row.get('allocated_peak_mib')}
        if row['status']=='completed':
            actual = checked_array(directory, record, row['plan']+'_logits')
            if actual.shape != expected or actual.dtype != np.float32:
                raise ValueError('Candidate shape/type differs from frozen contract')
            checks = compare_arrays(actual, reference, protocol['contract']['atol'], protocol['contract']['rtol'])
            stored = row['logit_checks']
            if len(stored)!=len(checks):
                raise ValueError('Incomplete per-position checks')
            for measured, saved in zip(checks, stored, strict=True):
                for key in ('passed','finite','violating_elements'):
                    if measured[key]!=saved[key]:
                        raise ValueError('Saved numerical decision disagrees with arrays')
                for key in ('max_abs','max_normalized_error'):
                    if measured[key] is None:
                        if saved[key] is not None:raise ValueError('Nonfinite metric mismatch')
                    elif not math.isclose(measured[key], saved[key], rel_tol=1e-10, abs_tol=1e-12):
                        raise ValueError('Saved numerical magnitude disagrees with arrays')
            passed = all(c['passed'] for c in checks)
            if row['correctness']['passed'] is not passed:
                raise ValueError('Saved pass flag disagrees with arrays')
            samples = row['samples_ms']
            if len(samples)!=case['generation_tokens']-1 or len(row['generated_token_ids'])!=case['generation_tokens']:
                raise ValueError('Incomplete free-generation history')
            values = samples+[row['generation_ms'],row['prefill_ms'],row['allocated_peak_mib']]
            if not all(isinstance(v,(int,float)) and math.isfinite(v) and v>0 for v in values):
                raise ValueError('Invalid timing or memory observation')
            if row['generation_ms']+1e-6 < row['prefill_ms']+sum(samples):
                raise ValueError('Generation duration excludes recorded work')
            if not math.isclose(row['median_ms'], statistics.median(samples), rel_tol=1e-12):
                raise ValueError('Sample median mismatch')
            if all(c['finite'] for c in checks):
                losses = [nll(a, int(ids[case['context']+i])) for i,a in enumerate(actual)]
                if not math.isclose(statistics.mean(losses), row['mean_nll'], rel_tol=1e-10, abs_tol=1e-12):
                    raise ValueError('Recorded NLL disagrees with arrays')
            arm.update(passed=passed, violating_elements=sum(c['violating_elements'] or 0 for c in checks),
                       max_abs=max((c['max_abs'] for c in checks if c['max_abs'] is not None), default=None))
        result['arms'].append(arm)
    controls = {r['plan']:r for r in result['arms'] if r['plan'] in ('native','resident')}
    controls_pass = len(controls)==2 and all(r['status']=='completed' and r['passed'] is True for r in controls.values())
    for arm, row in zip(result['arms'], rows, strict=True):
        arm['eligible'] = record['status']=='completed' and controls_pass and arm['passed'] is True
        if record['status']=='completed' and row['eligibility']['eligible'] is not arm['eligible']:
            raise ValueError('Saved eligibility disagrees with independent verification')
    return result


def build(study: Path, source_root: Path | None = None):
    protocol_path = study/'protocol.json'
    protocol = read_json(protocol_path)
    protocol['_sha256'] = file_digest(protocol_path)
    if source_root is not None:
        verify_files(source_root, protocol['sources'])
    ledger = read_json(study/'execution.json')
    if ledger['protocol_sha256'] != protocol['_sha256']:
        raise ValueError('Execution ledger protocol mismatch')
    registered = {c['case_id']:c for c in protocol['cases']}
    seen, cases = set(), []
    for entry in ledger['cases']:
        case_id = entry['case_id']
        if case_id not in registered or case_id in seen:
            raise ValueError('Duplicate or unregistered case in ledger')
        seen.add(case_id)
        directory = artifact_path(study, 'runs/'+entry['run_id'])
        path = directory/'result.json'
        if file_digest(path)!=entry['sha256']:
            raise ValueError('Raw run hash mismatch')
        record = read_json(path)
        interrupted=entry['status']=='interrupted' and record['status'] in ('running','queued') and entry.get('disposition')
        if record['id']!=entry['run_id'] or (record['status']!=entry['status'] and not interrupted):
            raise ValueError('Execution status or run identity mismatch')
        if interrupted:
            cases.append({'case_id':case_id, 'cohort':registered[case_id]['cohort'],
                          'context':registered[case_id]['context'], 'block':registered[case_id]['block'],
                          'book_id':registered[case_id]['book_id'], 'status':'interrupted', 'arms':[],
                          'disposition':entry['disposition']})
        else:
            cases.append(verify_record(directory, record, registered[case_id], protocol))
    comparisons = []
    for cohort in ('v2-replay','pg19-books'):
        for context in (32768,40000):
            selected = [r for r in cases if r['cohort']==cohort and r['context']==context]
            paired, eligible, reductions = [], [], []
            for case in selected:
                arms = {r['plan']:r for r in case['arms']}
                if not all(m in arms and arms[m]['status']=='completed' for m in ('native','hybrid50')):
                    continue
                native, hybrid = arms['native'], arms['hybrid50']
                ratio = native['generation_ms']/hybrid['generation_ms']
                paired.append(ratio)
                if native['eligible'] and hybrid['eligible']:
                    eligible.append(ratio)
                    reductions.append((1-hybrid['allocated_peak_mib']/native['allocated_peak_mib'])*100)
            comparisons.append({'cohort':cohort, 'context':context,
                                'planned':sum(c['cohort']==cohort and c['context']==context for c in registered.values()),
                                'recorded':len(selected), 'timing_pairs':len(paired), 'eligible_pairs':len(eligible),
                                'eligible_generation_ratios':eligible,
                                'eligible_ratio_median':statistics.median(eligible) if eligible else None,
                                'eligible_allocator_reduction_percent':reductions,
                                'interpretation':'Descriptive pairs only; no OOM rescue or physical-residency inference'})
    return {'schema_version':3, 'protocol_id':protocol['id'], 'protocol_sha256':protocol['_sha256'],
            'all_cases_recorded':len(seen)==len(registered),
            'planned_cases':len(registered), 'recorded_cases':len(seen),
            'missing_cases':sorted(set(registered)-seen),
            'verification':'Raw hashes, full-array comparisons, controls, token identities, timing arithmetic and NLL reconstructed independently of runtime comparator. Not proof of real execution or independent reproduction.',
            'cases':cases, 'comparisons':comparisons, 'limits':protocol['limits']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,default=Path(__file__).resolve().parents[1]/'docs/research/attention-v3')
    parser.add_argument('--source-root',type=Path)
    parser.add_argument('--write-summary',action='store_true')
    args = parser.parse_args()
    result = build(args.study, args.source_root)
    if args.write_summary:write_json(args.study/'summary.json',result)
    print(json.dumps({'verified':True, 'recorded':result['recorded_cases'], 'planned':result['planned_cases'], 'missing':result['missing_cases'], 'comparisons':result['comparisons']}, indent=2))


if __name__=='__main__':main()
