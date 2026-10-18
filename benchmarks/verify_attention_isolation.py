"""Independent NumPy arithmetic for the separately registered isolated-arm diagnostic."""
from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path
import statistics
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from atlas.attention.portable import artifact_path,file_digest,read_json,verify_files,write_json
from verify_attention_v3 import checked_array,compare_arrays


def check_saved_decisions(actual,reference,row,contract):
    checks=compare_arrays(actual,reference,contract['atol'],contract['rtol'])
    saved=row['logit_checks']
    if saved is None or len(saved)!=len(checks):raise ValueError('Incomplete isolated per-position decisions')
    for measured,stored in zip(checks,saved,strict=True):
        for key in ('passed','finite','violating_elements'):
            if measured[key]!=stored[key]:raise ValueError('Isolated numerical decision disagrees with arrays')
        for key in ('max_abs','max_normalized_error'):
            if measured[key] is None:
                if stored[key] is not None:raise ValueError('Nonfinite isolated metric mismatch')
            elif not math.isclose(measured[key],stored[key],rel_tol=1e-10,abs_tol=1e-12):raise ValueError('Isolated magnitude disagrees with arrays')
    passed=all(c['passed'] for c in checks)
    if row['correctness']['passed'] is not passed:raise ValueError('Isolated pass flag disagrees with arrays')
    return passed,sum(c['violating_elements'] or 0 for c in checks)


def build(directory,source_root=None):
    protocol_path=directory/'protocol.json';protocol=read_json(protocol_path)
    protocol_sha=file_digest(protocol_path)
    if source_root is not None:verify_files(source_root,protocol['sources'])
    ledger=read_json(directory/'execution.json')
    if ledger['protocol_sha256']!=protocol_sha:raise ValueError('Isolated ledger protocol mismatch')
    registered={c['case_id']:c for c in protocol['cases']}
    seen=set();records={};processes=set();rows=[]
    for entry in ledger['cases']:
        if entry['case_id'] not in registered or entry['case_id'] in seen:raise ValueError('Duplicate/unregistered isolated case')
        seen.add(entry['case_id']);case=registered[entry['case_id']]
        if entry['status'] in ('starting','running'):raise ValueError('Isolated diagnostic is still running')
        folder=artifact_path(directory,'runs/'+entry['run_id']);path=folder/'result.json'
        if file_digest(path)!=entry['sha256']:raise ValueError('Isolated raw hash mismatch')
        record=read_json(path)
        if record['id']!=entry['run_id'] or record['config']!=case:raise ValueError('Isolated record identity mismatch')
        if record['status']!=entry.get('raw_status',entry['status']):raise ValueError('Isolated raw status mismatch')
        result={'case_id':case['case_id'],'method':case['method'],'context':case['context'],
                'execution':entry['status'],'arm_status':'unavailable','passed':None,'eligible':False}
        rows.append(result)
        if entry['status']!='completed':continue
        if record['protocol_sha256']!=protocol_sha:raise ValueError('Isolated executed protocol mismatch')
        identity=(record['execution_process']['pid'],record['execution_process']['created_at'])
        if identity in processes:raise ValueError('Isolated arms reused an execution process')
        processes.add(identity)
        if not entry.get('owned_scope_closed'):raise ValueError('Isolated owned scope not closed')
        result['execution_process']=record['execution_process']
        for name,item in record['artifacts'].items():
            path=artifact_path(folder,item['path'])
            if not path.is_file() or file_digest(path)!=item['sha256']:raise ValueError('Isolated artifact missing/changed: '+name)
        for relative,expected in protocol['sources'].items():
            if relative=='benchmarks/attention_isolation.py' or relative=='lab.py' or relative.startswith('src/atlas/attention/'):
                if record['environment']['source_sha256'].get(relative)!=expected:raise ValueError('Isolated worker source mismatch: '+relative)
                if file_digest(artifact_path(folder,'source/'+relative))!=expected:raise ValueError('Isolated worker snapshot mismatch: '+relative)
        ids=checked_array(folder,record,'input_ids')
        if ids.shape!=(case['context']+32,) or ids.dtype.kind!='i':raise ValueError('Isolated input shape/type mismatch')
        if hashlib.sha256(ids.astype('<i8').tobytes()).hexdigest()!=case['input_case']['input_token_sha256']:raise ValueError('Isolated input token mismatch')
        if len(record['rows'])!=1 or record['rows'][0]['plan']!=case['method']:raise ValueError('An isolated worker ran extra or incorrect arms')
        row=record['rows'][0];result['arm_status']=row['status']
        result['host_allocator']=record['host_allocator']
        expected_shape=(32,protocol['vocabulary_size'])
        if case['method']=='native' and 'reference_logits' in record['artifacts']:
            available_reference=checked_array(folder,record,'reference_logits')
            if available_reference.shape!=expected_shape or available_reference.dtype!=np.float32 or not np.isfinite(available_reference).all():raise ValueError('Isolated native reference invalid')
            records[case['context']]=(available_reference,case['input_case']['input_token_sha256'])
        if row['status']!='completed':continue
        candidate=checked_array(folder,record,'candidate_logits')
        if candidate.shape!=expected_shape or candidate.dtype!=np.float32:raise ValueError('Isolated candidate shape/type mismatch')
        if case['method']=='native':
            if case['context'] not in records:raise ValueError('Completed isolated native lacks reference')
            reference=records[case['context']][0]
        else:
            reference_info=records.get(case['context']);reference=reference_info[0] if reference_info else None
            if reference_info and reference_info[1]!=case['input_case']['input_token_sha256']:raise ValueError('Isolated shared reference input mismatch')
        if reference is not None:
            result['passed'],result['violating_elements']=check_saved_decisions(candidate,reference,row,protocol['contract'])
        elif row['correctness']['passed'] is not None:raise ValueError('Unavailable reference was promoted to numerical pass')
        samples=row['samples_ms']
        if len(samples)!=15 or len(row['generated_token_ids'])!=16:raise ValueError('Incomplete isolated generation')
        if not all(isinstance(v,(int,float)) and math.isfinite(v) and v>0 for v in samples+[row['generation_ms'],row['prefill_ms'],row['allocated_peak_mib']]):raise ValueError('Invalid isolated timing/memory')
        if row['generation_ms']+1e-6<row['prefill_ms']+sum(samples):raise ValueError('Isolated generation duration excludes recorded work')
        if not math.isclose(row['median_ms'],statistics.median(samples),rel_tol=1e-12):raise ValueError('Isolated median mismatch')
        result.update(generation_ms=row['generation_ms'],allocated_peak_mib=row['allocated_peak_mib'])
    comparisons=[]
    for context in (32768,40000):
        by_method={r['method']:r for r in rows if r['context']==context}
        control=all(m in by_method and by_method[m]['passed'] is True for m in ('native','resident'))
        for item in by_method.values():item['eligible']=control and item['passed'] is True
        hybrid=by_method.get('hybrid50');native=by_method.get('native')
        qualified=bool(control and hybrid and hybrid['passed'] is True)
        comparisons.append({'context':context,'qualified':qualified,
                            'generation_ratio':native['generation_ms']/hybrid['generation_ms'] if qualified else None,
                            'allocator_reduction_percent':(1-hybrid['allocated_peak_mib']/native['allocated_peak_mib'])*100 if qualified else None,
                            'interpretation':'One post-hoc document, fresh process per arm; descriptive diagnostic only'})
    return {'schema_version':1,'protocol_id':protocol['id'],'protocol_sha256':protocol_sha,
            'all_cases_recorded':len(seen)==len(registered) and bool(ledger.get('finished_at')),'planned_cases':len(registered),'recorded_cases':len(seen),
            'missing_cases':sorted(set(registered)-seen),'cases':rows,'comparisons':comparisons,
            'verification':'Raw/source/array hashes, token histories, literal numerical decisions, unique execution-process identities, closed owned scopes and timing arithmetic reconstructed. Not independent human or device execution proof.',
            'limits':protocol['limits']}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,default=ROOT/'docs/research/attention-isolation')
    parser.add_argument('--source-root',type=Path)
    parser.add_argument('--write-summary',action='store_true')
    args=parser.parse_args();summary=build(args.study,args.source_root)
    if args.write_summary:write_json(args.study/'summary.json',summary)
    print(summary)
