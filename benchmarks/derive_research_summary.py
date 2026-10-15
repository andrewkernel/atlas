"""Recalculate descriptive headline arithmetic from committed experiment records."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def load(name: str) -> dict:
    return json.loads((Path('docs/research/experiments')/name).read_text(encoding='utf-8'))


def canonical_digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def run() -> dict:
    synthetic=load('cuda-memory-probe.json')
    fashion=load('fashion-pilot.json')
    replay=load('fashion-repeatability.json')
    synthetic_plans=[]
    for micro,checkpointed in ((16,False),(4,False),(16,True),(4,True)):
        intended=[trial for trial in synthetic['trials'] if trial['microbatch']==micro and trial['checkpointed']==checkpointed]
        completed=[trial for trial in intended if trial['status']=='completed']
        synthetic_plans.append({'microbatch':micro,'checkpointed':checkpointed,'intended':len(intended),'completed':len(completed),
                                'allocated_peak_mib':max((trial['measured_peak']['allocated_bytes']/2**20 for trial in completed),default=None),
                                'mean_of_process_median_update_seconds':statistics.mean(trial['median_update_seconds'] for trial in completed) if completed else None})
    fashion_arms=[]
    for candidate in (False,True):
        intended=[trial for trial in fashion['trials'] if trial['candidate']==candidate]
        completed=[trial for trial in intended if trial['status']=='completed']
        fashion_arms.append({'candidate':candidate,'intended':len(intended),'completed':len(completed),
                             'allocated_peak_mib':max((trial['whole_training_and_validation_peak']['allocated_bytes']/2**20 for trial in completed),default=None),
                             'mean_synchronized_update_seconds':statistics.mean(trial['summed_synchronized_update_seconds'] for trial in completed) if completed else None})
    pairs=[]
    for seed in fashion['config']['seeds']:
        selected={trial['candidate']:trial for trial in fashion['trials'] if trial['seed']==seed}
        if len(selected)==2 and all(trial['status']=='completed' for trial in selected.values()):
            left,right=selected[False],selected[True]
            pairs.append({'seed':seed,'accuracy_difference_percentage_points':100*(right['history'][-1]['validation_accuracy']-left['history'][-1]['validation_accuracy']),
                          'nll_difference':right['history'][-1]['validation_nll']-left['history'][-1]['validation_nll'],
                          'initialization_and_order_match':all(left[key]==right[key] for key in ('initial_state_digest','split_digest','sample_order_digest')),
                          'updates_per_arm':[left['successful_updates'],right['successful_updates']]})
    return {'schema_version':'1.0','derived_utc':datetime.now(timezone.utc).isoformat(),
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'input_canonical_json_sha256':{'cuda-memory-probe.json':canonical_digest(synthetic),'fashion-pilot.json':canonical_digest(fashion),'fashion-repeatability.json':canonical_digest(replay)},
            'synthetic_plans':synthetic_plans,'fashion_arms':fashion_arms,'fashion_pairs':pairs,
            'fashion_allocated_reduction_percent':100*(1-fashion_arms[1]['allocated_peak_mib']/fashion_arms[0]['allocated_peak_mib']),
            'fashion_update_time_increase_percent':100*(fashion_arms[1]['mean_synchronized_update_seconds']/fashion_arms[0]['mean_synchronized_update_seconds']-1),
            'replay_status_counts':dict(Counter(trial['status'] for trial in replay['trials'])),
            'limitation':'Descriptive arithmetic with explicit intended/completed counts. Replays add no independent seeds; no inference, equivalence, field risk or policy efficacy estimate.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=run()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({key:result[key] for key in ('fashion_allocated_reduction_percent','fashion_update_time_increase_percent','fashion_pairs')},indent=2))
