"""Fresh-process repeatability of the lowest frozen pilot seed, not new seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def run(output: Path) -> None:
    root=Path('artifacts/fashion-repeatability')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    root.mkdir(parents=True)
    original=json.loads(Path('docs/research/experiments/fashion-pilot.json').read_text(encoding='utf-8'))
    fixture=Path('benchmarks/fashion_pilot.py')
    if original['source_sha256']!=hashlib.sha256(fixture.read_bytes()).hexdigest():
        raise ValueError('Frozen pilot source differs')
    seed=min(original['config']['seeds'])
    results=[]
    for candidate in (False,True):
        target=root/f'{int(candidate)}.json'
        command=[sys.executable,str(fixture.resolve()),'--child','--seed',str(seed),'--output',str(target.resolve())]
        if candidate:
            command.append('--candidate')
        environment=os.environ.copy()
        environment['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
        try:
            process=subprocess.run(command,env=environment,capture_output=True,text=True,timeout=300)
            observed=json.loads(target.read_text(encoding='utf-8')) if target.exists() else {'status':'error','reason':'No child output'}
            previous=next(trial for trial in original['trials'] if trial['seed']==seed and trial['candidate']==candidate)
            equal={key:observed.get(key)==previous.get(key) for key in ('initial_state_digest','split_digest','sample_order_digest','final_state_digest')}
            outcomes_equal=observed.get('status')=='completed' and all(
                observed['history'][epoch][key]==previous['history'][epoch][key]
                for epoch in range(5) for key in ('validation_nll','validation_accuracy','validation_correct','successful_updates'))
            record={'seed':seed,'candidate':candidate,'status':observed['status'],'returncode':process.returncode,
                    'stderr':process.stderr[-3000:],'identities_equal_to_original':equal,
                    'all_epoch_validation_values_equal':outcomes_equal,'replayed_result':observed}
        except subprocess.TimeoutExpired:
            record={'seed':seed,'candidate':candidate,'status':'timeout'}
        results.append(record)
        report={'schema_version':'1.0','started_utc':STARTED,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'fixture_source_sha256':hashlib.sha256(fixture.read_bytes()).hexdigest(),'trials':results,
                'limitation':'Two repeat executions of the lowest existing pilot seed; no additional independent seeds, new hypothesis test or general cross-platform determinism guarantee.'}
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        print(json.dumps({key:record.get(key) for key in ('seed','candidate','status','identities_equal_to_original','all_epoch_validation_values_equal')}),flush=True)


STARTED=datetime.now(timezone.utc).isoformat()
if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    run(args.output)
