"""Check whether observed reuse failure requires a recomputing schedule."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
MODES=('planner_streaming','planner_deferred_separate','planner_deferred_joint')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--worker',choices=MODES);parser.add_argument('--output',type=Path);args=parser.parse_args()
    if args.worker:
        import rockmate
        import planner_reuse_control as fixture
        original=rockmate.from_config;summary={}
        def configured(*values,**settings):
            values=list(values);values[2]=64*1024**2;settings['keep_outputs']=True
            planned=original(*values,**settings)
            if planned.op_sched is not None:
                counts=Counter(op.name for op in planned.op_sched.op_list if type(op).__name__=='ComputeOp' and not op.disabled and getattr(op.target,'is_fwd',False))
                summary.update({'planned_repeated_forward_count':sum(max(value-1,0) for value in counts.values()),'enabled_forward_operation_counts':dict(counts),'reported_schedule_peak_bytes':float(planned.op_sched.peak_mem)})
            return planned
        rockmate.from_config=configured
        fixture.worker(args.worker,args.output)
        record=json.loads(args.output.read_text(encoding='utf-8'));record.update({'keep_outputs':True,'budget_mib':64,'schedule_summary':summary})
        args.output.write_text(json.dumps(record,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8');return
    source=ROOT/'.tools/rockmate-review';review_path=ROOT/'docs/research/competitors/rockmate-execution-review.json'
    review=json.loads(review_path.read_text(encoding='utf-8'))
    for item in review['files']:assert sha(source/item['path'])==item['sha256']
    directory=ROOT/'artifacts/planner-reuse-nonrecompute-control';directory.mkdir(parents=True,exist_ok=True)
    interpreter=ROOT/'.tools/planner-env/Scripts/python.exe';env={key:os.environ[key] for key in ('SystemRoot','WINDIR') if key in os.environ}
    env.update({'PATH':str(interpreter.parent)+os.pathsep+str(Path(os.environ['SystemRoot'])/'System32'),
                'PYTHONPATH':str(source/'rockmate/src')+os.pathsep+str(source/'rkgb/src'),
                'PYTHONNOUSERSITE':'1','PYTHONDONTWRITEBYTECODE':'1','PYTHONHASHSEED':'0',
                'TEMP':str(directory),'TMP':str(directory),'TORCH_HOME':str(directory/'torch-cache'),
                'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','CUBLAS_WORKSPACE_CONFIG':':4096:8'})
    rows=[]
    for mode in MODES:
        target=directory/(mode+'.json');started=time.perf_counter()
        with (directory/(mode+'.stdout.txt')).open('w',encoding='utf-8') as stdout,(directory/(mode+'.stderr.txt')).open('w',encoding='utf-8') as stderr:
            process=subprocess.run([str(interpreter),str(Path(__file__).resolve()),'--worker',mode,'--output',str(target)],cwd=directory,env=env,stdout=stdout,stderr=stderr,timeout=60,check=False)
        record=json.loads(target.read_text(encoding='utf-8')) if process.returncode==0 and target.exists() else {'mode':mode,'status':'worker_failure','returncode':process.returncode}
        record['outer_wall_seconds']=time.perf_counter()-started;rows.append(record);print(mode,record['status'],record.get('schedule_summary',{}).get('planned_repeated_forward_count'),flush=True)
    result={'captured_utc':datetime.now(timezone.utc).isoformat(),'source_sha256':sha(Path(__file__)),'fixture_source_sha256':sha(ROOT/'benchmarks/planner_reuse_control.py'),'review_sha256':sha(review_path),'modes':MODES,'trials':rows,
            'scope':'Exploratory larger-budget control, fixed keep_outputs=True. Static schedule repeats are recorded to test whether failure requires recomputation. No kernel trace, general root-cause claim or historical paper replication.'}
    (ROOT/'docs/research/experiments/planner-reuse-nonrecompute-control.json').write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8')


if __name__=='__main__':
    main()
