"""Test an existing output-retention option on the frozen reuse fixture."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
MODES = ("planner_streaming", "planner_deferred_separate", "planner_deferred_joint")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--worker',choices=MODES);parser.add_argument('--output',type=Path);args=parser.parse_args()
    if args.worker:
        import rockmate
        import planner_reuse_control as fixture
        original=rockmate.from_config
        def retained(*values,**settings):
            settings['keep_outputs']=True
            return original(*values,**settings)
        rockmate.from_config=retained
        fixture.worker(args.worker,args.output)
        record=json.loads(args.output.read_text(encoding='utf-8'));record['keep_outputs']=True
        args.output.write_text(json.dumps(record,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8')
        return
    source=ROOT/'.tools/rockmate-review';review_path=ROOT/'docs/research/competitors/rockmate-execution-review.json'
    review=json.loads(review_path.read_text(encoding='utf-8'))
    for item in review['files']:assert sha(source/item['path'])==item['sha256']
    directory=ROOT/'artifacts/planner-reuse-retention-control';directory.mkdir(parents=True,exist_ok=True)
    interpreter=ROOT/'.tools/planner-env/Scripts/python.exe'
    env={key:os.environ[key] for key in ('SystemRoot','WINDIR') if key in os.environ}
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
        record['outer_wall_seconds']=time.perf_counter()-started;rows.append(record);print(mode,record['status'],flush=True)
    result={'captured_utc':datetime.now(timezone.utc).isoformat(),'source_sha256':sha(Path(__file__)),'fixture_source_sha256':sha(ROOT/'benchmarks/planner_reuse_control.py'),
            'review_sha256':sha(review_path),'modes':MODES,'trials':rows,
            'scope':'Exploratory keep_outputs=True option control, unchanged model/seed/objective/tolerances. Passing execution is distinct from numerical agreement; no historical paper or general framework conclusion.'}
    (ROOT/'docs/research/experiments/planner-reuse-retention-control.json').write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8')


if __name__=='__main__':
    main()
