"""Audit exact tolerance/health semantics of the reused comparison machinery."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime,timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
CASES=(
    ('tensor_candidate_scale_accept','tensor','1','1.11',.1,0.,'tolerance',False),
    ('tensor_candidate_scale_reject','tensor','1','.905',.1,0.,'tolerance',False),
    ('scalar_max_scale','scalar','1','1.11',.1,0.,'tolerance',False),
    ('scalar_max_vs_additive','scalar','1','1.15',.1,.1,'tolerance',False),
    ('tensor_small_boundary','tensor','1','1.00001000005',1e-5,0.,'tolerance',False),
    ('same_infinity_exact','tensor','inf','inf',1e-5,1e-8,'exact',False),
    ('same_infinity_tolerance','tensor','inf','inf',1e-5,1e-8,'tolerance',False),
    ('same_nan_exact','tensor','nan','nan',1e-5,1e-8,'exact',False),
    ('same_nan_tolerance','tensor','nan','nan',1e-5,1e-8,'tolerance',False),
    ('same_nan_equal_nan','tensor','nan','nan',1e-5,1e-8,'tolerance',True),
)


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--worker',action='store_true');parser.add_argument('--output',type=Path);args=parser.parse_args()
    if args.worker:
        import platform,torch,trainparity
        from trainparity.comparison import ExactComparison,ToleranceComparison
        from trainparity.snapshot import Snapshot
        from trainparity.state import FullValueBackend
        from planner_reuse_control import finite_json
        backend=FullValueBackend();rows=[]
        for name,kind,a_text,b_text,rtol,atol,policy,equal_nan in CASES:
            a,b=float(a_text),float(b_text)
            left=torch.tensor([a],dtype=torch.float64) if kind=='tensor' else a
            right=torch.tensor([b],dtype=torch.float64) if kind=='tensor' else b
            baseline=Snapshot(0,backend.freeze({'value':left}));candidate=Snapshot(0,backend.freeze({'value':right}))
            comparator=ExactComparison() if policy=='exact' else ToleranceComparison(rtol=rtol,atol=atol,equal_nan=equal_nan)
            finite=math.isfinite(a) and math.isfinite(b)
            rows.append({'name':name,'representation':kind,'baseline_literal':a_text,'candidate_literal':b_text,'policy':policy,
                         'rtol':rtol,'atol':atol,'equal_nan':equal_nan,'comparison':asdict(comparator.compare(baseline,candidate)),
                         'both_finite':finite,'baseline_scaled_additive_rule':abs(a-b)<=atol+rtol*abs(a) if finite else None})
        result={'captured_utc':datetime.now(timezone.utc).isoformat(),'source_sha256':sha(Path(__file__)),
                'helper_source_sha256':sha(ROOT/'benchmarks/planner_reuse_control.py'),
                'review_sha256':sha(ROOT/'docs/research/competitors/trainparity-execution-review.json'),
                'environment':{'python':platform.python_version(),'torch':torch.__version__,'trainparity':trainparity.__version__,'device':'cpu','tensor_dtype':'float64'},
                'trials':rows,'scope':'Ten deterministic constructed boundary/health controls, not ML error prevalence, scientific equivalence or GPU-kernel validation.'}
        invalid=[];result=finite_json(result,invalid=invalid);result['replaced_nonfinite_scalar_fields']=invalid
        args.output.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8');return
    source=ROOT/'.tools/trainparity-reviewed';manifest=ROOT/'docs/research/competitors/trainparity-execution-review.json'
    for item in json.loads(manifest.read_text(encoding='utf-8'))['files']:assert sha(source/item['path'])==item['sha256']
    directory=ROOT/'artifacts/comparator-contract-control';directory.mkdir(parents=True,exist_ok=True)
    interpreter=ROOT/'.tools/checker-env/Scripts/python.exe';env={key:os.environ[key] for key in ('SystemRoot','WINDIR') if key in os.environ}
    env.update({'PATH':str(interpreter.parent)+os.pathsep+str(Path(os.environ['SystemRoot'])/'System32'),
                'PYTHONPATH':str(source/'src')+os.pathsep+str(ROOT/'benchmarks'),'PYTHONNOUSERSITE':'1','PYTHONDONTWRITEBYTECODE':'1',
                'TEMP':str(directory),'TMP':str(directory),'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
    output=ROOT/'docs/research/experiments/comparator-contract-control.json'
    with (directory/'stdout.txt').open('w',encoding='utf-8') as stdout,(directory/'stderr.txt').open('w',encoding='utf-8') as stderr:
        subprocess.run([str(interpreter),str(Path(__file__).resolve()),'--worker','--output',str(output)],cwd=directory,env=env,stdout=stdout,stderr=stderr,timeout=30,check=True)


if __name__=='__main__':main()
