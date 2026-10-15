"""Differential equality versus an externally stated finite objective."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from board_energy_control import ROOT,child_environment
D=ROOT/'docs/research/experiments'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def oracle(case):
    import torch
    from token_reference_cases import TokenCase
    state=case.build(2301,'cpu');batch=case.batch('cpu');loss=case.loss(state,batch).loss;loss.backward()
    expected=TokenCase().build(2301,'cpu');logits=expected.model(batch['x']);mask=batch['y']!=-100
    chosen=torch.log_softmax(logits,dim=-1)[mask].gather(1,batch['y'][mask,None]).sum()
    target=-chosen/int(mask.sum());target.backward()
    losses={'declared_token_oracle':target.item(),'adapter_objective':loss.item()}
    squared=norm=maximum=0.
    for (name,p),(other,q) in zip(state.model.named_parameters(),expected.model.named_parameters(),strict=True):
        assert name==other and torch.equal(p,q)
        difference=p.grad-q.grad;squared+=difference.square().sum().item();norm+=q.grad.square().sum().item();maximum=max(maximum,difference.abs().max().item())
    return {'losses':losses,'gradient_max_abs_error':maximum,'gradient_relative_l2_error':(squared/norm)**.5,'valid_token_counts':[int(x) for x in mask.sum(dim=1)],'initial_parameters_exactly_equal':True,'scope':'Independent authored log-softmax/gather token-objective comparison on the full fixed batch; not independent human labels, a model-quality study or a checker bug.'}
def main():
    p=argparse.ArgumentParser();p.add_argument('--internal',choices=['TokenCase','SequenceMeanCase']);p.add_argument('--output',type=Path);a=p.parse_args()
    if a.internal:
        from trainparity.api import AccumulationExecutionPlan,ToleranceComparison,check_accumulation
        from token_reference_cases import TokenCase,SequenceMeanCase
        result=check_accumulation('token_reference_cases:'+a.internal,candidate=AccumulationExecutionPlan(microbatch_count=2),comparison=ToleranceComparison(rtol=1e-10,atol=1e-12),device='cpu',seed=2301,cwd=ROOT,timeout=30,temporary_root=a.output/'temporary',report_path=a.output/'checker.json')
        obj=TokenCase() if a.internal=='TokenCase' else SequenceMeanCase()
        print(json.dumps({'case':a.internal,'checker':result.to_dict(),'objective_control':oracle(obj)},sort_keys=True,allow_nan=False),flush=True);return
    review=json.loads((ROOT/'docs/research/competitors/trainparity-execution-review.json').read_text(encoding='utf-8'));source=ROOT/'.tools/trainparity-reviewed'
    for entry in review['files']:assert sha(source/entry['path'])==entry['sha256']
    directory=ROOT/'artifacts/token-reference-control'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ');directory.mkdir(parents=True)
    result={'started_utc':datetime.now(timezone.utc).isoformat(),'source_sha256':sha(Path(__file__)),'case_source_sha256':sha(ROOT/'benchmarks/token_reference_cases.py'),
            'review_sha256':sha(ROOT/'docs/research/competitors/trainparity-execution-review.json'),'environment_helper_sha256':sha(ROOT/'benchmarks/board_energy_control.py'),
            'checker_revision':review['revision'],'seed':2301,'modes':['TokenCase','SequenceMeanCase'],'trials':[]}
    interpreter=ROOT/'.tools/checker-env/Scripts/python.exe'
    for name in result['modes']:
        scratch=directory/name;scratch.mkdir();env=child_environment(interpreter,scratch);env['PYTHONPATH']=str(source/'src')+os.pathsep+str(ROOT/'benchmarks')
        start=time.perf_counter()
        try:
            child=subprocess.run([str(interpreter),str(Path(__file__).resolve()),'--internal',name,'--output',str(scratch)],cwd=ROOT,env=env,capture_output=True,text=True,timeout=115)
            from planner_reuse_control import portable
            row=json.loads(child.stdout.strip().splitlines()[-1]) if child.returncode==0 else {'case':name,'status':'launcher_error','returncode':child.returncode,'stderr_excerpt':portable(child.stderr[-2000:])}
        except subprocess.TimeoutExpired:row={'case':name,'status':'timeout','timeout_seconds':115}
        row['outer_wall_seconds']=time.perf_counter()-start;result['trials'].append(row)
        (D/'token-reference-control.json').write_bytes((json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n').encode())
        print(json.dumps(row),flush=True)
if __name__=='__main__':main()
