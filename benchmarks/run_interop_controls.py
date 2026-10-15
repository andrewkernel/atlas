"""Shared supported-stack checks and a competent streaming capture adaptation."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
MODES=('native_accumulation_runner','planned_accumulation_runner','planned_streaming_capture')


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def internal(mode,output):
    import torch,trainparity
    from trainparity.api import check_accumulation,AccumulationExecutionPlan,ToleranceComparison
    from trainparity.snapshot import capture_snapshot
    from interop_cases import PlannedCase
    from planner_reuse_control import portable,finite_json
    row={'mode':mode,'status':'error','seed':2001,'rtol':1e-5,'atol':1e-7,
         'environment':{'torch':torch.__version__,'cuda':torch.version.cuda,'trainparity':trainparity.__version__}}
    started=time.perf_counter()
    try:
        if mode.endswith('accumulation_runner'):
            target='NativeCase' if mode.startswith('native_') else 'PlannedCase'
            result=check_accumulation('interop_cases:'+target,candidate=AccumulationExecutionPlan(microbatch_count=2),
                                      comparison=ToleranceComparison(rtol=1e-5,atol=1e-7),device='cuda',seed=2001,
                                      cwd=ROOT,timeout=30,temporary_root=output.parent/'temporary')
            row.update({'status':'reported','result':result.to_dict()})
        else:
            case=PlannedCase();reference=case.build(2001,'cuda');batch=case.batch('cuda')
            initial_reference=capture_snapshot(reference.model,step=0,optimizer=reference.optimizer,capture_rng=False)
            accounting=case.loss(reference,batch);reference.optimizer.zero_grad(set_to_none=True)
            accounting.value.backward();reference.optimizer.step()
            reference_loss=float(accounting.value.detach().cpu())
            post_reference=capture_snapshot(reference.model,step=1,optimizer=reference.optimizer,capture_rng=False,
                                             extras={'denominator':batch['target'].numel(),'loss':reference_loss})
            candidate=case.build(2001,'cuda')
            initial_candidate=capture_snapshot(candidate.model,step=0,optimizer=candidate.optimizer,capture_rng=False)
            candidate.optimizer.zero_grad(set_to_none=True);total=batch['target'].numel();loss_value=0.
            for x,target in zip(torch.tensor_split(batch['x'],2),torch.tensor_split(batch['target'],2)):
                part=case.loss(candidate,{'x':x,'target':target});loss=part.numerator/total
                loss_value+=float(loss.detach().cpu());loss.backward()
            candidate.optimizer.step()
            post_candidate=capture_snapshot(candidate.model,step=1,optimizer=candidate.optimizer,capture_rng=False,
                                             extras={'denominator':total,'loss':loss_value})
            policy=ToleranceComparison(rtol=1e-5,atol=1e-7)
            captures={'initial_reference':initial_reference,'initial_candidate':initial_candidate,'post_reference':post_reference,'post_candidate':post_candidate}
            row['capture_outcomes']={name:value.outcome.value for name,value in captures.items()}
            if any(value.snapshot is None for value in captures.values()):
                row['status']='capture_incomplete'
            else:
                row['initial_comparison']=asdict(policy.compare(initial_reference.snapshot,initial_candidate.snapshot))
                row['post_comparison']=asdict(policy.compare(post_reference.snapshot,post_candidate.snapshot))
                with torch.no_grad():next(candidate.model.parameters()).view(-1)[0].add_(.01)
                perturbed=capture_snapshot(candidate.model,step=1,optimizer=candidate.optimizer,capture_rng=False,
                                            extras={'denominator':total,'loss':loss_value})
                row['negative_capture_outcome']=perturbed.outcome.value
                row['negative_comparison']=None if perturbed.snapshot is None else asdict(policy.compare(post_reference.snapshot,perturbed.snapshot))
                row['status']='completed'
            row['data_sha256']=hashlib.sha256(batch['x'].cpu().numpy().tobytes()+batch['target'].cpu().numpy().tobytes()).hexdigest()
            row['rng_scope']='Pure fixed-input comparison; post-state snapshots deliberately exclude RNG. Construction uses the scoped Torch guard, not a generic state restoration claim.'
    except Exception as error:
        row.update({'error_type':type(error).__name__,'error_message':portable(str(error)),'traceback':portable(traceback.format_exc()[-5000:])})
    row['wall_seconds']=time.perf_counter()-started
    invalid=[];row=finite_json(row,invalid=invalid);row['nonfinite_scalar_fields']=invalid
    output.write_text(json.dumps(row,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--worker',choices=MODES);parser.add_argument('--output',type=Path);args=parser.parse_args()
    if args.worker:internal(args.worker,args.output);return
    reviews=[ROOT/'docs/research/competitors/trainparity-execution-review.json',ROOT/'docs/research/competitors/rockmate-execution-review.json']
    roots=[ROOT/'.tools/trainparity-reviewed',ROOT/'.tools/rockmate-review']
    for manifest,source in zip(reviews,roots):
        for item in json.loads(manifest.read_text(encoding='utf-8'))['files']:assert sha(source/item['path'])==item['sha256']
    directory=ROOT/'artifacts/interop-controls';directory.mkdir(parents=True,exist_ok=True)
    interpreter=ROOT/'.tools/interop-env/Scripts/python.exe'
    env={key:os.environ[key] for key in ('SystemRoot','WINDIR') if key in os.environ}
    env.update({'PATH':str(interpreter.parent)+os.pathsep+str(Path(os.environ['SystemRoot'])/'System32'),
                'PYTHONPATH':os.pathsep.join([str(roots[0]/'src'),str(roots[1]/'rockmate/src'),str(roots[1]/'rkgb/src'),str(ROOT/'benchmarks')]),
                'PYTHONNOUSERSITE':'1','PYTHONDONTWRITEBYTECODE':'1','PYTHONHASHSEED':'0',
                'TEMP':str(directory),'TMP':str(directory),'TORCH_HOME':str(directory/'torch-cache'),
                'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','CUBLAS_WORKSPACE_CONFIG':':4096:8'})
    rows=[]
    for mode in MODES:
        target=directory/(mode+'.json');started=time.perf_counter()
        with (directory/(mode+'.stdout.txt')).open('w',encoding='utf-8') as stdout,(directory/(mode+'.stderr.txt')).open('w',encoding='utf-8') as stderr:
            try:
                process=subprocess.run([str(interpreter),str(Path(__file__).resolve()),'--worker',mode,'--output',str(target)],cwd=directory,env=env,stdout=stdout,stderr=stderr,timeout=120,check=False)
                code=process.returncode
            except subprocess.TimeoutExpired:code=None
        row=json.loads(target.read_text(encoding='utf-8')) if code==0 and target.exists() else {'mode':mode,'status':'timeout' if code is None else 'worker_failure','returncode':code}
        row['outer_wall_seconds']=time.perf_counter()-started;rows.append(row);print(mode,row['status'],row.get('result',{}).get('outcome',''),flush=True)
    result={'captured_utc':datetime.now(timezone.utc).isoformat(),'source_sha256':sha(Path(__file__)),'case_source_sha256':sha(ROOT/'benchmarks/interop_cases.py'),
            'helper_source_sha256':sha(ROOT/'benchmarks/planner_reuse_control.py'),'review_sha256':{p.name:sha(p) for p in reviews},'modes':MODES,'trials':rows,
            'scope':'Declared-compatible shared GPU stack; selected accumulation runner and existing capture/comparison primitives with an authored streaming executor. No Atlas policy, historical paper reproduction or field reliability claim.'}
    (ROOT/'docs/research/experiments/interop-controls.json').write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8')


if __name__=='__main__':main()
