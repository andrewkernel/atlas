"""Existing native selective checkpointing on the unchanged original CUDA fixture."""
import argparse
from collections import Counter
from datetime import datetime,timezone
import functools
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import time
from board_energy_control import ROOT,child_environment
D=ROOT/'docs/research/experiments'
MODES=('identity','plain_checkpoint','selective_matmul')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def child(mode,micro,reference,write_reference):
    import torch
    import torch.utils.checkpoint as cp
    import cuda_memory_probe as fixture
    original=json.loads((D/'cuda-memory-probe.json').read_text(encoding='utf-8'))
    assert sha(Path(fixture.__file__))==original['source_sha256']
    counts=Counter();saved=Counter();regions=0
    if mode=='selective_matmul':
        ordinary=cp.checkpoint
        selected={torch.ops.aten.mm.default,torch.ops.aten.addmm.default,torch.ops.aten.bmm.default}
        def wrapped(function,*args,**kwargs):
            nonlocal regions
            record=regions<fixture.CONFIG['depth']*(fixture.CONFIG['logical_batch']//micro);regions+=1
            def policy(context,op,*operands,**keywords):
                keep=op in selected
                if record:
                    counts[str(op)]+=1
                    if keep:saved[str(op)]+=1
                return cp.CheckpointPolicy.MUST_SAVE if keep else cp.CheckpointPolicy.PREFER_RECOMPUTE
            kwargs['context_fn']=functools.partial(cp.create_selective_checkpoint_contexts,policy)
            return ordinary(function,*args,**kwargs)
        cp.checkpoint=wrapped
    try:
        result=fixture.child_trial(micro,mode!='identity',reference,write_reference)
    finally:
        if mode=='selective_matmul':cp.checkpoint=ordinary
    result.update({'mode':mode,'first_logical_update_policy_ops':dict(counts),'first_logical_update_must_save_ops':dict(saved),
                   'selective_checkpoint_region_calls':regions,'scope':'Original fixed-input pure residual MLP; first-update numerical observations, first/later peaks and finite update timings; no full-training equivalence or primary policy comparison.'})
    return result
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--child',action='store_true');parser.add_argument('--mode',choices=MODES);parser.add_argument('--micro',type=int);parser.add_argument('--reference',type=Path);parser.add_argument('--write-reference',action='store_true');parser.add_argument('--output',type=Path);args=parser.parse_args()
    if args.child:
        try:record=child(args.mode,args.micro,args.reference,args.write_reference)
        except Exception as e:
            from planner_reuse_control import portable
            record={'status':'error','error_type':type(e).__name__,'message':portable(str(e))}
        args.output.write_bytes((json.dumps(record,indent=2,sort_keys=True,allow_nan=False)+'\n').encode());return
    directory=ROOT/'artifacts/native-selective-control'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ');directory.mkdir(parents=True)
    reference=directory/'first-update-reference.pt';rng=random.Random(2201);order=[]
    for repetition in range(2):
        plans=[(mode,micro) for mode in MODES for micro in (16,4)];rng.shuffle(plans)
        if repetition==0:plans.remove(('identity',16));plans.insert(0,('identity',16))
        order.extend((repetition,mode,micro) for mode,micro in plans)
    result={'started_utc':datetime.now(timezone.utc).isoformat(),'source_sha256':sha(Path(__file__)),
            'fixture_source_sha256':sha(ROOT/'benchmarks/cuda_memory_probe.py'),'environment_helper_sha256':sha(ROOT/'benchmarks/board_energy_control.py'),
            'order_seed':2201,'order':[{'repetition':r,'mode':m,'micro':b} for r,m,b in order],'trials':[],
            'scope':'Two process repetitions x three native configurations x two microbatch sizes. Shares original pure synthetic fixture, no new optimizer or scientific-equivalence result.'}
    interpreter=ROOT/'.tools/cuda-env/Scripts/python.exe'
    for index,(rep,mode,micro) in enumerate(order):
        scratch=directory/str(index);scratch.mkdir();target=scratch/'result.json'
        command=[str(interpreter),str(Path(__file__).resolve()),'--child','--mode',mode,'--micro',str(micro),'--reference',str(reference),'--output',str(target)]
        if index==0:command.append('--write-reference')
        start=time.perf_counter()
        try:
            process=subprocess.run(command,cwd=ROOT,env=child_environment(interpreter,scratch),capture_output=True,text=True,timeout=180)
            record=json.loads(target.read_text(encoding='utf-8')) if target.exists() else {'status':'missing_result'}
            from planner_reuse_control import portable
            record.update({'returncode':process.returncode,'stderr_excerpt':portable(process.stderr[-2000:])})
        except subprocess.TimeoutExpired:record={'status':'timeout','timeout_seconds':180}
        record.update({'repetition':rep,'mode':mode,'microbatch':micro,'owned_child_wall_seconds':time.perf_counter()-start})
        result['trials'].append(record)
        (D/'native-selective-control.json').write_bytes((json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n').encode())
        print(json.dumps({k:record.get(k) for k in ('repetition','mode','microbatch','status','measured_peak','median_update_seconds')}),flush=True)
if __name__=='__main__':main()
