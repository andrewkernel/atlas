"""Original analogue of streamed versus deferred microbatch backward execution.

No external checker code runs. Same separable MLP geometry/initialization as the
earlier CUDA fixture; locally generated tensor-only reference supplied explicitly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def child(deferred: bool, cap_mib: int, reference: Path) -> dict:
    import torch
    from torch import nn

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.fp32_precision='ieee'
    torch.backends.cuda.matmul.fp32_precision='ieee'
    torch.backends.cudnn.fp32_precision='ieee'
    if cap_mib:
        torch.cuda.set_per_process_memory_fraction(cap_mib*2**20/torch.cuda.get_device_properties(0).total_memory)
    torch.manual_seed(884)

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm=nn.LayerNorm(512)
            self.up=nn.Linear(512,2048)
            self.down=nn.Linear(2048,512)

        def forward(self,x):
            return x+.1*self.down(nn.functional.gelu(self.up(self.norm(x))))

    model=nn.Sequential(*(Block() for _ in range(8))).to('cuda')
    data=torch.randn(16,256,512,generator=torch.Generator().manual_seed(885)).to('cuda')
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,betas=(.9,.999),eps=1e-8,weight_decay=.01,foreach=False,fused=False)
    baseline=torch.load(reference,map_location='cpu',weights_only=True)
    completed=0
    first_errors=None
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    try:
        for update in range(8):
            optimizer.zero_grad(set_to_none=True)
            if deferred:
                losses=[model(chunk).square().sum()/data.numel() for chunk in data.split(4)]
                for loss in losses:
                    loss.backward()
                del loss,losses
            else:
                for chunk in data.split(4):
                    (model(chunk).square().sum()/data.numel()).backward()
            optimizer.step()
            torch.cuda.synchronize()
            completed+=1
            if update==0:
                first_errors={'parameter':0.,'gradient':0.,'momentum1':0.,'momentum2':0.}
                for name,parameter in model.named_parameters():
                    values={'parameter':(parameter,'parameter/'+name),'gradient':(parameter.grad,'gradient/'+name),
                            'momentum1':(optimizer.state[parameter]['exp_avg'],'optimizer/'+name+'/exp_avg'),
                            'momentum2':(optimizer.state[parameter]['exp_avg_sq'],'optimizer/'+name+'/exp_avg_sq')}
                    for group,(value,key) in values.items():
                        observed=value.detach().cpu()
                        if observed.shape!=baseline[key].shape or observed.dtype!=baseline[key].dtype or not torch.isfinite(observed).all():
                            raise ValueError('Invalid first-update state')
                        first_errors[group]=max(first_errors[group],(observed-baseline[key]).abs().max().item())
        status='completed'
    except torch.cuda.OutOfMemoryError:
        status='allocator_oom'
    return {'status':status,'deferred_backward':deferred,'allocator_cap_mib':cap_mib or None,
            'completed_optimizer_updates':completed,'first_update_max_abs_errors':first_errors,
            'allocated_peak_bytes':torch.cuda.max_memory_allocated(),'reserved_peak_bytes':torch.cuda.max_memory_reserved(),
            'environment':{'torch':torch.__version__,'cuda_runtime':torch.version.cuda,'gpu':torch.cuda.get_device_name(0)},
            'scope':'Eight repeated-input updates, microbatch4/logical16, no checkpointing; original pattern analogue only'}


def run(reference: Path,output: Path) -> None:
    root=Path('artifacts/activation-liveness')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    root.mkdir(parents=True)
    trials=[]
    for cap,deferred in ((0,False),(0,True),(600,False),(600,True)):
        target=root/f'{len(trials)}.json'
        command=[sys.executable,str(Path(__file__).resolve()),'--child','--reference',str(reference.resolve()),
                 '--cap-mib',str(cap),'--output',str(target.resolve())]
        if deferred:
            command.append('--deferred')
        environment=os.environ.copy()
        environment['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
        try:
            process=subprocess.run(command,env=environment,capture_output=True,text=True,timeout=180)
            result=json.loads(target.read_text(encoding='utf-8')) if target.exists() else {'status':'error','reason':'No child output'}
            result.update({'returncode':process.returncode,'stderr':process.stderr[-3000:]})
        except subprocess.TimeoutExpired:
            result={'status':'timeout'}
        result.update({'deferred_backward':deferred,'allocator_cap_mib':cap or None})
        trials.append(result)
        record={'schema_version':'1.0','started_utc':STARTED,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'reference_sha256':hashlib.sha256(reference.read_bytes()).hexdigest(),'trials':trials,
                'limitation':'Original control, not TrainParity execution or a global capability result. Artificial allocator cap on12 GB GPU, no physical smaller-card claim or training outcome study.'}
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        print(json.dumps({key:result.get(key) for key in ('status','deferred_backward','allocator_cap_mib','completed_optimizer_updates','allocated_peak_bytes')}),flush=True)


STARTED=datetime.now(timezone.utc).isoformat()
if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--child',action='store_true')
    parser.add_argument('--deferred',action='store_true')
    parser.add_argument('--cap-mib',type=int,default=0)
    args=parser.parse_args()
    if args.child:
        try:
            result=child(args.deferred,args.cap_mib,args.reference)
        except Exception as error:
            result={'status':'error','error_type':type(error).__name__,'reason':str(error)[:2000]}
        args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        raise SystemExit(0 if result['status'] in ('completed','allocator_oom') else 1)
    run(args.reference,args.output)
