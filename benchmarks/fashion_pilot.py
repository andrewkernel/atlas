"""Original fixed five-epoch CNN pilot, not a planner or equivalence study."""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import os
import platform
import random
import struct
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

CONFIG = {'seeds': [1101,1102,1103], 'split_seed': 9001, 'epochs': 5, 'train_examples': 50000,
          'validation_examples': 10000, 'logical_batch': 128, 'candidate_microbatch': 32,
          'lr': .01, 'momentum': .9, 'weight_decay': 0.}


def tensor_digest(tensors) -> str:
    digest = hashlib.sha256()
    for name, tensor in tensors:
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def worker(seed: int, candidate: bool, data_root: Path, manifest_path: Path, output: Path) -> dict:
    import torch
    from torch import nn
    from torch.utils.checkpoint import checkpoint

    began = time.perf_counter()
    if not torch.cuda.is_available():
        raise RuntimeError('This pilot explicitly requires CUDA')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.fp32_precision = 'ieee'
    torch.backends.cuda.matmul.fp32_precision = 'ieee'
    torch.backends.cudnn.fp32_precision = 'ieee'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    expanded = {}
    for entry in manifest['files']:
        body = (data_root/entry['name']).read_bytes()
        if hashlib.sha256(body).hexdigest() != entry['compressed_sha256']:
            raise ValueError('Dataset bytes differ from frozen manifest')
        value = gzip.decompress(body)
        if hashlib.sha256(value).hexdigest() != entry['expanded_sha256']:
            raise ValueError('Expanded dataset bytes differ')
        expanded[entry['name']] = value
    image_bytes = expanded['train-images-idx3-ubyte.gz']
    if struct.unpack('>IIII', image_bytes[:16]) != (2051,60000,28,28):
        raise ValueError('Invalid image dimensions')
    images = torch.frombuffer(bytearray(image_bytes), dtype=torch.uint8, offset=16).clone().reshape(60000,1,28,28).float().div_(255)
    labels = torch.frombuffer(bytearray(expanded['train-labels-idx1-ubyte.gz']), dtype=torch.uint8, offset=8).clone().long()
    if len(labels) != 60000 or labels.min() < 0 or labels.max() > 9:
        raise ValueError('Invalid labels')
    del expanded, image_bytes
    split = torch.randperm(60000, generator=torch.Generator().manual_seed(CONFIG['split_seed']))
    train_ids, validation_ids = split[:50000], split[50000:]

    class CNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.block1 = nn.Sequential(nn.Conv2d(1,32,3), nn.ReLU())
            self.block2 = nn.Sequential(nn.Conv2d(32,64,3), nn.ReLU(), nn.MaxPool2d(2))
            self.head = nn.Linear(9216,10)

        def forward(self, x, checkpointed=False):
            for block in (self.block1,self.block2):
                x = checkpoint(block,x,use_reentrant=False,preserve_rng_state=True) if checkpointed else block(x)
            return self.head(x.flatten(1))

    torch.manual_seed(seed)
    initial = CNN()
    initial_hash = tensor_digest(initial.state_dict().items())
    orders = [train_ids[torch.randperm(50000,generator=torch.Generator().manual_seed(seed*100+epoch))] for epoch in range(5)]
    order_hash = tensor_digest((str(epoch),order) for epoch,order in enumerate(orders))
    split_hash = tensor_digest([('train',train_ids),('validation',validation_ids)])

    def make_state():
        model = copy.deepcopy(initial).to('cuda').train()
        optimizer = torch.optim.SGD(model.parameters(),lr=.01,momentum=.9,weight_decay=0,foreach=False)
        return model,optimizer

    def update(model, optimizer, ids, microbatch, checkpointed):
        optimizer.zero_grad(set_to_none=True)
        x,y = images[ids].to('cuda'),labels[ids].to('cuda')
        for start in range(0,len(ids),microbatch):
            logits = model(x[start:start+microbatch],checkpointed)
            loss = nn.functional.cross_entropy(logits,y[start:start+microbatch],reduction='sum')/len(ids)
            loss.backward()
        optimizer.step()

    def snapshot(model,optimizer):
        state = {}
        for name,p in model.named_parameters():
            state['parameter/'+name] = p.detach().cpu().clone()
            state['gradient/'+name] = p.grad.detach().cpu().clone()
            state['momentum/'+name] = optimizer.state[p]['momentum_buffer'].detach().cpu().clone()
        return state

    verification_began = time.perf_counter()
    reference_model,reference_optimizer = make_state()
    update(reference_model,reference_optimizer,orders[0][:128],128,False)
    torch.cuda.synchronize()
    reference_state = snapshot(reference_model,reference_optimizer)
    del reference_model,reference_optimizer
    torch.cuda.empty_cache()
    model,optimizer = make_state()
    torch.cuda.synchronize()
    preparation_seconds = time.perf_counter()-began
    reference_preparation_seconds = time.perf_counter()-verification_began
    torch.cuda.reset_peak_memory_stats()
    history = []
    first_comparison = None
    first_peak = None
    successful_updates = 0
    update_seconds = 0.
    verification_seconds = 0.
    microbatch = 32 if candidate else 128
    for epoch,order in enumerate(orders):
        epoch_start = time.perf_counter()
        model.train()
        for start in range(0,50000,128):
            ids = order[start:start+128]
            torch.cuda.synchronize()
            update_start = time.perf_counter()
            update(model,optimizer,ids,microbatch,candidate)
            torch.cuda.synchronize()
            update_seconds += time.perf_counter()-update_start
            successful_updates +=1
            if successful_updates ==1:
                first_peak = {'allocated_bytes':torch.cuda.max_memory_allocated(),'reserved_bytes':torch.cuda.max_memory_reserved()}
                check_start = time.perf_counter()
                observed = snapshot(model,optimizer)
                first_comparison = {}
                for key,value in observed.items():
                    error = (value-reference_state[key]).abs().max().item()
                    group = key.split('/')[0]
                    first_comparison[group] = max(first_comparison.get(group,0.),error)
                    if not torch.isfinite(value).all():
                        raise ValueError('Nonfinite first-update state')
                del observed,reference_state
                verification_seconds += time.perf_counter()-check_start
        train_wall = time.perf_counter()-epoch_start
        evaluation_start = time.perf_counter()
        model.eval()
        loss_sum,correct=0.,0
        with torch.no_grad():
            for start in range(0,10000,128):
                ids=validation_ids[start:start+128]
                logits=model(images[ids].to('cuda'))
                y=labels[ids].to('cuda')
                loss_sum+=nn.functional.cross_entropy(logits,y,reduction='sum').item()
                correct+=int((logits.argmax(1)==y).sum().item())
        torch.cuda.synchronize()
        history.append({'epoch':epoch+1,'successful_updates':successful_updates,'train_loop_wall_seconds':train_wall,
                        'validation_wall_seconds':time.perf_counter()-evaluation_start,
                        'validation_nll':loss_sum/10000,'validation_accuracy':correct/10000,'validation_correct':correct})
        output.write_text(json.dumps({'status':'running','seed':seed,'candidate':candidate,'history':history},indent=2)+'\n',encoding='utf-8')
    return {'status':'completed','seed':seed,'candidate':candidate,'microbatch':microbatch,'checkpointed':candidate,
            'environment':{'python':sys.version,'torch':torch.__version__,'cuda_runtime':torch.version.cuda,'platform':platform.platform(),
                           'gpu':torch.cuda.get_device_name(0),'device_total_bytes':torch.cuda.get_device_properties(0).total_memory,
                           'deterministic_algorithms':True,'cudnn_benchmark':False,'fp32_precision':'ieee'},
            'initial_state_digest':initial_hash,'split_digest':split_hash,'sample_order_digest':order_hash,
            'parameter_count':sum(p.numel() for p in model.parameters()),'first_update_max_abs_errors':first_comparison,
            'first_update_peak':first_peak,'whole_training_and_validation_peak':{'allocated_bytes':torch.cuda.max_memory_allocated(),'reserved_bytes':torch.cuda.max_memory_reserved()},
            'preparation_seconds':preparation_seconds,'reference_preparation_seconds_within_preparation':reference_preparation_seconds,
            'first_state_comparison_seconds':verification_seconds,'summed_synchronized_update_seconds':update_seconds,
            'history':history,'successful_updates':successful_updates,'samples_seen':50000*5,
            'final_state_digest':tensor_digest(model.state_dict().items()),'worker_wall_seconds':time.perf_counter()-began}


def run(data_root: Path,manifest: Path,output: Path,artifact_root: Path) -> None:
    artifact_root=artifact_root/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    artifact_root.mkdir(parents=True)
    records=[]
    order=[(seed,candidate) for seed in CONFIG['seeds'] for candidate in (False,True)]
    random.Random(9051).shuffle(order)
    for seed,candidate in order:
        child_output=artifact_root/f'{len(records)}.json'
        command=[sys.executable,str(Path(__file__).resolve()),'--child','--seed',str(seed),'--data-root',str(data_root.resolve()),
                 '--manifest',str(manifest.resolve()),'--output',str(child_output.resolve())]
        if candidate:
            command.append('--candidate')
        environment=os.environ.copy()
        environment['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
        began=time.perf_counter()
        try:
            process=subprocess.run(command,env=environment,capture_output=True,text=True,timeout=300)
            record=json.loads(child_output.read_text(encoding='utf-8')) if child_output.exists() else {'status':'error','reason':'No child output'}
            record.update({'returncode':process.returncode,'stderr':process.stderr[-3000:]})
        except subprocess.TimeoutExpired:
            record={'status':'timeout','timeout_seconds':300}
        record.update({'seed':seed,'candidate':candidate,'owned_child_wall_seconds':time.perf_counter()-began})
        records.append(record)
        result={'schema_version':'1.0','started_utc':STARTED,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'config':CONFIG,'dataset_manifest':json.loads(manifest.read_text(encoding='utf-8')),'trials':records,
                'limitation':'Three exploratory pilot seeds, no confirmatory equivalence inference, scientific conclusion, planner comparison, physical OOM rescue or canonical test-set evaluation.'}
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        print(json.dumps({'seed':seed,'candidate':candidate,'status':record['status'],'final':record.get('history',[None])[-1]},allow_nan=False),flush=True)


STARTED=datetime.now(timezone.utc).isoformat()
if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--data-root',type=Path,default=Path('data/fashion-mnist'))
    parser.add_argument('--manifest',type=Path,default=Path('docs/research/experiments/fashion-data-manifest.json'))
    parser.add_argument('--artifact-root',type=Path,default=Path('artifacts/fashion-pilot'))
    parser.add_argument('--child',action='store_true')
    parser.add_argument('--seed',type=int,default=1101)
    parser.add_argument('--candidate',action='store_true')
    args=parser.parse_args()
    if args.child:
        try:
            record=worker(args.seed,args.candidate,args.data_root,args.manifest,args.output)
        except Exception as error:
            record={'status':'error','error_type':type(error).__name__,'reason':str(error)[:2000]}
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        raise SystemExit(0 if record['status']=='completed' else 1)
    run(args.data_root,args.manifest,args.output,args.artifact_root)
