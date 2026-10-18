"""Diagnose legacy GPT-2 failure without altering its 1e-4 predicate."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import json
import uuid
import torch
from filelock import FileLock
from transformers import GPT2LMHeadModel,AutoTokenizer
import atlas.attention.model as integration
from atlas.attention.runtime import DecodeCache,PLANS,error_metrics
from atlas.attention.oracle import attention_oracle,distribution_metrics
from atlas.attention.research import configure
from atlas.attention.evidence import ROOT,RUNS,Recorder,sha
from atlas.attention.assets import file_sha


def main():
    configure();directory=RUNS/('diagnosis-v1-replay-'+uuid.uuid4().hex[:12]);rec=Recorder(directory,{'kind':'diagnosis','relation':'Original float32 atol=rtol=1e-4; legacy prompt only'})
    with FileLock(str(RUNS/'.gpu.lock'),timeout=0),torch.inference_mode():
        manifest=json.loads((integration.MODEL_DIR/'atlas-manifest.json').read_text())
        if manifest['revision']!=integration.MODEL_REVISION:raise ValueError('Diagnosis requires original pinned GPT-2 revision')
        for filename in ('model.safetensors','config.json','tokenizer.json'):
            if file_sha(integration.MODEL_DIR/filename)!=manifest['files_sha256'][filename]:raise ValueError('GPT-2 bytes changed')
        rec.doc['model']=manifest;rec.doc['diagnostic_source_sha256']=sha(Path(__file__).read_bytes())
        saved_source=directory/'source/benchmarks/diagnose_attention_v1.py';saved_source.parent.mkdir(parents=True,exist_ok=True);saved_source.write_bytes(Path(__file__).read_bytes())
        model=GPT2LMHeadModel.from_pretrained(integration.MODEL_DIR,local_files_only=True,use_safetensors=True,attn_implementation='sdpa',dtype=torch.float32).cuda().eval()
        tokenizer=AutoTokenizer.from_pretrained(integration.MODEL_DIR,local_files_only=True)
        tokens=tokenizer(integration.DEFAULT_PROMPT,return_tensors='pt').input_ids.cuda()
        rec.doc['input']={'tokens':tokens.shape[1],'input_sha256':sha(tokens.cpu().numpy().tobytes()),'prompt':integration.DEFAULT_PROMPT,'dtype':'float32'}
        state={'step':-1};captures={};handles=[]
        for index,block in enumerate(model.transformer.h):
            def hook(module,args,output,i=index):
                if state['step'] in (0,46,47):captures[(state['step'],i)]=(output[0] if isinstance(output,tuple) else output).float().cpu()
            handles.append(block.register_forward_hook(hook))
        history=[];reference=[];past=None
        for step in range(49):
            state['step']=step
            inp=tokens if step==0 else torch.tensor([[history[-1]]],device='cuda')
            result=model(inp,past_key_values=past,use_cache=True);past=result.past_key_values
            reference.append(result.logits[:,-1,:].cpu());history.append(int(result.logits[:,-1,:].argmax(-1)))
        native_captures=dict(captures);del result,past
        for method in ('resident','gpu_dense','hybrid50'):
            captures.clear();state['step']=-1;instances=[];local=[]
            class DiagnosticCache(DecodeCache):
                def __init__(self,*args,**kwargs):
                    super().__init__(*args,**kwargs);self.index=len(instances);instances.append(self)
                def attend(self,q,scale=None):
                    out=super().attend(q,scale)
                    if state['step']==47 and self.index in (0,5,11):
                        segments=list(self.segments());keys=[self.ck[...,:self.cpu_count,:]]+[x[0].cpu() for x in segments];vals=[self.cv[...,:self.cpu_count,:]]+[x[1].cpu() for x in segments]
                        ref=attention_oracle(q.cpu().numpy(),torch.cat(keys,dim=-2).numpy(),torch.cat(vals,dim=-2).numpy(),scale)
                        local.append({'layer':self.index,'step':47,'correctness':error_metrics(out,torch.from_numpy(ref))})
                    return out
            saved=integration.DecodeCache;integration.DecodeCache=DiagnosticCache
            try:
                with integration.GPT2Adapter(model,PLANS[method],tokens.shape[1]+48):
                    checks=[];distribution=[]
                    for step in range(49):
                        state['step']=step
                        inp=tokens if step==0 else torch.tensor([[history[step-1]]],device='cuda')
                        result=model(inp,use_cache=False,position_ids=(torch.arange(tokens.shape[1],device='cuda').unsqueeze(0) if step==0 else torch.tensor([[tokens.shape[1]+step-1]],device='cuda')))
                        logits=result.logits[:,-1,:].cpu();checks.append(error_metrics(logits,reference[step]));distribution.append(distribution_metrics(logits.numpy(),reference[step].numpy()))
                rec.doc['rows'].append({'plan':method,'context':tokens.shape[1],'failed_steps':[i for i,c in enumerate(checks) if not c['passed']],
                    'logit_checks':checks,'distribution_checks':distribution,'local_attention_oracle':local,
                    'layer_checks':[{'step':step,'layer':i,'correctness':error_metrics(out,native_captures[(step,i)])} for (step,i),out in captures.items()]})
                rec.save()
            finally:integration.DecodeCache=saved
        for h in handles:h.remove()
    rec.finish()
    dest=ROOT/'docs/research/attention-v2/diagnosis-v1-replay.json'
    if dest.exists():dest=dest.with_name(directory.name+'.json')
    dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((directory/'result.json').read_bytes())
    print(json.dumps({'diagnosis':str(dest),'methods':len(rec.doc['rows'])}))


if __name__=='__main__':main()
