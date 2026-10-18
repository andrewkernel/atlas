"""Registered research studies: real corpus, matched precision, full generation."""
from __future__ import annotations
import gc
import json
import random
import statistics
import time
import numpy as np
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from .assets import MODELS,model_dir,corpus_text,file_sha
from .benchmark import synchronize,percentile
from .evidence import ROOT,sha
from .oracle import attention_oracle,distribution_metrics
from .qwen import QwenAdapter
from .runtime import PLANS,error_metrics,DecodeCache,ShiftDecodeCache
from .telemetry import Sampler
from .validity import CONTRACTS,eligibility


def configure(threads=4):
    torch.set_num_threads(threads);torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False
    if not torch.cuda.is_available():raise RuntimeError('CUDA required; CPU results are not substituted')


def model_inputs(config,tokenizer):
    text,manifest=corpus_text(config.get('split','train'))
    # Split is read only after the protocol has been frozen for confirmation.
    ids=tokenizer(text,return_tensors='pt',truncation=False,verbose=False).input_ids
    context,steps=config['context'],config.get('new_tokens',8)
    offset=config.get('offset',0)
    if offset+context+steps>ids.shape[1]:raise ValueError('Corpus range outside pinned split')
    selected=ids[:,offset:offset+context+steps].clone()
    return selected[:,:context].cuda(),selected[:,context:].cuda(),{
        'split':config.get('split','train'),'offset':offset,'tokens':context,
        'input_token_sha256':sha(selected.numpy().tobytes()),'dataset':manifest,
        'scope':'Contiguous public text; finite continuation NLL only, not a downstream task suite'}


def forward(model,tokens,past,position,adapter):
    return model(tokens,past_key_values=past,use_cache=adapter is None,
                 position_ids=position,logits_to_keep=1)


def run_research(rec):
    config=rec.doc['config'];configure(config.get('threads',4))
    from .protocol import verify_protocol
    if config.get('protocol_id'):verify_protocol(config)
    if config.get('suite')=='cache':return cache_ablation(rec)
    if config.get('suite')=='layer':return layer_study(rec)
    name='qwen3-0.6b';folder=model_dir(name)
    manifest=json.loads((folder/'atlas-manifest.json').read_text())
    for filename,digest in manifest['files_sha256'].items():
        if file_sha(folder/filename)!=digest:raise ValueError('Pinned model bytes changed: '+filename)
    dtype=getattr(torch,config.get('dtype','float32'))
    load=time.perf_counter()
    tokenizer=AutoTokenizer.from_pretrained(folder,local_files_only=True,trust_remote_code=False)
    model=AutoModelForCausalLM.from_pretrained(folder,local_files_only=True,trust_remote_code=False,
           use_safetensors=True,attn_implementation='sdpa',dtype=dtype).cuda().eval()
    synchronize()
    rec.doc.update(schema_version=2,model={**manifest,'load_s':time.perf_counter()-load,'dtype':str(dtype),'parameter_count':sum(p.numel() for p in model.parameters())},
                   contract=CONTRACTS[config.get('dtype','float32')],uncertainty_unit='independent worker process, not adjacent tokens')
    tokens,targets,rec.doc['input']=model_inputs(config,tokenizer)
    context=tokens.shape[1];steps=config.get('new_tokens',8)
    methods=list(config.get('methods',['native','resident','gpu_dense','hybrid50']))
    base=['native','resident','gpu_dense','hybrid50']
    # Cyclic Latin rotations vary position without pilot timing.
    block=config.get('block',0);order=base[block%len(base):]+base[:block%len(base)]
    order=[m for m in order if m in methods]+[m for m in methods if m not in order]
    rec.doc['method_order']=order
    rec.doc['protocol']={'id':config.get('protocol_id'),'comparison':'Matched dtype native SDPA; same corpus history in separate validation replay',
        'timing':'Full free greedy generation: prefill, token selection, input preparation, append/eviction/transfers, synchronized forwards. Validation is a separate replay.',
        'new_tokens':steps,'prefill_logits':'Final token only for all arms','window_tokens':config.get('window',256),
        'precision':'FP16/BF16 KV storage, FP32 partition math; conversion costs included. Native SDPA uses matched model dtype.',
        'quality':'Real next-token NLL and distribution diagnostics over a bounded continuation; no general quality or training claim',
        'memory':'Allocator peaks, process RSS, board observations; Windows WDDM may spill. No hard quota or CXL hardware.'}
    rec.doc['limitations']=['One shared Windows consumer GPU; no CXL/DSA hardware or exclusive physical residency guarantee.',
        'Independent processes share one machine/session; no external replication or unbiased population estimate.',
        'Small public corpus and bounded continuations are not broad language quality evaluation.',
        'Native reference is precision matched; passing low-precision bounds does not establish equivalence to FP32.']
    positions=torch.arange(context,device='cuda').unsqueeze(0)
    with torch.inference_mode():
        warm=model(tokens[:,:min(context,128)],use_cache=False,logits_to_keep=1);del warm;synchronize()
        reference=[];past=result=None
        rec.progress(0,len(order),'Preparing native validation replay')
        # Reference generation and validation are not inside performance windows.
        if not config.get('pressure'):
            result=forward(model,tokens,None,positions,None)
            for step in range(steps):
                rec.check();reference.append(result.logits[:,-1,:].float().cpu())
                if step<steps-1:result=forward(model,targets[:,step:step+1],result.past_key_values,torch.tensor([[context+step]],device='cuda'),None)
        del result;gc.collect();torch.cuda.empty_cache()
        for number,method in enumerate(order):
            rec.progress(number,len(order),f'{context:,} tokens / {str(dtype)} / {method}')
            adapter=None if method=='native' else QwenAdapter(model,PLANS[method],context+steps,window=config.get('window',256))
            gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
            row={'plan':method,'label':('Native model SDPA' if method=='native' else PLANS[method].label),
                'context':context,'phase':'model','dtype':str(dtype),'block':block,'status':'running'}
            with Sampler(rec.directory/(method+'-telemetry.json')) as sampler:
                if adapter:adapter.__enter__()
                result=past=logits=None
                try:
                    sampler.phase=method+'/generation'
                    synchronize();began=time.perf_counter();t=began
                    result=forward(model,tokens,None,positions,adapter)
                    synchronize();prefill=(time.perf_counter()-t)*1000
                    samples=[];generated=[];decode_began=time.perf_counter()
                    for step in range(steps):
                        rec.check()
                        token=result.logits[:,-1,:].argmax(-1,keepdim=True);generated.append(int(token.item()))
                        if step<steps-1:
                            synchronize();t=time.perf_counter()
                            result=forward(model,token,result.past_key_values,torch.tensor([[context+step]],device='cuda'),adapter)
                            synchronize();samples.append((time.perf_counter()-t)*1000)
                    synchronize();ended=time.perf_counter()
                    row.update(generation_ms=(ended-began)*1000,decode_ms=(ended-decode_began)*1000,prefill_ms=prefill,
                        samples_ms=samples,median_ms=statistics.median(samples),p90_ms=percentile(samples,.9),
                        tokens_per_s=steps/(ended-began),generated_token_ids=generated,
                        allocated_peak_mib=torch.cuda.max_memory_allocated()/2**20,reserved_peak_mib=torch.cuda.max_memory_reserved()/2**20,
                        gpu_kv_mib=sum(c.gpu_kv_bytes for c in adapter.caches.values())/2**20 if adapter else 2*model.config.num_hidden_layers*model.config.num_key_value_heads*model.config.head_dim*(context+steps-1)*torch.empty((),dtype=dtype).element_size()/2**20,
                        cpu_kv_mib=sum(c.cpu_kv_bytes for c in adapter.caches.values())/2**20 if adapter else 0)
                except torch.OutOfMemoryError as exc:
                    row.update(status='oom',error=str(exc)[:1200],allocated_peak_mib=torch.cuda.max_memory_allocated()/2**20,reserved_peak_mib=torch.cuda.max_memory_reserved()/2**20,correctness={'passed':False,'finite':None},eligibility={'eligible':False,'reason':'OOM; numerical relation unavailable'})
                finally:
                    if adapter:adapter.__exit__()
                    del result,past,logits;gc.collect();torch.cuda.empty_cache()
                if row['status']!='oom' and config.get('pressure'):
                    row.update(status='completed',correctness={'passed':None,'finite':None},eligibility={'eligible':False,'reason':'Pressure feasibility run only; full-context numerical replay not measured'})
                if row['status']!='oom' and not config.get('pressure'):
                    sampler.phase=method+'/validation'
                    capture={} if context<=512 and method=='hybrid50' else None
                    adapter=None if method=='native' else QwenAdapter(model,PLANS[method],context+steps,window=config.get('window',256),capture=capture)
                    if adapter:adapter.__enter__()
                    try:
                        checks=[];distributions=[]
                        result=forward(model,tokens,None,positions,adapter)
                        for step in range(steps):
                            rec.check();logits=result.logits[:,-1,:].float().cpu()
                            checks.append(error_metrics(logits,reference[step],**rec.doc['contract']))
                            distributions.append(distribution_metrics(logits.numpy(),reference[step].numpy(),int(targets[0,step])))
                            if step<steps-1:result=forward(model,targets[:,step:step+1],result.past_key_values,torch.tensor([[context+step]],device='cuda'),adapter)
                        row.update(status='completed',logit_checks=checks,distribution_checks=distributions,
                            correctness={'passed':all(c['passed'] for c in checks),'finite':all(c['finite'] for c in checks),'max_abs':max(c['max_abs'] for c in checks),**rec.doc['contract']},
                            eligibility=eligibility(checks),mean_nll=statistics.mean(c['candidate_nll'] for c in distributions),
                            native_mean_nll=statistics.mean(c['reference_nll'] for c in distributions),max_kl=max(c['reference_kl'] for c in distributions),
                            greedy_agreement=statistics.mean(c['argmax_agreement'] for c in distributions))
                        if capture:
                            row['captured_layer_oracle']=[{'layer':i,'correctness':error_metrics(torch.from_numpy(out),torch.from_numpy(attention_oracle(q,k,v)),**rec.doc['contract'])} for i,(q,k,v,out) in capture.items()]
                            if any(not c['correctness']['passed'] for c in row['captured_layer_oracle']):row['eligibility']={'eligible':False,'reason':'Captured QKV independent oracle failed'}
                    finally:
                        if adapter:adapter.__exit__()
                        del result,logits;gc.collect();torch.cuda.empty_cache()
            row['telemetry']=sampler.rows
            rec.doc['rows'].append(row);rec.save()
    rec.progress(len(order),len(order),'Matched full-model comparison complete')


def cache_ablation(rec):
    """Matched shift/ring complete decode steps; initial setup is separately costed."""
    configure();seed=8159+rec.doc['config'].get('block',0);torch.manual_seed(seed);gen=torch.Generator().manual_seed(seed)
    rec.doc['seed']=seed
    for context in (512,4096,16384):
        k=torch.randn(1,12,context+16,64,generator=gen,device='cpu').cuda();v=torch.randn_like(k);q=torch.randn(1,12,1,64,device='cuda')
        for label,cls in (('shift',ShiftDecodeCache),('ring',DecodeCache)):
            synchronize();start=time.perf_counter();cache=cls(k[...,:context,:],v[...,:context,:],PLANS['hybrid50'],context+16)
            synchronize();setup=(time.perf_counter()-start)*1000;samples=[]
            for step in range(16):
                synchronize();start=time.perf_counter();cache.append(k[...,context+step:context+step+1,:],v[...,context+step:context+step+1,:]);out=cache.attend(q);synchronize();samples.append((time.perf_counter()-start)*1000)
            expected=attention_oracle(q.cpu().numpy(),k.cpu().numpy(),v.cpu().numpy())
            rec.doc['rows'].append({'plan':label,'label':label.title()+' complete decode','phase':'cache','context':context,'samples_ms':samples,'median_ms':statistics.median(samples),'p90_ms':percentile(samples,.9),'setup_ms':setup,'gpu_kv_mib':cache.gpu_kv_bytes/2**20,'correctness':error_metrics(out,torch.from_numpy(expected))})
            del cache;rec.save()
        del k,v,q


def layer_study(rec):
    from .validity import BoundPolicy
    config=rec.doc['config'];dtype=getattr(torch,config['dtype'])
    identity={k:rec.doc['environment'][k] for k in ('gpu','cuda','packages','source_sha256')}
    identity.update(dtype=config['dtype'],threads=4,heads=16,kv_heads=8,dim=128,batch=1,query=1,scope='layer-attend-only',implementation=DecodeCache.version)
    rec.doc.update(identity=identity,calibration_cost_s=0,policy_decisions=[],evaluation=[])
    start=time.perf_counter();done=0
    for phase,lengths in [('calibration',[768,3072,12288,32768]),('validation',[1792,7168,20480])]:
        if phase=='validation':
            rec.doc['calibration_cost_s']=time.perf_counter()-start
            policy=BoundPolicy(rec.doc['rows'],identity)
            rec.doc['fixed_plans']={}
            for budget in (48,2048):
                support=policy.choose(max(lengths),identity,budget).get('predictions',[])
                costs={p['plan']:statistics.mean(r['median_ms']/min(x['median_ms'] for x in rec.doc['rows'] if x['phase']=='calibration' and x['context']==r['context']) for r in rec.doc['rows'] if r['phase']=='calibration' and r['plan']==p['plan']) for p in support}
                rec.doc['fixed_plans'][str(budget)]=min(costs,key=costs.get) if costs else None
            for length in lengths:
                for budget in (48,2048):
                    rec.doc['policy_decisions'].append({'context':length,'budget_mib':budget,'selected':policy.choose(length,identity,budget),'rule':policy.budget_rule(length,identity,budget)})
        for length in lengths:
            gen=torch.Generator().manual_seed(config['seed']+length)
            q=torch.randn(1,16,1,128,generator=gen).to(dtype)
            k=torch.randn(1,8,length,128,generator=gen).to(dtype);v=torch.randn(1,8,length,128,generator=gen).to(dtype)
            ref=torch.from_numpy(attention_oracle(q.float().numpy(),k.float().numpy(),v.float().numpy()))
            order=list(PLANS);random.Random(config['seed']+length).shuffle(order)
            for plan in order:
                rec.progress(done,49,f'{phase} / {config["dtype"]} / {length} / {plan}')
                cache=DecodeCache(k,v,PLANS[plan],length,device='cuda');gq=q.cuda()
                for _ in range(3):cache.attend(gq)
                synchronize();samples=[]
                if plan=='resident' and length==768:
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:cache.attend(gq);synchronize()
                    rec.doc['native_sdpa_operations']=[e.key for e in prof.key_averages() if 'attention' in e.key]
                for _ in range(config['repeats']):
                    rec.check();synchronize();t=time.perf_counter();out=cache.attend(gq);synchronize();samples.append((time.perf_counter()-t)*1000)
                check=error_metrics(out,ref,**CONTRACTS[config['dtype']])
                rec.doc['rows'].append({'phase':phase,'context':length,'plan':plan,'label':PLANS[plan].label,'identity':identity,'dtype':config['dtype'],'block':config['block'],
                    'input_sha256':sha(q.float().numpy().tobytes()+k.float().numpy().tobytes()+v.float().numpy().tobytes()),
                    'samples_ms':samples,'median_ms':statistics.median(samples),'p90_ms':percentile(samples,.9),'gpu_kv_mib':cache.gpu_kv_bytes/2**20,
                    'cpu_kv_mib':cache.cpu_kv_bytes/2**20,'correctness':check,'eligibility':eligibility([check])})
                done+=1;del cache,gq,out;rec.save()
    # Retrospective oracle is used for regret only, never supplied to the policy.
    for decision in rec.doc['policy_decisions']:
        valid=[r for r in rec.doc['rows'] if r['phase']=='validation' and r['context']==decision['context'] and r['correctness']['passed'] and r['gpu_kv_mib']<=decision['budget_mib']]
        best=min(valid,key=lambda r:r['median_ms']) if valid else None
        fixed=rec.doc['fixed_plans'][str(decision['budget_mib'])]
        item={'context':decision['context'],'budget_mib':decision['budget_mib'],'oracle_plan':best['plan'] if best else None,'fixed_plan':fixed}
        for key in ('selected','rule'):
            selected=next((r for r in valid if r['plan']==decision[key].get('plan')),None)
            item[key]={'plan':decision[key].get('plan'),'eligible':selected is not None,'regret_percent':(selected['median_ms']/best['median_ms']-1)*100 if selected and best else None,'measured_ms':selected['median_ms'] if selected else None,'prediction_error_ms':selected['median_ms']-decision[key]['predicted_ms'] if selected else None}
        f=next((r for r in valid if r['plan']==fixed),None);item['fixed_ms']=f['median_ms'] if f else None
        rec.doc['evaluation'].append(item)
