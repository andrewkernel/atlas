"""Pinned safetensors GPT-2 preparation and full autoregressive integration.

The adapter owns caches before framework concatenation. Supported scope is one
unpadded sequence, eval, float32, GPT-2 standard scaling, no cross attention.
"""
from __future__ import annotations
import gc
import json
import random
import statistics
from pathlib import Path
import time
import types
import torch
from torch.nn import functional as F
from .benchmark import synchronize,percentile
from .evidence import ROOT,Recorder,atomic_json,sha
from .runtime import PLANS,DecodeCache,error_metrics,CalibratedPolicy

MODEL_ID="openai-community/gpt2"
MODEL_DIR=ROOT/"weights"/"gpt2-attention-lab"
DEFAULT_PROMPT="Memory systems connect processors to the data they need. A useful experiment measures both the cost of moving data and the time spent computing with it. The results suggest that"


def prepare_model():
    from huggingface_hub import HfApi,snapshot_download
    info=HfApi().model_info(MODEL_ID)
    revision=info.sha
    snapshot_download(MODEL_ID,revision=revision,local_dir=MODEL_DIR,
        allow_patterns=["config.json","generation_config.json","model.safetensors","tokenizer.json","tokenizer_config.json","vocab.json","merges.txt","README.md"])
    files={p.name:sha(p.read_bytes()) for p in MODEL_DIR.iterdir() if p.is_file()}
    manifest={"model_id":MODEL_ID,"revision":revision,"files_sha256":files,
        "loader":"builtin GPT2LMHeadModel, safetensors only, trust_remote_code=False, local_files_only=True",
        "source":"https://huggingface.co/openai-community/gpt2","license":"MIT"}
    atomic_json(MODEL_DIR/"atlas-manifest.json",manifest)
    print(json.dumps({"model_id":MODEL_ID,"revision":revision,"files":list(files)}))


class GPT2Adapter:
    def __init__(self,model,plan,capacity):
        if model.training or model.config.add_cross_attention or model.config.scale_attn_by_inverse_layer_idx:
            raise ValueError("Only standard eval GPT-2 without cross attention is supported")
        self.model,self.plan,self.capacity=model,plan,capacity
        self.caches={}
        self.original=[]

    def __enter__(self):
        for index,block in enumerate(self.model.transformer.h):
            attn=block.attn
            self.original.append((attn,attn.forward))
            def forward(module,hidden_states,past_key_values=None,attention_mask=None,encoder_hidden_states=None,encoder_attention_mask=None,output_attentions=False,_index=index,**kwargs):
                if hidden_states.shape[0]!=1 or module.training or encoder_hidden_states is not None:
                    raise ValueError("Adapter requires one unpadded inference sequence")
                q,k,v=module.c_attn(hidden_states).split(module.split_size,dim=2)
                def shape(x):return x.view(1,-1,module.num_heads,module.head_dim).transpose(1,2)
                q,k,v=shape(q),shape(k),shape(v)
                scale=module.head_dim**-.5 if module.scale_attn_weights else 1.0
                if _index not in self.caches:
                    out=F.scaled_dot_product_attention(q,k,v,is_causal=True,scale=scale)
                    self.caches[_index]=DecodeCache(k,v,self.plan,self.capacity)
                else:
                    self.caches[_index].append(k,v)
                    out=self.caches[_index].attend(q,scale=scale)
                out=out.transpose(1,2).reshape(*hidden_states.shape[:-1],module.embed_dim).contiguous()
                return module.resid_dropout(module.c_proj(out)),None
            attn.forward=types.MethodType(forward,attn)
        return self

    def __exit__(self,*args):
        for module,forward in self.original:module.forward=forward
        self.caches.clear()


def run_model(rec:Recorder):
    from transformers import AutoTokenizer,GPT2LMHeadModel
    if not (MODEL_DIR/"atlas-manifest.json").exists():
        raise RuntimeError("Model assets missing. Run: python lab.py prepare-model")
    config=rec.doc["config"]
    torch.set_num_threads(config.get("threads",4))
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32=False
    if not torch.cuda.is_available():raise RuntimeError("CUDA required")
    asset_manifest=json.loads((MODEL_DIR/"atlas-manifest.json").read_text())
    for name,digest in asset_manifest["files_sha256"].items():
        if name in ("model.safetensors","config.json","tokenizer.json") and sha((MODEL_DIR/name).read_bytes())!=digest:
            raise ValueError(f"Pinned model file changed: {name}")
    start=time.perf_counter()
    tokenizer=AutoTokenizer.from_pretrained(MODEL_DIR,local_files_only=True,trust_remote_code=False)
    model=GPT2LMHeadModel.from_pretrained(MODEL_DIR,local_files_only=True,trust_remote_code=False,
        use_safetensors=True,attn_implementation="sdpa",dtype=torch.float32).cuda().eval()
    synchronize()
    rec.doc["model"]={**asset_manifest,"load_s":time.perf_counter()-start,"dtype":"float32",
        "max_positions":model.config.n_positions,"parameter_count":sum(p.numel() for p in model.parameters())}
    text=config.get("prompt",DEFAULT_PROMPT)
    tokens=tokenizer(text,return_tensors="pt").input_ids.cuda()
    steps=config.get("new_tokens",24)
    if tokens.shape[1]+steps>model.config.n_positions:raise ValueError("Prompt plus continuation exceeds GPT-2 positional limit")
    rec.doc["prompt"]={"text":text,"tokens":tokens.shape[1],"sha256":sha(tokens.cpu().numpy().tobytes())}
    rec.doc["protocol"]={"comparison":"same teacher-forced native greedy token history for every method",
        "atol":1e-4,"rtol":1e-4,"decode_steps":steps,"timing":"synchronized wall time including cache append, eviction and transfers",
        "quality_scope":"finite full-vocabulary logit checks; no downstream quality benchmark",
        "prefill":"native causal SDPA for all methods; offload/cache preparation included in prefill time"}
    rec.doc["protocol"]["cache_window"]="Hybrid fractions set the initial prompt's GPU window; that token window stays fixed during decoding."
    calibration=[]
    candidates=[]
    for path in (ROOT/"artifacts"/"attention-lab").glob("*/result.json"):
        doc=json.loads(path.read_text())
        if (doc.get("status")=="completed" and doc.get("config",{}).get("kind")=="benchmark"
            and doc.get("config",{}).get("heads",12)==12 and doc["config"].get("threads",4)==config.get("threads",4)
            and doc.get("environment",{}).get("source_sha256",{}).get("src/atlas/attention/runtime.py")==sha((ROOT/"src/atlas/attention/runtime.py").read_bytes())):
            candidates.append((doc["created_at"],doc,path))
    if candidates:
        _,doc,path=max(candidates,key=lambda x:x[0])
        calibration=doc["rows"]
        rec.doc["policy_source"]={"id":doc["id"],"sha256":sha(path.read_bytes())}
    # Model is 12 heads x64, matching the authored synthetic suite. No hidden profiling.
    if calibration:
        decision=CalibratedPolicy(calibration).choose(tokens.shape[1],12,64,config.get("kv_budget_mib",2048))
        rec.doc["adaptive_decision"]=decision
    methods=["native","resident","gpu_dense","copy","cpu","hybrid25","hybrid50","hybrid75"]
    if calibration:methods.append("adaptive")
    reference_logits=[];continuation=[]
    with torch.inference_mode():
        # Warm model kernels before any timed method; no caches persist between arms.
        model(tokens,use_cache=False)
        synchronize()
        # Independent native reference history, generated before randomized timing.
        ref_started=time.perf_counter()
        result=model(tokens,use_cache=True)
        reference_logits.append(result.logits[:,-1,:].cpu())
        for _ in range(steps):
            rec.check()
            token=int(result.logits[:,-1,:].argmax(-1).item())
            continuation.append(token)
            result=model(torch.tensor([[token]],device="cuda"),past_key_values=result.past_key_values,use_cache=True)
            reference_logits.append(result.logits[:,-1,:].cpu())
        synchronize()
        rec.doc["reference_preparation_s"]=time.perf_counter()-ref_started
        del result
        random.Random(config.get("seed",2026)).shuffle(methods)
        rec.doc["method_order"]=methods
        for method_index,ident in enumerate(methods):
            rec.progress(method_index,len(methods),f"GPT-2 · {ident}")
            gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
            selected=rec.doc.get("adaptive_decision",{}).get("plan") if ident=="adaptive" else ident
            adapter=None if ident=="native" else GPT2Adapter(model,PLANS[selected],tokens.shape[1]+steps)
            if adapter:adapter.__enter__()
            result=past=logits=None
            try:
                synchronize();t=time.perf_counter()
                result=model(tokens,use_cache=adapter is None,position_ids=torch.arange(tokens.shape[1],device="cuda").unsqueeze(0))
                synchronize();prefill_ms=(time.perf_counter()-t)*1000
                past=result.past_key_values
                logits=result.logits[:,-1,:]
                samples=[];checks=[];agreements=[]
                checks.append(error_metrics(logits,reference_logits[0]))
                for step in range(steps):
                    rec.check()
                    agreements.append(int(logits.argmax(-1).item())==continuation[step])
                    next_token=torch.tensor([[continuation[step]]],device="cuda")
                    synchronize();t=time.perf_counter()
                    result=model(next_token,past_key_values=past,use_cache=adapter is None,
                        position_ids=torch.tensor([[tokens.shape[1]+step]],device="cuda"))
                    synchronize();samples.append((time.perf_counter()-t)*1000)
                    past=result.past_key_values
                    logits=result.logits[:,-1,:]
                    checks.append(error_metrics(logits,reference_logits[step+1]))
                row={"plan":ident,"selected_plan":selected,"label":"Native model SDPA" if ident=="native" else ("Atlas adaptive" if ident=="adaptive" else PLANS[ident].label),
                    "context":tokens.shape[1],"phase":"model","samples_ms":samples,"prefill_ms":prefill_ms,
                    "median_ms":statistics.median(samples),"p90_ms":percentile(samples,.9),
                    "decode_ms":sum(samples),"tokens_per_s":steps/(sum(samples)/1000),
                    "allocated_peak_mib":torch.cuda.max_memory_allocated()/2**20,
                    "greedy_agreement":sum(agreements)/len(agreements),"logit_checks":checks,
                    "correctness":{"passed":all(c["passed"] for c in checks),"finite":all(c["finite"] for c in checks),
                        "max_abs":max((c["max_abs"] for c in checks if c["max_abs"] is not None),default=None),"atol":1e-4,"rtol":1e-4,
                        "violating_elements":sum(c["violating_elements"] or 0 for c in checks),
                        "element_count":sum(c["element_count"] for c in checks)},
                    "gpu_kv_mib":sum(c.gpu_kv_bytes for c in adapter.caches.values())/2**20 if adapter else None,
                    "cpu_kv_mib":sum(c.cpu_kv_bytes for c in adapter.caches.values())/2**20 if adapter else None,
                    "gpu_window_tokens":next(iter(adapter.caches.values())).window if adapter else None}
                rec.doc["rows"].append(row)
                rec.save()
            finally:
                if adapter:adapter.__exit__()
                del result,past,logits
    rec.doc["continuation"]={"token_ids":continuation,"text":tokenizer.decode(continuation),
        "note":"Native greedy history, teacher-forced into all comparison arms; base-model text is not factual advice."}
    rec.progress(len(methods),len(methods),"Model comparison complete")
