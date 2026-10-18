"""Seeded CUDA benchmark with disjoint calibration and validation lengths."""
from __future__ import annotations
import gc
import random
import statistics
import time
import torch
from .evidence import Recorder, sha
from .runtime import PLANS, DecodeCache, CalibratedPolicy, error_metrics, partial_attention
from .oracle import attention_oracle


def synchronize():
    if torch.cuda.is_available():torch.cuda.synchronize()


def percentile(values:list[float],fraction:float)->float:
    xs=sorted(values)
    pos=(len(xs)-1)*fraction
    lower=int(pos)
    return xs[lower]+(xs[min(lower+1,len(xs)-1)]-xs[lower])*(pos-lower)


def run_benchmark(rec:Recorder):
    config=rec.doc["config"]
    torch.set_num_threads(config.get("threads",4))
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32=False
    if not torch.cuda.is_available():raise RuntimeError("This benchmark requires CUDA; no CPU results are substituted")
    calibration=[512,2048,8192,32768] if config.get("profile")=="full" else [256,1024,4096]
    validation=[1024,4096,16384] if config.get("profile")=="full" else [512,2048]
    heads,dim=config.get("heads",12),config.get("head_dim",64)
    repeats=config.get("repeats",9)
    rec.doc["protocol"]={"calibration_lengths":calibration,"validation_lengths":validation,
        "warmups":3,"repeats":repeats,"dtype":"float32","query_tokens":1,"batch":1,
        "reference":"independent float64 CPU dense softmax","atol":1e-4,"rtol":1e-4,
        "timing":"CUDA-synchronized wall time; query transfers, output return and fusion included; cache preparation separate",
        "selector":"linear-median-v1; calibration rows only; no held-out observations supplied"}
    rec.doc["protocol"]["budget_scope"]="Soft GPU KV selection budget only; sweep measures every plan, including ineligible plans; no whole-process quota"
    total=(len(calibration)+len(validation))*len(PLANS)
    done=0
    phase_started=time.perf_counter()
    for phase,lengths in (("calibration",calibration),("validation",validation)):
        if phase=="validation":
            rec.doc["calibration_elapsed_s"]=time.perf_counter()-phase_started
            policy=CalibratedPolicy(rec.doc["rows"])
            rec.doc["policy_decisions"]=[{"context":n,**policy.choose(n,heads,dim,config.get("kv_budget_mib",2048))} for n in lengths]
            # Freeze the fixed comparator using calibration only, normalized per shape.
            ratios={}
            for ident in PLANS:
                if CalibratedPolicy.kv_mib(ident,max(validation),heads,dim)>config.get("kv_budget_mib",2048):continue
                ratios[ident]=statistics.mean(next(r["median_ms"] for r in rec.doc["rows"] if r["context"]==n and r["plan"]==ident)/min(r["median_ms"] for r in rec.doc["rows"] if r["context"]==n) for n in calibration)
            rec.doc["fixed_plan"]=min(ratios,key=lambda p:(ratios[p],p))
        for length in lengths:
            gen=torch.Generator().manual_seed(config.get("seed",2026)+length)
            q=torch.randn(1,heads,1,dim,generator=gen)
            k=torch.randn(1,heads,length,dim,generator=gen)
            v=torch.randn(1,heads,length,dim,generator=gen)
            reference=torch.from_numpy(attention_oracle(q.numpy(),k.numpy(),v.numpy()))
            input_hash=sha(q.numpy().tobytes()+k.numpy().tobytes()+v.numpy().tobytes())
            order=list(PLANS)
            random.Random(config.get("seed",2026)+length).shuffle(order)
            for ident in order:
                rec.progress(done,total,f"{phase.title()} · {length:,} tokens · {PLANS[ident].label}")
                gc.collect()
                torch.cuda.empty_cache()
                synchronize()
                setup=time.perf_counter()
                gq=q.cuda()
                cache=DecodeCache(k.cuda(),v.cuda(),PLANS[ident],length)
                synchronize()
                setup_ms=(time.perf_counter()-setup)*1000
                for _ in range(3):cache.attend(gq)
                synchronize()
                if ident=="resident" and length==calibration[0]:
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
                        cache.attend(gq);synchronize()
                    rec.doc["native_sdpa_operations"]=[event.key for event in prof.key_averages() if 'attention' in event.key]
                torch.cuda.reset_peak_memory_stats()
                base_alloc=torch.cuda.memory_allocated()
                samples=[]
                for _ in range(repeats):
                    rec.check()
                    synchronize()
                    t=time.perf_counter()
                    out=cache.attend(gq)
                    synchronize()
                    samples.append((time.perf_counter()-t)*1000)
                peak=torch.cuda.max_memory_allocated()
                row={"phase":phase,"context":length,"heads":heads,"head_dim":dim,"plan":ident,
                    "label":PLANS[ident].label,"samples_ms":samples,"median_ms":statistics.median(samples),
                    "p90_ms":percentile(samples,.9),"min_ms":min(samples),"setup_ms":setup_ms,
                    "gpu_kv_mib":cache.gpu_kv_bytes/2**20,"cpu_kv_mib":cache.cpu_kv_bytes/2**20,
                    "allocated_peak_mib":peak/2**20,"scratch_peak_mib":(peak-base_alloc)/2**20,
                    "correctness":error_metrics(out,reference),"input_sha256":input_hash}
                rec.doc["rows"].append(row)
                done+=1
                del cache,gq,out
                rec.save()
    rec.doc["evaluation"]=[]
    for decision in rec.doc["policy_decisions"]:
        rows=[r for r in rec.doc["rows"] if r["phase"]=="validation" and r["context"]==decision["context"]]
        chosen=next(r for r in rows if r["plan"]==decision["plan"])
        eligible=[r for r in rows if r["correctness"]["passed"] and CalibratedPolicy.kv_mib(r["plan"],r["context"],heads,dim)<=config.get("kv_budget_mib",2048)]
        best=min(eligible,key=lambda r:r["median_ms"])
        fixed=next(r for r in rows if r["plan"]==rec.doc["fixed_plan"])
        rec.doc["evaluation"].append({"context":decision["context"],"selected":decision["plan"],
            "selected_ms":chosen["median_ms"],"measured_best":best["plan"],"best_ms":best["median_ms"],
            "regret_percent":(chosen["median_ms"]/best["median_ms"]-1)*100,
            "fixed_ms":fixed["median_ms"],"vs_fixed_ratio":fixed["median_ms"]/chosen["median_ms"],
            "correctness_passed":chosen["correctness"]["passed"]})
    rec.progress(total,total,"Benchmark complete")
