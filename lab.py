"""Atlas Attention Lab CLI. Run with the documented CUDA worker environment."""
from pathlib import Path
import argparse
import json
import sys
import uuid
sys.path.insert(0,str(Path(__file__).resolve().parent/"src"))


def main():
    parser=argparse.ArgumentParser(description="Atlas exact dense attention research workspace")
    sub=parser.add_subparsers(dest="command",required=True)
    sub.add_parser("prepare-model")
    assets=sub.add_parser('prepare-research')
    assets.add_argument('--model',choices=['qwen3-0.6b','qwen3-4b','corpus'],required=True)
    serve=sub.add_parser("serve");serve.add_argument("--port",type=int,default=8765)
    for name in ("benchmark","model","research"):
        p=sub.add_parser(name)
        p.add_argument("--profile",choices=["quick","full"],default="full")
        p.add_argument("--threads",type=int,choices=[1,2,4,6],default=4)
        p.add_argument("--repeats",type=int,default=9)
        p.add_argument("--new-tokens",type=int,default=24)
        p.add_argument("--prompt",default=None)
        p.add_argument("--seed",type=int,default=2026)
        p.add_argument("--kv-budget-mib",type=float,default=2048)
        if name=='research':
            p.add_argument('--suite',choices=['pilot','cache'],default='pilot')
            p.add_argument('--context',type=int,choices=[128,512,2048],default=512)
            p.add_argument('--dtype',choices=['float32','float16','bfloat16'],default='float32')
    worker=sub.add_parser("worker");worker.add_argument("--run-id",required=True)
    args=parser.parse_args()
    if args.command=='prepare-research':
        from atlas.attention.assets import prepare,prepare_corpus
        prepare_corpus() if args.model=='corpus' else prepare(args.model)
        return
    if args.command=="prepare-model":
        from atlas.attention.model import prepare_model
        prepare_model();return
    if args.command=="serve":
        import uvicorn
        from atlas.attention.server import app
        uvicorn.run(app,host="127.0.0.1",port=args.port,log_level="warning");return
    from atlas.attention.evidence import RUNS,Recorder,Cancelled
    if args.command=="worker":
        if not args.run_id.replace('-','').isalnum():raise ValueError("Invalid run ID")
        directory=RUNS/args.run_id
        config=json.loads((directory/"config.json").read_text())
    else:
        directory=RUNS/(args.command+"-"+uuid.uuid4().hex[:12])
        config={"kind":args.command,"profile":args.profile,"threads":args.threads,"repeats":args.repeats,"new_tokens":args.new_tokens,"seed":args.seed,"kv_budget_mib":args.kv_budget_mib}
        if args.prompt:config["prompt"]=args.prompt
        if args.command=='research':config.update(suite=args.suite,context=args.context,dtype=args.dtype,split='train',methods=['native','resident','gpu_dense','hybrid50'],window=256,new_tokens=8)
    rec=Recorder(directory,config)
    try:
        from filelock import FileLock
        import torch
        rec.check()
        with FileLock(str(RUNS/".gpu.lock"),timeout=0),torch.inference_mode():
            if config["kind"]=="benchmark":
                from atlas.attention.benchmark import run_benchmark
                run_benchmark(rec)
            elif config["kind"]=="model":
                from atlas.attention.model import run_model
                run_model(rec)
            elif config['kind']=='research':
                from atlas.attention.research import run_research
                run_research(rec)
            else:raise ValueError("Unregistered experiment")
        rec.finish()
        print(json.dumps({"run":directory.name,"status":"completed","rows":len(rec.doc['rows'])}))
    except Cancelled as exc:
        rec.finish("cancelled",str(exc))
    except Exception as exc:
        import traceback
        rec.finish("failed",f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        sys.exit(1)


if __name__=="__main__":main()
