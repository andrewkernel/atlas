"""Finite native/planner module-reuse and accumulation integration controls."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
MODES = ("native_streaming", "planner_streaming", "planner_deferred_separate", "planner_deferred_joint", "planner_single")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def worker(mode: str, output: Path) -> None:
    import copy
    import torch
    from torch import nn
    phase = "fixture"
    row = {"mode": mode, "status": "error", "seed": 1901, "rtol": 1e-5, "atol": 1e-7}
    try:
        torch.set_num_threads(1)
        torch.manual_seed(1901)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

        class Stack(nn.Module):
            def __init__(self):
                super().__init__()
                blocks = []
                for _ in range(4):
                    blocks.extend([nn.Linear(64,128), nn.ReLU(), nn.Linear(128,64), nn.ReLU()])
                self.layers = nn.Sequential(*blocks)

            def forward(self, x):
                return self.layers(x)

        template = Stack()
        count = 1 if mode == "planner_single" else 2
        inputs = [torch.randn(128,64,64,device="cuda") for _ in range(2)]
        targets = [torch.randn_like(x) for x in inputs]
        total = sum(value.numel() for value in targets[:count])
        row["input_target_sha256"] = hashlib.sha256(b"".join(value.cpu().numpy().tobytes() for value in [*inputs,*targets])).hexdigest()
        initial = {key:value.detach().clone() for key,value in template.state_dict().items()}

        def run(module, original, style):
            optimizer = torch.optim.SGD(original.parameters(),lr=.01,momentum=.9)
            original.zero_grad(set_to_none=True)
            reported_losses = []
            if style == "streaming":
                for x,target in zip(inputs[:count],targets[:count]):
                    result = module(x)
                    result = result[0] if isinstance(result,tuple) else result
                    loss = (result-target).square().sum()/total
                    reported_losses.append(float(loss.detach().cpu()))
                    loss.backward()
            else:
                losses=[]
                for x,target in zip(inputs[:count],targets[:count]):
                    result=module(x)
                    result=result[0] if isinstance(result,tuple) else result
                    loss=(result-target).square().sum()/total
                    reported_losses.append(float(loss.detach().cpu()))
                    losses.append(loss)
                if style == "deferred_separate":
                    for loss in losses: loss.backward()
                else:
                    sum(losses).backward()
            gradient={name:None if p.grad is None else p.grad.detach().cpu().clone() for name,p in original.named_parameters()}
            optimizer.step()
            parameters={name:p.detach().cpu().clone() for name,p in original.named_parameters()}
            momentum={name:optimizer.state[p].get('momentum_buffer',torch.empty(0)).detach().cpu().clone() for name,p in original.named_parameters()}
            return {"gradient":gradient,"parameter":parameters,"momentum":momentum},reported_losses

        phase="native_reference"
        reference_model=copy.deepcopy(template).cuda()
        reference,reference_losses=run(reference_model,reference_model,"deferred_joint")
        del reference_model
        row["reference_losses"]=reference_losses
        model=copy.deepcopy(template).cuda()
        phase="planner_construction"
        if mode.startswith("planner"):
            import rockmate
            config=rockmate.generate_config("rotor")
            config.solver.top.rotor.force_python=True
            config.partitioner.sequence.sub_partitioner="base"
            with torch.random.fork_rng(devices=[0]):
                module=rockmate.from_config(model,inputs[0],32*1024**2,config=config,
                                             gpu_optim=None,cpu_optim=None,minor_offload_size=0)
            row["schedule_present"]=module.op_sched is not None
            if module.op_sched is None:
                raise RuntimeError("No schedule found")
        else:
            module=model
        model.load_state_dict(initial)
        row["initial_parameter_max_abs_error"]=max(float((p.detach().cpu()-initial[name]).abs().max()) for name,p in model.named_parameters())
        style="streaming" if mode in ("native_streaming","planner_streaming") else "deferred_separate" if mode=="planner_deferred_separate" else "deferred_joint"
        phase="logical_update_"+style
        observed,losses=run(module,model,style)
        row["observed_losses"]=losses
        comparisons={}
        for section in ("gradient","parameter","momentum"):
            left,right=reference[section],observed[section]
            finite=True;passed=set(left)==set(right);max_abs=0.;error2=0.;reference2=0.;none_mismatches=[]
            for name,a in left.items():
                b=right.get(name)
                if a is None or b is None:
                    if not(a is None and b is None):none_mismatches.append(name);passed=False
                    continue
                if a.shape!=b.shape:passed=False;continue
                finite=finite and bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all())
                delta=(a-b).double()
                if a.numel():max_abs=max(max_abs,float(delta.abs().max()))
                error2+=float(delta.square().sum());reference2+=float(a.double().square().sum())
                passed=passed and bool(torch.allclose(a,b,rtol=1e-5,atol=1e-7))
            comparisons[section]={"within_declared_tolerance":bool(passed and finite),"finite":finite,"max_abs_error":max_abs,
                                  "relative_l2_error":(error2/reference2)**.5 if reference2 else (0. if error2==0 else None),
                                  "zero_reference_norm":reference2==0,"none_mismatches":none_mismatches}
        row["comparisons"]=comparisons
        row["status"]="completed"
        row["environment"]={"torch":torch.__version__,"cuda":torch.version.cuda,"gpu":torch.cuda.get_device_name(0)}
    except Exception as error:
        row.update({"failed_phase":phase,"error_type":type(error).__name__,"error_message":str(error)[:2000],"traceback":traceback.format_exc()[-5000:]})
    for key in ("error_message","traceback"):
        if key in row:row[key]=row[key].replace(str(ROOT),"<atlas>").replace(str(Path.home()),"<profile>")
    output.write_text(json.dumps(row,indent=2,sort_keys=True,allow_nan=False)+"\n",encoding="utf-8")


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument("--worker",choices=MODES);parser.add_argument("--output",type=Path);args=parser.parse_args()
    if args.worker:
        worker(args.worker,args.output);return
    source=ROOT/".tools/rockmate-review"
    review_path=ROOT/"docs/research/competitors/rockmate-execution-review.json"
    review=json.loads(review_path.read_text(encoding="utf-8"))
    for item in review["files"]:assert sha(source/item["path"])==item["sha256"]
    directory=ROOT/"artifacts/planner-reuse-control";directory.mkdir(parents=True,exist_ok=True)
    interpreter=ROOT/".tools/planner-env/Scripts/python.exe"
    env={key:os.environ[key] for key in ("SystemRoot","WINDIR") if key in os.environ}
    env.update({"PATH":str(interpreter.parent)+os.pathsep+str(Path(os.environ["SystemRoot"])/"System32"),
                "PYTHONPATH":str(source/"rockmate/src")+os.pathsep+str(source/"rkgb/src"),
                "PYTHONNOUSERSITE":"1","PYTHONDONTWRITEBYTECODE":"1","PYTHONHASHSEED":"0",
                "TEMP":str(directory),"TMP":str(directory),"TORCH_HOME":str(directory/"torch-cache"),
                "OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1","CUBLAS_WORKSPACE_CONFIG":":4096:8"})
    rows=[]
    for mode in MODES:
        target=directory/(mode+".json");started=time.perf_counter()
        with (directory/(mode+".stdout.txt")).open('w',encoding='utf-8') as stdout,(directory/(mode+".stderr.txt")).open('w',encoding='utf-8') as stderr:
            subprocess.run([str(interpreter),str(Path(__file__).resolve()),"--worker",mode,"--output",str(target)],cwd=directory,env=env,stdout=stdout,stderr=stderr,timeout=60,check=True)
        row=json.loads(target.read_text(encoding='utf-8'));row['outer_wall_seconds']=time.perf_counter()-started;rows.append(row)
        print(mode,row['status'],row.get('failed_phase',''),flush=True)
    result={"captured_utc":datetime.now(timezone.utc).isoformat(),"source_sha256":sha(Path(__file__)),"review_sha256":sha(review_path),"modes":MODES,"trials":rows,
            "scope":"Exploratory pure module-reuse/accumulation relation. Investigator cases, not independent labels, paper-environment replication, bug prevalence, scientific equivalence or Atlas policy evaluation."}
    (ROOT/'docs/research/experiments/planner-reuse-control.json').write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8')


if __name__=="__main__":
    main()
