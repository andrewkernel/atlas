"""Loopback-only experiment service. Fixed trusted workers; no arbitrary code."""
from __future__ import annotations
import atexit
import csv
import io
import json
import os
from pathlib import Path
import secrets
import signal
import subprocess
import sys
import threading
import time
import uuid
from urllib.parse import urlsplit
from fastapi import FastAPI,HTTPException,Request
from fastapi.responses import FileResponse,JSONResponse,Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel,ConfigDict,Field
from typing import Literal
from .evidence import ROOT,RUNS,atomic_json,utc

TOKEN=secrets.token_urlsafe(32)
app=FastAPI(title="Atlas Attention Lab",docs_url=None,redoc_url=None,openapi_url=None)


class Experiment(BaseModel):
    model_config=ConfigDict(extra="forbid")
    kind:Literal["benchmark","model"]="benchmark"
    profile:Literal["quick","full"]="full"
    threads:Literal[1,2,4,6]=4
    repeats:int=Field(default=9,ge=5,le=30)
    heads:Literal[12]=12
    head_dim:Literal[64]=64
    seed:int=Field(default=2026,ge=0,le=2**31-1)
    kv_budget_mib:float=Field(default=2048,ge=0,le=4096,allow_inf_nan=False)
    new_tokens:int=Field(default=24,ge=4,le=64)
    prompt:str=Field(default="Memory systems connect processors to the data they need. A useful experiment measures both the cost of moving data and the time spent computing with it. The results suggest that",min_length=1,max_length=1800)


def result_path(ident):
    if not ident or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-" for c in ident):
        raise HTTPException(404,"Unknown run")
    path=RUNS/ident/"result.json"
    if not path.exists():raise HTTPException(404,"Unknown run")
    return path


class Jobs:
    wall_timeout_s=600
    cancel_grace_s=5
    def __init__(self):
        self.lock=threading.Lock()
        self.process=None
        self.scope=None
        self.active=None
        self.cancel_at=None

    def start(self,config):
        with self.lock:
            if self.active:raise HTTPException(409,"An experiment is already running")
            ident=config["kind"]+"-"+uuid.uuid4().hex[:12]
            directory=RUNS/ident
            directory.mkdir(parents=True)
            atomic_json(directory/"config.json",config)
            atomic_json(directory/"result.json",{"schema_version":1,"id":ident,"created_at":utc(),"status":"queued","config":config,"rows":[],"progress":{"done":0,"total":1,"message":"Starting owned worker"}})
            env={k:v for k,v in os.environ.items() if k.upper() in {"PATH","SYSTEMROOT","WINDIR","TEMP","TMP","USERPROFILE","LOCALAPPDATA","APPDATA","COMSPEC"}}
            env.update(HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",TOKENIZERS_PARALLELISM="false",PYTHONUNBUFFERED="1")
            log=open(directory/"worker.log","wb")
            process=None;scope=None
            try:
                flags=0x08000000|0x00000004 if os.name=="nt" else 0
                process=subprocess.Popen([sys.executable,str(ROOT/"lab.py"),"worker","--run-id",ident],cwd=ROOT,env=env,
                    stdout=log,stderr=subprocess.STDOUT,creationflags=flags,start_new_session=os.name!="nt")
                scope=None
                if os.name=="nt":
                    import psutil
                    sys.path.insert(0,str(ROOT/"benchmarks"))
                    from windows_job_scope import WindowsJobScope
                    scope=WindowsJobScope()
                    owned=psutil.Process(process.pid)
                    try:
                        scope.assign_owned(process.pid,owned.create_time())
                        scope.resume_owned_single_thread(owned)
                    except BaseException:
                        scope.terminate();scope.close();process.kill();process.wait(timeout=5)
                        raise
                self.process,self.scope,self.active,self.cancel_at=process,scope,ident,None
            except Exception as exc:
                if process and process.poll() is None:
                    process.kill();process.wait(timeout=5)
                if scope:scope.close()
                doc=json.loads((directory/"result.json").read_text())
                doc.update(status="failed",finished_at=utc(),error=f"Worker startup failed: {type(exc).__name__}")
                atomic_json(directory/"result.json",doc)
                raise HTTPException(500,"Worker startup failed; see saved run") from exc
            finally:log.close()
            threading.Thread(target=self.watch,args=(ident,process,scope),daemon=True).start()
            return ident

    def terminate(self,process,scope):
        if process.poll() is None:
            if scope:scope.terminate()
            elif os.name!="nt":os.killpg(process.pid,signal.SIGKILL)
            else:process.kill()

    def watch(self,ident,process,scope):
        began=time.monotonic()
        forced=None
        try:
            while process.poll() is None:
                if time.monotonic()-began>self.wall_timeout_s:
                    forced="timed_out";self.terminate(process,scope);break
                if self.cancel_at and time.monotonic()-self.cancel_at>self.cancel_grace_s:
                    forced="cancelled";self.terminate(process,scope);break
                time.sleep(.2)
            process.wait(timeout=10)
            path=RUNS/ident/"result.json"
            doc=json.loads(path.read_text())
            if forced or doc["status"] in ("queued","running"):
                doc.update(status=forced or "failed",finished_at=utc(),error=forced or f"Worker exited {process.returncode} before writing completion")
                atomic_json(path,doc)
        finally:
            if scope:scope.close()
            with self.lock:
                self.process,self.scope,self.active,self.cancel_at=None,None,None,None

    def cancel(self,ident):
        with self.lock:
            if ident!=self.active:raise HTTPException(409,"This run is not active")
            (RUNS/ident/"cancel").touch()
            self.cancel_at=time.monotonic()

    def close(self):
        with self.lock:
            if self.process:self.terminate(self.process,self.scope)


jobs=Jobs()
atexit.register(jobs.close)


@app.middleware("http")
async def local_boundary(request:Request,call_next):
    host=request.headers.get("host","")
    try:hostname=urlsplit("http://"+host).hostname
    except ValueError:hostname=None
    if hostname not in ("127.0.0.1","localhost","testserver"):
        return JSONResponse({"detail":"Local host required"},status_code=403)
    if request.method not in ("GET","HEAD","OPTIONS"):
        origin=request.headers.get("origin")
        if origin and origin not in ("http://"+host,"https://"+host):
            return JSONResponse({"detail":"Same origin required"},status_code=403)
        if not secrets.compare_digest(request.headers.get("x-atlas-token",""),TOKEN):
            return JSONResponse({"detail":"Session token required"},status_code=403)
    response=await call_next(request)
    response.headers["Cache-Control"]="no-store"
    response.headers["X-Content-Type-Options"]="nosniff"
    response.headers["Content-Security-Policy"]="default-src 'self'; script-src 'self'; style-src 'self'; font-src 'self'; img-src 'self' data:; connect-src 'self'; object-src 'none'; frame-ancestors 'none'; base-uri 'none'"
    return response


@app.get("/api/session")
def session():
    return {"token":TOKEN,"active_run":jobs.active,"model_ready":(ROOT/"weights/gpt2-attention-lab/atlas-manifest.json").exists()}


@app.get("/api/runs")
def list_runs():
    records=[]
    for p in RUNS.glob("*/result.json"):
        try:
            doc=json.loads(p.read_text())
            records.append({k:doc.get(k) for k in ("id","created_at","status","config","progress","elapsed_s","error")})
        except (OSError,json.JSONDecodeError):continue
    return sorted(records,key=lambda r:r.get("created_at", ""),reverse=True)


@app.get("/api/runs/{ident}")
def get_run(ident:str):return json.loads(result_path(ident).read_text())


@app.post("/api/runs",status_code=202)
def start_run(experiment:Experiment):
    if experiment.kind=="model" and not (ROOT/"weights/gpt2-attention-lab/atlas-manifest.json").exists():
        raise HTTPException(409,"Prepare the pinned model first: python lab.py prepare-model")
    return {"id":jobs.start(experiment.model_dump())}


@app.post("/api/runs/{ident}/cancel")
def cancel_run(ident:str):jobs.cancel(ident);return {"status":"cancellation_requested"}


@app.get("/api/runs/{ident}/export")
def export(ident:str,format:Literal["json","csv"]="json"):
    path=result_path(ident)
    if format=="json":return FileResponse(path,media_type="application/json",filename=ident+".json")
    doc=json.loads(path.read_text())
    out=io.StringIO()
    fields=["phase","context","plan","median_ms","p90_ms","gpu_kv_mib","cpu_kv_mib","allocated_peak_mib"]
    writer=csv.DictWriter(out,fieldnames=fields,extrasaction="ignore")
    writer.writeheader();writer.writerows(doc.get("rows",[]))
    return Response(out.getvalue(),media_type="text/csv",headers={"Content-Disposition":f'attachment; filename="{ident}.csv"'})


@app.get("/api/architecture")
def architecture():return {"markdown":(ROOT/"docs/architecture/hybrid-attention-lab.md").read_text(encoding="utf-8")}


@app.get("/papers/{ident}")
def paper(ident:str):
    files={"hgca":"hgca-2507.03153.pdf","beyond":"beyond-hpdc2026.pdf","nomad":"nomad-osdi24.pdf","miku":"miku-2503.17864v2.pdf","dsa":"dsa-2lm-atc25.pdf"}
    if ident not in files:raise HTTPException(404)
    path=ROOT/"artifacts/cxl-papers"/files[ident]
    if not path.exists():raise HTTPException(404,"Paper not downloaded on this machine")
    return FileResponse(path,media_type="application/pdf")


app.mount("/archive",StaticFiles(directory=ROOT/"frontend",html=True),name="archive")
app.mount("/assets",StaticFiles(directory=ROOT/"frontend/assets"),name="assets")
app.mount("/",StaticFiles(directory=ROOT/"lab-ui",html=True),name="dashboard")
