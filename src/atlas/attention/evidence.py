"""Portable run identities and atomic, inspectable evidence."""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import time
import torch

ROOT = Path(__file__).resolve().parents[3]
RUNS = ROOT / "artifacts" / "attention-lab"


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def identities() -> dict:
    def git(*args):
        return subprocess.check_output(["git", *args],cwd=ROOT,stderr=subprocess.DEVNULL)
    try:
        head = git("rev-parse","HEAD").decode().strip()
        diff = sha(git("diff","HEAD","--binary"))
    except (OSError,subprocess.CalledProcessError):
        head, diff = None, None
    files = sorted((ROOT/"src"/"atlas"/"attention").glob("*.py"))
    files.append(ROOT/"lab.py")
    try:
        telemetry=subprocess.check_output(["nvidia-smi","--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu,temperature.gpu","--format=csv,noheader,nounits"],text=True,timeout=5).strip()
    except (OSError,subprocess.SubprocessError):telemetry=None
    return {
        "git_head":head, "dirty_diff_sha256":diff,
        "source_sha256":{str(p.relative_to(ROOT)).replace('\\','/'):sha(p.read_bytes()) for p in files},
        "packages":{p:importlib.metadata.version(p) for p in ("torch","transformers","numpy","fastapi","psutil")},
        "python":platform.python_version(),"platform":platform.system(),
        "gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_total_mib":torch.cuda.get_device_properties(0).total_memory/2**20 if torch.cuda.is_available() else None,
        "cuda":torch.version.cuda,"cpu":platform.processor(),"cpu_logical_count":os.cpu_count(),
        "tf32":torch.backends.cuda.matmul.allow_tf32,
        "gpu_telemetry_start_csv":telemetry,
    }


class Cancelled(Exception):
    pass


class Recorder:
    def __init__(self, directory: Path, config: dict):
        self.directory=directory
        self.started=time.perf_counter()
        self.doc={"schema_version":1,"id":directory.name,"created_at":utc(),"status":"running","config":config,
                  "environment":identities(),"rows":[],"progress":{"done":0,"total":1,"message":"Preparing"},
                  "limitations":["Single consumer PC with shared desktop GPU; no CXL hardware.",
                    "Float32 dense attention retains all entries; this is not a reproduction of sparse HGCA/BEYOND.",
                    "GPU allocation is a PyTorch allocator observation, not exclusive physical VRAM residency.",
                    "Finite numerical agreement does not establish downstream language quality."]}
        for relative in self.doc["environment"]["source_sha256"]:
            dest=directory/"source"/relative
            dest.parent.mkdir(parents=True,exist_ok=True)
            dest.write_bytes((ROOT/relative).read_bytes())
        self.save()

    def check(self):
        if (self.directory/"cancel").exists():
            raise Cancelled("Cancelled between bounded operations")

    def save(self):
        self.doc["elapsed_s"]=time.perf_counter()-self.started
        atomic_json(self.directory/"result.json",self.doc)

    def progress(self, done:int,total:int,message:str):
        self.check()
        self.doc["progress"]={"done":done,"total":total,"message":message}
        self.save()

    def finish(self,status="completed",error=None):
        self.doc.update(status=status,finished_at=utc())
        if error:self.doc["error"]=error
        self.save()
