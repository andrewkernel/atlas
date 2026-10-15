"""Read-only board telemetry availability; never changes GPU settings."""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--worker',action='store_true');parser.add_argument('--output',type=Path);args=parser.parse_args()
    if args.worker:
        import pynvml as nv
        result={'captured_utc':datetime.now(timezone.utc).isoformat(),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'python':platform.python_version(),'nvidia_ml_py':importlib.metadata.version('nvidia-ml-py'),
                'binding_sha256':hashlib.sha256(Path(nv.__file__).read_bytes()).hexdigest(),
                'scope':'Read-only uncontrolled board telemetry availability, not an idle calibration, workload energy measurement or per-process/whole-host energy estimate.'}
        initialized=False
        try:
            nv.nvmlInit();initialized=True
            device=nv.nvmlDeviceGetHandleByIndex(0)
            result['device_count']=nv.nvmlDeviceGetCount();result['gpu']=nv.nvmlDeviceGetName(device);result['driver']=nv.nvmlSystemGetDriverVersion()
            result['status']='initialized';samples=[];began=time.perf_counter_ns()
            for index in range(3):
                sample={'elapsed_ns':time.perf_counter_ns()-began}
                for name,getter in [('power_mw',lambda:nv.nvmlDeviceGetPowerUsage(device)),('energy_mj',lambda:nv.nvmlDeviceGetTotalEnergyConsumption(device))]:
                    try:sample[name]={'status':'available','value':getter()}
                    except nv.NVMLError as error:sample[name]={'status':'unavailable','error_type':type(error).__name__,'message':str(error),'value':None}
                samples.append(sample)
                if index<2:time.sleep(.25)
            result['samples']=samples
        except nv.NVMLError as error:
            result.update({'status':'unavailable','error_type':type(error).__name__,'message':str(error)})
        finally:
            if initialized:nv.nvmlShutdown()
        args.output.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8');return
    directory=ROOT/'artifacts/nvml-availability';directory.mkdir(parents=True,exist_ok=True)
    interpreter=ROOT/'.tools/telemetry-env/Scripts/python.exe'
    env={key:os.environ[key] for key in ('SystemRoot','WINDIR') if key in os.environ}
    env.update({'PATH':str(interpreter.parent)+os.pathsep+str(Path(os.environ['SystemRoot'])/'System32'),
                'PYTHONNOUSERSITE':'1','PYTHONDONTWRITEBYTECODE':'1','TEMP':str(directory),'TMP':str(directory)})
    output=ROOT/'docs/research/experiments/nvml-availability.json'
    with (directory/'stdout.txt').open('w',encoding='utf-8') as stdout,(directory/'stderr.txt').open('w',encoding='utf-8') as stderr:
        subprocess.run([str(interpreter),str(Path(__file__).resolve()),'--worker','--output',str(output)],cwd=directory,env=env,stdout=stdout,stderr=stderr,timeout=15,check=True)


if __name__=='__main__':main()
