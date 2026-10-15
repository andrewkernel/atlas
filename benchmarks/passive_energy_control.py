"""Read-only no-owned-training telemetry consistency diagnostic."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time
from board_energy_control import ROOT,child_environment
D=ROOT/'docs/research/experiments'
def main():
    p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true');a=p.parse_args()
    if not a.worker:
        directory=ROOT/'artifacts/passive-energy-control';directory.mkdir(parents=True,exist_ok=True)
        interpreter=ROOT/'.tools/telemetry-env/Scripts/python.exe'
        subprocess.run([str(interpreter),str(Path(__file__).resolve()),'--worker'],cwd=ROOT,env=child_environment(interpreter,directory),timeout=40,check=True);return
    import pynvml as nv
    expected=json.loads((D/'nvml-availability.json').read_text(encoding='utf-8'))['binding_sha256']
    assert hashlib.sha256(Path(nv.__file__).read_bytes()).hexdigest()==expected
    nv.nvmlInit()
    try:
        device=nv.nvmlDeviceGetHandleByIndex(0);began=time.perf_counter_ns();rows=[]
        while time.perf_counter_ns()-began<20*10**9:
            before=time.perf_counter_ns();power=nv.nvmlDeviceGetPowerUsage(device);energy=nv.nvmlDeviceGetTotalEnergyConsumption(device);after=time.perf_counter_ns()
            rows.append({'elapsed_ns':(before+after)//2-began,'query_span_ns':after-before,'power_mw':power,'energy_mj':energy})
            time.sleep(.2)
        duration=(rows[-1]['elapsed_ns']-rows[0]['elapsed_ns'])/1e9
        assert all(a['energy_mj']<=b['energy_mj'] for a,b in zip(rows,rows[1:]))
        counter=(rows[-1]['energy_mj']-rows[0]['energy_mj'])/1000
        integral=sum((a['power_mw']+b['power_mw'])*(b['elapsed_ns']-a['elapsed_ns'])/(2*10**12) for a,b in zip(rows,rows[1:]))
        result={'captured_utc':datetime.now(timezone.utc).isoformat(),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'helper_source_sha256':hashlib.sha256((ROOT/'benchmarks/board_energy_control.py').read_bytes()).hexdigest(),'binding_sha256':expected,
            'duration_seconds':duration,'gross_counter_j':counter,'integrated_reported_power_j':integral,'integral_excess_percent':100*(integral/counter-1),
            'counter_mean_w':counter/duration,'reported_power_mean_w':integral/duration,'samples':rows,
            'scope':'20-second passive read without an owned training child. Background board use is uncontrolled; neither API is a calibration truth. Tests only whether workload-window disagreement also appears here.'}
        (D/'passive-energy-control.json').write_bytes((json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n').encode())
        print(json.dumps({k:v for k,v in result.items() if k!='samples'}))
    finally:nv.nvmlShutdown()
if __name__=='__main__':main()
