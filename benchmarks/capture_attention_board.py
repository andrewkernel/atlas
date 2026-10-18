"""Standalone shared-board observation. No process residency/energy attribution."""
import json
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from atlas.attention.evidence import ROOT,RUNS,atomic_json,utc,sha
from atlas.attention.protocol import DIRECTORY,EXECUTION_PATH,PROTOCOL_PATH


def main():
    path=DIRECTORY/'board-observations.json'
    if path.exists():raise ValueError('Preserve existing board trace')
    doc={'started_at':utc(),'scope':'Shared board, external observer started after in-worker NVML unavailability diagnosed. Earlier absent samples remain unknown. No per-process residency or energy claim.',
         'source_sha256':sha(Path(__file__).read_bytes()),'observer':'Official nvidia-smi, inherited normal environment; 1s target spacing','samples':[]}
    protocol=json.loads(PROTOCOL_PATH.read_text())
    while True:
        began=time.perf_counter();now=utc()
        recent=max(RUNS.glob('research-*/result.json'),key=lambda p:p.stat().st_mtime)
        try:run=json.loads(recent.read_text())
        except (OSError,json.JSONDecodeError):time.sleep(.2);continue
        row={'at':now,'run_id':run['id'],'case_id':run['config'].get('case_id'),'phase':run.get('progress',{}).get('message'),'execution_status':run['status']}
        try:
            values=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu,temperature.gpu','--format=csv,noheader,nounits'],text=True,timeout=3,creationflags=0x08000000).strip().split(',')
            row.update(board_used_mib=float(values[0]),utilization_percent=float(values[1]),temperature_c=float(values[2]))
        except (OSError,ValueError,subprocess.SubprocessError):row['device_observation']='unavailable'
        row['observer_wall_ms']=(time.perf_counter()-began)*1000;doc['samples'].append(row);atomic_json(path,doc)
        ledger=json.loads(EXECUTION_PATH.read_text())
        if len(ledger['cases'])==len(protocol['cases']):break
        time.sleep(max(0,1-(time.perf_counter()-began)))
    doc['finished_at']=utc();atomic_json(path,doc)
    print(json.dumps({'trace':str(path),'samples':len(doc['samples'])}))


if __name__=='__main__':main()
