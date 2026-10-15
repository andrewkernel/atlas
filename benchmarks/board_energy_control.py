"""Gross shared-board telemetry around unchanged, frozen CNN replays."""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
CONFIG={'seed':1101,'pairs':3,'order_seed':2101,'sample_interval_seconds':.05,
        'pre_seconds':2.,'post_seconds':2.,'child_timeout_seconds':300}


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def integrated_joules(samples):
    if len(samples)<2 or any(value['power_mw'] is None for value in samples):return None
    return sum((a['power_mw']+b['power_mw'])/2*(b['elapsed_ns']-a['elapsed_ns'])/1e12 for a,b in zip(samples,samples[1:]))


def child_environment(interpreter,directory):
    profile=directory/'private-home';profile.mkdir(parents=True,exist_ok=True)
    env={key:os.environ[key] for key in ('SystemRoot','WINDIR') if key in os.environ}
    env.update({'PATH':str(interpreter.parent)+os.pathsep+str(Path(os.environ['SystemRoot'])/'System32'),
                'PYTHONNOUSERSITE':'1','PYTHONDONTWRITEBYTECODE':'1','PYTHONHASHSEED':'0',
                'USERPROFILE':str(profile),'TEMP':str(directory),'TMP':str(directory),
                'TORCH_HOME':str(directory/'torch-cache'),'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
                'CUBLAS_WORKSPACE_CONFIG':':4096:8'})
    return env


def supervise(output):
    import pynvml as nv
    from planner_reuse_control import portable
    fixture=ROOT/'benchmarks/fashion_pilot.py'
    original_path=ROOT/'docs/research/experiments/fashion-pilot.json'
    original=json.loads(original_path.read_text(encoding='utf-8'))
    assert sha(fixture)==original['source_sha256']
    availability=json.loads((ROOT/'docs/research/experiments/nvml-availability.json').read_text(encoding='utf-8'))
    assert sha(Path(nv.__file__))==availability['binding_sha256']
    directory=ROOT/'artifacts/board-energy-control'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ');directory.mkdir(parents=True)
    interpreter=ROOT/'.tools/cuda-env/Scripts/python.exe'
    order=[];rng=random.Random(CONFIG['order_seed'])
    for pair in range(CONFIG['pairs']):
        arms=[False,True];rng.shuffle(arms);order.extend((pair,arm) for arm in arms)
    result={'started_utc':datetime.now(timezone.utc).isoformat(),'source_sha256':sha(Path(__file__)),
            'fixture_source_sha256':sha(fixture),'reference_canonical_json_sha256':hashlib.sha256(json.dumps(original,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
            'binding_sha256':sha(Path(nv.__file__)),'config':CONFIG,'order':[{'pair':p,'candidate':c} for p,c in order],
            'trials':[],'constant_power_unit_control_j':integrated_joules([{'elapsed_ns':0,'power_mw':100000},{'elapsed_ns':2000000000,'power_mw':100000}]),
            'scope':'Gross shared GPU-board energy over 2s pre + complete child workflow + 2s post. Includes unrelated device activity; not per-process, whole-host, causal energy savings, independent training seeds or a policy comparison.'}
    nv.nvmlInit()
    try:
        device=nv.nvmlDeviceGetHandleByIndex(0)
        result['hardware']={'gpu':nv.nvmlDeviceGetName(device),'driver':nv.nvmlSystemGetDriverVersion(),'device_count':nv.nvmlDeviceGetCount()}
        for index,(pair,candidate) in enumerate(order):
            trial_directory=directory/str(index);trial_directory.mkdir()
            target=trial_directory/'training.json';samples=[];began=time.perf_counter_ns()
            def sample(phase):
                before=time.perf_counter_ns();row={'phase':phase,'power_mw':None,'energy_mj':None,'gpu_utilization_percent':None,'temperature_c':None,'errors':[]}
                for name,getter in [('power_mw',lambda:nv.nvmlDeviceGetPowerUsage(device)),('energy_mj',lambda:nv.nvmlDeviceGetTotalEnergyConsumption(device)),
                                    ('gpu_utilization_percent',lambda:nv.nvmlDeviceGetUtilizationRates(device).gpu),('temperature_c',lambda:nv.nvmlDeviceGetTemperature(device,nv.NVML_TEMPERATURE_GPU))]:
                    try:row[name]=getter()
                    except nv.NVMLError as error:row['errors'].append({'field':name,'error_type':type(error).__name__,'message':str(error)})
                after=time.perf_counter_ns();row['elapsed_ns']=(before+after)//2-began;row['query_span_ns']=after-before;samples.append(row)
            pre_end=time.perf_counter()+CONFIG['pre_seconds']
            while time.perf_counter()<pre_end:sample('pre');time.sleep(CONFIG['sample_interval_seconds'])
            command=[str(interpreter),str(fixture),'--child','--seed',str(CONFIG['seed']),'--data-root',str(ROOT/'data/fashion-mnist'),
                     '--manifest',str(ROOT/'docs/research/experiments/fashion-data-manifest.json'),'--output',str(target)]
            if candidate:command.append('--candidate')
            child_start=time.perf_counter();timed_out=False
            with (trial_directory/'stdout.txt').open('w',encoding='utf-8') as stdout,(trial_directory/'stderr.txt').open('w',encoding='utf-8') as stderr:
                process=subprocess.Popen(command,cwd=ROOT,env=child_environment(interpreter,trial_directory),stdout=stdout,stderr=stderr)
                while process.poll() is None:
                    sample('child')
                    if time.perf_counter()-child_start>CONFIG['child_timeout_seconds']:
                        process.kill();process.wait();timed_out=True;break
                    time.sleep(CONFIG['sample_interval_seconds'])
            child_wall=time.perf_counter()-child_start
            post_end=time.perf_counter()+CONFIG['post_seconds']
            while time.perf_counter()<post_end:sample('post');time.sleep(CONFIG['sample_interval_seconds'])
            sample('post_endpoint')
            observed=json.loads(target.read_text(encoding='utf-8')) if target.exists() else {'status':'missing_result'}
            previous=next(t for t in original['trials'] if t['seed']==CONFIG['seed'] and t['candidate']==candidate)
            identities={key:observed.get(key)==previous.get(key) for key in ('initial_state_digest','split_digest','sample_order_digest','final_state_digest')}
            metrics_equal=observed.get('status')=='completed' and all(observed['history'][epoch][key]==previous['history'][epoch][key]
                for epoch in range(5) for key in ('validation_nll','validation_accuracy','validation_correct','successful_updates'))
            counters=[s['energy_mj'] for s in samples]
            monotonic=all(value is not None for value in counters) and all(a<=b for a,b in zip(counters,counters[1:]))
            counter_j=(counters[-1]-counters[0])/1000 if monotonic else None
            gaps=[(b['elapsed_ns']-a['elapsed_ns'])/1e9 for a,b in zip(samples,samples[1:])]
            trial={'pair':pair,'candidate':candidate,'status':'timeout' if timed_out else observed.get('status'),'returncode':process.returncode,
                   'child_wall_seconds':child_wall,'window_seconds':(samples[-1]['elapsed_ns']-samples[0]['elapsed_ns'])/1e9,
                   'counter_monotonic_and_complete':monotonic,'gross_counter_j':counter_j,'integrated_reported_power_j':integrated_joules(samples),
                   'max_sample_gap_seconds':max(gaps),'sample_count':len(samples),'samples':samples,
                   'identities_equal_to_original':identities,'all_epoch_validation_values_equal':metrics_equal,
                   'replayed_result':observed,'stderr_excerpt':portable((trial_directory/'stderr.txt').read_text(encoding='utf-8')[-2000:])}
            result['trials'].append(trial)
            output.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8')
            print(json.dumps({'pair':pair,'candidate':candidate,'status':trial['status'],'gross_counter_j':counter_j,'metrics_match':metrics_equal}),flush=True)
    finally:nv.nvmlShutdown()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--supervise',action='store_true');parser.add_argument('--output',type=Path);args=parser.parse_args()
    if args.supervise:supervise(args.output);return
    directory=ROOT/'artifacts/board-energy-supervisor';directory.mkdir(parents=True,exist_ok=True)
    interpreter=ROOT/'.tools/telemetry-env/Scripts/python.exe'
    target=ROOT/'docs/research/experiments/board-energy-control.json'
    subprocess.run([str(interpreter),str(Path(__file__).resolve()),'--supervise','--output',str(target)],cwd=ROOT,
                   env=child_environment(interpreter,directory),timeout=2000,check=True)


if __name__=='__main__':main()
