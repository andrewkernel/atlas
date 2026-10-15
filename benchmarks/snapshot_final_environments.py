"""Read-only selected environment identity checks without importing ML packages."""
import argparse
import base64
from datetime import datetime,timezone
import hashlib
import importlib.metadata as meta
import json
from pathlib import Path
import platform
import re
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'docs/research/experiments'
PROFILES={
 'core':('.venv/Scripts/python.exe',None,'uv.lock'),
 'original_cuda':('.tools/cuda-env/Scripts/python.exe','cuda-probe-environment.json','configs/cuda-probe-windows-py312-requirements.txt'),
 'checker':('.tools/checker-env/Scripts/python.exe','checker-environment.json','configs/checker-windows-py311-requirements.txt'),
 'planner':('.tools/planner-env/Scripts/python.exe','planner-environment.json','configs/planner-windows-py312-requirements.txt'),
 'interop':('.tools/interop-env/Scripts/python.exe','interop-environment.json','configs/interop-windows-py311-requirements.txt'),
 'telemetry':('.tools/telemetry-env/Scripts/python.exe',None,'configs/telemetry-windows-py312-requirements.txt')}
SELECTED={'torch':('torch/utils/checkpoint.py','torch/optim/sgd.py','torch/optim/adamw.py','torch/nn/functional.py','torch/_dynamo/eval_frame.py','torch/fx/graph.py'),'nvidia-ml-py':('pynvml.py',)}
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def normalized(name):return re.sub(r'[-_.]+','-',name).lower()
def worker():
    packages={normalized(d.metadata['Name']):d.version for d in meta.distributions()}
    files=[]
    for name,selected in SELECTED.items():
        if name not in packages:continue
        dist=meta.distribution(name);entries={str(f).replace('\\','/'):f for f in dist.files or []}
        for relative in selected:
            entry=entries.get(relative)
            if entry is None:files.append({'distribution':name,'path':relative,'status':'missing_record_entry'});continue
            raw=dist.locate_file(entry).read_bytes();digest=hashlib.sha256(raw).digest()
            check=entry.hash is not None and entry.hash.mode=='sha256' and base64.urlsafe_b64encode(digest).decode().rstrip('=')==entry.hash.value
            files.append({'distribution':name,'path':relative,'sha256':digest.hex(),'matches_installed_record':check,'status':'read'})
    print(json.dumps({'python':platform.python_version(),'packages':packages,'selected_files':files},sort_keys=True,allow_nan=False))
def main():
    p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true');a=p.parse_args()
    if a.worker:worker();return
    result={'captured_utc':datetime.now(timezone.utc).isoformat(),'source_sha256':sha(Path(__file__)),'profiles':{},
            'scope':'Post-experiment distribution identity and selected Python-source consistency checks. Installed wheel RECORD is not a trusted signature, whole-package integrity proof, contemporaneous attestation or scientific-correctness validation. No ML package is imported by this probe.'}
    from board_energy_control import child_environment
    for name,(executable,prior,requirements) in PROFILES.items():
        interpreter=ROOT/executable;scratch=ROOT/'artifacts/final-environment-audit'/name;scratch.mkdir(parents=True,exist_ok=True)
        process=subprocess.run([str(interpreter),str(Path(__file__).resolve()),'--worker'],env=child_environment(interpreter,scratch),cwd=ROOT,capture_output=True,text=True,timeout=30,check=True)
        row=json.loads(process.stdout);row['requirements_file']=requirements;row['requirements_sha256']=sha(ROOT/requirements)
        if prior:
            earlier=json.loads((D/prior).read_text(encoding='utf-8'));old=earlier['packages']
            expected={normalized(p['name']):p['version'] for p in old} if old and isinstance(old[0],dict) else {normalized(k):v for k,v in (x.split('==',1) for x in old)}
            row['earlier_record']=prior;row['earlier_record_canonical_json_sha256']=hashlib.sha256(json.dumps(earlier,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
            row['earlier_package_version_mismatches']={key:{'earlier':version,'current':row['packages'].get(key)} for key,version in expected.items() if row['packages'].get(key)!=version}
            row['additional_distributions_since_earlier']=sorted(set(row['packages'])-set(expected))
        result['profiles'][name]=row
    result['environment_helper_sha256']=sha(ROOT/'benchmarks/board_energy_control.py')
    (D/'final-environment-inventory.json').write_bytes((json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n').encode())
    print(json.dumps({name:{'python':row['python'],'distributions':len(row['packages']),'selected_files':len(row['selected_files']),'record_mismatches':sum(not f.get('matches_installed_record',False) for f in row['selected_files']),'earlier_version_mismatches':row.get('earlier_package_version_mismatches')} for name,row in result['profiles'].items()},indent=2))
if __name__=='__main__':main()
