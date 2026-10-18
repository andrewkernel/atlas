"""Finish the registered study handoff after its owned controller terminates.

Waits only for an explicit PID+creation identity; never restarts or kills a run.
Incomplete ledgers fail closed. This is local packaging, not publication.
"""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
from pathlib import Path,PurePosixPath
import subprocess
import sys
import time
import uuid
import zipfile

import psutil

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from atlas.attention.portable import file_digest,read_json,write_json

ROOT=Path(__file__).resolve().parents[1]
STUDY=ROOT/'docs/research/attention-v3'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--driver-pid',type=int)
    parser.add_argument('--driver-created-at',type=float)
    args=parser.parse_args()
    if bool(args.driver_pid)!=(args.driver_created_at is not None):
        raise ValueError('A waiting controller needs both PID and creation time')
    began=time.monotonic();last=None
    while args.driver_pid:
        try:
            process=psutil.Process(args.driver_pid)
            alive=process.create_time()==args.driver_created_at and process.is_running()
        except psutil.NoSuchProcess:alive=False
        ledger=read_json(STUDY/'execution.json')
        terminal=sum(e['status']!='running' for e in ledger['cases'])
        if terminal!=last:
            print({'terminal_cases':terminal,'controller_live':alive},flush=True);last=terminal
        if not alive:break
        if time.monotonic()-began>7200:raise TimeoutError('Controller observation limit; no run was restarted or killed')
        time.sleep(1)
    protocol=read_json(STUDY/'protocol.json');ledger=read_json(STUDY/'execution.json')
    if (not ledger.get('finished_at') or {e['case_id'] for e in ledger['cases']}!={c['case_id'] for c in protocol['cases']}
            or any(e['status']=='running' for e in ledger['cases'])):
        raise ValueError('Study is incomplete; retain its outcomes and investigate before handoff')
    for script in ('report_attention_v3.py','package_attention_v3.py'):
        subprocess.run([sys.executable,str(ROOT/'benchmarks'/script)],cwd=ROOT,check=True)
    receipt=read_json(STUDY/'bundle-receipt.json')
    archive=ROOT/'lab-ui/downloads'/receipt['filename']
    if file_digest(archive)!=receipt['sha256']:raise ValueError('Archive changed before extracted-kit check')
    destination=ROOT/'artifacts/attention-v3-handoff'/('extract-'+uuid.uuid4().hex[:12])
    destination.mkdir(parents=True)
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            name=PurePosixPath(member.filename)
            if name.is_absolute() or '..' in name.parts or '\\' in member.filename or ':' in member.filename:
                raise ValueError('Unsafe archive member')
            if not (destination/name).resolve().is_relative_to(destination.resolve()):raise ValueError('Archive path escape')
        bundle.extractall(destination)
    extracted=destination/'atlas-researcher-kit'
    subprocess.run([sys.executable,str(extracted/'benchmarks/verify_researcher_bundle.py'),str(extracted)],cwd=extracted,check=True)
    subprocess.run([sys.executable,str(extracted/'benchmarks/verify_attention_v3.py'),
                    '--study',str(extracted/'docs/research/attention-v3'),'--source-root',str(extracted)],cwd=extracted,check=True)
    verified={'created_at':datetime.now(timezone.utc).isoformat(), 'archive_sha256':receipt['sha256'],
              'extracted_directory':extracted.relative_to(ROOT).as_posix(), 'shipped_bytes_verified':True,
              'full_array_verification_from_extracted_source':True,
              'cold_environment_install':False, 'fresh_model_execution':False,
              'external_reproduction':False,
              'scope':'Local extraction and independent arithmetic using the existing pinned Python environment. Fresh execution and browser flow remain separate gates.'}
    write_json(STUDY/'handoff-verification.json',verified)
    print(verified,flush=True)


if __name__=='__main__':main()
