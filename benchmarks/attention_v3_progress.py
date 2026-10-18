"""Verify a terminal-case snapshot without altering the live execution ledger.

This publishes explicitly partial evidence. Running/unattempted cases remain
missing from verification and visible in the UI. No inputs or bounds change.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
import uuid

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from atlas.attention.portable import file_digest,read_json,write_json
from verify_attention_v3 import build

ROOT=Path(__file__).resolve().parents[1]
STUDY=ROOT/'docs/research/attention-v3'


def main():
    ledger=read_json(STUDY/'execution.json')
    terminal=[e for e in ledger['cases'] if e['status']!='running']
    if not terminal:raise ValueError('No terminal outcomes to verify')
    snapshot=ROOT/'artifacts/attention-v3-audits'/('partial-'+uuid.uuid4().hex[:12])
    snapshot.mkdir(parents=True)
    shutil.copyfile(STUDY/'protocol.json',snapshot/'protocol.json')
    write_json(snapshot/'execution.json',{**ledger,'cases':terminal})
    for entry in terminal:
        source=STUDY/'runs'/entry['run_id']
        for path in source.rglob('*'):
            if path.is_file():
                target=snapshot/'runs'/entry['run_id']/path.relative_to(source)
                target.parent.mkdir(parents=True,exist_ok=True)
                # Hard links avoid duplicating GB of read-only arrays on this disk.
                # This verifier never modifies linked raw/source/array artifacts.
                os.link(path,target)
    result=build(snapshot,ROOT)
    result['progress_audit']={'created_at':datetime.now(timezone.utc).isoformat(),
                             'scope':'Only terminal outcomes in the captured ledger; running and unattempted cases are unverified, never counted as passes.',
                             'snapshot_ledger_sha256':file_digest(snapshot/'execution.json'),
                             'analysis_script_sha256':file_digest(Path(__file__)),
                             'interference':'Read-only analysis should be scheduled outside free-generation timing windows; shared desktop activity remains a limitation.'}
    write_json(snapshot/'summary.json',result)
    write_json(STUDY/'summary.json',result)
    print(json.dumps({'verified_cases':result['recorded_cases'],'planned':result['planned_cases'],
                      'snapshot':snapshot.relative_to(ROOT).as_posix(),'comparisons':result['comparisons']},indent=2))


if __name__=='__main__':main()
