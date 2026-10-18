"""Project immutable reference evidence into the app's normal read-only history.

No model is executed and no measurement is invented or rewritten. Identical
hardlinks avoid duplicating large arrays; copy fallback supports other filesystems.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from atlas.attention.portable import artifact_path,file_digest,read_json,write_json


def prepare(root=ROOT):
    study=root/'docs/research/attention-v3'
    runtime=root/'artifacts/attention-lab';runtime.mkdir(parents=True,exist_ok=True)
    index={'origin':'Recorded reference study; these are not fresh executions on this workspace',
           'runs':[],'unavailable_runs':[]}
    if not (study/'execution.json').exists():
        write_json(root/'lab-ui/data/reference-history.json',index);return index
    ledger=read_json(study/'execution.json')
    protocol=read_json(study/'protocol.json')
    if ledger['protocol_sha256']!=file_digest(study/'protocol.json'):
        raise ValueError('Reference history protocol identity mismatch')
    registered={c['case_id']:c for c in protocol['cases']}
    index['protocol_sha256']=ledger['protocol_sha256']
    for entry in ledger['cases']:
        if entry['case_id'] not in registered:raise ValueError('Unregistered reference case')
        ident=entry['run_id']
        if not ident or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-' for c in ident):
            raise ValueError('Unsafe reference run ID')
        source=study/'runs'/ident
        result=source/'result.json'
        if entry['status'] in ('starting','running') or not result.is_file():
            index['unavailable_runs'].append(ident);continue
        if file_digest(result)!=entry['sha256']:raise ValueError('Reference raw record changed: '+ident)
        record=read_json(result)
        if record['id']!=ident:raise ValueError('Reference raw run ID mismatch')
        if record['config']!=registered[entry['case_id']]:raise ValueError('Reference configuration changed')
        complete=True
        for item in record.get('artifacts',{}).values():
            path=artifact_path(source,item['path'])
            if not path.is_file():complete=False;break
            if file_digest(path)!=item['sha256']:raise ValueError('Reference array changed: '+ident)
        if not complete:
            index['unavailable_runs'].append(ident);continue
        target=runtime/ident;target.mkdir(parents=True,exist_ok=True)
        for path in source.rglob('*'):
            if not path.is_file():continue
            relative=path.relative_to(source).as_posix()
            actual_source=artifact_path(source,relative)
            destination=artifact_path(target,relative)
            if destination.exists():
                if not os.path.samefile(actual_source,destination) and file_digest(actual_source)!=file_digest(destination):
                    raise ValueError('Reference history conflicts with an existing local run: '+ident)
                continue
            destination.parent.mkdir(parents=True,exist_ok=True)
            try:os.link(actual_source,destination)
            except OSError:shutil.copy2(actual_source,destination)
        index['runs'].append(ident)
    write_json(root/'lab-ui/data/reference-history.json',index)
    return index


if __name__=='__main__':
    index=prepare()
    print(json.dumps({'reference_records':len(index['runs']),'unavailable_records':len(index['unavailable_runs']),
                      'scope':'Imported recorded evidence only; no model execution'}))
