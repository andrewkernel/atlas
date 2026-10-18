"""Prepare, freeze or run the separate long-context study with retained attempts."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from atlas.attention.long_protocol import DIRECTORY, PROTOCOL, ROOT, freeze, utc, verify
from atlas.attention.portable import file_digest, read_json, write_json


def retain_attempt(output, entry, directory):
    """Copy once, including incomplete raw bytes, without rewriting worker evidence."""
    record = read_json(directory/'result.json')
    destination = output/'runs'/entry['run_id']
    destination.mkdir(parents=True,exist_ok=True)
    for path in directory.rglob('*'):
        if path.is_file():
            target = destination/path.relative_to(directory)
            target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists() and file_digest(target)!=file_digest(path):
                raise ValueError('Previously retained attempt changed: '+str(target))
            if not target.exists():shutil.copyfile(path,target)
    status = record['status']
    entry.update(status=status if status not in ('queued','running') else 'interrupted',
                 raw_status=status, sha256=file_digest(destination/'result.json'),
                 elapsed_s=record.get('elapsed_s'), error=record.get('error'), finished_at=utc())
    if entry['status']=='interrupted':
        entry['disposition']='Worker identity confirmed absent; raw queued/running bytes retained. No numerical eligibility.'


def recover_pending(output, ledger, ledger_path, runs):
    import psutil
    for entry in ledger['cases']:
        if entry['status']!='running':continue
        try:
            process = psutil.Process(entry['pid'])
            if process.create_time()==entry['process_created_at']:
                raise RuntimeError('Registered worker is still alive; do not restart: '+entry['run_id'])
        except psutil.NoSuchProcess:
            pass
        retain_attempt(output,entry,runs/entry['run_id'])
        write_json(ledger_path,ledger)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare','freeze','preflight','run','analyze'])
    parser.add_argument('--replication-id')
    parser.add_argument('--case-id')
    args = parser.parse_args()
    if args.replication_id and (not args.replication_id.replace('-','').isalnum() or len(args.replication_id)>48):
        raise ValueError('Use a short letters/digits/hyphens replication ID')
    if args.action=='prepare':
        from atlas.attention.long_assets import prepare_books, restore_books
        if PROTOCOL.exists():restore_books(read_json(PROTOCOL)['assets']['books'])
        else:prepare_books()
        return
    if args.action=='freeze':
        print(freeze()['id'])
        return
    output = ROOT/'artifacts/attention-v3-replications'/args.replication_id if args.replication_id else DIRECTORY
    if args.action=='analyze':
        from verify_attention_v3 import build
        result = build(output, ROOT)
        write_json(output/'summary.json', result)
        print({'recorded':result['recorded_cases'], 'planned':result['planned_cases']})
        return
    protocol = verify()
    if args.action=='preflight':
        import torch, psutil
        if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no substituted measurements')
        print({'source_and_assets_verified':True, 'gpu':torch.cuda.get_device_name(0),
               'torch':torch.__version__, 'cuda':torch.version.cuda,
               'available_ram_gib':psutil.virtual_memory().available/2**30,
               'free_disk_gib':shutil.disk_usage(ROOT).free/2**30,
               'warning':'Different hardware/software is a new replication, not identical conditions. No hard residency guarantee.'})
        return
    if args.case_id and not any(c['case_id']==args.case_id for c in protocol['cases']):
        raise ValueError('Unregistered case')
    ledger_path = output/'execution.json'
    ledger = read_json(ledger_path) if ledger_path.exists() else {
        'protocol_sha256':file_digest(PROTOCOL), 'started_at':utc(), 'cases':[]}
    if ledger['protocol_sha256'] != file_digest(PROTOCOL):raise ValueError('Ledger belongs to a different protocol')
    output.mkdir(parents=True,exist_ok=True)
    if not (output/'protocol.json').exists():shutil.copyfile(PROTOCOL,output/'protocol.json')
    from atlas.attention.server import Jobs
    from atlas.attention.evidence import RUNS
    import psutil
    recover_pending(output,ledger,ledger_path,RUNS)
    supervisor = Jobs()
    supervisor.wall_timeout_s = 1200
    try:
        for case in protocol['cases']:
            if args.case_id and case['case_id']!=args.case_id:continue
            if any(r['case_id']==case['case_id'] for r in ledger['cases']):continue
            ident = supervisor.start(case)
            entry = {'case_id':case['case_id'], 'run_id':ident, 'status':'running', 'started_at':utc(),
                     'pid':supervisor.process.pid,
                     'process_created_at':psutil.Process(supervisor.process.pid).create_time()}
            # Save dispatch before waiting: interrupted controllers cannot hide attempts.
            ledger['cases'].append(entry)
            write_json(ledger_path,ledger)
            print(entry,flush=True)
            while supervisor.active:time.sleep(.5)
            directory = RUNS/ident
            retain_attempt(output,entry,directory)
            write_json(ledger_path,ledger)
            print(entry,flush=True)
        ledger['finished_at'] = utc()
        write_json(ledger_path,ledger)
    finally:
        supervisor.close()


if __name__=='__main__':main()
