"""Freeze or execute all registered cases through the owned worker supervisor."""
import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from atlas.attention.protocol import DIRECTORY,PROTOCOL_PATH,EXECUTION_PATH,freeze
from atlas.attention.evidence import ROOT,RUNS,atomic_json,sha,utc


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run','analyze']);p.add_argument('--replication-id');p.add_argument('--case-id');args=p.parse_args()
    if args.replication_id and (not args.replication_id.replace('-','').isalnum() or len(args.replication_id)>48):raise ValueError('Replication ID must be 1-48 letters/digits/hyphens')
    output=ROOT/'artifacts/attention-replications'/args.replication_id if args.replication_id else DIRECTORY
    if args.case_id and not args.replication_id:raise ValueError('Partial repeats need a separate replication ID')
    if args.action=='freeze':print(freeze());return
    if args.action=='analyze':
        from attention_v2_report import build
        build(output if args.replication_id else None);return
    from atlas.attention.server import Jobs
    protocol=json.loads(PROTOCOL_PATH.read_text())
    if args.case_id and not any(c['case_id']==args.case_id for c in protocol['cases']):raise ValueError('Unregistered case ID')
    ledger_path=output/'execution.json' if args.replication_id else EXECUTION_PATH
    ledger=json.loads(ledger_path.read_text()) if ledger_path.exists() else {'protocol_id':protocol['id'],'started_at':utc(),'cases':[]}
    if len(ledger['cases'])==len(protocol['cases']) and ledger.get('finished_at'):
        print('Study already complete; use a new --replication-id for fresh evidence.');return
    if not ledger['cases']:
        import atlas.attention.server as controller
        ledger['supervisor_source_sha256']=sha(Path(__file__).read_bytes())
        ledger['controller_source_sha256']=sha(Path(controller.__file__).read_bytes())
    supervisor=Jobs();supervisor.wall_timeout_s=420
    try:
        for case in protocol['cases']:
            if args.case_id and case['case_id']!=args.case_id:continue
            if any(r['case_id']==case['case_id'] for r in ledger['cases']):continue
            config={**case,'protocol_id':protocol['id']}
            ident=supervisor.start(config)
            print(json.dumps({'starting':case['case_id'],'run':ident,'suite':case['suite'],'context':case.get('context'),'dtype':case.get('dtype'),'block':case['block']}),flush=True)
            while supervisor.active:time.sleep(.5)
            path=RUNS/ident/'result.json';doc=json.loads(path.read_text())
            # Preserve exact raw bytes, including controlled failures.
            dest=output/'runs'/f'{ident}.json';dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(path.read_bytes())
            ledger['cases'].append({'case_id':case['case_id'],'run_id':ident,'status':doc['status'],'sha256':sha(path.read_bytes()),'elapsed_s':doc.get('elapsed_s'),'error':doc.get('error')})
            atomic_json(ledger_path,ledger)
            print(json.dumps(ledger['cases'][-1]),flush=True)
        ledger['finished_at']=utc();atomic_json(ledger_path,ledger)
    finally:supervisor.close()


if __name__=='__main__':main()
