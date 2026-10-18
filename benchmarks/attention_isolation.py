"""Registered post-hoc controls: one arm per owned CUDA process, no simulated runs."""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import gc
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from atlas.attention.portable import artifact_path,file_digest,read_json,verify_files,write_json

DIRECTORY=ROOT/'docs/research/attention-isolation'
PROTOCOL=DIRECTORY/'protocol.json'


def utc():return datetime.now(timezone.utc).isoformat()


def freeze():
    if PROTOCOL.exists():raise ValueError('Preserve the frozen diagnostic; changes need a new version')
    from atlas.attention.long_protocol import verify
    base=verify()
    cases=[]
    for method in ('native','resident','hybrid50'):
        for context in (32768,40000):
            source=next(c for c in base['cases'] if c['book_id']=='2383' and c['context']==context)
            cases.append({'case_id':f'isolated-{len(cases):03d}','kind':'isolated-attention',
                          'method':method,'context':context,'input_case':source,
                          'validation_tokens':32,'generation_tokens':16,'window':256,'threads':4})
    doc={'id':'atlas-isolated-arms-01','schema_version':1,'created_at':utc(),
         'development':'Post-hoc: book 2383 chosen because v3 showed both literal numerical failure and later-arm OOM. No unseen/confirmatory claim.',
         'base_protocol_sha256':file_digest(ROOT/'docs/research/attention-v3/protocol.json'),
         'sources':{**base['sources'],
                    'benchmarks/attention_isolation.py':file_digest(Path(__file__)),
                    'benchmarks/verify_attention_isolation.py':file_digest(ROOT/'benchmarks/verify_attention_isolation.py')},
         'contract':base['contract'],'vocabulary_size':base['vocabulary_size'],'cases':cases,
         'analysis':'Each method/context owns a fresh process. Same short warmup; generation precedes validation/reference. Require native repeat + resident controls. Compare matched contexts only, retain every outcome. One book and one shared session are not a population estimate.',
         'limits':['Post-hoc diagnostic, one document, one model, FP32, no external replication.',
                   'Process isolation does not reset OS file caches or reserve hardware.',
                   'Disappearing OOM does not by itself identify a particular allocator cause.',
                   'Prior numerical failures and OOMs are preserved, not repaired retrospectively.']}
    write_json(PROTOCOL,doc);return doc


def verify():
    from atlas.attention.long_protocol import verify as verify_base
    verify_base()
    doc=read_json(PROTOCOL)
    if doc['base_protocol_sha256']!=file_digest(ROOT/'docs/research/attention-v3/protocol.json'):
        raise ValueError('Base protocol changed')
    verify_files(ROOT,doc['sources'])
    return doc


def reference_array(output,context,protocol):
    import numpy as np
    ledger=read_json(output/'execution.json')
    case=next(c for c in protocol['cases'] if c['method']=='native' and c['context']==context)
    entry=next((e for e in ledger['cases'] if e['case_id']==case['case_id']),None)
    if not entry or entry.get('status')!='completed':return None
    folder=output/'runs'/entry['run_id'];record=read_json(folder/'result.json')
    if file_digest(folder/'result.json')!=entry['sha256'] or record['config']!=case:
        raise ValueError('Native reference record identity mismatch')
    item=record.get('artifacts',{}).get('reference_logits')
    if not item:return None
    path=artifact_path(folder,item['path'])
    if file_digest(path)!=item['sha256']:raise ValueError('Native reference bytes changed')
    return np.load(path,allow_pickle=False)


def execute_worker(output,case_id,run_id):
    import numpy as np
    import torch
    import psutil
    import importlib.metadata
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from filelock import FileLock
    from atlas.attention.evidence import Recorder
    from atlas.attention.long_protocol import input_ids,token_digest
    from atlas.attention.long_validation import cleanup,replay,save_array,Observations
    from atlas.attention.runtime import error_metrics
    protocol=verify();case=next(c for c in protocol['cases'] if c['case_id']==case_id)
    rec=Recorder(output/'runs'/run_id,case)
    relative=Path(__file__).relative_to(ROOT).as_posix()
    rec.doc['environment']['source_sha256'][relative]=file_digest(Path(__file__))
    snapshot=rec.directory/'source'/relative;snapshot.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(__file__,snapshot)
    rec.doc.update(schema_version=4,protocol_sha256=file_digest(PROTOCOL),artifacts={},
                   limitations=protocol['limits'],rows=[],host_allocator={},
                   execution_process={'pid':os.getpid(),'created_at':psutil.Process().create_time()})
    try:
        if not torch.cuda.is_available():raise RuntimeError('CUDA required; no substituted measurements')
        torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
        rec.doc['configured_precision']={'dtype':'float32','tf32':torch.backends.cuda.matmul.allow_tf32,'matmul_precision':torch.get_float32_matmul_precision()}
        rec.doc['environment']['packages'].update({p:importlib.metadata.version(p) for p in ('pyarrow','safetensors','tokenizers','filelock')})
        def host():
            try:return dict(torch.cuda.memory.host_memory_stats())
            except (RuntimeError,AttributeError) as exc:return {'unavailable':type(exc).__name__+': '+str(exc)}
        with FileLock(str(ROOT/'artifacts/attention-lab/.gpu.lock'),timeout=0),torch.inference_mode():
            folder=ROOT/'weights/qwen3-0.6b'
            tokenizer=AutoTokenizer.from_pretrained(folder,local_files_only=True,trust_remote_code=False)
            ids=input_ids(case['input_case'],tokenizer)
            if token_digest(ids)!=case['input_case']['input_token_sha256']:raise ValueError('Input token identity mismatch')
            save_array(rec,'input_ids',np.asarray(ids,dtype='<i8'))
            prompt=torch.tensor([ids[:case['context']]],device='cuda',dtype=torch.long)
            targets=torch.tensor([ids[case['context']:]],device='cuda',dtype=torch.long)
            model=AutoModelForCausalLM.from_pretrained(folder,local_files_only=True,trust_remote_code=False,
                    use_safetensors=True,attn_implementation='sdpa',dtype=torch.float32).cuda().eval()
            method=case['method'];row={'plan':method,'status':'running','correctness':{'passed':None}}
            rec.doc['rows'].append(row)
            with Observations() as observation:
                try:
                    rec.progress(0,2,f'{method} / short warmup, isolated process')
                    replay(model,prompt[:,:128],targets[:,:2],method,256,rec);cleanup()
                    rec.doc['host_allocator']['before_generation']=host()
                    torch.cuda.reset_peak_memory_stats();observation.phase=method+'/generation'
                    rec.progress(0,2,f'{method} / {case["context"]:,} tokens / generation')
                    _,measured=replay(model,prompt,targets[:,:16],method,256,rec,generation=True)
                    row.update(measured);cleanup();rec.doc['host_allocator']['after_generation_cleanup']=host();rec.save()
                    observation.phase=method+'/validation';rec.progress(1,2,f'{method} / identical-history validation')
                    actual,_=replay(model,prompt,targets,method,256,rec);actual=np.concatenate(actual,axis=0)
                    cleanup()
                    if method=='native':
                        reference=actual;save_array(rec,'reference_logits',reference)
                        repeat,_=replay(model,prompt,targets,'native',256,rec);actual=np.concatenate(repeat,axis=0);cleanup()
                    else:reference=reference_array(output,case['context'],protocol)
                    save_array(rec,'candidate_logits',actual)
                    checks=None if reference is None else [error_metrics(torch.from_numpy(a),torch.from_numpy(r),atol=1e-4,rtol=1e-4) for a,r in zip(actual,reference,strict=True)]
                    row.update(status='completed',logit_checks=checks,
                               correctness={'passed':all(c['passed'] for c in checks) if checks is not None else None,
                                            'finite':bool(np.isfinite(actual).all())},
                               eligibility={'eligible':False,'reason':'Cross-process controls require independent study analysis'})
                    rec.doc['host_allocator']['after_validation_cleanup']=host()
                except torch.OutOfMemoryError as exc:
                    row.update(status='oom',error=str(exc)[:1200]);cleanup()
                finally:
                    rec.doc['observations']=observation.rows;rec.save()
            rec.progress(2,2,'Isolated arm finished; inspect status and cross-process controls')
        rec.finish()
    except BaseException as exc:
        rec.finish('failed',f'{type(exc).__name__}: {exc}');raise


def run(output):
    import psutil
    protocol=verify();output.mkdir(parents=True,exist_ok=True)
    if not (output/'protocol.json').exists():shutil.copyfile(PROTOCOL,output/'protocol.json')
    ledger_path=output/'execution.json'
    ledger=read_json(ledger_path) if ledger_path.exists() else {'protocol_sha256':file_digest(PROTOCOL),'started_at':utc(),'cases':[]}
    if ledger['protocol_sha256']!=file_digest(PROTOCOL):raise ValueError('Diagnostic ledger identity mismatch')
    if any(e['status'] in ('starting','running') for e in ledger['cases']):raise RuntimeError('An unfinished attempt needs authoritative process inspection before resuming; do not replace it')
    for case in protocol['cases']:
        if any(e['case_id']==case['case_id'] for e in ledger['cases']):continue
        run_id='isolated-'+uuid.uuid4().hex[:12];directory=output/'runs'/run_id;directory.mkdir(parents=True)
        environment={k:v for k,v in os.environ.items() if k.upper() in {'PATH','SYSTEMROOT','WINDIR','TEMP','TMP','USERPROFILE','LOCALAPPDATA','APPDATA','COMSPEC','PROGRAMFILES','PROGRAMW6432'}}
        environment.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1')
        scope=None;process=None
        entry={'case_id':case['case_id'],'run_id':run_id,'status':'starting','started_at':utc()}
        ledger['cases'].append(entry);write_json(ledger_path,ledger)
        try:
            with (directory/'worker.log').open('wb') as log:
                process=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'worker','--case-id',case['case_id'],'--run-id',run_id,'--output',str(output)],cwd=ROOT,env=environment,
                    stdout=log,stderr=subprocess.STDOUT,creationflags=0x08000000|0x00000004 if os.name=='nt' else 0,start_new_session=os.name!='nt')
            owned=psutil.Process(process.pid)
            entry.update(status='running',pid=process.pid,process_created_at=owned.create_time())
            write_json(ledger_path,ledger)
            if os.name=='nt':
                from windows_job_scope import WindowsJobScope
                scope=WindowsJobScope();scope.assign_owned(process.pid,owned.create_time());scope.resume_owned_single_thread(owned)
            print(entry,flush=True)
            start=time.monotonic();timed_out=False
            while process.poll() is None:
                if time.monotonic()-start>1200:
                    timed_out=True
                    if scope:scope.terminate()
                    elif os.name!='nt':os.killpg(process.pid,signal.SIGKILL)
                    else:process.kill()
                    break
                time.sleep(.5)
            process.wait(timeout=10)
            if scope:scope.close();scope=None
            path=directory/'result.json'
            if not path.exists():write_json(path,{'id':run_id,'config':case,'status':'failed','rows':[],'error':'Worker exited before initializing evidence'})
            record=read_json(path)
            entry.update(status='timed_out' if timed_out else record['status'] if record['status'] not in ('running','queued') else 'interrupted',
                         raw_status=record['status'],sha256=file_digest(path),exit_code=process.returncode,
                         owned_scope_closed=True,finished_at=utc())
            write_json(ledger_path,ledger);print(entry,flush=True)
        except BaseException as exc:
            entry['controller_error']=f'{type(exc).__name__}: {exc}'
            raise
        finally:
            if scope:scope.terminate();scope.close()
            if process and process.poll() is None:
                if os.name!='nt':os.killpg(process.pid,signal.SIGKILL)
                else:process.kill()
                process.wait(timeout=10)
            if entry['status'] in ('starting','running'):
                path=directory/'result.json'
                if not path.exists():write_json(path,{'id':run_id,'config':case,'status':'failed','rows':[],
                                                      'error':entry.get('controller_error','Controller interrupted before worker initialization')})
                record=read_json(path)
                entry.update(status='interrupted' if record['status'] in ('running','queued') else record['status'],
                             raw_status=record['status'],sha256=file_digest(path),finished_at=utc(),owned_scope_closed=True)
                write_json(ledger_path,ledger)
    ledger['finished_at']=utc();write_json(ledger_path,ledger)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['freeze','run','worker','analyze'])
    parser.add_argument('--replication-id');parser.add_argument('--case-id');parser.add_argument('--run-id')
    parser.add_argument('--output',type=Path);parser.add_argument('--wait-for-primary',action='store_true')
    args=parser.parse_args()
    if args.replication_id and (not args.replication_id.replace('-','').isalnum() or len(args.replication_id)>48):raise ValueError('Invalid replication ID')
    if args.output is not None and args.action!='worker':raise ValueError('Custom output accepted only for owned internal worker')
    output=ROOT/'artifacts/attention-isolation-replications'/args.replication_id if args.replication_id else DIRECTORY
    if args.action=='freeze':print(freeze()['id']);return
    if args.action=='worker':
        if not args.run_id or not args.run_id.replace('-','').isalnum():raise ValueError('Invalid run ID')
        return execute_worker(args.output,args.case_id,args.run_id)
    if args.action=='analyze':
        from verify_attention_isolation import build
        summary=build(output,ROOT);write_json(output/'summary.json',summary);print(summary);return
    if not args.replication_id:
        began=time.monotonic()
        while True:
            primary=read_json(ROOT/'docs/research/attention-v3/execution.json')
            if primary.get('finished_at') and len(primary['cases'])==18 and all(e['status']!='running' for e in primary['cases']):break
            if not args.wait_for_primary:raise RuntimeError('Primary study is still running; preserve its measurement windows')
            if time.monotonic()-began>7200:raise TimeoutError('Primary study remains incomplete; no run was restarted')
            time.sleep(1)
    run(output)


if __name__=='__main__':main()
