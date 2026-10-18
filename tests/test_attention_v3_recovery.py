"""Controller-crash bookkeeping controls, never benchmark evidence."""
from pathlib import Path
import sys

import psutil
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'benchmarks'))
from atlas.attention.portable import file_digest,read_json,write_json
from attention_v3_study import recover_pending


def test_recovery_preserves_interrupted_raw_bytes_instead_of_faking_completion(tmp_path):
    runs=tmp_path/'workers';directory=runs/'test-fixture';directory.mkdir(parents=True)
    write_json(directory/'result.json',{'status':'running','rows':[]})
    original=(directory/'result.json').read_bytes()
    ledger={'cases':[{'run_id':'test-fixture','status':'running','pid':psutil.Process().pid,
                       'process_created_at':-1}]}
    output=tmp_path/'study';ledger_path=output/'execution.json'
    recover_pending(output,ledger,ledger_path,runs)
    assert ledger['cases'][0]['status']=='interrupted'
    assert (output/'runs/test-fixture/result.json').read_bytes()==original
    assert ledger['cases'][0]['sha256']==file_digest(directory/'result.json')
    assert 'absent' in ledger['cases'][0]['disposition']


def test_recovery_refuses_to_restart_an_authoritatively_live_worker(tmp_path):
    process=psutil.Process()
    ledger={'cases':[{'run_id':'test-fixture','status':'running','pid':process.pid,
                       'process_created_at':process.create_time()}]}
    with pytest.raises(RuntimeError,match='still alive'):
        recover_pending(tmp_path/'study',ledger,tmp_path/'study/execution.json',tmp_path/'workers')
