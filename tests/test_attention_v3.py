"""Invented tiny control fixtures; never published as model/hardware runs."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'benchmarks'))
from atlas.attention.portable import artifact_path, digest, file_digest, read_json, verify_files, write_json
from verify_attention_v3 import compare_arrays, nll, verify_record


def test_semantic_identity_survives_order_indentation_and_windows_newlines(tmp_path):
    value={'files':{'b':'2','a':'1'},'revision':'fixed'}
    windows=json.dumps(value,indent=2).replace('\n','\r\n')
    path=tmp_path/'manifest.json';path.write_bytes(windows.encode())
    assert digest(read_json(path))==digest({'revision':'fixed','files':{'a':'1','b':'2'}})
    write_json(path,value)
    assert b'\r' not in path.read_bytes()


@pytest.mark.parametrize('relative',['../secret','/absolute','C:/secret','a\\b','a//b','a/./b','a/../../b',''])
def test_bundle_refuses_unsafe_paths(tmp_path,relative):
    with pytest.raises(ValueError):artifact_path(tmp_path,relative)


def test_byte_identity_fails_closed_for_missing_or_changed_files(tmp_path):
    path=tmp_path/'asset';path.write_bytes(b'actual')
    verify_files(tmp_path,{'asset':file_digest(path)})
    with pytest.raises(ValueError):verify_files(tmp_path,{'missing':'0'*64})
    with pytest.raises(ValueError):verify_files(tmp_path,{'asset':'0'*64})


def fixture_record(tmp_path):
    # Two positions, three vocabulary elements: explicit control data, no timings
    # derived from GPU execution and never entered in a real execution ledger.
    values=np.array([[1.,2.,3.],[3.,2.,1.]],dtype=np.float32)
    ids=np.array([0,1,2,1],dtype='<i8')
    case={'kind':'long-validation','case_id':'long-000','cohort':'test-fixture',
          'block':0,'context':2,'book_id':None,'validation_tokens':2,
          'generation_tokens':2,'methods':['native','resident','hybrid50'],
          'input_token_sha256':hashlib.sha256(ids.tobytes()).hexdigest()}
    protocol={'_sha256':'test-only','vocabulary_size':3,'contract':{'atol':1e-4,'rtol':1e-4}}
    record={'id':'test-fixture','config':case,'protocol_sha256':'test-only',
            'status':'completed','artifacts':{},'rows':[]}
    for name,array in [('input_ids',ids),('reference_logits',values)]+[(m+'_logits',values) for m in case['methods']]:
        path=tmp_path/(name+'.npy');np.save(path,array,allow_pickle=False)
        record['artifacts'][name]={'path':path.name,'sha256':file_digest(path),'shape':list(array.shape),'dtype':str(array.dtype)}
    for method in case['methods']:
        record['rows'].append({'plan':method,'status':'completed','generation_ms':3.,'prefill_ms':1.,
          'samples_ms':[1.],'median_ms':1.,'allocated_peak_mib':1.,'generated_token_ids':[0,1],
          'logit_checks':compare_arrays(values,values,1e-4,1e-4),'correctness':{'passed':True},
          'eligibility':{'eligible':True},'mean_nll':np.mean([nll(a,int(ids[2+i])) for i,a in enumerate(values)])})
    return record,case,protocol


def test_independent_verifier_accepts_control_and_rejects_lying_pass_flag(tmp_path):
    record,case,protocol=fixture_record(tmp_path)
    result=verify_record(tmp_path,record,case,protocol)
    assert all(r['eligible'] for r in result['arms'])
    path=tmp_path/'hybrid50_logits.npy'
    altered=np.load(path,allow_pickle=False);altered[1,1]+=1
    np.save(path,altered,allow_pickle=False)
    # Even a recomputed file hash cannot make the lying numerical flag pass.
    record['artifacts']['hybrid50_logits']['sha256']=file_digest(path)
    with pytest.raises(ValueError,match='numerical decision'):verify_record(tmp_path,record,case,protocol)


def test_independent_verifier_rejects_missing_array_and_incomplete_positions(tmp_path):
    record,case,protocol=fixture_record(tmp_path)
    record['rows'][0]['logit_checks'].pop()
    with pytest.raises(ValueError,match='Incomplete per-position'):verify_record(tmp_path,record,case,protocol)
    record,case,protocol=fixture_record(tmp_path)
    (tmp_path/'reference_logits.npy').unlink()
    with pytest.raises(ValueError,match='Missing or changed'):verify_record(tmp_path,record,case,protocol)


def test_native_control_failure_blocks_candidate_even_if_candidate_agrees(tmp_path):
    record,case,protocol=fixture_record(tmp_path)
    path=tmp_path/'native_logits.npy'
    actual=np.load(path,allow_pickle=False);actual[0,0]+=1
    reference=np.load(tmp_path/'reference_logits.npy',allow_pickle=False)
    np.save(path,actual,allow_pickle=False)
    record['artifacts']['native_logits']['sha256']=file_digest(path)
    row=record['rows'][0]
    row.update(logit_checks=compare_arrays(actual,reference,1e-4,1e-4),correctness={'passed':False},
               mean_nll=np.mean([nll(a,t) for a,t in zip(actual,[2,1],strict=True)]))
    for row in record['rows']:row['eligibility']['eligible']=False
    assert not any(r['eligible'] for r in verify_record(tmp_path,record,case,protocol)['arms'])


def test_nonfinite_candidate_is_never_a_pass():
    actual=np.array([[float('nan'),1]],dtype=np.float32)
    checks=compare_arrays(actual,np.ones_like(actual),1e-4,1e-4)
    assert checks[0]['finite'] is False and checks[0]['passed'] is False
    assert checks[0]['max_abs'] is None


def test_validation_api_refuses_unregistered_work_without_launching(monkeypatch):
    pytest.importorskip('fastapi')
    from fastapi.testclient import TestClient
    from atlas.attention import server
    client=TestClient(server.app)
    assert client.post('/api/validation/repeat',json={'case_id':'long-000'}).status_code==403
    headers={'X-Atlas-Token':server.TOKEN}
    for payload in ({'case_id':'../bad'},{'case_id':'long-000','context':1},{'case_id':'long-000','script':'anything'}):
        assert client.post('/api/validation/repeat',json=payload,headers=headers).status_code==422
