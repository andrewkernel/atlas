"""Invented import-control records; never model or hardware evidence."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'benchmarks'))
from atlas.attention.portable import file_digest,read_json,write_json
from prepare_attention_history import prepare


def fixture(root):
    study=root/'docs/research/attention-v3'
    source=study/'runs/reference-fixture';source.mkdir(parents=True)
    config={'case_id':'control-case','kind':'invented-test-fixture'}
    write_json(study/'protocol.json',{'cases':[config]})
    write_json(source/'result.json',{'id':'reference-fixture','config':config,'status':'completed','rows':[]})
    write_json(study/'execution.json',{'protocol_sha256':file_digest(study/'protocol.json'),'cases':[
        {'case_id':'control-case','run_id':'reference-fixture','status':'completed','sha256':file_digest(source/'result.json')}]})
    return source


def test_history_projection_preserves_bytes_and_explicit_reference_origin(tmp_path):
    source=fixture(tmp_path);before=(source/'result.json').read_bytes()
    index=prepare(tmp_path)
    assert index['runs']==['reference-fixture'] and 'not fresh' in index['origin']
    target=tmp_path/'artifacts/attention-lab/reference-fixture/result.json'
    assert target.read_bytes()==before and (source/'result.json').read_bytes()==before
    assert prepare(tmp_path)==index


def test_history_projection_refuses_to_overwrite_a_conflicting_run(tmp_path):
    fixture(tmp_path)
    target=tmp_path/'artifacts/attention-lab/reference-fixture/result.json'
    write_json(target,{'id':'reference-fixture','status':'different fixture'})
    with pytest.raises(ValueError,match='conflicts'):prepare(tmp_path)
    assert read_json(target)['status']=='different fixture'


def test_missing_reference_payload_does_not_become_a_fake_local_run(tmp_path):
    source=fixture(tmp_path)
    record=read_json(source/'result.json');record['artifacts']={'missing':{'path':'missing.npy','sha256':'0'*64}}
    write_json(source/'result.json',record)
    ledger=read_json(source.parent.parent/'execution.json');ledger['cases'][0]['sha256']=file_digest(source/'result.json')
    write_json(source.parent.parent/'execution.json',ledger)
    index=prepare(tmp_path)
    assert index['runs']==[] and index['unavailable_runs']==['reference-fixture']
    assert not (tmp_path/'artifacts/attention-lab/reference-fixture/result.json').exists()


def test_matching_raw_hash_cannot_bless_a_mutated_registered_configuration(tmp_path):
    source=fixture(tmp_path);record=read_json(source/'result.json')
    record['config']['kind']='mutated-fixture';write_json(source/'result.json',record)
    ledger=read_json(source.parent.parent/'execution.json');ledger['cases'][0]['sha256']=file_digest(source/'result.json')
    write_json(source.parent.parent/'execution.json',ledger)
    with pytest.raises(ValueError,match='configuration changed'):prepare(tmp_path)
