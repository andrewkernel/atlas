import sys
import time
import json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import pytest
pytest.importorskip('fastapi')
from fastapi.testclient import TestClient
from atlas.attention import server


@pytest.fixture
def client():return TestClient(server.app)


def test_api_boundary_refuses_cross_origin_and_untrusted_host(client):
    assert client.get('/api/runs',headers={'Host':'evil.example'}).status_code==403
    assert client.post('/api/runs',json={}).status_code==403
    assert client.post('/api/runs',json={},headers={'Origin':'http://evil.example','X-Atlas-Token':server.TOKEN}).status_code==403


@pytest.mark.parametrize('payload',[{'kind':'shell'},{'repeats':100000},{'prompt':'a'*1801},{'heads':1024},{'script':'os.remove'},{'threads':128},{'kv_budget_mib':-1}])
def test_job_schema_rejects_arbitrary_or_unbounded_work(client,payload):
    assert client.post('/api/runs',json=payload,headers={'X-Atlas-Token':server.TOKEN}).status_code==422


def test_downloads_and_results_do_not_escape_output_registry(client):
    assert client.get('/api/runs/not-real').status_code==404
    assert client.get('/papers/not-real').status_code==404
    assert client.get('/api/runs/a.b').status_code==404


def test_launch_contract_and_serialization(client,monkeypatch):
    configs=[]
    monkeypatch.setattr(server.jobs,'start',lambda config:configs.append(config) or 'benchmark-test')
    response=client.post('/api/runs',json={'kind':'benchmark','profile':'quick'},headers={'X-Atlas-Token':server.TOKEN})
    assert response.status_code==202
    assert response.json()['id']=='benchmark-test'
    assert configs[0]['threads']==4 and configs[0]['heads']==12
    manager=server.Jobs();manager.active='owned'
    with pytest.raises(server.HTTPException) as exc:manager.start(configs[0])
    assert exc.value.status_code==409


def test_exports_are_actual_record_values(client,monkeypatch,tmp_path):
    monkeypatch.setattr(server,'RUNS',tmp_path)
    server.atomic_json(tmp_path/'benchmark-test'/'result.json',{'id':'benchmark-test','rows':[{'phase':'validation','context':1024,'plan':'cpu','median_ms':.25,'p90_ms':.3,'gpu_kv_mib':0}]})
    assert client.get('/api/runs/benchmark-test').json()['rows'][0]['median_ms']==.25
    csv=client.get('/api/runs/benchmark-test/export?format=csv')
    assert 'validation,1024,cpu,0.25,0.3,0' in csv.text


def test_real_owned_worker_timeout_releases_registry():
    manager=server.Jobs()
    manager.wall_timeout_s=.05
    ident=manager.start(server.Experiment(profile='quick').model_dump())
    process=manager.process
    until=time.monotonic()+10
    while manager.active and time.monotonic()<until:time.sleep(.05)
    try:
        assert manager.active is None
        assert process.poll() is not None
        doc=json.loads((server.RUNS/ident/'result.json').read_text())
        assert doc['status']=='timed_out'
    finally:manager.close()
