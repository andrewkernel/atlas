import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import pytest
pytest.importorskip('fastapi')
from fastapi.testclient import TestClient
from atlas.attention import server


def test_study_registry_is_readable_but_mutations_require_auth():
    client=TestClient(server.app)
    response=client.get('/api/study')
    assert response.status_code==200
    assert client.post('/api/study/repeat',json={'case_id':'case-000'}).status_code==403
    headers={'X-Atlas-Token':server.TOKEN}
    for payload in ({'case_id':'../../escape'},{'case_id':'case-000','script':'anything'},{'case_id':'case-000','dtype':'float64'}):
        assert client.post('/api/study/repeat',json=payload,headers=headers).status_code==422


def test_repeat_passes_only_exact_registered_case(monkeypatch,tmp_path):
    from atlas.attention.protocol import PROTOCOL_PATH
    if not PROTOCOL_PATH.exists():pytest.skip('Local protocol not present')
    configs=[];monkeypatch.setattr(server,'RUNS',tmp_path)
    monkeypatch.setattr(server.jobs,'start',lambda c:configs.append(c) or 'research-test')
    client=TestClient(server.app);headers={'X-Atlas-Token':server.TOKEN}
    assert client.post('/api/study/repeat',json={'case_id':'case-999'},headers=headers).status_code==404
    result=client.post('/api/study/repeat',json={'case_id':'case-012'},headers=headers)
    assert result.status_code==202
    assert configs[0]['suite']=='cache' and configs[0]['block']==0
    assert configs[0]['protocol_id']=='atlas-attention-v2-93518-a1'
