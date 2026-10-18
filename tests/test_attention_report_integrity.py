import json
import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'benchmarks'))
from attention_v2_report import build
from atlas.attention.protocol import PROTOCOL_PATH


def test_report_refuses_corrupted_raw_record_before_summarizing(tmp_path):
    if not PROTOCOL_PATH.exists():pytest.skip('Study artifacts not available')
    (tmp_path/'runs').mkdir()
    (tmp_path/'runs/research-fake.json').write_text('{"config":{}}')
    (tmp_path/'execution.json').write_text(json.dumps({'cases':[{'run_id':'research-fake','case_id':'case-000','sha256':'0'*64}]}))
    with pytest.raises(ValueError,match='Raw run hash mismatch'):build(tmp_path)
