"""Small invented bundle controls; no fabricated scientific run data."""
import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'benchmarks'))
from verify_researcher_bundle import verify


def test_standalone_bundle_check_rejects_changed_and_unmanifested_files(tmp_path):
    path=tmp_path/'actual.txt';path.write_bytes(b'control fixture')
    manifest={'files_sha256':{'actual.txt':hashlib.sha256(path.read_bytes()).hexdigest()}}
    (tmp_path/'bundle-manifest.json').write_text(json.dumps(manifest))
    assert verify(tmp_path)['verified_files']==1
    extra=tmp_path/'unexpected.txt';extra.write_bytes(b'extra')
    with pytest.raises(ValueError,match='Unmanifested'):verify(tmp_path)
    extra.unlink()
    path.write_bytes(b'changed')
    with pytest.raises(ValueError,match='Changed bundle member'):verify(tmp_path)


def test_standalone_bundle_check_rejects_path_traversal(tmp_path):
    (tmp_path/'bundle-manifest.json').write_text(json.dumps({'files_sha256':{'../outside':'0'*64}}))
    with pytest.raises(ValueError,match='Unsafe bundle'):verify(tmp_path)


def test_runtime_outputs_do_not_hide_modified_shipped_files_or_new_source(tmp_path):
    path=tmp_path/'source.py';path.write_bytes(b'# shipped fixture\n')
    (tmp_path/'bundle-manifest.json').write_text(json.dumps({
        'files_sha256':{'source.py':hashlib.sha256(path.read_bytes()).hexdigest()}}))
    output=tmp_path/'artifacts/attention-lab/test-fixture/result.json'
    output.parent.mkdir(parents=True);output.write_text('{}')
    weights=tmp_path/'weights/test-fixture';weights.mkdir(parents=True)
    (weights/'downloaded').write_bytes(b'not authenticated by bundle verifier')
    assert verify(tmp_path)['verified_files']==1
    source=tmp_path/'src/injected.py';source.parent.mkdir();source.write_text('# unexpected')
    with pytest.raises(ValueError,match='Unmanifested'):verify(tmp_path)
    source.unlink();path.write_bytes(b'# altered shipped source\n')
    with pytest.raises(ValueError,match='Changed bundle member'):verify(tmp_path)
