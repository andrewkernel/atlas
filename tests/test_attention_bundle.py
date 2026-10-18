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
