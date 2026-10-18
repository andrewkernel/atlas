"""Standalone standard-library bundle integrity check; runs no experiment code."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath

GENERATED_PREFIXES=('.tools/','.venv/','weights/','data/',
                    'artifacts/attention-lab/','artifacts/attention-lab-server/',
                    'artifacts/attention-v3-replications/','artifacts/attention-v3-audits/')


def verify(root):
    root = root.resolve()
    manifest = json.loads((root/'bundle-manifest.json').read_text(encoding='utf-8'))
    files = manifest['files_sha256']
    if not files:raise ValueError('Empty bundle manifest')
    for name,expected in files.items():
        posix = PurePosixPath(name)
        if (not name or '\\' in name or ':' in name or posix.is_absolute()
                or any(x in ('','.','..') for x in name.split('/'))):
            raise ValueError('Unsafe bundle member: '+name)
        path = root.joinpath(*posix.parts)
        if not path.resolve().is_relative_to(root) or not path.is_file():
            raise ValueError('Missing or escaping bundle member: '+name)
        hasher = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(8*1024**2),b''):hasher.update(block)
        if hasher.hexdigest()!=expected:raise ValueError('Changed bundle member: '+name)
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()
              and '__pycache__' not in p.parts and p.name!='bundle-manifest.json'}
    missing=set(files)-actual
    extra={name for name in actual-set(files) if not name.startswith(GENERATED_PREFIXES)}
    if missing or extra:
        raise ValueError('Unmanifested or missing files: '+repr(sorted(missing|extra)))
    return {'verified_files':len(files),
            'scope':'Shipped-byte consistency only; numerical verification and fresh execution are separate.',
            'generated_paths_not_authenticated':list(GENERATED_PREFIXES)}


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    print(json.dumps(verify(parser.parse_args().directory),indent=2))
