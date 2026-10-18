"""Standard-library identities shared by execution and read-only verification."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath


def canonical_bytes(value) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False,
                       separators=(',', ':'), allow_nan=False) + '\n').encode('utf-8')


def digest(value) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            result.update(block)
    return result.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_bytes(canonical_bytes(value))
    os.replace(temporary, path)


def read_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def artifact_path(root: Path, relative: str) -> Path:
    name = PurePosixPath(relative)
    if (not relative or '\\' in relative or ':' in relative or name.is_absolute()
            or any(part in ('', '.', '..') for part in relative.split('/'))):
        raise ValueError('Unsafe artifact path')
    path = root.joinpath(*name.parts)
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('Artifact escapes its bundle')
    return path


def verify_files(folder: Path, files: dict) -> None:
    if not files:
        raise ValueError('Empty asset identity')
    for name, expected in files.items():
        path = artifact_path(folder, name)
        if not path.is_file() or file_digest(path) != expected:
            raise ValueError('Missing or changed artifact: ' + name)
