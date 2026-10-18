"""Pure identities, bounded paths and lossless content-addressed artifacts."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import re
import subprocess

from atlas.attention.portable import artifact_path, canonical_bytes, digest, file_digest, read_json, write_json

ROOT = Path(__file__).resolve().parents[3]
STORE = ROOT / 'artifacts/atlas-tuning'
BASE_SHA = '9db4132e3d09a471f9aa2a2d79103d797a1f42dcde0429c563e4e835942f0cea'
BASE_PATH = 'docs/research/attention-v3/protocol.json'
NEW_SOURCES = ['tune.py'] + ['src/atlas/tuning/' + name + '.py' for name in
    ('__init__', 'common', 'registry', 'executor', 'policy', 'verify', 'controller', 'profiles', 'inference')]


def utc():
    return datetime.now(timezone.utc).isoformat()


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,90}', value):
        raise ValueError('Invalid registered identifier')
    return value


def campaign_dir(ident, home=STORE):
    return home / 'campaigns' / identifier(ident)


def source_map(root=ROOT):
    base = root / BASE_PATH
    if file_digest(base) != BASE_SHA:
        raise ValueError('Frozen base protocol identity changed')
    sources = {**read_json(base)['sources'], BASE_PATH: BASE_SHA}
    for name, expected in sources.items():
        if file_digest(artifact_path(root, name)) != expected:
            raise ValueError('Previously frozen source changed: ' + name)
    return {**sources, **{name: file_digest(root / name) for name in NEW_SOURCES}}


def check_sources(sources, root=ROOT):
    if sources != source_map(root):
        raise ValueError('Tuning execution source differs from frozen campaign')


def packages():
    return {name: importlib.metadata.version(name) for name in
        ('torch', 'transformers', 'numpy', 'tokenizers', 'safetensors', 'pyarrow', 'filelock', 'psutil')}


def runtime_identity():
    """Query the driver externally; the controller must not initialize CUDA."""
    import psutil
    if os.name != 'nt':
        raise RuntimeError('The validated owned CUDA runtime currently requires Windows')
    command = ['nvidia-smi', '--id=0', '--query-gpu=uuid,name,driver_version,memory.total,compute_cap',
               '--format=csv,noheader,nounits']
    row = next(csv.reader(subprocess.check_output(command, text=True, timeout=8,
                        creationflags=0x08000000).strip().splitlines()))
    if len(row) != 5:
        raise ValueError('Unexpected driver identity response')
    device, name, driver, memory, capability = [s.strip() for s in row]
    import winreg
    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'SOFTWARE\Microsoft\Cryptography') as key:
        machine = winreg.QueryValueEx(key, 'MachineGuid')[0]
    hardware = {'gpu': name, 'driver': driver, 'reported_device_mib': float(memory),
                'compute_capability': capability, 'cpu': platform.processor(),
                'physical_cpus': psutil.cpu_count(logical=False), 'logical_cpus': psutil.cpu_count(),
                'ram_bytes': psutil.virtual_memory().total, 'platform': platform.system(),
                'os_version': platform.version(), 'machine_device_fingerprint': digest([machine, device])}
    return {'hardware': hardware, 'packages': packages(), 'python': platform.python_version()}


def domain(settings):
    return digest({k: settings[k] for k in ('model', 'dtype', 'context', 'generation_tokens')})


def save_array(folder, values):
    import numpy as np
    array = np.ascontiguousarray(values)
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    payload = buffer.getvalue()
    import hashlib
    sha = hashlib.sha256(payload).hexdigest()
    path = folder / 'blobs' / (sha + '.npy')
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if file_digest(path) != sha:
            raise ValueError('Existing content-addressed artifact is corrupt')
    else:
        temporary = path.with_suffix('.tmp')
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    return {'path': path.relative_to(folder).as_posix(), 'sha256': sha,
            'shape': list(array.shape), 'dtype': str(array.dtype)}


def load_array(folder, item):
    import numpy as np
    path = artifact_path(folder, item['path'])
    if file_digest(path) != item['sha256']:
        raise ValueError('Changed array artifact')
    array = np.load(path, allow_pickle=False)
    if list(array.shape) != item['shape'] or str(array.dtype) != item['dtype']:
        raise ValueError('Array descriptor mismatch')
    return array


def frozen_json(path, value):
    payload = canonical_bytes(value)
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError('Refusing to rewrite a frozen artifact: ' + path.name)
    else:
        write_json(path, value)


def snapshot(folder, sources, root=ROOT):
    for name, sha in sources.items():
        source = artifact_path(root, name)
        if file_digest(source) != sha:
            raise ValueError('Execution source changed: ' + name)
        target = artifact_path(folder / 'source', name)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_bytes(source.read_bytes())
        if file_digest(target) != sha:
            raise ValueError('Source snapshot changed: ' + name)


def public_error(exc):
    return type(exc).__name__ + ': ' + str(exc)[:1200]
