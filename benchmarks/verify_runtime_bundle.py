"""Portable public bundle consistency check; no model execution or PyTorch import."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys


def verify(root):
    manifest = json.loads((root / 'bundle-manifest.json').read_text(encoding='utf-8'))
    for name, expected in manifest['files_sha256'].items():
        path = root / name
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError('Manifest path escapes extracted bundle')
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('Missing or changed bundle file: ' + name)
    result = subprocess.run([sys.executable, str(root / 'tune.py'), 'verify', manifest['campaign']],
                            cwd=root, capture_output=True, text=True, timeout=120)
    if result.returncode:
        raise ValueError('Arithmetic verification failed: ' + result.stderr[-1800:])
    verified = json.loads(result.stdout)
    if verified['evidence_sha256'] != manifest['evidence_sha256']:
        raise ValueError('Reconstructed evidence differs from manifest')
    return {'campaign': manifest['campaign'], 'files_checked': len(manifest['files_sha256']),
            'evidence_sha256': verified['evidence_sha256'], 'decision': verified['decision'],
            'scope': 'Byte/array/decision consistency; not independent model replication or author authentication'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('root', type=Path)
    print(json.dumps(verify(parser.parse_args().root.resolve())))
