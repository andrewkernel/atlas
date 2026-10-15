"""Fetch only two pinned Fashion-MNIST data files and their MIT notice.

No downloaded Python or model code is executed. Data stay in ignored data/.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import struct
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

REVISION = 'b2617bb6d3ffa2e429640350f613e3291e10b141'
FILES = {'train-images-idx3-ubyte.gz': (26421880, 47040016), 'train-labels-idx1-ubyte.gz': (29515, 60008)}


def run(destination: Path, manifest: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = f'https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/{REVISION}/'
    records = []
    for name, (compressed_size, expanded_size) in FILES.items():
        url = root+'data/fashion/'+name
        with urlopen(url, timeout=60) as response:
            body = response.read(compressed_size+1)
        if len(body) != compressed_size:
            raise ValueError('Unexpected compressed data length')
        with gzip.GzipFile(fileobj=io.BytesIO(body)) as stream:
            expanded = stream.read(expanded_size+1)
        if len(expanded) != expanded_size:
            raise ValueError('Unexpected expanded data length')
        magic, count = struct.unpack('>II', expanded[:8])
        if count != 60000 or magic != (2051 if 'images' in name else 2049):
            raise ValueError('Unexpected IDX header')
        (destination/name).write_bytes(body)
        records.append({'name': name, 'source_url': url, 'compressed_bytes': len(body),
                        'compressed_sha256': hashlib.sha256(body).hexdigest(), 'expanded_bytes': len(expanded),
                        'expanded_sha256': hashlib.sha256(expanded).hexdigest(), 'magic': magic, 'examples': count})
    with urlopen(root+'LICENSE', timeout=30) as response:
        license_text = response.read(20000)
    if b'The MIT License' not in license_text or b'Zalando' not in license_text:
        raise ValueError('Unexpected repository license notice')
    (destination/'LICENSE').write_bytes(license_text)
    record = {'schema_version': '1.0', 'retrieved_utc': datetime.now(timezone.utc).isoformat(),
              'source_revision': REVISION, 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'files': records, 'license_url': root+'LICENSE', 'license_sha256': hashlib.sha256(license_text).hexdigest(),
              'scope': 'Official training partition only; canonical test data not downloaded. Dataset/code notice retained locally. No downloaded code executed.'}
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'examples': 60000, 'download_bytes': sum(item['compressed_bytes'] for item in records)}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--destination', type=Path, default=Path('data/fashion-mnist'))
    parser.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args()
    run(args.destination, args.manifest)
