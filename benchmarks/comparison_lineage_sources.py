"""Read-only primary-source capture for policy/validation lineage; no code execution."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    'gardner14.pdf': 'https://proceedings.mlr.press/v32/gardner14.pdf',
    'gardner14.html': 'https://proceedings.mlr.press/v32/gardner14.html',
    'gelbart14-v1.html': 'https://arxiv.org/html/1403.5607v1',
    'alive2-pldi21.pdf': 'https://web.ist.utl.pt/nuno.lopes/pubs/alive2-pldi21.pdf',
    'alive2-readme.md': 'https://raw.githubusercontent.com/AliveToolkit/alive2/master/README.md',
}


def main():
    directory = ROOT / '.tools/papers'
    directory.mkdir(parents=True, exist_ok=True)
    result = {'captured_utc': datetime.now(timezone.utc).isoformat(),
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'records': {},
              'scope': 'Primary bytes retrieved without execution; retrieval is not full-paper reading, reproduction, proof validation or a complete novelty search.'}
    for name, url in SOURCES.items():
        row = {'url': url}
        try:
            request = Request(url, headers={'User-Agent': 'Atlas bounded primary-source research'})
            with urlopen(request, timeout=30) as response:
                content = response.read(4 * 1024**2 + 1)
                if len(content) > 4 * 1024**2:
                    raise ValueError('Source exceeds fixed 4 MiB retrieval bound')
                row.update({'resolved_url': response.url, 'content_type': response.headers.get_content_type()})
            if name.endswith('.pdf') and not content.startswith(b'%PDF'):
                raise ValueError('Expected PDF header absent')
            target = directory / name
            if target.exists() and target.read_bytes() != content:
                raise ValueError('Different bytes already exist at this source cache name; preserve them')
            target.write_bytes(content)
            row.update({'status': 'retrieved', 'bytes': len(content), 'sha256': hashlib.sha256(content).hexdigest()})
        except Exception as error:
            row.update({'status': 'retrieval_error', 'error_type': type(error).__name__, 'message': str(error)[:500]})
        result['records'][name] = row
        print(json.dumps({'name': name, 'status': row['status'], 'bytes': row.get('bytes')}), flush=True)
    target = ROOT / 'docs/research/comparison-lineage-sources.json'
    target.write_bytes((json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n').encode())


if __name__ == '__main__':
    main()
