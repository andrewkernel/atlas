"""Check source-bound data and a first-load byte budget; no foreign code or ML runtime."""
import gzip
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import sys
from build_snapshot import make_snapshot

FRONTEND = Path(__file__).resolve().parents[1]
INITIAL = ['index.html', 'app.css', 'app.js', 'data/snapshot.json', 'assets/atlas.svg', 'assets/geist.woff2']


class Assets(HTMLParser):
    def __init__(self):
        super().__init__()
        self.assets = []
    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if tag in {'script','img'} and values.get('src'):
            self.assets.append(values['src'])
        if tag == 'link' and values.get('href'):
            self.assets.append(values['href'])


def check():
    failures = []
    recorded = json.loads((FRONTEND/'data/snapshot.json').read_text(encoding='utf-8'))
    if recorded != make_snapshot():
        failures.append('Dashboard snapshot is stale; regenerate from recorded source inputs.')
    parser = Assets()
    parser.feed((FRONTEND/'index.html').read_text(encoding='utf-8'))
    for name in parser.assets:
        if not (FRONTEND/name).is_file():
            failures.append('Missing page asset: '+name)
    font = json.loads((FRONTEND/'assets/font-source.json').read_text(encoding='utf-8'))
    for entry in font['files']:
        data = (FRONTEND/'assets'/entry['file']).read_bytes()
        if hashlib.sha256(data).hexdigest() != entry['sha256'] or len(data) != entry['bytes']:
            failures.append('Font/license identity mismatch: '+entry['file'])
    sizes = {name:(FRONTEND/name).stat().st_size for name in INITIAL}
    total = sum(sizes.values())
    if total > 128_000:
        failures.append('First-load source assets exceed the 128,000-byte budget.')
    if len(recorded['configurations']) != 8 or sum(c['completed']*c['updatesPerProcess'] for c in recorded['configurations']) != 128:
        failures.append('Resource configuration/update denominator changed.')
    if len(recorded['environments']) != 8 or sum(e['checkedSources'] for e in recorded['environments']) != 54:
        failures.append('Environment coverage changed.')
    result = {'first_load_source_bytes':total,'budget_bytes':128_000,'asset_bytes':sizes,
              'gzip_estimate_bytes':sum(len(gzip.compress((FRONTEND/name).read_bytes(),mtime=0)) for name in INITIAL),
              'scope':'Uncompressed local source bytes and gzip estimate; not observed network transfer, LCP or an internet-latency guarantee.', 'failures':failures}
    print(json.dumps(result,indent=2))
    return bool(failures)


if __name__ == '__main__':
    sys.exit(check())
