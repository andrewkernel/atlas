"""Read-only HTTP/title checks for each paper note's first primary link.

Availability and title checks do not establish scientific correctness. Bot/error
pages are recorded, not bypassed. No external code or PDF is executed.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


class Title(HTMLParser):
    def __init__(self):
        super().__init__()
        self.inside=False
        self.parts=[]
        self.citation_metadata={}

    def handle_starttag(self,tag,attrs):
        if tag.lower()=='title':
            self.inside=True
        if tag.lower()=='meta':
            attributes=dict(attrs)
            name=attributes.get('name','').lower()
            if name in ('citation_author','citation_title','citation_date','citation_publication_date'):
                self.citation_metadata.setdefault(name,[]).append(attributes.get('content'))

    def handle_endtag(self,tag):
        if tag.lower()=='title':
            self.inside=False

    def handle_data(self,data):
        if self.inside:
            self.parts.append(data)


def inspect(note: Path) -> dict:
    source=note.read_text(encoding='utf-8')
    urls=re.findall(r'https://[^\s)>]+',source)
    url=urls[0].rstrip('.,') if urls else None
    record={'note':note.name,'note_heading':source.splitlines()[0].lstrip('# '),
            'note_sha256_at_retrieval':hashlib.sha256(note.read_bytes()).hexdigest(),'requested_url':url}
    if url is None:
        return {**record,'status':'no_link'}
    try:
        with urlopen(url,timeout=15) as response:
            body=response.read(2_000_000)
            record.update({'http_status':response.status,'resolved_url':response.geturl(),'content_type':response.headers.get('Content-Type')})
        if body.startswith(b'%PDF'):
            record.update({'status':'pdf_header_retrieved','title':None})
        else:
            parser=Title()
            parser.feed(body.decode('utf-8',errors='replace'))
            title=html.unescape(' '.join(''.join(parser.parts).split()))
            record.update({'status':'html_title_retrieved' if title else 'no_html_title','title':title,'citation_metadata':parser.citation_metadata})
    except (HTTPError,URLError,TimeoutError,OSError) as error:
        record.update({'status':'retrieval_unresolved','error_type':type(error).__name__,'http_status':getattr(error,'code',None)})
    return record


def run(output: Path) -> None:
    notes=sorted(path for path in Path('docs/research/papers').glob('*.md') if path.name!='README.md')
    with ThreadPoolExecutor(max_workers=4) as executor:
        results=list(executor.map(inspect,notes))
    record={'schema_version':'1.0','retrieved_utc':datetime.now(timezone.utc).isoformat(),
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'notes':results,
            'limitation':'First linked primary URL per note; HTTP/title availability only. Failed/bot retrieval is not proof of a missing source. Title agreement does not verify paper claims, author lists or publication status.'}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps([{key:item.get(key) for key in ('note','status','title','http_status')} for item in results],indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    run(args.output)
