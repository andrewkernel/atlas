"""Small PG-19 subset from the original public bucket; never executes dataset code."""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen

from .portable import file_digest, read_json, verify_files, write_json

ROOT = Path(__file__).resolve().parents[3]
DIRECTORY = ROOT / 'data/pg19-attention-v3'
LIST_URL = ('https://storage.googleapis.com/storage/v1/b/deepmind-gutenberg/o'
            '?prefix=validation%2F&maxResults=1000')


def fetch(url: str) -> bytes:
    with urlopen(url, timeout=60) as response:
        data = response.read(16 * 1024**2 + 1)
    if len(data) > 16 * 1024**2:
        raise ValueError('PG-19 file exceeds bounded download size')
    return data


def prepare_books() -> dict:
    manifest_path = DIRECTORY / 'atlas-manifest.json'
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        verify_files(DIRECTORY, manifest['files_sha256'])
        return manifest
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(ROOT/'weights/qwen3-0.6b',
                                              local_files_only=True, trust_remote_code=False)
    listing = json.loads(fetch(LIST_URL))
    if listing.get('nextPageToken'):
        raise ValueError('Unexpected truncated PG-19 listing')
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    # This selection reads length only. Model inference occurs after protocol freeze.
    considered, selected, files = [], [], {}
    for item in sorted(listing['items'], key=lambda x: x['name']):
        name = item['name']
        if not name.startswith('validation/') or not name.endswith('.txt'):
            continue
        book_id = name.split('/')[-1][:-4]
        if not book_id.isdigit():
            raise ValueError('Unexpected book name')
        url = ('https://storage.googleapis.com/deepmind-gutenberg/' + quote(name)
               + '?generation=' + item['generation'])
        data = fetch(url)
        actual_md5 = base64.b64encode(hashlib.md5(data).digest()).decode('ascii')
        if len(data) != int(item['size']) or actual_md5 != item['md5Hash']:
            raise ValueError('Original bucket checksum mismatch')
        path = DIRECTORY / (book_id + '.txt')
        path.write_bytes(data)
        ids = tokenizer(data.decode('utf-8'), truncation=False, verbose=False).input_ids
        eligible = len(ids) >= 40032
        entry = {'book_id':book_id, 'url':url, 'generation':item['generation'],
                 'sha256':file_digest(path), 'token_count':len(ids), 'selected':eligible}
        considered.append(entry)
        files[path.name] = entry['sha256']
        if eligible:
            selected.append(book_id)
        if len(selected) == 6:
            break
    if len(selected) != 6:
        raise ValueError('Insufficient eligible books; do not substitute after outcomes')
    manifest = {'dataset':'PG-19', 'split':'validation', 'source':LIST_URL,
                'homepage':'https://github.com/google-deepmind/pg19',
                'license_as_declared_upstream':'Apache-2.0',
                'selection':'First six lexicographically ordered original validation files with >=40032 Qwen3 tokens; length only, before model outcomes',
                'limitations':'Historical books have corpus/style biases. Not broad language quality or word-level PG-19 perplexity.',
                'tokenizer_revision':'c1899de289a04d12100db370d81485cdf75e47ca',
                'selected':selected, 'considered':considered, 'files_sha256':files}
    write_json(manifest_path, manifest)
    return manifest


def restore_books(manifest: dict) -> None:
    """Download the already-frozen subset without relisting or reselecting it."""
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    for item in manifest['considered']:
        path = DIRECTORY / (item['book_id'] + '.txt')
        if not path.exists() or file_digest(path) != item['sha256']:
            data = fetch(item['url'])
            if hashlib.sha256(data).hexdigest() != item['sha256']:
                raise ValueError('Pinned PG-19 bytes no longer available')
            path.write_bytes(data)
    verify_files(DIRECTORY, manifest['files_sha256'])
    write_json(DIRECTORY / 'atlas-manifest.json', manifest)
