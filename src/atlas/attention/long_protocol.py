"""Separate frozen registry for long-context correctness and researcher handoff."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import itertools
from pathlib import Path

from .portable import digest, file_digest, read_json, verify_files, write_json

ROOT = Path(__file__).resolve().parents[3]
DIRECTORY = ROOT / 'docs/research/attention-v3'
PROTOCOL = DIRECTORY / 'protocol.json'
ID = 'atlas-long-context-v3-01'


def utc():
    return datetime.now(timezone.utc).isoformat()


def input_ids(case, tokenizer):
    if case['cohort'] == 'v2-replay':
        from .assets import corpus_text
        text, _ = corpus_text('test')
    else:
        text = (ROOT/'data/pg19-attention-v3'/(case['book_id']+'.txt')).read_bytes().decode('utf-8')
    ids = tokenizer(text, truncation=False, verbose=False).input_ids
    begin, end = case['offset'], case['offset'] + case['context'] + case['validation_tokens']
    if end > len(ids):
        raise ValueError('Registered token range unavailable')
    return ids[begin:end]


def token_digest(ids):
    # Explicit little endian int64, independent of native NumPy/platform byte order.
    import struct
    return hashlib.sha256(b''.join(struct.pack('<q', i) for i in ids)).hexdigest()


def freeze():
    if PROTOCOL.exists():
        raise ValueError('Preserve frozen protocol; changes require a new version')
    from transformers import AutoTokenizer
    from .long_assets import DIRECTORY as BOOKS
    books = read_json(BOOKS/'atlas-manifest.json')
    model = read_json(ROOT/'weights/qwen3-0.6b/atlas-manifest.json')
    corpus = read_json(ROOT/'data/wikitext-attention-v2/atlas-manifest.json')
    verify_files(BOOKS, books['files_sha256'])
    verify_files(ROOT/'weights/qwen3-0.6b', model['files_sha256'])
    verify_files(ROOT/'data/wikitext-attention-v2', corpus['files_sha256'])
    tokenizer = AutoTokenizer.from_pretrained(ROOT/'weights/qwen3-0.6b', local_files_only=True, trust_remote_code=False)
    cases = []
    orders = list(itertools.permutations(['native', 'resident', 'hybrid50']))
    for cohort, count in [('v2-replay', 3), ('pg19-books', 6)]:
        for block in range(count):
            for context in (32768, 40000):
                case = {'kind':'long-validation', 'case_id':f'long-{len(cases):03d}',
                        'cohort':cohort, 'block':block, 'context':context,
                        'offset':5003 + 997*block if cohort=='v2-replay' else 0,
                        'book_id':books['selected'][block] if cohort=='pg19-books' else None,
                        'dtype':'float32', 'threads':4, 'window':256,
                        'validation_tokens':32, 'generation_tokens':16,
                        'methods':list(orders[block]), 'protocol_id':ID}
                case['input_token_sha256'] = token_digest(input_ids(case, tokenizer))
                cases.append(case)
    files = list((ROOT/'src/atlas/attention').glob('*.py'))
    files += [ROOT/'lab.py', ROOT/'benchmarks/attention_v3_study.py',
              ROOT/'benchmarks/verify_attention_v3.py', ROOT/'benchmarks/windows_job_scope.py',
              ROOT/'configs/attention-v2-windows-py312-requirements.txt',
              ROOT/'configs/attention-v2-windows-py312-freeze.txt']
    doc = {'schema_version':3, 'id':ID, 'created_at':utc(),
           'vocabulary_size':read_json(ROOT/'weights/qwen3-0.6b/config.json')['vocab_size'],
           'sources':{p.relative_to(ROOT).as_posix():file_digest(p) for p in sorted(files)},
           'assets':{'model':model, 'wikitext':corpus, 'books':books},
           'contract':{'atol':1e-4, 'rtol':1e-4, 'relation':'Every full-vocabulary logit element on identical teacher-forced histories; all 32 positions must pass'},
           'cases':cases,
           'analysis':{'unit':'process case; pair context/method within document, never treat tokens as independent',
                       'cohorts':'Report retrospective WikiText separately from six distinct PG-19 books. Contexts within a book are nested.',
                       'timing':'One free-generation window per arm after identical 128-token warmup. Prefill, greedy selection, input construction, cache updates/transfers and synchronized forwards included. Separate numerical replay excluded.',
                       'control':'Native validation replay compared with unchanged native repeat and persistent native SDPA. All controls must pass to qualify a candidate.',
                       'aggregation':'Retain all registered case outcomes. Paired ratios only where both timings completed; promote no correctness-qualified ratio with missing/failed controls. Descriptive ranges/medians, no population inference.',
                       'uncertainty':'Six book blocks are one shared machine/session; no independent human replication, confidence coverage guarantee or universal language equivalence.',
                       'failure':'No post-outcome input replacement, bound changes or selective reruns. New IDs retain attempts. Unknown is never pass.'},
           'limits':['Self-produced evidence; hashes detect byte changes, not fabricated execution.',
                     'One model family and FP32. Prior low-precision failures remain unresolved.',
                     'Historical-book selection and length filtering create corpus bias.',
                     'No CXL/DSA hardware, exclusive VRAM guarantee, broad quality, energy or novelty claim.',
                     'Native reference always precedes tested arms; all six arm orders balanced only in PG-19 cohort.']}
    write_json(PROTOCOL, doc)
    return doc


def verify(config=None):
    doc = read_json(PROTOCOL)
    if config is not None and config not in doc['cases']:
        raise ValueError('Unregistered or modified long-context case')
    verify_files(ROOT, doc['sources'])
    locations = {'model':'weights/qwen3-0.6b', 'wikitext':'data/wikitext-attention-v2', 'books':'data/pg19-attention-v3'}
    for key, relative in locations.items():
        # Manifest formatting is deliberately not an asset identity.
        manifest = read_json(ROOT/relative/'atlas-manifest.json')
        if digest(manifest) != digest(doc['assets'][key]):
            raise ValueError('Semantic asset manifest mismatch: ' + key)
        verify_files(ROOT/relative, manifest['files_sha256'])
    return doc
