"""Pre-outcome finite search registry and fixed workload identities."""
from __future__ import annotations

import hashlib
import importlib.metadata
import math
from pathlib import Path

from .common import (ROOT, BASE_PATH, check_sources, digest, file_digest, frozen_json,
                     load_array, read_json, runtime_identity, save_array, source_map)
from atlas.attention.portable import verify_files

DTYPES = ('float32', 'float16', 'bfloat16')
CONTEXTS = (512, 2048, 8192, 32768)
OUTPUTS = (16, 32, 64)
RULES = {'confirmation_repetitions': 3, 'minimum_ratio': 1.05,
         'worker_wall_s': 1200, 'minimum_available_ram_bytes': 2 * 1024**3,
         'target_shape_warmup': True, 'aggregation': 'geometric mean of two calibration windows',
         'pruning': 'None; preserve all registered calibration outcomes',
         'budget_scope': 'Maximum observed warmup/generation CUDA allocated peak; not a hard quota'}


def settings(value):
    allowed = {'model', 'dtype', 'context', 'generation_tokens', 'goal', 'budget_mib',
               'auto_apply', 'max_wall_s'}
    if set(value) - allowed:
        raise ValueError('Unregistered tuning option')
    result = {'model': 'qwen3-0.6b', 'dtype': 'float32', 'context': 2048,
              'generation_tokens': 16, 'goal': 'fastest', 'budget_mib': None,
              'auto_apply': True, 'max_wall_s': 14400, **value}
    if result['model'] != 'qwen3-0.6b' or result['dtype'] not in DTYPES:
        raise ValueError('Unsupported model/precision')
    if type(result['context']) is not int or result['context'] not in CONTEXTS:
        raise ValueError('Unsupported context')
    if type(result['generation_tokens']) is not int or result['generation_tokens'] not in OUTPUTS:
        raise ValueError('Unsupported fixed output length')
    if result['goal'] not in ('fastest', 'budget') or type(result['auto_apply']) is not bool:
        raise ValueError('Unsupported objective')
    if type(result['max_wall_s']) is not int or not 60 <= result['max_wall_s'] <= 21600:
        raise ValueError('Campaign wall bound must be 60..21600 seconds')
    budget = result['budget_mib']
    if result['goal'] == 'budget':
        if type(budget) not in (int, float) or not math.isfinite(budget) or not 512 <= budget <= 65536:
            raise ValueError('A finite allocation budget of 512..65536 MiB is required')
    elif budget is not None:
        raise ValueError('The fastest objective has no allocation budget')
    return result


def profiles(context):
    result = [{'id': f'native-t{n}', 'method': 'native', 'threads': n, 'window': 0}
              for n in (1, 2, 4, 6)]
    result += [{'id': 'resident-t4', 'method': 'resident', 'threads': 4, 'window': 0},
               {'id': 'offloaded-t4', 'method': 'offloaded', 'threads': 4, 'window': 0}]
    result += [{'id': f'hybrid256-t{n}', 'method': 'hybrid', 'threads': n, 'window': 256}
               for n in (1, 2, 4, 6)]
    if context > 1024:
        result.append({'id': 'hybrid1024-t4', 'method': 'hybrid', 'threads': 4, 'window': 1024})
    return result


def token_sha(values):
    import numpy as np
    return hashlib.sha256(np.asarray(values, dtype='<i8').tobytes()).hexdigest()


def assets(root=ROOT):
    expected = read_json(root / BASE_PATH)['assets']
    for key, relative in (('model', 'weights/qwen3-0.6b'), ('wikitext', 'data/wikitext-attention-v2')):
        folder = root / relative
        manifest = read_json(folder / 'atlas-manifest.json')
        if digest(manifest) != digest(expected[key]):
            raise ValueError('Pinned asset identity differs: ' + key)
        verify_files(folder, manifest['files_sha256'])
    return {key: expected[key] for key in ('model', 'wikitext')}


def freeze(folder, request, baseline='native-t4', expected_active=None, root=ROOT):
    import numpy as np
    import psutil
    from transformers import AutoTokenizer
    from atlas.attention.assets import corpus_text
    config = settings(request)
    sources = source_map(root)
    identity = runtime_identity()
    # No CUDA context belongs to the controller; only disposable model workers.
    import torch
    if torch.cuda.is_initialized():
        raise RuntimeError('Controller unexpectedly initialized CUDA')
    expected = {'torch': '2.14.1+cu130', 'transformers': '5.19.0', 'numpy': '2.5.4'}
    if any(importlib.metadata.version(name) != version for name, version in expected.items()):
        raise ValueError('Use the pinned Windows attention environment')
    pinned = assets(root)
    tokenizer = AutoTokenizer.from_pretrained(root / 'weights/qwen3-0.6b',
                        local_files_only=True, trust_remote_code=False)
    inputs = []
    for split, prefix in [('train', 'cal'), ('validation', 'hold')]:
        text, _ = corpus_text(split)
        ids = tokenizer(text, truncation=False, verbose=False).input_ids
        for number, offset in enumerate((5003, 5003 + config['context'] + 32 + 997)):
            values = ids[offset:offset + config['context'] + 32]
            if len(values) != config['context'] + 32:
                raise ValueError('Registered text window is unavailable')
            descriptor = save_array(folder, np.asarray(values, dtype='<i8'))
            inputs.append({'id': f'{prefix}-{number}', 'split': split, 'offset': offset,
                           'token_sha256': token_sha(values), 'artifact': descriptor})
    choices = profiles(config['context'])
    if baseline not in {p['id'] for p in choices}:
        baseline = 'native-t4'
    # Reserve an uncompressed upper bound; deduplication can save space, not evidence.
    max_cases = 4 + 2 * len(choices) + 4 + 12
    needed = (max_cases + 4) * 32 * 151936 * 4 + 128 * 1024**2
    import shutil
    if shutil.disk_usage(folder).free < needed:
        raise RuntimeError(f'Insufficient output space: need {needed // 2**20} MiB free')
    if psutil.virtual_memory().available < 4 * 1024**3:
        raise RuntimeError('Insufficient available system RAM for a bounded campaign')
    protocol = {'schema_version': 1, 'id': folder.name, 'settings': config, 'baseline': baseline,
                'expected_active': expected_active, 'identity': identity, 'sources': sources,
                'assets': pinned, 'inputs': inputs, 'profiles': choices, 'vocabulary_size': 151936,
                'contract': {'atol': 1e-4, 'rtol': 1e-4, 'positions': 32},
                'rules': RULES,
                'limits': ['One shared desktop and finite windows; no population confidence or universal optimum.',
                           '32-position literal relation is not broad answer quality or a guarantee for every future prompt.',
                           'Model startup, full-shape warmup and validation are outside scored generation and retained as costs.',
                           'Profile/array hashes prove consistency, not author honesty or independent execution.']}
    frozen_json(folder / 'protocol.json', protocol)
    return protocol


def verify_protocol(folder, root=ROOT, with_assets=False):
    protocol = read_json(folder / 'protocol.json')
    if protocol['schema_version'] != 1 or settings(protocol['settings']) != protocol['settings']:
        raise ValueError('Unsupported campaign protocol')
    if protocol['profiles'] != profiles(protocol['settings']['context']):
        raise ValueError('Profile registry changed')
    if protocol['contract'] != {'atol': 1e-4, 'rtol': 1e-4, 'positions': 32}:
        raise ValueError('Numerical contract changed')
    if protocol['rules'] != RULES or protocol['vocabulary_size'] != 151936:
        raise ValueError('Predeclared decision or workload rules changed')
    if protocol['id'] != folder.name or protocol['baseline'] not in {p['id'] for p in protocol['profiles']}:
        raise ValueError('Campaign/baseline identity differs')
    expected_inputs = [(f'{prefix}-{n}', split, offset) for split, prefix in
                       [('train', 'cal'), ('validation', 'hold')]
                       for n, offset in enumerate((5003, 5003 + protocol['settings']['context'] + 32 + 997))]
    if [(i['id'], i['split'], i['offset']) for i in protocol['inputs']] != expected_inputs:
        raise ValueError('Separate fixed selection/confirmation inputs are required')
    check_sources(protocol['sources'], root)
    if with_assets:
        if assets(root) != protocol['assets']:
            raise ValueError('Campaign assets changed')
    for item in protocol['inputs']:
        values = load_array(folder, item['artifact'])
        if values.shape != (protocol['settings']['context'] + 32,) or values.dtype != '<i8':
            raise ValueError('Registered input shape/type differs')
        if token_sha(values) != item['token_sha256'] or values.min() < 0 or values.max() >= 151936:
            raise ValueError('Registered token identity differs')
    return protocol


def calibration_cases(protocol):
    import random
    result = []
    for n in range(2):
        choices = list(protocol['profiles'])
        random.Random(2026).shuffle(choices)
        if n:
            choices.reverse()
        for profile in choices:
            result.append({'id': f'cal-{n}-{profile["id"]}', 'stage': 'calibration',
                           'input': f'cal-{n}', 'profile': profile, 'repetition': 0})
    return result


def reference_cases(prefix):
    return [{'id': f'ref-{prefix}-{n}', 'stage': 'reference', 'input': f'{prefix}-{n}',
             'profile': {'id': 'native-t4', 'method': 'native', 'threads': 4, 'window': 0},
             'repetition': 0} for n in range(2)]


def confirmation_cases(protocol, nominated):
    choices = {p['id']: p for p in protocol['profiles']}
    result = []
    for n in range(2):
        for name in ('native-t4', 'resident-t4'):
            result.append({'id': f'guard-{n}-{name}', 'stage': 'guard', 'input': f'hold-{n}',
                           'profile': choices[name], 'repetition': 0})
        for rep in range(3):
            order = list(dict.fromkeys([protocol['baseline'], nominated]))
            if (n + rep) % 2:
                order.reverse()
            for name in order:
                result.append({'id': f'confirm-{n}-{rep}-{name}', 'stage': 'confirmation',
                               'input': f'hold-{n}', 'profile': choices[name], 'repetition': rep})
    return result
