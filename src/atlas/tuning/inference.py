"""Apply compatible local settings to a real request; keep prompt records private."""
from __future__ import annotations

import time

from .common import (ROOT, STORE, campaign_dir, domain, identifier, load_array, public_error,
                     read_json, save_array, utc, write_json)
from . import profiles


def run(ident, home=STORE):
    from filelock import FileLock
    from .executor import Cancelled, configure, load_model, replay
    from .registry import settings
    folder = home / 'private-inference' / identifier(ident)
    request = read_json(folder / 'request.json')
    config = settings(request['settings'])
    start = time.perf_counter()
    doc = {'id': ident, 'status': 'running', 'created_at': utc(), 'settings': config,
           'profile_numerical_evidence': 'unknown', 'request_numerical_check': 'unknown',
           'private': True, 'artifacts': {}}

    def save():
        doc['total_request_s'] = time.perf_counter()-start
        write_json(folder / 'result.json', doc)

    def check():
        if (folder / 'cancel').exists():
            raise Cancelled('Request cancelled between model operations')

    save()
    try:
        with FileLock(str(ROOT / 'artifacts/attention-lab/.gpu.lock'), timeout=0):
            active = profiles.state(domain(config), home)
            doc['active_state_at_start'] = active
            profile = None
            if active['profile']:
                try:
                    profile = profiles.validate(active['profile'], home)
                    if profile['settings']['goal'] != config['goal'] or profile['settings']['budget_mib'] != config['budget_mib']:
                        raise ValueError('Active profile was validated for a different objective/budget')
                    doc['profile_numerical_evidence'] = 'passed finite campaign windows'
                except (ValueError, OSError, RuntimeError) as exc:
                    doc['fallback_reason'] = public_error(exc)
                    profile = None
            if config['goal'] == 'budget' and profile is None:
                doc.update(status='abstained', error='No compatible verified profile qualifies for this allocation target')
                return doc
            chosen = profile['execution'] if profile else {'id': 'native-t4', 'method': 'native', 'threads': 4, 'window': 0}
            doc.update(applied_profile=active['profile'] if profile else None, execution=chosen,
                       native_fallback=profile is None)
            doc['configured'] = configure(chosen, config['dtype'])
            check()
            loading = time.perf_counter()
            model, tokenizer = load_model(config['dtype'])
            doc['model_load_s'] = time.perf_counter()-loading
            prompt = request.get('prompt')
            if prompt is not None:
                ids = tokenizer(prompt, truncation=False, verbose=False).input_ids
                if len(ids) != config['context']:
                    raise ValueError(f'Prompt has {len(ids)} tokens; this exact profile requires {config["context"]}. No truncation performed')
                doc['prompt'] = prompt
            else:
                source = campaign_dir(request['example_campaign'], home)
                protocol = read_json(source / 'protocol.json')
                if domain(protocol['settings']) != domain(config):
                    raise ValueError('Registered example does not match the requested workload')
                ids = load_array(source, next(i['artifact'] for i in protocol['inputs'] if i['id'] == 'hold-0')).tolist()[:config['context']]
                doc['registered_example'] = {'campaign': source.name, 'input': 'hold-0'}
            doc['artifacts']['input_ids'] = save_array(folder, __import__('numpy').asarray(ids, dtype='<i8'))
            doc['warmup'], _ = replay(model, ids, config['generation_tokens'], chosen, check)
            doc['measurement'], _ = replay(model, ids, config['generation_tokens'], chosen, check)
            doc['output_token_ids'] = doc['measurement']['generated_token_ids']
            doc['text'] = tokenizer.decode(doc['output_token_ids'])
            save()
            if request.get('check_native', False):
                checking = time.perf_counter()
                native = {'id': 'native-t4', 'method': 'native', 'threads': 4, 'window': 0}
                import torch
                from atlas.attention.runtime import error_metrics
                torch.set_num_threads(4)
                reference_gen, expected = replay(model, ids, 32, native, check, capture=True)
                torch.set_num_threads(chosen['threads'])
                _, actual = replay(model, ids, 32, chosen, check,
                                   teacher=reference_gen['generated_token_ids'], capture=True)
                doc['artifacts']['reference_logits'] = save_array(folder, expected)
                doc['artifacts']['candidate_logits'] = save_array(folder, actual)
                checks = [error_metrics(torch.from_numpy(a), torch.from_numpy(b)) for a, b in zip(actual, expected, strict=True)]
                # Fresh request also receives an independent arithmetic check.
                from .verify import compare, check_saved_checks
                independent = compare(actual, expected)
                check_saved_checks(independent, checks)
                passed = all(c['passed'] for c in independent)
                doc.update(logit_checks=checks, request_numerical_check='passed' if passed else 'failed',
                           native_check_s=time.perf_counter()-checking,
                           check_scope='32 full-vocabulary positions on native greedy history; finite relation, not answer quality')
                if not passed:
                    if config['goal'] == 'budget':
                        doc.update(status='abstained', text=None,
                                   error='Fresh check failed; no fallback above the allocation target is allowed')
                        return doc
                    doc['rejected_candidate_text'] = doc['text']
                    torch.set_num_threads(4)
                    doc['fallback_measurement'], _ = replay(model, ids, config['generation_tokens'], native, check)
                    doc.update(text=tokenizer.decode(doc['fallback_measurement']['generated_token_ids']),
                               output_token_ids=doc['fallback_measurement']['generated_token_ids'], native_fallback=True,
                               fallback_reason='Candidate failed the fresh literal numerical relation', execution=native)
            peak = max(doc['warmup']['allocated_peak_mib'], doc['measurement']['allocated_peak_mib'])
            if config['goal'] == 'budget' and peak > config['budget_mib']:
                doc.update(status='abstained', text=None, error='Actual request exceeded the measured allocation target')
                return doc
            check()
            doc['status'] = 'completed'
    except BaseException as exc:
        doc.update(status='cancelled' if isinstance(exc, Cancelled) else 'failed', error=public_error(exc))
    finally:
        doc['finished_at'] = utc()
        save()
    return doc
