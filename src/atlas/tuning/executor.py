"""Trusted full-model worker. Timing excludes verification, never cache transfers."""
from __future__ import annotations

from contextlib import nullcontext
import gc
import os
import time

from .common import (ROOT, check_sources, file_digest, load_array, public_error,
                     read_json, runtime_identity, save_array, utc, write_json, observation)


class Cancelled(RuntimeError):
    pass


def configure(profile, dtype):
    import torch
    torch.set_num_threads(profile['threads'])
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for this registered runtime')
    if dtype == 'bfloat16' and not torch.cuda.is_bf16_supported():
        raise RuntimeError('This device does not support registered BF16 execution')
    return {'dtype': dtype, 'threads': torch.get_num_threads(),
            'interop_threads': torch.get_num_interop_threads(), 'tf32': torch.backends.cuda.matmul.allow_tf32,
            'matmul_precision': torch.get_float32_matmul_precision()}


def load_model(dtype):
    import torch
    from transformers import AutoTokenizer, Qwen3ForCausalLM
    path = ROOT / 'weights/qwen3-0.6b'
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    model = Qwen3ForCausalLM.from_pretrained(path, local_files_only=True, trust_remote_code=False,
                use_safetensors=True, attn_implementation='sdpa', dtype=getattr(torch, dtype)).cuda().eval()
    torch.cuda.synchronize()
    return model, tokenizer


def replay(model, prompt, steps, profile, check=lambda: None, teacher=None, capture=False):
    """Exactly steps outputs. Corpus teacher history is separate from free generation."""
    import numpy as np
    import torch
    from transformers.cache_utils import DynamicCache
    from atlas.attention.qwen import QwenAdapter
    from atlas.attention.runtime import PLANS
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    adapter = (QwenAdapter(model, PLANS['resident' if profile['method'] == 'resident' else 'hybrid50'],
               len(prompt) + steps, window=profile['window'] or None)
               if profile['method'] in ('resident', 'hybrid') else None)
    context = adapter if adapter is not None else nullcontext()
    result = past = logits = token = position = tokens = None
    outputs, samples, arrays = [], [], []
    try:
        with context, torch.inference_mode():
            past = DynamicCache(config=model.config, offloading=True) if profile['method'] == 'offloaded' else None
            tokens = torch.tensor([prompt], device='cuda', dtype=torch.long)
            position = torch.arange(len(prompt), device='cuda').unsqueeze(0)
            check()
            result = model(tokens, position_ids=position, past_key_values=past,
                           use_cache=adapter is None, logits_to_keep=1)
            torch.cuda.synchronize()
            prefill_ms = (time.perf_counter() - started) * 1000
            past, logits = result.past_key_values, result.logits[:, -1, :]
            for index in range(steps):
                check()
                if capture:
                    arrays.append(logits.float().cpu().numpy().copy()[0])
                value = int(logits.argmax(-1).item())
                outputs.append(value)
                if index == 0:
                    first_token_ms = (time.perf_counter() - started) * 1000
                if index == steps - 1:
                    break
                torch.cuda.synchronize()
                tick = time.perf_counter()
                value = int(teacher[index]) if teacher is not None else value
                token = torch.tensor([[value]], device='cuda', dtype=torch.long)
                position = torch.tensor([[len(prompt) + index]], device='cuda', dtype=torch.long)
                result = None
                result = model(token, position_ids=position, past_key_values=past,
                               use_cache=adapter is None, logits_to_keep=1)
                past, logits = result.past_key_values, result.logits[:, -1, :]
                torch.cuda.synchronize()
                samples.append((time.perf_counter() - tick) * 1000)
    finally:
        # Adapter exits before the timer ends; cache release and synchronization count.
        result = past = logits = token = position = tokens = None
        torch.cuda.synchronize()
    value = {'generation_ms': (time.perf_counter()-started)*1000, 'prefill_ms': prefill_ms,
             'first_token_ms': first_token_ms, 'samples_ms': samples, 'generated_token_ids': outputs,
             'allocated_peak_mib': torch.cuda.max_memory_allocated()/2**20,
             'reserved_peak_mib': torch.cuda.max_memory_reserved()/2**20}
    return value, np.stack(arrays).astype(np.float32) if capture else None


def run_worker(folder, case_id):
    import numpy as np
    import psutil
    start = time.perf_counter()
    run = folder / 'runs' / case_id
    config = read_json(run / 'config.json')
    doc = {'id': case_id, 'campaign': folder.name, 'config': config, 'status': 'running',
           'created_at': utc(), 'artifacts': {}, 'execution_process':
           {'pid': os.getpid(), 'created_at': psutil.Process().create_time()}}

    def save():
        doc['elapsed_s'] = time.perf_counter()-start
        write_json(run / 'result.json', doc)

    def check():
        if (folder / 'cancel').exists():
            raise Cancelled('Cancellation accepted between model operations')

    try:
        protocol = read_json(folder / 'protocol.json')
        check_sources(protocol['sources'])
        from .registry import assets
        if assets() != protocol['assets']:
            raise ValueError('Pinned model, tokenizer or public text bytes changed')
        identity = runtime_identity()
        if identity != protocol['identity']:
            raise RuntimeError('Hardware, driver or package identity changed')
        doc.update(protocol_sha256=file_digest(folder / 'protocol.json'),
                   environment={'source_sha256': protocol['sources'], 'identity': identity})
        doc['configured'] = configure(config['profile'], protocol['settings']['dtype'])
        source = next(i for i in protocol['inputs'] if i['id'] == config['input'])
        doc['input_artifact'] = source['artifact']
        ids = load_array(folder, source['artifact']).tolist()
        length, steps = protocol['settings']['context'], protocol['settings']['generation_tokens']
        check()
        loaded = time.perf_counter()
        model, _ = load_model(protocol['settings']['dtype'])
        doc['model_load_s'] = time.perf_counter()-loaded
        doc['model'] = {'class': type(model).__name__, 'parameter_count': sum(p.numel() for p in model.parameters()),
                        'parameter_dtypes': sorted({str(p.dtype) for p in model.parameters()})}
        doc['phase_times'] = {}
        doc['observations'] = {'before_execution': observation()}
        save()
        if config['stage'] == 'reference':
            _, logits = replay(model, ids[:length], 32, config['profile'], check,
                               teacher=ids[length:], capture=True)
            if not np.isfinite(logits).all():
                raise ValueError('Native reference contains nonfinite logits')
            doc['artifacts']['reference_logits'] = save_array(folder, logits)
            doc['observations']['after_reference'] = observation()
        else:
            phases = doc['phase_times']
            phases['warm_start'] = time.perf_counter()-start
            doc['warmup'], _ = replay(model, ids[:length], steps, config['profile'], check)
            phases['warm_end'] = time.perf_counter()-start
            doc['observations']['after_warmup'] = observation()
            save()
            phases['measure_start'] = time.perf_counter()-start
            doc['measurement'], _ = replay(model, ids[:length], steps, config['profile'], check)
            phases['measure_end'] = time.perf_counter()-start
            doc['observations']['after_measurement'] = observation()
            phases['validation_start'] = time.perf_counter()-start
            _, logits = replay(model, ids[:length], 32, config['profile'], check,
                               teacher=ids[length:], capture=True)
            phases['validation_end'] = time.perf_counter()-start
            doc['observations']['after_validation'] = observation()
            doc['artifacts']['candidate_logits'] = save_array(folder, logits)
            ref_path = folder / 'runs' / ('ref-' + config['input']) / 'result.json'
            reference = read_json(ref_path)
            if reference['status'] != 'completed':
                raise ValueError('Native reference is unavailable')
            doc['reference_sha256'] = file_digest(ref_path)
            native = load_array(folder, reference['artifacts']['reference_logits'])
            # Producer uses PyTorch; verifier reconstructs the relation separately in NumPy.
            import torch
            from atlas.attention.runtime import error_metrics
            doc['logit_checks'] = [error_metrics(torch.from_numpy(a), torch.from_numpy(b))
                                  for a, b in zip(logits, native, strict=True)]
            doc['correctness'] = {'passed': all(c['passed'] for c in doc['logit_checks'])}
            values = logits.astype(np.float64)
            maximum = values.max(axis=1)
            loss = np.log(np.exp(values-maximum[:, None]).sum(axis=1)) + maximum
            loss -= values[np.arange(32), ids[length:length+32]]
            doc['mean_nll'] = float(loss.mean()) if np.isfinite(loss).all() else None
        check()
        doc['status'] = 'completed'
    except BaseException as exc:
        import torch
        doc.update(status='cancelled' if isinstance(exc, Cancelled) else
                   'oom' if isinstance(exc, torch.cuda.OutOfMemoryError) else 'failed', error=public_error(exc))
    finally:
        doc['finished_at'] = utc()
        save()
    return doc
