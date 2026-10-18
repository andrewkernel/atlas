"""Real long-context execution. Arrays, failures and unknowns are retained."""
from __future__ import annotations

from contextlib import nullcontext
import gc
import importlib.metadata
import os
import statistics
import subprocess
import threading
import time

import numpy as np
import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .long_protocol import ROOT, input_ids, token_digest, verify
from .oracle import distribution_metrics
from .portable import file_digest
from .qwen import QwenAdapter
from .runtime import PLANS, error_metrics


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()


class Observations:
    """Portable passive observations; missing values carry a reason, never zero."""
    def __init__(self):
        self.rows, self.phase = [], 'setup'
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.sample, daemon=True)
        self.start = time.perf_counter()

    def sample(self):
        process = psutil.Process()
        while not self.stop.is_set():
            row = {'elapsed_s':time.perf_counter()-self.start, 'phase':self.phase,
                   'rss_mib':process.memory_info().rss/2**20,
                   'available_ram_mib':psutil.virtual_memory().available/2**20}
            try:
                options = {'creationflags':0x08000000} if os.name=='nt' else {}
                values = subprocess.check_output(['nvidia-smi', '--id=0',
                    '--query-gpu=memory.used,utilization.gpu,temperature.gpu',
                    '--format=csv,noheader,nounits'], text=True, timeout=3, **options).strip().split(',')
                row.update(board_used_mib=float(values[0]), utilization_percent=float(values[1]), temperature_c=float(values[2]))
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                row['board_observation_error'] = type(exc).__name__ + ': ' + str(exc)[:300]
            self.rows.append(row)
            self.stop.wait(.5)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join(timeout=4)


def replay(model, prompt, targets, method, window, rec, generation=False):
    steps = targets.shape[1]
    context = prompt.shape[1]
    adapter = None if method=='native' else QwenAdapter(model, PLANS[method], context+steps, window=window)
    outputs, generated, samples = [], [], []
    result = None
    with adapter if adapter else nullcontext():
        try:
            positions = torch.arange(context, device='cuda').unsqueeze(0)
            torch.cuda.synchronize()
            start = time.perf_counter()
            result = model(prompt, use_cache=adapter is None, position_ids=positions, logits_to_keep=1)
            torch.cuda.synchronize()
            prefill_ms = (time.perf_counter()-start)*1000
            for step in range(steps):
                rec.check()
                if generation:
                    token = result.logits[:,-1,:].argmax(-1, keepdim=True)
                    generated.append(int(token.item()))
                else:
                    outputs.append(result.logits[:,-1,:].float().cpu().numpy().copy())
                    token = targets[:,step:step+1]
                if step < steps-1:
                    torch.cuda.synchronize()
                    tick = time.perf_counter()
                    result = model(token, past_key_values=result.past_key_values,
                                   use_cache=adapter is None,
                                   position_ids=torch.tensor([[context+step]], device='cuda'),
                                   logits_to_keep=1)
                    torch.cuda.synchronize()
                    samples.append((time.perf_counter()-tick)*1000)
            torch.cuda.synchronize()
            measured = {'generation_ms':(time.perf_counter()-start)*1000,
                        'prefill_ms':prefill_ms, 'samples_ms':samples,
                        'median_ms':statistics.median(samples),
                        'allocated_peak_mib':torch.cuda.max_memory_allocated()/2**20,
                        'reserved_peak_mib':torch.cuda.max_memory_reserved()/2**20,
                        'generated_token_ids':generated}
            return outputs, measured
        finally:
            del result


def save_array(rec, name, values):
    path = rec.directory/(name+'.npy')
    np.save(path, np.asarray(values), allow_pickle=False)
    artifact = {'path':path.name, 'sha256':file_digest(path),
                'shape':list(np.asarray(values).shape), 'dtype':str(np.asarray(values).dtype)}
    rec.doc.setdefault('artifacts', {})[name] = artifact
    rec.save()
    return artifact


def run_validation(rec):
    config = rec.doc['config']
    protocol = verify(config)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; do not replace this study with simulated timings')
    torch.set_num_threads(config['threads'])
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    rec.doc.update(schema_version=3, protocol_sha256=file_digest(ROOT/'docs/research/attention-v3/protocol.json'),
                   contract=protocol['contract'], limitations=protocol['limits'],
                   method_order=config['methods'], artifacts={},
                   configured_precision={'dtype':'float32', 'tf32':torch.backends.cuda.matmul.allow_tf32,
                                         'matmul_precision':torch.get_float32_matmul_precision()},
                   validation_scope='32 full-vocabulary teacher-forced positions; not broad language quality')
    rec.doc['environment']['packages'].update({p:importlib.metadata.version(p) for p in ('pyarrow','safetensors','tokenizers','filelock')})
    folder = ROOT/'weights/qwen3-0.6b'
    tokenizer = AutoTokenizer.from_pretrained(folder, local_files_only=True, trust_remote_code=False)
    ids = input_ids(config, tokenizer)
    if token_digest(ids) != config['input_token_sha256']:
        raise ValueError('Actual token history differs from frozen protocol')
    save_array(rec, 'input_ids', np.asarray(ids, dtype='<i8'))
    prompt = torch.tensor([ids[:config['context']]], device='cuda', dtype=torch.long)
    targets = torch.tensor([ids[config['context']:]], device='cuda', dtype=torch.long)
    model = AutoModelForCausalLM.from_pretrained(folder, local_files_only=True, trust_remote_code=False,
              use_safetensors=True, attn_implementation='sdpa', dtype=torch.float32).cuda().eval()
    rec.doc['model'] = {'repo':protocol['assets']['model']['repo'], 'revision':protocol['assets']['model']['revision'],
                        'parameter_count':sum(p.numel() for p in model.parameters())}
    with Observations() as observation:
        try:
            # Same short warmup for every arm. Costs are retained, outside long timing.
            rec.doc['warmup_s'] = {}
            for method in config['methods']:
                tick = time.perf_counter()
                replay(model, prompt[:,:128], targets[:,:2], method, config['window'], rec)
                cleanup()
                rec.doc['warmup_s'][method] = time.perf_counter()-tick
            observation.phase = 'native-reference/validation'
            rec.progress(0, 3, 'Full native reference / '+str(config['context'])+' tokens')
            reference, _ = replay(model, prompt, targets, 'native', config['window'], rec)
            reference = np.concatenate(reference, axis=0)
            save_array(rec, 'reference_logits', reference)
            cleanup()
            for number, method in enumerate(config['methods']):
                rec.progress(number, 3, f'{config["context"]:,} tokens / {method} / generation and validation')
                row = {'plan':method, 'label':'Hybrid / 256 GPU tokens' if method=='hybrid50' else method,
                       'phase':'model', 'context':config['context'], 'status':'running',
                       'correctness':{'passed':None}, 'eligibility':{'eligible':False, 'reason':'Validation incomplete'}}
                rec.doc['rows'].append(row)
                rec.save()
                try:
                    cleanup()
                    torch.cuda.reset_peak_memory_stats()
                    observation.phase = method+'/generation'
                    _, measured = replay(model, prompt, targets[:,:config['generation_tokens']], method, config['window'], rec, generation=True)
                    row.update(measured)
                    cleanup()
                    observation.phase = method+'/validation'
                    actual, _ = replay(model, prompt, targets, method, config['window'], rec)
                    actual = np.concatenate(actual, axis=0)
                    save_array(rec, method+'_logits', actual)
                    checks = [error_metrics(torch.from_numpy(a), torch.from_numpy(r),
                                            atol=protocol['contract']['atol'], rtol=protocol['contract']['rtol'])
                              for a, r in zip(actual, reference, strict=True)]
                    diagnostics = [distribution_metrics(a, r, ids[config['context']+step])
                                   for step, (a, r) in enumerate(zip(actual, reference, strict=True))]
                    row.update(status='completed', logit_checks=checks, distribution_checks=diagnostics,
                               correctness={'passed':all(c['passed'] for c in checks),
                                            'finite':all(c['finite'] for c in checks),
                                            'max_abs':max((c['max_abs'] for c in checks if c['max_abs'] is not None), default=None),
                                            'violating_elements':sum(c['violating_elements'] or 0 for c in checks)},
                               mean_nll=statistics.mean(d['candidate_nll'] for d in diagnostics) if all(d['finite'] for d in diagnostics) else None)
                    del actual
                except torch.OutOfMemoryError as exc:
                    row.update(status='oom', error=str(exc)[:1200])
                finally:
                    cleanup()
                    rec.save()
            controls_pass = all(r['status']=='completed' and r['correctness']['passed'] is True
                                for r in rec.doc['rows'] if r['plan'] in ('native','resident'))
            for row in rec.doc['rows']:
                passed = controls_pass and row['status']=='completed' and row['correctness']['passed'] is True
                row['eligibility'] = {'eligible':passed,
                                      'reason':'All 32 literal logit checks and both controls passed' if passed else 'Numerical/control failure or unavailable result'}
            rec.progress(3, 3, 'Long-context replay complete; inspect numerical status separately')
        finally:
            rec.doc['observations'] = observation.rows
            rec.save()
