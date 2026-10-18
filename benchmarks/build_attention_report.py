"""Independently reconstruct selected saved measurements; publish small evidence.

Does not run any GPU work. All reported numbers are derived from raw samples.
"""
from pathlib import Path
import argparse
import hashlib
import json
import statistics

ROOT=Path(__file__).resolve().parents[1]
RUNS=ROOT/'artifacts/attention-lab'
OUTPUT=ROOT/'docs/research/attention-lab'


def validate(doc):
    assert doc['status']=='completed'
    for row in doc['rows']:
        assert len(row['samples_ms'])>=5
        assert abs(statistics.median(row['samples_ms'])-row['median_ms'])<1e-10
        xs=sorted(row['samples_ms']);pos=(len(xs)-1)*.9;i=int(pos)
        p90=xs[i]+(xs[min(i+1,len(xs)-1)]-xs[i])*(pos-i)
        assert abs(p90-row['p90_ms'])<1e-10
        if 'logit_checks' in row:
            assert row['correctness']['passed']==all(c['passed'] for c in row['logit_checks'])
            assert len(row['logit_checks'])==doc['config']['new_tokens']+1
            assert row['correctness']['violating_elements']==sum(c['violating_elements'] or 0 for c in row['logit_checks'])
        elif row['correctness']['finite']:
            assert row['correctness']['passed']==(row['correctness']['violating_elements']==0)
    for e in doc.get('evaluation',[]):
        matches=[r for r in doc['rows'] if r['phase']=='validation' and r['context']==e['context']]
        chosen=next(r for r in matches if r['plan']==e['selected'])
        best=next(r for r in matches if r['plan']==e['measured_best'])
        assert abs(e['selected_ms']-chosen['median_ms'])<1e-10
        assert abs(e['regret_percent']-(chosen['median_ms']/best['median_ms']-1)*100)<1e-10


def build(ids):
    OUTPUT.mkdir(parents=True,exist_ok=True)
    (OUTPUT/'runs').mkdir(exist_ok=True)
    docs=[];manifest=[]
    for ident in ids:
        assert all(c in 'abcdefghijklmnopqrstuvwxyz0123456789-' for c in ident)
        data=(RUNS/ident/'result.json').read_bytes()
        doc=json.loads(data);validate(doc);docs.append(doc)
        (OUTPUT/'runs'/f'{ident}.json').write_bytes(data)
        manifest.append({'id':ident,'sha256':hashlib.sha256(data).hexdigest(),'rows':len(doc['rows'])})
    (OUTPUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    benchmarks=[d for d in docs if d['config']['kind']=='benchmark' and d['config'].get('kv_budget_mib',2048)==2048]
    budget=[d for d in docs if d['config']['kind']=='benchmark' and d['config'].get('kv_budget_mib',2048)<2048]
    models=[d for d in docs if d['config']['kind']=='model']
    lines=['# Atlas Attention Lab: measured engineering study','',
      'The end-to-end prototype runs persistent exact dense CPU/GPU attention, calibrated selection and pretrained GPT-2 inference behind a local interactive dashboard. This is an exploratory engineering study on one shared Windows PC, not a novel CXL system or a replication of the optimized sparse HGCA/BEYOND runtime.','',
      '## What the evidence says','',
      f'The selected package contains {len(benchmarks)} unrestricted full sweeps, {len(budget)} constrained-selection sweep{'s' if len(budget)!=1 else ''} and {len(models)} pretrained-model comparisons. Each unrestricted sweep uses 30 samples per plan and disjoint calibration/held-out lengths. All numbers below are reconstructed from saved synchronized wall-time samples.','',
      'Direct GPU dense attention is an essential comparator. Initial SDPA-only pilots made hybrid execution appear more advantageous; these stronger-baseline records supersede that interpretation. Per-layer improvements do not establish end-to-end model improvement.','',
      '## Held-out attention latency','',
      'Median of per-run medians in milliseconds, across unrestricted runs. These are descriptive aggregates, not confidence intervals.','',
      '| Context | Native SDPA GPU | Direct dense GPU | CPU dense | Transfer to GPU | Hybrid 75% GPU |',
      '| ---: | ---: | ---: | ---: | ---: | ---: |']
    for n in (1024,4096,16384):
        values=[]
        for p in ('resident','gpu_dense','cpu','copy','hybrid75'):
            xs=[r['median_ms'] for d in benchmarks for r in d['rows'] if r['context']==n and r['plan']==p]
            values.append(f'{statistics.median(xs):.3f}')
        lines.append(f'| {n:,} | '+' | '.join(values)+' |')
    lines+=['','The native float32 SDPA probe recorded the following operations: '+', '.join(f'`{x}`' for x in benchmarks[0].get('native_sdpa_operations',[]))+'. This does not compare all low-precision fused kernels.','',
      '## Adaptive policy','',
      '| Run | KV selection budget (MiB) | Held-out selections | Regret versus eligible measured optimum | Calibration cost |',
      '| --- | ---: | --- | --- | ---: |']
    for d in benchmarks+budget:
        lines.append(f"| {d['id']} | {d['config'].get('kv_budget_mib',2048):g} | "+', '.join(f"{e['context']}: {e['selected']}" for e in d['evaluation'])+' | '+', '.join(f"{e['regret_percent']:.1f}%" for e in d['evaluation'])+f" | {d['calibration_elapsed_s']:.2f} s |")
    lines+=['','Selection uses calibration rows only. The fixed comparator is selected on calibration and must fit the same maximum held-out KV budget. The budget applies to policy selection; every plan is benchmarked. It is not a process memory cap. Calibration overhead is real cost, separate from selected-call latency. Matching a fixed plan is not an adaptive-policy advantage.','',
      '## Full pretrained-model comparisons','',
      'Pinned openai-community/gpt2 safetensors; float32; native SDPA reference. History preparation is separate; timed method order is randomized. The same native greedy token history is teacher-forced into every arm. Numerical bounds are elementwise `atol=1e-4, rtol=1e-4`. GPU windows for hybrid plans are fixed from the initial prompt.','']
    for d in models:
        lines += [f"### {d['id']} — {d['prompt']['tokens']} prompt tokens, {d['config']['new_tokens']} decode steps",'',
          '| Plan | Median forward ms | Decode tok/s | Logit vectors within bounds | Violating elements | Greedy agreement |',
          '| --- | ---: | ---: | ---: | ---: | ---: |']
        for r in d['rows']:
            passed=sum(c['passed'] for c in r['logit_checks'])
            lines.append(f"| {r['label']} | {r['median_ms']:.3f} | {r['tokens_per_s']:.1f} | {passed}/{len(r['logit_checks'])} | {r['correctness']['violating_elements']:,} | {r['greedy_agreement']:.0%} |")
        lines+=['',f"Model loading took {d['model']['load_s']:.2f} s; native reference preparation took {d.get('reference_preparation_s',0):.2f} s. These are excluded from decode throughput, as are numerical validation, greedy selection and token-input preparation. Prefill, setup, raw timings and source identities remain in the JSON record.",'']
    lines+=['## Interpretation and next steps','',
      '- The system executes the actual workload and exposes raw evidence, including negative results. It does not establish that hybrid attention generally accelerates consumer LLM inference.',
      '- Some methods produce matching greedy tokens while failing the stricter logit predicate on a small number of vectors. Those literal failures remain failures; no tolerance was relaxed after seeing results. Many vocabulary elements can violate a bound in one vector. These records do not establish downstream quality equivalence.',
      '- A policy calibrated on independent synthetic attention can fail a full-model numerical contract or mispredict complete model costs. Calibration results do not certify later inference.',
      '- Real CXL/DDR placement, sparse attention, long-context quality, low-precision kernel comparisons, multiple devices and useful OOM rescue remain future studies.',
      '- A useful discussion with Jia Rao would focus on the observed kernel/transfer crossover, the gap between layer and whole-model behavior, and whether application hints merit a separate real-CXL experiment. No advisor collaboration or endorsement is implied.','',
      '## Provenance','',
      'Raw selected records are under `runs/`; `manifest.json` binds their bytes. Recent local run directories additionally retain source snapshots. Environment hashes, data hashes, model revision, package versions, all timing samples, preparation costs and failed predicates are retained. Shared desktop interference and Windows/WDDM scheduling limit generalization. The individual runs, not this report prose, are authoritative.','']
    for m in manifest:lines.append(f"- [{m['id']}](runs/{m['id']}.json) · SHA-256 `{m['sha256']}`")
    (OUTPUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'validated_runs':len(docs),'rows':sum(len(d['rows']) for d in docs),'report':'docs/research/attention-lab/report.md'}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('runs',nargs='+');args=p.parse_args();build(args.runs)
