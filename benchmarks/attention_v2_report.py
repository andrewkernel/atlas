"""Recompute portable study summaries from exact raw records, including failures."""
from __future__ import annotations
import json
import math
from pathlib import Path
import statistics
import time
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from atlas.attention.evidence import ROOT,RUNS,atomic_json,sha,utc
from atlas.attention.protocol import DIRECTORY,PROTOCOL_PATH,EXECUTION_PATH
from atlas.attention.selector import evaluate_calibrated


def interval(values):
    if not values:return None
    if len(values)<3:return {'estimate':statistics.median(values),'low':None,'high':None,'n_process_blocks':len(values),'raw_block_values':values,'method':'Interval pending all three process blocks; no uncertainty claim from one/two blocks'}
    rng=np.random.default_rng(93517)
    draws=np.median(rng.choice(values,size=(10000,len(values)),replace=True),axis=1)
    return {'estimate':statistics.median(values),'low':float(np.quantile(draws,.025)),'high':float(np.quantile(draws,.975)),
            'n_process_blocks':len(values),'raw_block_values':values,'method':'Paired process-block percentile bootstrap of median; 10,000 resamples. Small n; interval is descriptive.'}


def build(replication=None):
    output=replication or DIRECTORY
    protocol=json.loads(PROTOCOL_PATH.read_text());ledger=json.loads((output/'execution.json' if replication else EXECUTION_PATH).read_text())
    amendment=json.loads((DIRECTORY/'selector-amendment.json').read_text())
    if sha((ROOT/'src/atlas/attention/selector.py').read_bytes())!=amendment['source_sha256']:raise ValueError('Selector analysis source changed after amendment')
    for filename,key in [('model-manifest.json','model_manifest_sha256'),('corpus-manifest.json','corpus_manifest_sha256')]:
        if sha((DIRECTORY/filename).read_bytes())!=protocol[key]:raise ValueError('Frozen asset manifest mismatch: '+filename)
    documents=[];checks=0
    for entry in ledger['cases']:
        path=output/'runs'/(entry['run_id']+'.json')
        if sha(path.read_bytes())!=entry['sha256']:raise ValueError('Raw run hash mismatch: '+entry['run_id'])
        doc=json.loads(path.read_text());case=next(c for c in protocol['cases'] if c['case_id']==entry['case_id'])
        if any(doc['config'].get(k)!=v for k,v in case.items()):raise ValueError('Recorded case differs from registration')
        for name,digest in protocol['code_sha256'].items():
            if doc['environment']['source_sha256'].get('src/atlas/attention/'+name)!=digest:raise ValueError('Recorded worker core differs from frozen source: '+name)
        for row in doc['rows']:
            if row.get('samples_ms'):
                if not math.isclose(statistics.median(row['samples_ms']),row['median_ms'],rel_tol=1e-10):raise ValueError('Median arithmetic mismatch')
                checks+=1
        documents.append(doc)
    model_rows=[(d,r) for d in documents if d['config']['suite']=='repeated' for r in d['rows']]
    comparisons=[]
    for dtype in protocol['contracts']:
        for context in (512,2048,8192):
            for method in ('resident','gpu_dense','hybrid50'):
                generation=[];prefill=[];decode=[];eligible=[];nll=[]
                for block in range(3):
                    pair=[r for d,r in model_rows if d['config']['dtype']==dtype and r['context']==context and d['config']['block']==block]
                    native=next((r for r in pair if r['plan']=='native' and r.get('generation_ms')),None)
                    selected=next((r for r in pair if r['plan']==method and r.get('generation_ms')),None)
                    if native and selected:
                        generation.append(native['generation_ms']/selected['generation_ms'])
                        prefill.append(native['prefill_ms']/selected['prefill_ms']);decode.append(native['decode_ms']/selected['decode_ms'])
                        eligible.append(selected.get('eligibility',{}).get('eligible') is True)
                        if selected.get('mean_nll') is not None:nll.append(selected['mean_nll']-selected['native_mean_nll'])
                comparisons.append({'dtype':dtype,'context':context,'plan':method,'label':'Hybrid / 256-token GPU window' if method=='hybrid50' else method,'generation_speedup':interval(generation),'prefill_speedup':interval(prefill),'decode_speedup':interval(decode),
                    'eligible_blocks':sum(eligible),'planned_blocks':3,'all_blocks_eligible':len(eligible)==3 and all(eligible),'nll_delta':interval(nll),
                    'interpretation':'Finite matched-case evidence only; slower-than-native values below 1 remain negative results.'})
    pressure=[]
    board_path=DIRECTORY/'board-observations.json'
    board=json.loads(board_path.read_text()) if board_path.exists() else {'samples':[]}
    for d in documents:
        if d['config']['suite']!='pressure':continue
        r=d['rows'][0] if d['rows'] else {}
        samples=r.get('telemetry',[])
        if not samples:
            extra=RUNS/d['id']/(d['config']['methods'][0]+'-telemetry.json')
            if extra.exists():samples=json.loads(extra.read_text()).get('samples',[])
        pressure.append({'case_id':d['config']['case_id'],'run_id':d['id'],'block':d['config']['block'],'context':d['config']['context'],'plan':d['config']['methods'][0],
            'label':'Hybrid / 256-token GPU window' if d['config']['methods'][0]=='hybrid50' else 'Native model SDPA',
            'execution_status':d['status'],'outcome':r.get('status',d['status']),'generation_ms':r.get('generation_ms'),'prefill_ms':r.get('prefill_ms'),'decode_ms':r.get('decode_ms'),
            'allocated_peak_mib':r.get('allocated_peak_mib'),'gpu_kv_mib':r.get('gpu_kv_mib'),'cpu_kv_mib':r.get('cpu_kv_mib'),
            'board_observed_peak_mib':max([s['board_used_mib'] for s in samples if s.get('board_used_mib') is not None]+[s['board_used_mib'] for s in board['samples'] if s['run_id']==d['id'] and s.get('board_used_mib') is not None],default=None),
            'board_external_samples':sum(s['run_id']==d['id'] for s in board['samples']),
            'host_rss_peak_mib':max((s.get('rss_mib',0) for s in samples),default=None),'telemetry_samples':len(samples),
            'observer_wall_ms_total':sum(s.get('observer_wall_ms',0) for s in samples),'error':d.get('error') or r.get('error'),
            'numerical_relation':'Unknown at full pressure context; feasibility only; never eligible for correctness recommendation'})
    pressure_comparisons=[]
    for context in (32768,40000):
        ratios=[];allocation=[];kv=[]
        for block in range(3):
            pair=[r for r in pressure if r['context']==context and r['block']==block and r['outcome']=='completed']
            native=next((r for r in pair if r['plan']=='native'),None);hybrid=next((r for r in pair if r['plan']=='hybrid50'),None)
            if native and hybrid:
                ratios.append(native['generation_ms']/hybrid['generation_ms'])
                allocation.append(100*(1-hybrid['allocated_peak_mib']/native['allocated_peak_mib']))
                kv.append(100*(1-hybrid['gpu_kv_mib']/native['gpu_kv_mib']))
        pressure_comparisons.append({'context':context,'generation_ratio':interval(ratios),'allocated_peak_reduction_percent':interval(allocation),'gpu_kv_reduction_percent':interval(kv),
            'scope':'Exploratory feasibility only. Full-context numerical replay unknown; no correctness recommendation or exclusive-residency claim.'})
    ablation=[]
    for n in (512,4096,16384):
        ratios=[]
        for d in documents:
            if d['config']['suite']!='cache':continue
            rows=[r for r in d['rows'] if r['context']==n]
            if len(rows)==2:ratios.append(next(r['median_ms'] for r in rows if r['plan']=='shift')/next(r['median_ms'] for r in rows if r['plan']=='ring'))
        ablation.append({'context':n,'shift_over_ring':interval(ratios)})
    selectors=[]
    for d in documents:
        if d['config']['suite']!='layer':continue
        if 'identity' not in d:continue
        analysis_started=time.perf_counter();corrected=evaluate_calibrated(d);evaluations=corrected['evaluation']
        reanalysis_s=time.perf_counter()-analysis_started
        savings=sum(max(0,e.get('fixed_ms',0)-e['selected']['measured_ms']) for e in evaluations if e.get('fixed_ms') and e['selected'].get('measured_ms'))
        selectors.append({'run_id':d['id'],'dtype':d['config']['dtype'],'block':d['config']['block'],'calibration_cost_s':d.get('calibration_cost_s'),
            'evaluation':evaluations,'policy':corrected['policy'],'budget_scope':corrected['budget_scope'],'calibration_decisions':corrected['decisions'],
            'additional_selector_reanalysis_s':reanalysis_s,'reanalysis_scope':'Observed report-time upper bound including decision recomputation and retrospective evaluation; not an isolated production selector latency',
            'amortization_note':'Calibration costs all measured calibration plans, setup and oracle work. No net adaptation benefit claimed.',
            'optimistic_equal_mix_calls_to_amortize':(d['calibration_cost_s']*1000/(savings/len(evaluations))) if savings>0 and evaluations else None})
    selector_comparisons=[]
    for dtype in protocol['contracts']:
        for budget in (48,2048):
            rule_ratios=[];fixed_ratios=[];costs=[];regrets=[]
            for entry in selectors:
                if entry['dtype']!=dtype:continue
                rows=[e for e in entry['evaluation'] if e['budget_mib']==budget]
                if len(rows)!=3 or not all(e['selected']['eligible'] and e['rule']['eligible'] and e.get('fixed_ms') for e in rows):continue
                selected=sum(e['selected']['measured_ms'] for e in rows)
                rule_ratios.append(sum(e['rule']['measured_ms'] for e in rows)/selected)
                fixed_ratios.append(sum(e['fixed_ms'] for e in rows)/selected)
                costs.append(entry['calibration_cost_s']);regrets.append(statistics.mean(e['selected']['regret_percent'] for e in rows))
            selector_comparisons.append({'dtype':dtype,'budget_mib':budget,'vs_rule':interval(rule_ratios),'vs_fixed':interval(fixed_ratios),
                'calibration_s':interval(costs),'mean_shape_regret_percent':interval(regrets),
                'scope':'Exploratory equal mix of three layer-attention lengths, calibration cost reported separately; no full-model routing evidence or net advantage claim'})
    diagnosis_path=DIRECTORY/'diagnosis-v1-replay.json'
    if not diagnosis_path.exists():diagnosis_path=DIRECTORY/'diagnosis-v1.json'
    diagnosis=json.loads(diagnosis_path.read_text())
    diagnosis_summary=[]
    for r in diagnosis['rows']:
        diagnosis_summary.append({'plan':r['plan'],'failed_steps':r['failed_steps'],'local_oracle_passed':all(x['correctness']['passed'] for x in r['local_attention_oracle']),
            'max_kl':max(x['reference_kl'] for x in r['distribution_checks']),
            'max_centered_logit_error':max(x['centered_logit_max_abs'] for x in r['distribution_checks']),
            'earliest_checked_layer_failure':next(({'step':x['step'],'layer':x['layer'],'max_abs':x['correctness']['max_abs']} for x in r['layer_checks'] if not x['correctness']['passed']),None)})
    summary={'schema_version':2,'built_at':utc(),'protocol_id':protocol['id'],'protocol_sha256':sha(PROTOCOL_PATH.read_bytes()),
        'planned_cases':len(protocol['cases']),'recorded_cases':len(ledger['cases']),'complete':len(ledger['cases'])==len(protocol['cases']),
        'status_counts':{s:sum(d['status']==s for d in documents) for s in sorted({d['status'] for d in documents})},
        'arithmetic_checks':checks,'numerical_pass_rows':sum(r.get('correctness',{}).get('passed') is True for d in documents for r in d['rows']),
        'numerical_failed_rows':sum(r.get('correctness',{}).get('passed') is False for d in documents for r in d['rows']),
        'numerical_unknown_rows':sum(r.get('correctness',{}).get('passed') is None for d in documents for r in d['rows']),
        'model_comparisons':comparisons,'pressure':pressure,'pressure_comparisons':pressure_comparisons,'cache_ablation':ablation,'selectors':selectors,'selector_comparisons':selector_comparisons,'selector_amendment':amendment,'diagnosis':diagnosis_summary,
        'backend_operations':sorted({op for d in documents for op in d.get('native_sdpa_operations',[])}),
        'backend_controls':json.loads((DIRECTORY/'backend-controls.json').read_text()),
        'requirements_sha256':sha((ROOT/'configs/attention-v2-windows-py312-requirements.txt').read_bytes()),
        'board_trace':{'started_at':board.get('started_at'),'finished_at':board.get('finished_at'),'sha256':sha(board_path.read_bytes()) if board_path.exists() else None,
                       'scope':board.get('scope'),'samples':len(board['samples']),'observer_wall_ms_total':sum(s['observer_wall_ms'] for s in board['samples'])},
        'uncertainty':'Three fresh process blocks, with different contiguous corpus offsets and cyclic arm order, on one shared machine/session. Corpus windows overlap; blocks are not independent document samples or machine/session replications. Adjacent tokens are not treated as independent replications. Intervals are descriptive and do not guarantee 95% population coverage.',
        'bias_limits':['Small, non-random public corpus selection, overlapping windows and short continuations','One modern model family, one GPU and shared desktop load','Method order only partially balanced with three blocks and four arms; cache ablation uses fixed shift-then-ring order','Native warmup/reference replay precede timing; custom decoder first-call costs remain included. These are registered generation windows, not all-method steady-state timings.','Low-precision tolerances are finite relations, not universal equivalence','In-worker NVML unavailable in primary study; external shared-board trace starts later. Earlier absent measurements remain unknown. Observer placement is not randomized.','No actual CXL hardware, independent human replication, energy data or novelty proof'],
        'diagnosis_source':{'file':diagnosis_path.name,'sha256':sha(diagnosis_path.read_bytes())},'manifest':ledger['cases']}
    atomic_json(output/'summary.json',summary)
    lines=['# Atlas Attention Research v2','',f"Protocol: `{protocol['id']}`. Recorded {summary['recorded_cases']}/{summary['planned_cases']} planned cases. Execution: {summary['status_counts']}.",'',
        '**These benchmarks are not 100% accurate or unbiased.** The records support bounded numerical and performance observations on this machine. Passing execution and numerical eligibility are separate. Original failures and the interrupted first protocol attempt are preserved.','',
        '## Matched model results','', 'Native generation time divided by candidate time; values below 1 mean the candidate is slower. Brackets show a descriptive 95% paired process-block bootstrap interval. Three blocks give limited uncertainty resolution. Failed numerical candidates are blocked even when timing is available. The historical method ID `hybrid50` uses the explicitly configured fixed 256-token GPU window in all model cases; it does not mean 50% GPU placement there. Raw labels are preserved, and displayed model labels use the actual window. Synthetic layer fractions remain fractional.','',
        '| Dtype | Context | Method | Generation ratio [95% CI] | Eligible blocks |','| --- | ---: | --- | --- | ---: |']
    for c in comparisons:
        x=c['generation_speedup'];value=(f"{x['estimate']:.3f} [{x['low']:.3f}, {x['high']:.3f}]" if x['low'] is not None else f"{x['estimate']:.3f} [interval pending]") if x else 'Unmeasured'
        lines.append(f"| {c['dtype']} | {c['context']} | {c['plan']} | {value} | {c['eligible_blocks']}/3 |")
    lines+=['','## Baselines and precision','',
        'Native SDPA dispatch was observed as: '+', '.join(summary['backend_operations'])+'. Flash attention is unavailable in this installed build; no Flash performance baseline is claimed. Exact build flags and the external shared-session driver observation are in `backend-controls.json`.',
        '', 'FP32 retains atol=rtol=1e-4. New FP16 (atol=0.02, rtol=0.02) and BF16 (atol=0.15, rtol=0.05) relations were declared before their confirmation cases. They compare with precision-matched native logits, not FP32 equivalence. Finite real-token NLL/softmax diagnostics supplement rather than replace these literal gates. Native warmup and reference replay precede timing; custom first-call costs remain included. This does not measure universal steady-state performance.']
    lines+=['','## Real memory pressure','', 'Full generation includes prefill, selection, decode/cache movement and synchronization. No allocator quota is imposed. Windows WDDM can spill into system memory, so allocator and board observations do not prove physical residency or a hard hardware frontier. Full-context numerical replay is unknown and therefore these rows are not correctness-eligible.','',
        '| Context | Block | Method | Outcome | Generation ms | Allocator peak MiB | Board observed MiB | Host RSS MiB |','| ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |']
    for r in pressure:
        def show(key):return f"{r[key]:.1f}" if r.get(key) is not None else 'unknown'
        lines.append(f"| {r['context']} | {r['block']} | {r['plan']} | {r['outcome']} | {show('generation_ms')} | {show('allocated_peak_mib')} | {show('board_observed_peak_mib')} | {show('host_rss_peak_mib')} |")
    lines+=['','## Correctness diagnosis','', 'The legacy relation remains atol=rtol=1e-4. Captured local attention agrees with the independent NumPy float64 oracle where reported, while some full-model logits differ from native. This bounds the diagnosis to arithmetic and propagation differences; it does not prove every operation correct. Shift-invariant KL and centered-logit diagnostics are reported alongside the literal gate, never as a replacement.','',json.dumps(diagnosis_summary,indent=2),'',
        '## Cache and adaptation','', 'The ring has O(1) append/eviction and stores its GPU window once. Its measured complete-step ablation still includes attention, CPU copies and fusion; O(1) append alone does not guarantee a faster full step. Calibration binds implementation, software, GPU, precision, shape and timing scope, abstains outside support, and is compared with a budget rule and calibration-selected fixed plan. The held-out oracle is retrospective only. No net adaptive advantage is claimed.','',
        '## Remaining limits','']+[f'- {x}' for x in summary['bias_limits']]+['', 'The source-bound reproduction kit and arithmetic validation are self-produced evidence, not external independent reproduction. See `reproduce.md`, the frozen protocol, exact raw run hashes in `execution-amendment-1.json`, and `summary.json` for every metric.','']
    (output/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({k:summary[k] for k in ('complete','recorded_cases','planned_cases','status_counts','arithmetic_checks','numerical_pass_rows','numerical_failed_rows','numerical_unknown_rows')}))
    return summary


if __name__=='__main__':build()
