"""Independent exact-rational reconstruction of published telemetry traces."""
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
from statistics import mean
ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'docs/research/experiments'
def canonical(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def derive():
    raw=json.loads((D/'board-energy-control.json').read_text(encoding='utf-8'))
    original=json.loads((D/'fashion-pilot.json').read_text(encoding='utf-8'))
    assert raw['reference_canonical_json_sha256']==canonical(original)
    assert [(t['pair'],t['candidate']) for t in raw['trials']]==[(t['pair'],t['candidate']) for t in raw['order']]
    assert {(t['pair'],t['candidate']) for t in raw['trials']}=={(p,a) for p in range(3) for a in (False,True)}
    rows=[]
    for trial in raw['trials']:
        samples=trial['samples'];times=[s['elapsed_ns'] for s in samples]
        assert all(a<b for a,b in zip(times,times[1:]))
        assert not any(s['errors'] for s in samples)
        assert all(a['energy_mj']<=b['energy_mj'] for a,b in zip(samples,samples[1:]))
        counter=Fraction(samples[-1]['energy_mj']-samples[0]['energy_mj'],1000)
        integral=sum(Fraction((a['power_mw']+b['power_mw'])*(b['elapsed_ns']-a['elapsed_ns']),2*10**12) for a,b in zip(samples,samples[1:]))
        window=Fraction(times[-1]-times[0],10**9)
        assert float(counter)==trial['gross_counter_j']
        assert math.isclose(float(integral),trial['integrated_reported_power_j'],rel_tol=1e-14)
        assert float(window)==trial['window_seconds']
        replay=trial['replayed_result']
        ref=next(t for t in original['trials'] if t['seed']==raw['config']['seed'] and t['candidate']==trial['candidate'])
        keys=('initial_state_digest','split_digest','sample_order_digest','final_state_digest')
        assert all(replay[k]==ref[k] for k in keys)
        for a,b in zip(replay['history'],ref['history'],strict=True):
            assert all(a[k]==b[k] for k in ('validation_nll','validation_accuracy','validation_correct','successful_updates'))
        assert len(replay['history'])==5 and replay['status']=='completed' and trial['returncode']==0
        rows.append({'pair':trial['pair'],'candidate':trial['candidate'],'counter_j':float(counter),'power_integral_j':float(integral),
            'window_seconds':float(window),'integral_excess_percent':float(100*(integral/counter-1)),
            'counter_mean_w':float(counter/window),'reported_power_mean_w':float(integral/window),
            'max_gap_seconds':max(b-a for a,b in zip(times,times[1:]))/1e9,'sample_count':len(samples),
            'unchanged_counter_intervals':sum(a['energy_mj']==b['energy_mj'] for a,b in zip(samples,samples[1:])),
            'identical_training_replay':True})
    pairs=[]
    for pair in range(3):
        native=next(r for r in rows if r['pair']==pair and not r['candidate']);candidate=next(r for r in rows if r['pair']==pair and r['candidate'])
        pairs.append({'pair':pair,'counter_increase_percent':100*(candidate['counter_j']/native['counter_j']-1),
                      'window_increase_percent':100*(candidate['window_seconds']/native['window_seconds']-1)})
    arms={name:{field:mean(r[field] for r in rows if r['candidate']==candidate) for field in ('counter_j','power_integral_j','window_seconds','integral_excess_percent')} for name,candidate in [('native',False),('candidate',True)]}
    out={'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'input_canonical_json_sha256':{'board-energy-control.json':canonical(raw),'fashion-pilot.json':canonical(original)},
         'trials':rows,'pairs':pairs,'arm_means':arms,'scope':'Independent arithmetic and replay identity verification; same-seed repetitions, gross shared-board windows, no calibrated or causal process-energy inference.'}
    (D/'board-energy-derived.json').write_bytes((json.dumps(out,indent=2,sort_keys=True,allow_nan=False)+'\n').encode())
    print(json.dumps(out,indent=2))
if __name__=='__main__':derive()
