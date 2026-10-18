"""Predeclared contracts and conservative calibration applicability."""
from __future__ import annotations
import statistics

# These are finite numerical relations, not downstream quality equivalence.
# Keep the original FP32 tolerance; low-precision contracts are new relations.
CONTRACTS={
    'float32':{'atol':1e-4,'rtol':1e-4},
    'float16':{'atol':0.02,'rtol':0.02},
    'bfloat16':{'atol':0.15,'rtol':0.05},
}


def eligibility(checks):
    if not checks:return {'eligible':False,'reason':'No numerical evidence'}
    if any(c.get('passed') is not True for c in checks):
        return {'eligible':False,'reason':'Failed or unknown numerical relation'}
    return {'eligible':True,'reason':'Passed the declared finite relation in this measured case'}


class BoundPolicy:
    """Interpolation only inside a numerically validated calibration domain."""
    version='domain-bound-linear-v2'
    def __init__(self,rows,identity):
        self.identity=identity
        domain=[r for r in rows if r.get('phase')=='calibration' and r.get('identity')==identity]
        rejected={r['plan'] for r in domain if r.get('correctness',{}).get('passed') is not True}
        self.rows=[r for r in domain if r['plan'] not in rejected]

    def choose(self,length,identity,budget_mib):
        if identity!=self.identity:return {'plan':None,'abstained':True,'reason':'Execution identity differs'}
        predictions=[]
        for plan in sorted({r['plan'] for r in self.rows}):
            points=sorted((r for r in self.rows if r['plan']==plan),key=lambda r:r['context'])
            if len({r['context'] for r in points})<2 or not points[0]['context']<=length<=points[-1]['context']:continue
            # Repeated calibration processes are aggregated by context first.
            points=[{'context':n,'median_ms':statistics.median(r['median_ms'] for r in points if r['context']==n),
                     'gpu_kv_mib':max(r['gpu_kv_mib'] for r in points if r['context']==n)} for n in sorted({r['context'] for r in points})]
            lo=max((r for r in points if r['context']<=length),key=lambda r:r['context'])
            hi=min((r for r in points if r['context']>=length),key=lambda r:r['context'])
            w=(length-lo['context'])/(hi['context']-lo['context']) if hi['context']!=lo['context'] else 0
            memory=lo['gpu_kv_mib']*(1-w)+hi['gpu_kv_mib']*w
            if memory<=budget_mib:predictions.append({'plan':plan,'predicted_ms':lo['median_ms']*(1-w)+hi['median_ms']*w,'estimated_kv_mib':memory})
        if not predictions:return {'plan':None,'abstained':True,'reason':'Outside validated lengths or budget support'}
        return {**min(predictions,key=lambda r:(r['predicted_ms'],r['plan'])),'abstained':False,'policy':self.version,'predictions':predictions}

    def budget_rule(self,length,identity,budget_mib):
        decision=self.choose(length,identity,budget_mib)
        if decision['abstained']:return decision
        options=decision['predictions']
        # Native first, then dense GPU, then highest GPU fraction that fits.
        rank={'resident':0,'gpu_dense':1,'hybrid75':2,'hybrid50':3,'hybrid25':4,'copy':5,'cpu':6}
        return {**min(options,key=lambda r:rank.get(r['plan'],99)),'abstained':False,'policy':'budget-aware-rule-v1'}
