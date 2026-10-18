"""Conservative execution-KV applicability; adds temporary materializations.

This is a soft KV working-set bound, not a process or physical VRAM quota.
Scores, attention workspace, query/output tensors and model weights are outside
this explicitly named budget. Model execution must abstain on layer calibration.
"""
from copy import deepcopy
from .runtime import PLANS
from .validity import BoundPolicy


def execution_kv_mib(plan,length,identity):
    dtype=identity['dtype'];size={'float32':4,'float16':2,'bfloat16':2}[dtype]
    p=PLANS[plan];n=length if p.executor in ('resident','dense_gpu','copy') else max(1,int(length*p.gpu_fraction)) if p.executor=='hybrid' else 0
    elements=2*identity['batch']*identity['kv_heads']*n*identity['dim']
    storage=elements*size
    # Current partition executor promotes both K and V on the GPU; native/copy
    # SDPA does not perform that explicit promotion. Include live converted KV.
    if dtype!='float32' and p.executor in ('dense_gpu','hybrid'):storage+=elements*4
    return storage/2**20


class ExecutionKVPolicy(BoundPolicy):
    version='domain-bound-execution-kv-v3'
    def __init__(self,rows,identity):
        values=deepcopy(rows)
        if identity.get('scope')!='layer-attend-only':values=[]
        for row in values:row['gpu_kv_mib']=execution_kv_mib(row['plan'],row['context'],identity)
        super().__init__(values,identity)


def evaluate_calibrated(doc):
    """Freeze choices from calibration first, then inspect held-out outcomes."""
    identity=doc['identity'];policy=ExecutionKVPolicy(doc['rows'],identity)
    lengths=[1792,7168,20480];decisions=[];fixed={}
    cal=[r for r in doc['rows'] if r['phase']=='calibration']
    for budget in (48,2048):
        support=policy.choose(max(lengths),identity,budget).get('predictions',[])
        costs={p['plan']:sum(r['median_ms']/min(x['median_ms'] for x in cal if x['context']==r['context']) for r in cal if r['plan']==p['plan']) for p in support}
        fixed[budget]=min(costs,key=lambda p:(costs[p],p)) if costs else None
        for n in lengths:decisions.append({'context':n,'budget_mib':budget,'selected':policy.choose(n,identity,budget),'rule':policy.budget_rule(n,identity,budget)})
    evaluations=[]
    for d in decisions:
        eligible=[r for r in doc['rows'] if r['phase']=='validation' and r['context']==d['context'] and r['correctness']['passed'] and execution_kv_mib(r['plan'],r['context'],identity)<=d['budget_mib']]
        best=min(eligible,key=lambda r:(r['median_ms'],r['plan'])) if eligible else None
        item={'context':d['context'],'budget_mib':d['budget_mib'],'oracle_plan':best['plan'] if best else None,'fixed_plan':fixed[d['budget_mib']]}
        for key in ('selected','rule'):
            prediction=d[key];row=next((r for r in eligible if r['plan']==prediction.get('plan')),None)
            item[key]={'plan':prediction.get('plan'),'eligible':row is not None,'measured_ms':row['median_ms'] if row else None,
                'regret_percent':(row['median_ms']/best['median_ms']-1)*100 if row and best else None,
                'prediction_error_ms':row['median_ms']-prediction['predicted_ms'] if row else None,
                'estimated_execution_kv_mib':prediction.get('estimated_kv_mib')}
        f=next((r for r in eligible if r['plan']==item['fixed_plan']),None);item['fixed_ms']=f['median_ms'] if f else None
        evaluations.append(item)
    return {'policy':policy.version,'decisions':decisions,'evaluation':evaluations,'fixed_plans':fixed,
            'budget_scope':'GPU KV storage plus explicit execution KV copies/promotions. Excludes weights, query/output, scores and workspace; no physical-residency guarantee.'}
