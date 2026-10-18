"""Prospective case registry. Freeze before evaluation; changes need a new ID."""
import json
from .evidence import ROOT,atomic_json,sha,utc
from .assets import model_dir
from .validity import CONTRACTS

DIRECTORY=ROOT/'docs/research/attention-v2'
PROTOCOL_PATH=DIRECTORY/'protocol-amendment-1.json'
EXECUTION_PATH=DIRECTORY/'execution-amendment-1.json'
CORE=['runtime.py','oracle.py','research.py','qwen.py','validity.py','assets.py','protocol.py','telemetry.py']


def cases():
    rows=[]
    for block in range(3):
        for dtype in CONTRACTS:
            rows.append({'kind':'research','suite':'layer','block':block,'dtype':dtype,'threads':4,'seed':93518+block,'repeats':9})
            for n in (512,2048,8192):
                rows.append({'kind':'research','suite':'repeated','block':block,'dtype':dtype,'threads':4,'context':n,'split':'test','offset':block*997+5003,'new_tokens':8,'window':256,'methods':['native','resident','gpu_dense','hybrid50']})
        rows.append({'kind':'research','suite':'cache','block':block,'threads':4})
        for n in (32768,40000):
            for method in ('native','hybrid50'):
                rows.append({'kind':'research','suite':'pressure','block':block,'dtype':'float32','threads':4,'context':n,'split':'test','offset':block*997+5003,'new_tokens':4,'window':256,'methods':[method],'pressure':True})
    for i,row in enumerate(rows):row['case_id']=f'case-{i:03d}'
    return rows


def freeze():
    path=PROTOCOL_PATH
    if path.exists():raise ValueError('Protocol already frozen; preserve it and make an explicit amendment')
    doc={'schema_version':2,'created_at':utc(),'id':'atlas-attention-v2-93518-a1','contracts':CONTRACTS,
        'amendment':{'prior':'protocol.json','reason':'Layer constructor used generic cuda while query uses cuda:0; normalize device before comparison. First layer case failed; next model case interrupted before analysis. Preserve both attempts. New seed and corpus offsets; no tolerance or algorithm tuning to evaluation.',
          'prior_execution':'execution.json','interrupted_run':'research-dc3db1bde1f8'},
        'code_sha256':{name:sha((ROOT/'src/atlas/attention'/name).read_bytes()) for name in CORE},
        'model_manifest_sha256':sha((model_dir('qwen3-0.6b')/'atlas-manifest.json').read_bytes()),
        'corpus_manifest_sha256':sha((ROOT/'data/wikitext-attention-v2/atlas-manifest.json').read_bytes()),
        'cases':cases(),'analysis':{'unit':'independent worker process, paired block','blocks':3,'bootstrap_draws':10000,'bootstrap_seed':93517,
          'estimator':'Median of paired per-process generation ratios, percentile block bootstrap 95% CI',
          'small_n_warning':'Three blocks give limited uncertainty resolution; no p-values or population generalization',
          'failures':'All planned cases retained. Missing/OOM/timed-out arms preclude speedup comparison and remain in feasibility denominators.',
          'eligibility':'Measured case numerical pass required. Unknown/failed rows blocked. Layer evidence never certifies full model.',
          'selector':'Calibration-only interpolation; budget rule; calibration-selected fixed; evaluation oracle is retrospective only.',
          'primary':'Full generation latency versus matched native at each dtype/context; no pooling precision or context.',
          'secondary':'Prefill/decode, NLL diagnostics, cache ablation, layer selector regret; exploratory labels retained.'},
        'development':'Legacy GPT-2 plus WikiText train-only pilot. Test cases unseen before freeze.',
        'hardware_limits':'32GB RAM, RTX3060 12GB shared Windows WDDM. Qwen4B download failed due disk space; Qwen0.6B FP32 32k/40k KV study is predeclared instead.',
        'prohibited_claims':['100% unbiased or accurate','actual CXL/DSA evidence','external independent reproduction','broad downstream language equivalence','novel algorithm established']}
    atomic_json(path,doc)
    return path


def verify_protocol(config):
    path=PROTOCOL_PATH;doc=json.loads(path.read_text())
    if config.get('protocol_id')!=doc['id']:raise ValueError('Protocol identity mismatch')
    case=next((r for r in doc['cases'] if r['case_id']==config.get('case_id')),None)
    if case is None or any(config.get(k)!=v for k,v in case.items()) or set(config)-set(case)-{'protocol_id'}:raise ValueError('Unregistered/mutated confirmatory case')
    for name,digest in doc['code_sha256'].items():
        if sha((ROOT/'src/atlas/attention'/name).read_bytes())!=digest:raise ValueError('Frozen execution source changed: '+name)
    if sha((model_dir('qwen3-0.6b')/'atlas-manifest.json').read_bytes())!=doc['model_manifest_sha256']:raise ValueError('Model manifest changed')
    if sha((ROOT/'data/wikitext-attention-v2/atlas-manifest.json').read_bytes())!=doc['corpus_manifest_sha256']:raise ValueError('Corpus manifest changed')
    return doc
