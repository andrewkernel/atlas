"""Independent head-wise NumPy float64 oracle; never calls runtime attention.

Intentionally simple, separately implemented and unsuitable as a speed baseline.
GQA maps each consecutive query-head group to its corresponding KV head.
"""
from __future__ import annotations
import numpy as np


def attention_oracle(query, key, value, scale=None):
    q=np.asarray(query,dtype=np.float64)
    k=np.asarray(key,dtype=np.float64)
    v=np.asarray(value,dtype=np.float64)
    if q.ndim!=4 or k.ndim!=4 or v.shape!=k.shape:
        raise ValueError('Expected B,H,Q,D queries and matching B,KVH,T,D keys/values')
    batch,heads,queries,dim=q.shape
    if k.shape[0]!=batch or k.shape[3]!=dim or heads%k.shape[1] or k.shape[2]==0:
        raise ValueError('Incompatible head/sequence dimensions')
    factor=1/np.sqrt(dim) if scale is None else float(scale)
    result=np.empty_like(q)
    groups=heads//k.shape[1]
    for b in range(batch):
        for h in range(heads):
            kh=h//groups
            for t in range(queries):
                # NumPy dot and normalization are independent of PyTorch kernels.
                scores=np.dot(k[b,kh],q[b,h,t])*factor
                weights=np.exp(scores-np.max(scores))
                weights/=np.sum(weights,dtype=np.float64)
                result[b,h,t]=np.dot(weights,v[b,kh])
    return result


def distribution_metrics(actual,reference,target=None):
    """Shift-invariant distribution diagnostics, separate from strict logit gate."""
    a=np.asarray(actual,dtype=np.float64).reshape(-1)
    r=np.asarray(reference,dtype=np.float64).reshape(-1)
    if a.shape!=r.shape or not np.isfinite(a).all() or not np.isfinite(r).all():
        return {'finite':False}
    def log_probs(x):
        shifted=x-x.max()
        return shifted-np.log(np.exp(shifted).sum())
    la,lr=log_probs(a),log_probs(r)
    p=np.exp(lr)
    answer={'finite':True,'reference_kl':float(np.sum(p*(lr-la))),
            'total_variation':float(np.abs(np.exp(la)-p).sum()/2),
            'argmax_agreement':bool(np.argmax(a)==np.argmax(r)),
            'centered_logit_max_abs':float(np.abs((a-a.mean())-(r-r.mean())).max()),
            'mean_logit_shift':float((a-r).mean())}
    if target is not None:
        answer.update(reference_nll=float(-lr[target]),candidate_nll=float(-la[target]),nll_delta=float(lr[target]-la[target]))
    return answer
