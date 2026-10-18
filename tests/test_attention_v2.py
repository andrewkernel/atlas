import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pytest
import torch
from atlas.attention.oracle import attention_oracle,distribution_metrics
from atlas.attention.runtime import DecodeCache,PLANS,error_metrics,partial_attention
from atlas.attention.validity import BoundPolicy,eligibility


@pytest.mark.parametrize('dtype',[torch.float32,torch.float16,torch.bfloat16])
@pytest.mark.parametrize('plan',['resident','gpu_dense','cpu','hybrid50','copy'])
def test_ring_matches_independent_oracle_across_repeated_wraps(dtype,plan):
    gen=torch.Generator().manual_seed(601)
    k=torch.randn(2,2,17,8,generator=gen).to(dtype);v=torch.randn(2,2,17,8,generator=gen).to(dtype)
    cache=DecodeCache(k[...,:5,:],v[...,:5,:],PLANS[plan],17)
    ptr=cache.gk.data_ptr()
    for t in range(5,17):
        cache.append(k[...,t:t+1,:],v[...,t:t+1,:])
        q=torch.randn(2,4,1,8,generator=gen).to(dtype)
        expected=torch.from_numpy(attention_oracle(q.float().numpy(),k[...,:t+1,:].float().numpy(),v[...,:t+1,:].float().numpy()))
        assert error_metrics(cache.attend(q),expected,atol=.01 if dtype!=torch.float32 else 1e-5,rtol=.01 if dtype!=torch.float32 else 1e-5)['passed']
        assert cache.gk.data_ptr()==ptr
    with pytest.raises(ValueError):cache.append(k[...,:1,:],v[...,:1,:])


def test_extreme_scores_and_corruption_detected():
    q=torch.tensor([[[[1000.,-1000.]]]])
    k=torch.tensor([[[[1000.,1000.],[1000.,-1000.],[-1000.,1000.]]]])
    v=torch.tensor([[[[1.,2.],[3.,7.],[-1.,-2.]]]])
    ref=torch.from_numpy(attention_oracle(q.numpy(),k.numpy(),v.numpy()))
    assert error_metrics(partial_attention(q,k,v)[0],ref)['passed']
    assert not error_metrics(partial_attention(q,k,v)[0]+1,ref)['passed']
    assert not eligibility([{'passed':False}])['eligible']
    assert not eligibility([])['eligible']


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA not installed')
def test_host_initialization_resolves_default_cuda_device_and_gqa():
    k=torch.randn(1,2,19,8);v=torch.randn_like(k);q=torch.randn(1,4,1,8).cuda()
    for plan in PLANS.values():
        cache=DecodeCache(k,v,plan,19,device='cuda')
        expected=torch.from_numpy(attention_oracle(q.cpu().numpy(),k.numpy(),v.numpy()))
        assert error_metrics(cache.attend(q),expected)['passed']


def test_distribution_diagnoses_constant_shift_without_relaxing_gate():
    a=np.array([1.,3.,7.]);b=a+2
    assert distribution_metrics(b,a)['total_variation']<1e-14
    assert not error_metrics(torch.from_numpy(b),torch.from_numpy(a))['passed']


def test_bound_policy_abstains_and_budget_rule_has_no_heldout_dependency():
    identity={'scope':'layer','dtype':'float32'}
    rows=[{'phase':'calibration','identity':identity,'context':n,'plan':p,'median_ms':ms*n/100,'gpu_kv_mib':mem*n/100,'correctness':{'passed':True}} for n in (100,1000) for p,ms,mem in [('resident',1,10),('cpu',5,0)]]
    policy=BoundPolicy(rows,identity)
    assert policy.choose(50,identity,100)['abstained']
    assert policy.choose(500,{'scope':'model','dtype':'float32'},100)['abstained']
    assert policy.choose(500,identity,1)['plan']=='cpu'
    assert policy.budget_rule(500,identity,100)['plan']=='resident'
