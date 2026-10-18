import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import pytest
import torch
from atlas.attention.runtime import PLANS,DecodeCache,partial_attention,fuse,error_metrics,CalibratedPolicy


@pytest.mark.parametrize("cut",[1,3,16])
def test_fusion_is_global_softmax_not_partition_average(cut):
    torch.manual_seed(4)
    q=torch.randn(2,3,1,8,dtype=torch.float64)*10
    k=torch.randn(2,3,17,8,dtype=torch.float64)*10
    v=torch.randn_like(k)
    actual=fuse(partial_attention(q,k[...,:cut,:],v[...,:cut,:]),partial_attention(q,k[...,cut:,:],v[...,cut:,:]))
    torch.testing.assert_close(actual,partial_attention(q,k,v)[0],atol=1e-10,rtol=1e-10)


@pytest.mark.parametrize("ident",list(PLANS))
def test_persistent_append_matches_full_history(ident):
    torch.manual_seed(9)
    k=torch.randn(1,2,13,8)
    v=torch.randn_like(k)
    q=torch.randn(1,2,1,8)
    cache=DecodeCache(k[...,:7,:],v[...,:7,:],PLANS[ident],13)
    for n in range(7,13):
        cache.append(k[...,n:n+1,:],v[...,n:n+1,:])
        assert error_metrics(cache.attend(q),partial_attention(q.double(),k[...,:n+1,:].double(),v[...,:n+1,:].double())[0])["passed"]
    with pytest.raises(ValueError):cache.append(k[...,:1,:],v[...,:1,:])


def test_errors_fail_nonfinite_and_wrong_fusion():
    assert not error_metrics(torch.tensor([float("nan")]),torch.tensor([1.]))["passed"]
    assert not error_metrics(torch.tensor([.5]),torch.tensor([.8]))["passed"]
    with pytest.raises(ValueError):partial_attention(torch.zeros(1,1,1,2),torch.zeros(1,1,0,2),torch.zeros(1,1,0,2))
    with pytest.raises(ValueError):error_metrics(torch.ones(1),torch.ones(4))


def test_policy_cannot_learn_from_validation_or_failed_rows():
    rows=[{"phase":"calibration","context":n,"heads":2,"head_dim":8,"plan":p,"median_ms":m,"correctness":{"passed":True}} for n in (10,100) for p,m in (("cpu",2),("resident",1))]
    rows.append({**rows[0],"phase":"validation","median_ms":0.0001})
    assert CalibratedPolicy(rows).choose(50,2,8)["plan"]=="resident"
    assert CalibratedPolicy(rows).choose(50,2,8,budget_mib=0)["plan"]=="cpu"
    with pytest.raises(ValueError):CalibratedPolicy([]).choose(50,2,8)


@pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA integration control")
def test_cuda_hybrid_and_copy():
    k=torch.randn(1,2,256,32,device="cuda")
    v=torch.randn_like(k)
    q=torch.randn(1,2,1,32,device="cuda")
    expected=partial_attention(q.cpu().double(),k.cpu().double(),v.cpu().double())[0]
    for plan in PLANS.values():
        cache=DecodeCache(k,v,plan,258)
        if cache.ck.numel():assert cache.ck.is_pinned() and cache.cv.is_pinned()
        assert error_metrics(cache.attend(q),expected)["passed"]
