"""Small independent native-model comparison: causal prefill and growing decode."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import pytest
import torch
pytest.importorskip("transformers")
from transformers import GPT2Config,GPT2LMHeadModel
from atlas.attention.model import GPT2Adapter
from atlas.attention.runtime import PLANS,error_metrics


@pytest.mark.parametrize("plan",list(PLANS))
def test_adapter_matches_native_causal_model_and_restores(plan):
    torch.manual_seed(81)
    config=GPT2Config(vocab_size=100,n_positions=32,n_embd=16,n_layer=2,n_head=2,
        resid_pdrop=0,embd_pdrop=0,attn_pdrop=0)
    config._attn_implementation="sdpa"
    model=GPT2LMHeadModel(config).eval()
    prefix=torch.tensor([[1,9,4,2,8]])
    native=[]
    history=[7,2,4,9,8]
    original=model.transformer.h[0].attn.forward
    with torch.inference_mode():
        result=model(prefix,use_cache=True)
        native.append(result.logits[:,-1,:])
        for token in history:
            result=model(torch.tensor([[token]]),past_key_values=result.past_key_values,use_cache=True)
            native.append(result.logits[:,-1,:])
        with GPT2Adapter(model,PLANS[plan],10) as adapter:
            result=model(prefix,use_cache=False)
            assert error_metrics(result.logits[:,-1,:],native[0])["passed"]
            for step,token in enumerate(history):
                result=model(torch.tensor([[token]]),use_cache=False,position_ids=torch.tensor([[5+step]]))
                assert error_metrics(result.logits[:,-1,:],native[step+1])["passed"]
            assert all(c.length==10 for c in adapter.caches.values())
    assert model.transformer.h[0].attn.forward==original


def test_training_refused():
    model=GPT2LMHeadModel(GPT2Config(n_embd=16,n_head=2,n_layer=1))
    with pytest.raises(ValueError):GPT2Adapter(model,PLANS['cpu'],10)
