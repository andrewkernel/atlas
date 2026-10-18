"""Owned persistent caches around the installed builtin Qwen3 attention."""
import types
import torch
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from .runtime import DecodeCache


class QwenAdapter:
    def __init__(self,model,plan,capacity,window=None,capture=None):
        if model.training or model.config.sliding_window is not None:
            raise ValueError('Only evaluation full attention Qwen3 is supported')
        self.model,self.plan,self.capacity,self.window=model,plan,capacity,window
        self.caches={};self.original=[];self.capture=capture
    def __enter__(self):
        for index,block in enumerate(self.model.model.layers):
            attn=block.self_attn;self.original.append((attn,attn.forward))
            def forward(module,hidden_states,position_embeddings,attention_mask=None,past_key_values=None,_index=index,**kwargs):
                if hidden_states.shape[0]!=1 or module.training:raise ValueError('One unpadded evaluation sequence required')
                shape=(*hidden_states.shape[:-1],-1,module.head_dim)
                q=module.q_norm(module.q_proj(hidden_states).view(shape)).transpose(1,2)
                k=module.k_norm(module.k_proj(hidden_states).view(shape)).transpose(1,2)
                v=module.v_proj(hidden_states).view(shape).transpose(1,2)
                q,k=apply_rotary_pos_emb(q,k,*position_embeddings)
                if _index not in self.caches:
                    out,_=sdpa_attention_forward(module,q,k,v,attention_mask,dropout=0.,scaling=module.scaling,is_causal=True)
                    self.caches[_index]=DecodeCache(k,v,self.plan,self.capacity,window=self.window)
                else:
                    cache=self.caches[_index];cache.append(k,v)
                    out=cache.attend(q,scale=module.scaling).transpose(1,2)
                    if self.capture is not None and _index in (0,13,27) and _index not in self.capture:
                        # Bounded capture, outside the separate performance pass.
                        g=list(cache.segments())
                        keys=[cache.ck[...,:cache.cpu_count,:]]+[x[0].cpu() for x in g]
                        vals=[cache.cv[...,:cache.cpu_count,:]]+[x[1].cpu() for x in g]
                        self.capture[_index]=(q.float().cpu().numpy(),torch.cat(keys,dim=-2).float().numpy(),torch.cat(vals,dim=-2).float().numpy(),out.transpose(1,2).float().cpu().numpy())
                out=out.reshape(*hidden_states.shape[:-1],-1).contiguous()
                return module.o_proj(out),None
            attn.forward=types.MethodType(forward,attn)
        return self
    def __exit__(self,*args):
        for module,forward in self.original:module.forward=forward
        self.caches.clear()
