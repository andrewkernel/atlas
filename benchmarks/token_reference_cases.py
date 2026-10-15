"""Authored finite CPU adapters; one deliberately misstates token-weighted intent."""
import torch
from torch import nn
from trainparity.protocols import LossAccounting,TrainingState
class TokenCase:
    equivalence='Declared intended objective is mean over all nonignored target tokens'
    def build(self,seed,device):
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.manual_seed(seed)
        model=nn.Sequential(nn.Embedding(7,3),nn.Linear(3,7)).to(device=device,dtype=torch.float64)
        return TrainingState(model=model,optimizer=torch.optim.SGD(model.parameters(),lr=.05,momentum=.9))
    def batch(self,device):
        return {'x':torch.tensor([[1,2,3,4],[2,4,5,0],[5,0,0,0],[1,3,6,0]],device=device),
                'y':torch.tensor([[2,3,4,5],[3,5,-100,-100],[6,-100,-100,-100],[2,4,0,-100]],device=device)}
    def loss(self,state,batch):
        logits=state.model(batch['x']);y=batch['y']
        total=torch.nn.functional.cross_entropy(logits.reshape(-1,7),y.reshape(-1),ignore_index=-100,reduction='sum')
        count=int((y!=-100).sum())
        return LossAccounting(total/count,numerator=total,denominator=count)
class SequenceMeanCase(TokenCase):
    def loss(self,state,batch):
        logits=state.model(batch['x']);y=batch['y']
        token=torch.nn.functional.cross_entropy(logits.reshape(-1,7),y.reshape(-1),ignore_index=-100,reduction='none').reshape(y.shape)
        total=(token.sum(dim=1)/(y!=-100).sum(dim=1)).sum();count=y.shape[0]
        return LossAccounting(total/count,numerator=total,denominator=count)
