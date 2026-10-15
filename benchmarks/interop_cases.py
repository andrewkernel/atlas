"""Original shared cases for pinned TrainParity/Rockmate GPU integration."""
from __future__ import annotations

import torch
from torch import nn
from trainparity.protocols import LossAccounting, TrainingState


class Stack(nn.Module):
    def __init__(self):
        super().__init__()
        values=[]
        for _ in range(4):
            values.extend([nn.Linear(64,128),nn.ReLU(),nn.Linear(128,64),nn.ReLU()])
        self.layers=nn.Sequential(*values)

    def forward(self,x):
        return self.layers(x)


class ConditionalExecution(nn.Module):
    def __init__(self,core,compiled):
        super().__init__();self.core=core
        # Register the scientific parameters once; the compiled runtime refers
        # to this same core but is not another parameter namespace.
        object.__setattr__(self,'compiled',compiled)

    def forward(self,x):
        if x.shape[0]==256:
            return self.core(x)
        if x.shape[0]!=128:
            raise ValueError('This reviewed case supports only 256/full or 128/micro')
        value=self.compiled(x)
        return value[0] if isinstance(value,tuple) else value


class NativeCase:
    equivalence='One fixed logical-256 FP32 MSE/SGD update, explicit denominator and named state'
    planned=False

    def build(self,seed,device):
        if not device.startswith('cuda'):
            raise ValueError('GPU-only integration control')
        torch.set_num_threads(1);torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        model=Stack().to(device)
        if self.planned:
            import rockmate
            config=rockmate.generate_config('rotor');config.solver.top.rotor.force_python=True
            config.partitioner.sequence.sub_partitioner='base'
            with torch.random.fork_rng(devices=[0]):
                compiled=rockmate.from_config(model,torch.zeros(128,64,64,device=device),32*1024**2,
                                             config=config,gpu_optim=None,cpu_optim=None,minor_offload_size=0)
            if compiled.op_sched is None:
                raise RuntimeError('No plan found in integration control')
            model=ConditionalExecution(model,compiled)
        optimizer=torch.optim.SGD(model.parameters(),lr=.01,momentum=.9)
        return TrainingState(model=model,optimizer=optimizer)

    def batch(self,device):
        return {'x':torch.randn(256,64,64,device=device),'target':torch.randn(256,64,64,device=device)}

    def loss(self,state,batch):
        numerator=(state.model(batch['x'])-batch['target']).square().sum()
        denominator=batch['target'].numel()
        return LossAccounting(numerator/denominator,numerator=numerator,denominator=denominator)


class PlannedCase(NativeCase):
    planned=True
