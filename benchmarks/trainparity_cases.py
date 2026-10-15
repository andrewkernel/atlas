"""Original, small CPU cases for the pinned external-checker comparison.

Import only in the separately reviewed Python 3.11 checker environment.
These investigator-authored controls are not an independent workload sample.
"""
from __future__ import annotations

import torch
from torch import nn
from trainparity.protocols import LossAccounting, TrainingState


class LinearCase:
    equivalence = "One float64 logical MSE/SGD update, named state and declared scheduler clock"
    size = 7
    explicit = True

    def build(self, seed: int, device: str) -> TrainingState:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(seed)
        model = nn.Linear(2, 1, bias=False, dtype=torch.float64, device=device)
        with torch.no_grad():
            model.weight.copy_(torch.tensor([[0.25, -0.4]], dtype=torch.float64, device=device))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
        return TrainingState(model=model, optimizer=optimizer, scheduler=scheduler)

    def batch(self, device: str) -> object:
        x = torch.tensor(
            [[1., -2.], [-1., 3.], [2., .5], [4., 1.], [-3., -2.], [.5, 5.], [6., -1.]],
            dtype=torch.float64, device=device,
        )[:self.size]
        y = 0.7 * x[:, :1] - 0.2 * x[:, 1:] + 0.3
        return {"x": x, "y": y}

    def loss(self, state: TrainingState, batch: object) -> LossAccounting:
        assert isinstance(batch, dict)
        residual = state.model(batch["x"]) - batch["y"]
        numerator = residual.square().sum()
        count = residual.numel()
        if self.explicit:
            return LossAccounting(numerator / count, numerator=numerator, denominator=count)
        return LossAccounting(numerator / count)


class BalancedCase(LinearCase):
    size = 6


class NoAccountingCase(BalancedCase):
    explicit = False


class MetadataCase(LinearCase):
    def batch(self, device: str) -> object:
        batch = super().batch(device)
        assert isinstance(batch, dict)
        return {**batch, "metadata": "unsupported string leaf"}


linear_case = LinearCase()
balanced_case = BalancedCase()
no_accounting_case = NoAccountingCase()
metadata_case = MetadataCase()
