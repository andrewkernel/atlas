"""Finite composite/moving-denominator controls; no new general verifier."""
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
CONFIG = {'seed': 2601, 'dtype': 'float64', 'partition': [4, 3], 'ignore_index': 5,
          'sgd_lr': .05, 'rtol': 1e-10, 'atol': 1e-12, 'finite_difference_step': 1e-6}


def tensor_rows(model, attribute=None):
    return {name: (getattr(p, attribute) if attribute else p).detach().flatten().tolist()
            for name, p in model.named_parameters()}


def maximum_error(left, right):
    assert left.keys() == right.keys()
    return max(abs(a - b) for name in left for a, b in zip(left[name], right[name], strict=True))


def head_case(labels):
    torch.manual_seed(CONFIG['seed'])
    original = torch.nn.Linear(3, 10).to(torch.float64)
    x = torch.arange(21, dtype=torch.float64).reshape(7, 3) / 13 - .5
    targets = torch.tensor(labels).T.contiguous()
    global_counts = (targets != 5).sum(0)
    if bool((global_counts == 0).any()):
        return {'status': 'unsupported_undefined_global_head_mean', 'global_valid_counts': global_counts.tolist()}
    outcomes = {}
    for mode in ('full', 'global_head_sums', 'naive_example_weighted_means'):
        model = copy.deepcopy(original)
        optimizer = torch.optim.SGD(model.parameters(), lr=CONFIG['sgd_lr'])
        chunks = [(0, 7)] if mode == 'full' else [(0, 4), (4, 7)]
        total_loss = 0.
        local_counts = []
        local_losses = []
        for begin, stop in chunks:
            logits = model(x[begin:stop]).reshape(stop - begin, 2, 5)
            target = targets[begin:stop]
            counts = (target != 5).sum(0)
            local_counts.append(counts.tolist())
            if mode == 'global_head_sums':
                loss = sum(F.cross_entropy(logits[:, head], target[:, head], ignore_index=5, reduction='sum') /
                           global_counts[head] / 2 for head in range(2))
            else:
                loss = sum(F.cross_entropy(logits[:, head], target[:, head], ignore_index=5, reduction='mean') / 2
                           for head in range(2)) * ((stop - begin) / 7)
            value = float(loss.detach())
            local_losses.append(value if math.isfinite(value) else None)
            total_loss += value
            loss.backward()
        gradients = tensor_rows(model, 'grad')
        healthy = math.isfinite(total_loss) and all(math.isfinite(v) for values in gradients.values() for v in values)
        if healthy:
            optimizer.step()
        outcomes[mode] = {'loss': total_loss if math.isfinite(total_loss) else None,
                          'loss_finite': math.isfinite(total_loss), 'local_loss_values': local_losses,
                          'local_valid_counts': local_counts, 'gradients': gradients,
                          'gradients_finite': all(math.isfinite(v) for values in gradients.values() for v in values),
                          'optimizer_step_performed': healthy, 'post_parameters': tensor_rows(model)}
    for mode in ('global_head_sums', 'naive_example_weighted_means'):
        outcomes[mode]['gradient_max_abs_error'] = maximum_error(outcomes['full']['gradients'], outcomes[mode]['gradients'])
        outcomes[mode]['post_parameter_max_abs_error'] = maximum_error(outcomes['full']['post_parameters'], outcomes[mode]['post_parameters'])
    return {'status': 'observed', 'global_valid_counts': global_counts.tolist(), 'outcomes': outcomes}


def moving_denominator():
    x = torch.linspace(-1.2, 1.3, 7, dtype=torch.float64)
    target = torch.cos(1.7 * x)
    initial = torch.tensor([.4, -.2], dtype=torch.float64)

    def terms(theta, begin=0, stop=7):
        prediction = theta[0] * x[begin:stop] + theta[1]
        weights = F.softplus(prediction)
        return (weights * (prediction - target[begin:stop]).square()).sum(), weights.sum()

    theta = initial.clone().requires_grad_(True)
    numerator, denominator = terms(theta)
    full_loss = numerator / denominator
    full_gradient, = torch.autograd.grad(full_loss, theta)
    theta = initial.clone().requires_grad_(True)
    numerator, denominator = terms(theta)
    detached_loss = numerator / denominator.detach()
    detached_gradient, = torch.autograd.grad(detached_loss, theta)
    numerator_value = denominator_value = 0.
    numerator_gradient = torch.zeros_like(initial)
    denominator_gradient = torch.zeros_like(initial)
    for begin, stop in ((0, 4), (4, 7)):
        theta = initial.clone().requires_grad_(True)
        numerator, denominator = terms(theta, begin, stop)
        grad_s, = torch.autograd.grad(numerator, theta, retain_graph=True)
        grad_d, = torch.autograd.grad(denominator, theta)
        numerator_value += float(numerator.detach())
        denominator_value += float(denominator.detach())
        numerator_gradient += grad_s
        denominator_gradient += grad_d
    quotient_gradient = numerator_gradient / denominator_value - numerator_value * denominator_gradient / denominator_value**2
    finite_difference = []
    for index in range(2):
        plus, minus = initial.clone(), initial.clone()
        plus[index] += CONFIG['finite_difference_step']
        minus[index] -= CONFIG['finite_difference_step']
        s_plus, d_plus = terms(plus)
        s_minus, d_minus = terms(minus)
        finite_difference.append(float((s_plus / d_plus - s_minus / d_minus) / (2 * CONFIG['finite_difference_step'])))
    return {'full_loss': float(full_loss.detach()), 'detached_denominator_loss': float(detached_loss.detach()),
            'streamed_forward_value': numerator_value / denominator_value, 'full_gradient': full_gradient.tolist(),
            'detached_denominator_gradient': detached_gradient.tolist(), 'streamed_quotient_gradient': quotient_gradient.tolist(),
            'finite_difference_gradient': finite_difference,
            'detached_denominator_gradient_max_abs_error': float((detached_gradient - full_gradient).abs().max()),
            'streamed_gradient_max_abs_error': float((quotient_gradient - full_gradient).abs().max()),
            'finite_difference_max_abs_error': float((torch.tensor(finite_difference, dtype=torch.float64) - full_gradient).abs().max())}


def run_controls():
    equal = [[0, 1, 2, 3, 5, 5, 0], [0, 5, 5, 1, 2, 3, 4]]
    unequal = [equal[0], [0, 5, 5, 1, 2, 3, 5]]
    empty_local = [[5, 5, 5, 5, 0, 1, 2], equal[1]]
    undefined = [[5] * 7, equal[1]]
    return {'equal_global_counts': head_case(equal), 'unequal_global_counts': head_case(unequal),
            'empty_local_head': head_case(empty_local), 'undefined_global_head': head_case(undefined),
            'parameter_dependent_denominator': moving_denominator()}


def main():
    torch.set_num_threads(1)
    result = {'captured_utc': datetime.now(timezone.utc).isoformat(),
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'config': CONFIG, 'environment': {'python': platform.python_version(), 'torch': torch.__version__, 'device': 'cpu'},
              'controls': run_controls(),
              'scope': 'Authored finite float64 objective controls, known normalization/quotient-rule mathematics; no new checker or real-BERT fidelity claim.'}
    target = ROOT / 'docs/research/experiments/composite-objective-control.json'
    target.write_bytes((json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n').encode())
    print(json.dumps({name: {'status': value.get('status'),
                             'naive_gradient_error': value.get('outcomes', {}).get('naive_example_weighted_means', {}).get('gradient_max_abs_error')}
                      for name, value in result['controls'].items()}))


if __name__ == '__main__':
    main()
