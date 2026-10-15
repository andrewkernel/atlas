"""Finite head-bias invariance and floating-point boundary observations."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]


def observe(dtype, bias, weight_shift=0.):
    head = torch.nn.Linear(3, 2, bias=True, dtype=dtype)
    with torch.no_grad():
        head.weight.copy_(torch.tensor([[.2, -.1, .05], [-.3, .15, .4]], dtype=dtype))
        head.weight[0, 0].add_(weight_shift)
        head.bias.copy_(torch.tensor(bias, dtype=dtype))
    inputs = torch.arange(30, dtype=dtype).reshape(2, 5, 3) / 17 - .7
    logits = head(inputs)
    probabilities = logits.softmax(dim=1)
    centered = logits - logits.mean(dim=1, keepdim=True)
    loss = (F.cross_entropy(logits[:, :, 0], torch.tensor([1, 3])) +
            F.cross_entropy(logits[:, :, 1], torch.tensor([2, 4]))) / 2
    loss.backward()
    return {'loss': float(loss.detach()), 'logits': logits.detach().tolist(),
            'centered_logits': centered.detach().tolist(), 'probabilities': probabilities.detach().tolist(),
            'argmax_positions': logits.argmax(dim=1).tolist(), 'weight': head.weight.detach().tolist(),
            'bias': head.bias.detach().tolist(), 'weight_gradient': head.weight.grad.tolist(),
            'bias_gradient': head.bias.grad.tolist(),
            'all_observed_values_finite': all(bool(torch.isfinite(value).all()) for value in (
                loss, logits, probabilities, centered, head.weight, head.bias, head.weight.grad, head.bias.grad))}


def compare(reference, candidate):
    values = {name + '_max_abs_difference': float((torch.tensor(reference[name], dtype=torch.float64) -
                                                  torch.tensor(candidate[name], dtype=torch.float64)).abs().max())
              for name in ('logits', 'centered_logits', 'probabilities', 'weight', 'bias', 'weight_gradient', 'bias_gradient')}
    values.update({'absolute_loss_difference': abs(reference['loss'] - candidate['loss']),
                   'argmax_equal': reference['argmax_positions'] == candidate['argmax_positions']})
    return values


def run_controls():
    reference64 = observe(torch.float64, [0., 0.])
    reference32 = observe(torch.float32, [0., 0.])
    cases = {'repeat_float64': observe(torch.float64, [0., 0.]),
             'bias_shift_float64': observe(torch.float64, [.3, -.2]),
             'weight_change_float64': observe(torch.float64, [0., 0.], weight_shift=.1),
             'large_bias_float32': observe(torch.float32, [1e8, 1e8])}
    return {'reference_float64': reference64, 'reference_float32': reference32,
            'cases': {name: {'observations': value, 'comparison': compare(reference32 if name.endswith('float32') else reference64, value)}
                      for name, value in cases.items()}}


def main():
    torch.set_num_threads(1)
    result = {'captured_utc': datetime.now(timezone.utc).isoformat(),
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'environment': {'python': platform.python_version(), 'torch': torch.__version__, 'device': 'cpu'},
              'controls': run_controls(), 'scope': 'Tiny finite relation/bias-invariance controls, including an artificial FP32 boundary; no BERT outcome, policy or general functional-equivalence claim.'}
    target = ROOT / 'docs/research/experiments/qa-gauge-control.json'
    target.write_bytes((json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n').encode())
    print(json.dumps({name: value['comparison'] for name, value in result['controls']['cases'].items()}))


if __name__ == '__main__':
    main()
