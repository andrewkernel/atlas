# Atlas source

`attention/` implements the owner-authorized local Attention Lab: dense persistent CPU/GPU KV caches, benchmarks, a calibrated selector, a pretrained GPT-2 adapter, evidence records, and a loopback API. Run `lab.py` from the repository root; see [the guide](../../docs/attention-lab.md).

The earlier analysis/profiling/optimization/training-fidelity architecture remains a separate research specification. Attention Lab does not implement that entire training optimizer.
