# Primary source register

Sources below were retrieved during this session. Descriptions identify documented capabilities, not independent replications. A live documentation URL is not a pinned software environment.

| ID | Source | Evidence used | Read depth |
|---|---|---|---|
| P01 | [Checkmate](https://arxiv.org/abs/1910.02653) | Profile-guided tensor rematerialization planning | Abstract |
| P02 | [Dynamic Tensor Rematerialization](https://arxiv.org/abs/2006.09616) | Online eviction/recomputation for dynamic models | Abstract |
| P03 | [Rockmate](https://arxiv.org/abs/2307.01236) | Automated PyTorch rematerialization | Abstract |
| P04 | [EasyScale](https://arxiv.org/abs/2208.14228) | Accuracy-consistent resource elasticity | Abstract |
| P05 | [Alpa](https://arxiv.org/abs/2201.12023) | Automated distributed parallel plans | Abstract |
| D01 | [PyTorch checkpointing](https://docs.pytorch.org/docs/2.14/checkpoint.html) | RNG preservation boundaries; determinism check is metadata-only | Relevant API sections |
| D02 | [PyTorch AMP examples](https://docs.pytorch.org/docs/2.14/notes/amp_examples.html) | Accumulation, unscaling, clipping and update timing | Relevant sections |
| D03 | [PyTorch reproducibility](https://docs.pytorch.org/docs/2.14/notes/randomness.html) | Platform/release limits; RNG and deterministic operations | Relevant sections |
| D04 | [PyTorch SDPA](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html) | Kernel-dependent numerics and dispatch | Relevant API sections |
| D05 | [DeepSpeed Autotuning](https://github.com/deepspeedai/DeepSpeed/blob/master/deepspeed/autotuning/README.md) | Profiles and searches ZeRO/microbatch configuration; documented offload exclusion | Workflow and scope |
| D06 | [DeepSpeed ZeRO-Offload](https://www.deepspeed.ai/tutorials/zero-offload/) | CPU optimizer offload; reference example needs 32 GB V100 | Overview and environment |
| D07 | [Colossal-Auto](https://colossalai.org/docs/Colossal-Auto/get_started/run_demo/) | Automatic activation checkpoint search | Search excerpt; expand before implementation |
| D08 | [Colossal-AI booster plugins](https://colossalai.org/docs/basics/booster_plugins/) | Gemini heterogeneous memory and feature compatibility caveats | Relevant API sections |

Retrieval date: 2026-10-14 America/Chicago. More sources and exact experimental versions will be added as verified.
