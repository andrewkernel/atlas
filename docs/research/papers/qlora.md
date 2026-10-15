# QLoRA: Efficient Finetuning of Quantized LLMs

- **Authors:** Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, Luke Zettlemoyer.
- **Year / venue:** 2023; NeurIPS 2023.
- **Verified primary URL:** [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314).
- **Read depth:** Abstract and proceedings metadata.
- **Objective:** Reduce LLM fine-tuning memory.
- **Technical method:** Frozen quantized base weights with trainable low-rank adapters and memory techniques.
- **Main finding used:** Large fine-tuning memory reductions can be practical.
- **Limitations / Atlas assessment:** Trainable subspace/representation differs from full-parameter fine-tuning.
- **Relevance:** Important alternative, not a strict-preserving default.
- **Code status:** Paper links implementation; not installed or audited.
- **Open question:** Which original hypotheses remain meaningful under adapters?
- **Requires further verification:** No hardware or quality replication here.

Retrieved in this research session; no independent replication is implied.
