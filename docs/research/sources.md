# Source register and evidence depth

This is a targeted primary-source scoping review. Most sources were retrieved on 2026-10-14 America/Chicago (2026-10-15 UTC); machine-readable snapshots carry exact retrieval times. Source claims are not independent replications. Paper abstracts, selected sections and repository documentation have different evidence depth; the table states that distinction.

Search families and limitations are in the [literature review](02-literature-review.md). Existing systems were not executed. Analytical judgments, proposed contracts and plans are Atlas proposals, not facts attributed to the papers. A recent preprint is identified as such. No search establishes global absence of prior work.

## Papers

| Verified primary paper link | Research note | Read depth |
|---|---|---|
| [ActNN: Reducing Training Memory Footprint via 2-Bit Activation Compressed Training](https://arxiv.org/abs/2104.14129) | [Note](papers/actnn.md) | Abstract only; no theorem audit. |
| [Adafactor: Adaptive Learning Rates with Sublinear Memory Cost](https://proceedings.mlr.press/v80/shazeer18a.html) | [Note](papers/adafactor.md) | Proceedings abstract and introduction. |
| [Alpa: Automating Inter- and Intra-Operator Parallelism for Distributed Deep Learning](https://arxiv.org/abs/2201.12023) | [Note](papers/alpa.md) | Abstract and official architecture overview. |
| [Ansor: Generating High-Performance Tensor Programs for Deep Learning](https://arxiv.org/abs/2006.06762) | [Note](papers/ansor.md) | Abstract only. |
| [Capuchin: Tensor-based GPU Memory Management for Deep Learning](https://alchem.usc.edu/portal/static/download/capuchin.pdf) | [Note](papers/capuchin.md) | Author-hosted paper metadata and first-party overview. |
| [Checkmate: Breaking the Memory Wall with Optimal Tensor Rematerialization](https://proceedings.mlsys.org/paper_files/paper/2020/hash/0b816ae8f06f8dd3543dc3d9ef196cab-Abstract.html) | [Note](papers/checkmate.md) | Paper formulation and evaluation setup inspected; not replicated. |
| [Chisel: Reliability- and Accuracy-Aware Optimization of Approximate Computational Kernels](https://people.csail.mit.edu/rinard/paper/oopsla14.pdf) | [Note](papers/chisel.md) | Abstract and first-page design context. |
| [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) | [Note](papers/compute-optimal.md) | Abstract only. |
| [A Modular Benchmarking Infrastructure for High-Performance and Reproducible Deep Learning](https://arxiv.org/abs/1901.10183) | [Note](papers/deep500.md) | Abstract and repository README. |
| [When More Memory Fails: Hidden Feasibility Holes in Dynamic Tensor Rematerialization](https://arxiv.org/abs/2610.14369) | [Note](papers/dtr-feasibility-holes.md) | Abstract only; exploratory context, not foundation of verdict. |
| [Dynamic Tensor Rematerialization](https://ztatlock.net/pubs/2021-iclr-dtr/) | [Note](papers/dtr.md) | Author abstract and project metadata inspected. |
| [EasyScale: Elastic Training with Consistent Accuracy and Improved Utilization on GPUs](https://arxiv.org/html/2208.14228v2) | [Note](papers/easyscale.md) | Design and consistency/overhead evaluation sections inspected. |
| [Equivalence Tests: A Practical Primer for t Tests, Correlations, and Meta-Analyses](https://doi.org/10.1177/1948550617697177) | [Note](papers/equivalence.md) | Publisher abstract and TOST explanation. |
| [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) | [Note](papers/flashattention.md) | Abstract and native SDPA documentation. |
| [Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization](https://www.jmlr.org/papers/volume18/16-558/16-558.pdf) | [Note](papers/hyperband.md) | Abstract and introductory methodology. |
| [KernelBench: Can LLMs Write Efficient GPU Kernels?](https://arxiv.org/abs/2502.10517) | [Note](papers/kernelbench.md) | Abstract and repository evaluation description. |
| [Learning to learn by gradient descent by gradient descent](https://arxiv.org/abs/1606.04474) | [Note](papers/learned-optimization.md) | Abstract only. |
| [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/abs/2104.04473) | [Note](papers/megatron.md) | Abstract and official repository overview. |
| [Mixed Precision Training](https://arxiv.org/abs/1710.03740) | [Note](papers/mixed-precision.md) | Abstract and current AMP guidance. |
| [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) | [Note](papers/qlora.md) | Abstract and proceedings metadata. |
| [Rockmate: an Efficient, Fast, Automatic and Generic Tool for Re-materialization in PyTorch](https://proceedings.mlr.press/v202/zhao23b.html) | [Note](papers/rockmate.md) | Proceedings abstract; repository metadata. |
| [Training with Confidence: Catching Silent Errors in Deep Learning Training with Automated Proactive Checks](https://www.usenix.org/conference/osdi25/presentation/jiang) | [Note](papers/traincheck.md) | Paper introduction/design and official technical/AE guides. |
| [Accounting for Variance in Machine Learning Benchmarks](https://arxiv.org/abs/2103.03098) | [Note](papers/variance.md) | Abstract and comparison-method passages. |
| [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054) | [Note](papers/zero.md) | Abstract; DeepSpeed documentation inspected separately. |

## Official documentation, repositories, models and datasets

| ID | Primary source | Inspection / limitation |
|---|---|---|
| D01 | [PyTorch checkpoint API](https://docs.pytorch.org/docs/2.14/checkpoint.html) | Relevant recomputation/RNG/determinism sections |
| D02 | [PyTorch AMP examples](https://docs.pytorch.org/docs/2.14/notes/amp_examples.html) | Accumulation, clipping, scaling sections |
| D03 | [PyTorch reproducibility](https://docs.pytorch.org/docs/2.14/notes/randomness.html) | Release/platform limits and deterministic operations |
| D04 | [PyTorch SDPA](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html) | Backend numerical differences and parameters |
| D05 | [PyTorch CUDA semantics](https://docs.pytorch.org/docs/2.14/notes/cuda.html) | Allocated/reserved memory distinction |
| D06 | [PyTorch numerical accuracy](https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html) | Floating-point and batched/sliced limitations |
| D07 | [PyTorch FX](https://docs.pytorch.org/docs/2.14/fx.html) | Symbolic-tracing pipeline and dynamic-control-flow limits |
| D08 | [PyTorch FSDP2](https://docs.pytorch.org/docs/2.14/distributed.fsdp.fully_shard.html) | Sharding/communication/offload API |
| D09 | [PyTorch selective checkpoint and memory budget](https://pytorch.org/blog/activation-checkpointing-techniques/) | First-party mechanisms and version-sensitive examples |
| D10 | [PyTorch profiler](https://docs.pytorch.org/docs/2.14/profiler.html) | API overview; implementation profiling not performed |
| D11 | [PyTorch zero_grad](https://docs.pytorch.org/docs/2.14/generated/torch.optim.Optimizer.zero_grad.html) | None-versus-zero optimizer behavior |
| D12 | [DeepSpeed Autotuning](https://github.com/deepspeedai/DeepSpeed/blob/master/deepspeed/autotuning/README.md) | Workflow, search scope, offload exclusion |
| D13 | [DeepSpeed training overview](https://www.deepspeed.ai/training/) | Mechanism overview |
| D14 | [DeepSpeed configuration](https://www.deepspeed.ai/docs/config-json/) | Batch accounting and offload fields |
| D15 | [DeepSpeed ZeRO-Offload](https://www.deepspeed.ai/tutorials/zero-offload/) | Overview and cited hardware environment |
| D16 | [Colossal-Auto](https://colossalai.org/docs/Colossal-Auto/get_started/run_demo/) | Automatic checkpoint-search documentation excerpt |
| D17 | [Colossal-AI plugins](https://colossalai.org/docs/basics/booster_plugins/) | Gemini/API compatibility sections; current build not tested |
| D18 | [Accelerate accumulation](https://huggingface.co/docs/accelerate/en/usage_guides/gradient_accumulation) | Current guide, update and token-accounting behavior |
| D19 | [Hugging Face accumulation correction](https://huggingface.co/blog/gradient_accumulation) | Primary bug explanation and summed-loss denominator correction |
| D20 | [Megatron repository](https://github.com/NVIDIA/Megatron-LM) | Official overview and metadata |
| D21 | [TrainCheck technical guide](https://orderlab.io/TrainCheck/technical-doc/) | Instrumentation/inference/checking stages |
| D22 | [TrainCheck artifact guide](https://orderlab.io/TrainCheck/ae/) | Full artifact hardware requirements; not reproduced |
| D23 | [TrainCheck quick tutorial](https://orderlab.io/TrainCheck/5-min-tutorial/) | CPU feasibility and scoped non-triggered checks |
| D24 | [TrainParity](https://github.com/intelland/TrainParity) | README, source accumulation phases/initial-state checks at captured head |
| D25 | [TrainParity comparison](https://github.com/intelland/TrainParity/blob/main/docs/comparison-with-traincheck.md) | First-party related-work distinctions; independently cross-checked TrainCheck sources |
| D26 | [Deep500](https://github.com/deep500/deep500) | README, interfaces and license metadata |
| D27 | [KernelBench](https://github.com/ScalingIntelligence/KernelBench) | README/evaluation scope |
| D28 | [FPBench](https://github.com/FPBench/FPBench) | README, benchmark/measurement standards |
| D29 | [Herbie](https://fpbench.cs.washington.edu/) | Project overview; no numerical tool executed |
| D30 | [MLflow tracking](https://mlflow.org/docs/latest/ml/tracking/) | Tracking overview; package not installed |
| D31 | [Docker security](https://docs.docker.com/engine/security/) | Daemon, capability and configuration attack surfaces |
| D32 | [Docker resource limits](https://docs.docker.com/engine/containers/resource_constraints/) | CPU/host memory controls; no sandbox validated |
| D33 | [NVML device queries](https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlDeviceQueries.html) | Energy API discovery; device support not tested |
| D34 | [Ollama structured output](https://docs.ollama.com/capabilities/structured-outputs) | Schema response capability; model correctness not established |
| D35 | [Ollama FAQ](https://docs.ollama.com/faq) | Lifecycle and memory/concurrency controls |
| D36 | [FastAPI types](https://fastapi.tiangolo.com/python-types/) | Typed boundary overview |
| D37 | [Pydantic models](https://docs.pydantic.dev/latest/concepts/models/) | Validation boundary overview |
| D38 | [uv lock/sync](https://docs.astral.sh/uv/concepts/projects/sync/) | Locked project synchronization |
| M01 | [Qwen2.5-Coder-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct) | First-party card; research-license distinction |
| M02 | [Qwen2.5-Coder-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct) | Card architecture/context/license; no Atlas inference evaluation |
| M03 | [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) | Card context/modes/license; no Atlas inference evaluation |
| M04 | [Phi-4-mini-instruct](https://huggingface.co/microsoft/Phi-4-mini-instruct) | Card parameters/context/function format/license |
| B01 | [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) | Dataset counts/download requirements and MIT declaration |
| B02 | [WikiText](https://huggingface.co/datasets/Salesforce/wikitext) | Dataset card; conflicting license fields explicitly recorded |
| B03 | [SST-2](https://huggingface.co/datasets/stanfordnlp/sst2) | Unknown license metadata; not default workload |
| B04 | [DistilBERT](https://huggingface.co/distilbert/distilbert-base-uncased) | Model card and Apache-2.0 metadata |
| B05 | [PyTorch examples](https://github.com/pytorch/examples) | Pinned directory inventory and language-model class snippets, not executed |
| V01 | [ACM Student Research Competition](https://src.acm.org/?searchterm=src) | Official description; no deadline claim |
| V02 | [MLSys conference](https://mlsys.org/Conferences/2026) | Official conference context; no workshop/acceptance promise |
| V03 | [JOSS](https://joss.theoj.org/about) | Official software-journal scope; this specification is not a submission |

## Reproducible source identities

Continuation sources: [HiLoRe primary preprint](https://arxiv.org/abs/2609.33570v1), selected HTML method/theory/evaluation sections ([note](papers/hilore.md)); pinned [FairScale source](competitors/fairscale.md), selected checkpoint functions; [TOSTER exact-power reference](https://aaroncaldwell.us/TOSTERpkg/reference/power_t_TOST.html), paired-SD convention and exact-power methods. Neither HiLoRe/FairScale nor the TOSTER R package was executed. A failed generic TOSTER introduction URL was replaced by the working specific reference page. The original metadata snapshot is historical and was not rewritten to imply it already covered these additions.

CUDA continuation used the official [PyTorch wheel index](https://download.pytorch.org/whl/cu130/torch/), [2.14 release information](https://pytorch.org/blog/pytorch-2-14-release-blog/), [NVIDIA minor-version compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html), and [PyTorch CUDA precision/allocator API](https://docs.pytorch.org/docs/2.14/notes/cuda.html). Actual compatibility is supported only by the recorded smoke and synthetic fixture, not inferred from the documentation alone. No driver installation was performed.

- [GitHub metadata snapshot](competitors/repository-snapshot.json):15 public repositories, exact heads/archive/license metadata; authenticated read-only API.
- [Hugging Face snapshot](asset-snapshot.json): model/data revision hashes and card license metadata; no data/weights downloaded.
- [Dependency snapshot](dependency-snapshot.json): PyPI metadata, Python requirement and installed-versus-optional distinction. These primary endpoints are recorded per package.
- CPU PyTorch package source and artifact hashes are pinned in [uv.lock](../../uv.lock).

Later targeted source inspection: TrainParity accumulation.py confirms declared phases and initial-state/self-consistency abstention at its captured commit; TrainCheck pyproject.toml declares version0.1.2, Python>=3.10 and Apache-2.0 text. These observations do not validate runtime compatibility. The snapshot's NOASSERTION license detector is preserved rather than overwritten by an inferred repository-wide license.

## Retrieval problems and exclusions

Some browser opens returned internal errors, including two PyTorch example directory pages and an initial NVML path. Directory contents were subsequently checked through GitHub's primary API; the NVML latest API path was located by search. An incorrectly guessed arXiv identifier resolved to an unrelated mathematics paper and was discarded. No claim or citation relies on it. Hyperband has a2016 preprint and 2018 JMLR publication; EasyScale's2022 preprint title differs from its2023 SC version. Notes preserve these distinctions.

No paper PDF, third-party code, dataset or weights were committed. For an implementation or publication claim, repeat the literature search and inspect current/pinned code rather than assuming a documentation absence means a capability is absent.
