# Dependency and license verification status

The [machine-readable snapshot](dependency-snapshot.json) records exact installed versions or inspected optional package metadata, Python requirements, license fields, source endpoints and its actual retrieval time. [uv.lock](../../uv.lock) pins the research environment, including the explicit official PyTorch CPU index. This is not a validated CUDA or full-application environment.

| Component | Observed version | Python requirement in inspected metadata | License information / validation |
|---|---|---|---|
| PyTorch CPU |2.14.1+cpu |>=3.10 | Composite license expression recorded; do not reduce all bundled notices to one guessed SPDX identifier; CPU probes executed |
| NumPy |2.5.4 |>=3.12 | Composite BSD/0BSD/MIT/Zlib/CC0 expression; environment installed/imported |
| SciPy |1.18.1 |>=3.12 | BSD classifier and license text; installed/imported, no statistical outcome analysis executed |
| pytest |9.1.1 |>=3.10 | MIT; test runner executed |
| Hypothesis |6.169.3 |>=3.11 | MPL-2.0; generated controls executed |
| psutil |7.2.2 |>=3.6 | BSD-3-Clause metadata; installed, production sampler not implemented |
| uv |0.13.1 |>=3.8 | MIT OR Apache-2.0; lock/sync/run executed |
| FastAPI |0.143.1 metadata only |>=3.10 | MIT; not installed or integration-tested |
| Pydantic |2.14.1 metadata only |>=3.10 | MIT; not installed or integration-tested |
| MLflow |3.17.1 metadata only |>=3.10 | Apache classifier/text; not installed or integration-tested |
| nvidia-ml-py |13.615.71 executed separately |No requirement declared | BSD metadata; read-only counter/power APIs run on Python 3.12.10 / RTX 3060 / driver 595.95; absolute accuracy unresolved |

Python 3.12.10 is admitted by explicit requirements for the listed packages with requirements. **Missing** requirement metadata is unknown, not a compatibility guarantee. Successful environment resolution and CPU tests are stronger evidence for that narrow setup than metadata alone, but do not validate all combinations of optional components.

GitHub's repository license detector returned `NOASSERTION` for some mixed-license repositories; see the [repository snapshot](competitors/repository-snapshot.json). This must not become “unlicensed” or an inferred permission. Inspect actual notices before incorporating code. Pinned MIT research source was captured in ignored local reference directories and executed only within the reviewed control scopes; it is not vendored into Atlas's tracked source. Public licensing for Atlas itself remains an owner decision; the repository was created private.

Separate [checker](experiments/checker-environment.json) and [planner](experiments/planner-environment.json) environment records now identify the exact installed packages and hash-pinned requirements used for external controls. They do not change the CPU lock or earlier CUDA fixture environment. Planner dependencies include the source-required PuLP 2.7.0 and explicitly added PyYAML; no commercial solver license or C extension build was required for these Python-solver controls. Bundled binary/dependency notices remain with the original distributions; a future redistributed application needs its own complete notice inventory.

Model licenses were checked individually: Qwen3-4B and Qwen2.5-Coder-7B are tagged Apache-2.0; Phi-4-mini MIT; Qwen2.5-Coder-3B has distinct research terms and is excluded as the default. Quantized distributions need separate provenance/digest/terms verification. No weights were downloaded.

Fashion-MNIST declares MIT. WikiText's pinned card has inconsistent licensing metadata/prose; resolve before acquiring/redistributing planned inputs. SST-2 metadata says unknown and was not selected as the default. These findings are recorded so future sessions do not silently treat online availability as established reuse rights.

A third [shared interop environment](experiments/interop-environment.json) pins Python 3.11.17/Torch 2.13.0+cu130/NumPy 2.4.6 within both inspected version ranges. Its official CUDA wheel/dependency hashes and actual finite controls are recorded separately; no historical environment or core lock was modified.

[Read-only telemetry](experiments/board-energy-control.md) uses its own one-package [hash-pinned requirements](../../configs/telemetry-windows-py312-requirements.txt). The recorded installed binding source hash binds the finite availability/workload/passive captures; it does not establish general device support or calibration.

The [post-experiment environment audit](experiments/final-environment-audit.md) reconciles six environments, earlier distribution versions and 31 selected installed Python-source RECORD hashes. It explicitly limits those checks and retains an initial name-normalization mistake; no package/driver change follows.

The second continuation adds a separate [QA stack](workload-screen/qa-environment.json): Python 3.11.17, Torch 2.13.0+cu130, Transformers 5.19.0 and NumPy 2.4.6, with 35 pinned binary distributions. Local BERT safetensors/config/tokenizer and SQuAD training bytes are now acquired under the recorded Apache-2.0 model and CC BY-SA 4.0 dataset declarations; the earlier blanket absence of weights is historical for these additions. No local-LLM weights or inference are added. [Acquisition identities](workload-screen/qa-asset-manifest.json) and local notices retain provenance.

An optional isolated plotting environment uses 11 hash-pinned packages, including Matplotlib 3.11.2 (Python >=3.11; [PSF-based license](https://github.com/matplotlib/matplotlib/blob/main/doc/project/license.rst)) to render the measured resource figure. It is not a new core or ML-worker dependency. [Post-study reconciliation](workload-screen/post-study-environments.md) now covers all eight profiles, selected source and QA/reference assets; it retains the missing contemporaneous per-QA-worker driver/cuDNN metadata rather than treating a late read as execution-time evidence.
