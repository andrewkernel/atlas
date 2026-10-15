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
| nvidia-ml-py |13.615.71 metadata only |No requirement declared | BSD classifier; actual driver/API support and Python compatibility untested |

Python 3.12.10 is admitted by explicit requirements for the listed packages with requirements. **Missing** requirement metadata is unknown, not a compatibility guarantee. Successful environment resolution and CPU tests are stronger evidence for that narrow setup than metadata alone, but do not validate all combinations of optional components.

GitHub's repository license detector returned `NOASSERTION` for some mixed-license repositories; see the [repository snapshot](competitors/repository-snapshot.json). This must not become “unlicensed” or an inferred permission. Inspect the actual notices before incorporating code. No third-party research source was copied into Atlas during this session. Public licensing for Atlas itself remains an owner decision; the repository was created private.

Model licenses were checked individually: Qwen3-4B and Qwen2.5-Coder-7B are tagged Apache-2.0; Phi-4-mini MIT; Qwen2.5-Coder-3B has distinct research terms and is excluded as the default. Quantized distributions need separate provenance/digest/terms verification. No weights were downloaded.

Fashion-MNIST declares MIT. WikiText's pinned card has inconsistent licensing metadata/prose; resolve before acquiring/redistributing planned inputs. SST-2 metadata says unknown and was not selected as the default. These findings are recorded so future sessions do not silently treat online availability as established reuse rights.
