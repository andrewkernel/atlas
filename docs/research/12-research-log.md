# Chronological research log

Times are observed UTC times unless stated otherwise; local date at session start was 2026-10-14 America/Chicago. The requested five-hour allocation is a prioritization guide. No five-hour claim is made.

## 2026-10-15 04:39 UTC — environment and prior art

- Read the complete attached assignment. Working directory is a ChatGPT project mirror, not a Git checkout. `sources/` contains no files. Existing root AGENTS.md is read-only and was preserved.
- GitHub connector repository search returned an access/search error. Authenticated GitHub CLI worked; no Atlas repository was present in the account listing.
- Initial literature searches found Checkmate, DTR, Rockmate, Alpa and EasyScale. Broad automatic-memory-optimization novelty is not credible; resource-consistency preservation also has established prior art.
- Created a separate local `atlas/` repository. User explicitly authorized creating a GitHub repository named `atlas`; created private `andrewkernel/atlas` successfully.

## 2026-10-15 04:40 UTC — available hardware and initial boundaries

- Observed Windows host, AMD Ryzen 5 7600X, 12 logical processors, 33,412,775,936 bytes system RAM; NVIDIA RTX 3060, 12,288 MiB VRAM, driver 595.95. Availability does not establish CUDA/PyTorch compatibility.
- Python 3.12.10 is installed; system default is Python 3.14.6. PyTorch was absent in both. Python 3.12 already has NumPy, SciPy and pytest available outside the proposed isolated environment.
- Ollama CLI reports no downloaded models. Its listing command started its local service automatically. No model was downloaded or inference benchmark performed.
- PyTorch live `stable` documentation redirected to 2.14; this does not determine the experimental package version. Exact versions must be recorded separately.
- Checkpoint API's default determinism check compares metadata, not values. This is material to the proposed verifier: passing that check cannot certify numerical or scientific fidelity.

## 2026-10-15 04:43 UTC — isolated environment and original probes

- Installed uv into a local virtual environment, then official PyTorch CPU wheel and the small research dependency set. Python 3.12.10 / Torch 2.14.1+cpu; no CUDA build or model weights installed. `uv.lock` resolved and synced successfully.
- At 04:43:58 UTC, original CPU probes found correct weighted accumulation within float64 rounding, denominator/BatchNorm counterexamples, dropout gradient drift without checkpoint RNG preservation, and BatchNorm-buffer drift despite identical gradients. Raw evidence and source hash saved.
- Initial eight tests passed, including 40 deterministic Hypothesis partition examples. These are bounded semantics controls, not full-training validation.
- Found TrainParity and TrainCheck as close prior art. Revised the contribution away from a generic correctness checker toward contract-conditioned plan selection and interactions. Added composed native+checker workflow as an essential baseline.

## 2026-10-15 04:44 UTC — primary-source snapshots

- Captured15 repository heads/archive/license metadata and seven model/dataset revisions. Alpa's repository is archived; current compatibility of historical memory systems remains unknown.
- Qwen2.5-Coder-3B has research terms distinct from the 7B model; excluded it as the default. WikiText's card has conflicting license metadata/prose; SST-2 license metadata is unknown. Recorded these as planned-evaluation prerequisites, not cleared rights.
- No research repository code, public dataset or model weights were executed/downloaded as a workload.

## 2026-10-15 04:49 UTC — second Git checkpoint

- Committed and pushed the prior-art assessment, initial fidelity framework, original probes, tests, environment lock and source metadata on `research/atlas-spec` (commit 4dcab28). Initial repository/source checkpoint is 4a3dcda on main.
- Accepted adapter-based, native finite planning and scoped evidence boundaries. No full Atlas application development started.

## 2026-10-15 04:57 UTC — synthesis and dependency metadata

- Completed the initial architecture/API/security design, benchmark/statistical plan, six research candidate rankings, three explicitly authored adversarial review perspectives and roadmap.
- Inspected primary package metadata for selected/optional components; recorded installed versus metadata-only status. CPU package compatibility does not validate CUDA/API/MLflow integration.
- Selected source inspection later confirmed TrainParity's accumulation phases and baseline/initial-state abstention logic at its pinned head. TrainCheck's pyproject declares Python>=3.10 and Apache-2.0 metadata. This is a targeted code read, not a code audit or replication.

## 2026-10-15 05:05 UTC — composed-update controls and consistency review

- At 05:05:10 UTC, a second original CPU experiment tested two AdamW logical updates under checkpointing plus3+4 accumulation. Correct composition differed in final parameters by 5.55e-17; clipping each partial gradient caused 0.033198 error; stepping each microbatch caused 0.024045 error and doubled update counters.
- Expanded suite: 11 tests passed, including the original 40 generated partition controls. Original evidence files were preserved; the composition experiment has its own code hash and result file.
- Initial offline audit found no missing required documents, broken local links, invalid JSON, stale probe hashes or unresolved placeholder markers. Final validation and repository status are recorded separately at completion.

## Limits and unfinished research

This is a bounded initial research/engineering investigation, not five hours of research. Read depth varies:24 individual paper notes include many abstract-level notes and selected deeper reads of the closest systems. There was no systematic forward-citation review, independent expert review/labeling, competing-system execution, native CUDA study, full-training equivalence study, LLM inference comparison or energy measurement. The scoped specification is complete as an initial blueprint; these evidence gaps remain explicitly open in `11-open-questions.md`.

## 2026-10-15 05:08 UTC — validation and handoff

- Complete specification checkpoint45477f4 was committed and pushed. GitHub Actions run38685042339 completed successfully on Linux and Windows; local11-test suite and offline specification audit passed.
- Tightened the primary question to an explicit safety/coverage comparison with native mechanisms plus existing checks under a fixed initial search budget. Added an expanded-access utility gate, while keeping full scientific-outcome claims separate.
- Final recorded validation clock05:08:45 UTC, about30 minutes after the first04:39:05 clock. No five-hour duration claim. Handoff publication preserves the research branch and fast-forwards main; exact final revision is in Git history.
- [Repository handoff](repository-status.md) records delivered files, evidence limits, CI link and immediate next coding task.

## 2026-10-15 05:10 UTC — published-main validation and CI maintenance

- Main and research branch were confirmed at9d5f687 with a clean local worktree. Main run38685249751 passed on both Linux and Windows.
- CI emitted an actual Node20-action deprecation warning and an upcoming ubuntu-latest image migration notice. Inspected current official release metadata/action manifests; pinned checkout v7.0.1 and setup-python v7.0.0 by commit, both using Node24. Selected ubuntu-24.04 explicitly. This is CI maintenance, not additional scientific evidence.
