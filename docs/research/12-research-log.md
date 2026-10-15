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
