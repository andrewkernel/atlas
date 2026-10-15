# Working on Atlas

Atlas studies whether explicit training contracts improve selection of memory-saving plans on local hardware. Read `docs/research/00-executive-assessment.md`, `04-scientific-fidelity.md`, `06-novelty-analysis.md`, architecture docs and ADRs before architectural changes. Broad novelty is not established; compare against native optimization plus existing checks.

## Priorities and organization

- Research specification and reproducible evaluation first. Do not implement the entire product at once.
- `docs/research/`: evidence, paper notes, competitors, decisions and chronological log.
- `docs/architecture/` and `docs/roadmap/`: implementation contracts and acceptance gates.
- `src/atlas/`: future analysis, profiling, optimization, fidelity, experiments, inference and API modules.
- `benchmarks/`: reviewed workload/probe code; `tests/`: meaningful positive/negative controls; `configs/`: versioned settings; `frontend/`: later inspection UI.

## Stack and conventions

Python 3.12, typed maintainable modules, PyTorch 2.x with explicit CPU/CUDA support. FastAPI/Pydantic, SQLite, MLflow and structured JSON/optional Parquet when needed. Optional replaceable Ollama backend. Later React/TypeScript/Vite/Tailwind. No mandatory proprietary API, distributed infrastructure or LLM authority.

Use uv and `pyproject.toml`/`uv.lock`; run with `--locked`. Add only necessary dependencies, inspect compatibility/license and record tested versions. CPU lock does not establish CUDA support. Keep clear units, deterministic serialization and explicit errors. Use trusted registered adapters; static AST findings are advisory. No arbitrary-project rewriting in MVP.

## Scientific requirements

- Separate resource feasibility, mathematical/numerical/statistical/methodological and conclusion claims.
- Each claim needs scope, assumptions, artifact identities and evidence. Unknown is not pass. Do not use a scalar scientific score.
- Preserve data, loss normalization, logical batches, RNG relation, model buffers and update clocks required by the contract.
- Test compositions; gradient parity alone does not certify buffer/state or full-training outcomes.
- Predeclare tolerances/margins and analysis before confirmation. Nonsignificance is not equivalence. Record negative and inconclusive results.
- Never fabricate timings, hardware tiers, completed runs, energy savings or external reviewer endorsement.

## Safety and reproducibility

Execute reviewed trusted code only in MVP. Imports, profiling, tracing and checkpoint loading can execute code. Refuse untrusted execution without a separately validated policy/isolation design. Docker is not a guarantee. Do not use elevated privileges, remote installer scripts, inherited secrets or broad writable mounts. Bound runtime/logs/resources and terminate only owned processes. Never modify user source/data for a candidate.

Record code+dirty diff, contract/config/plan hashes, data/tokenizer revisions and bytes, seeds, environment/lock, hardware and measurement windows. Keep weights/data/binaries/local artifacts out of Git; commit small sanitized summaries deliberately. Avoid machine-local paths and secrets in portable reports. Project `sources/` mirrors outside this repository remain read-only.

## Done means

Applicable acceptance criteria pass; meaningful positive, negative, unknown and failure controls exist; required checks run; docs and source citations are current; actual evidence is distinct from proposed work; artifacts are inspectable and large/sensitive files excluded. Use dedicated branches and meaningful commits; never force-push or delete unrelated work. Preserve the research log and revisit ADRs when assumptions change. Repository public licensing has not been selected by the owner.
