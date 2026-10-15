# Working on Atlas

Atlas currently develops a reproducible evaluation package for scoped fidelity evidence and resource tradeoffs on local hardware. Its selected immediate research question measures the coverage of a competent native-plus-existing-checker composition on a future independently reviewed fixed corpus. Policy-superiority research is conditional on specifying a distinct algorithm. Read `docs/research/00-executive-assessment.md`, `04-scientific-fidelity.md`, `06-novelty-analysis.md`, architecture docs and ADRs before architectural changes. Broad novelty is not established.

Read `docs/research/14-updated-recommendation.md` and the amendment ledger before expanding scope. Real-data GPU pilots now exist, but standalone optimizer/verifier novelty remains unconfirmed. Generic state/buffer/gradient checking substantially overlaps existing TrainParity; do not rebuild it under a novelty claim without the required comparison.

Read `docs/research/15-comparison-identifiability.md`, ADR 007, `docs/architecture/objective-and-evidence-boundaries.md` and `claim-applicability.md` for the current benchmark-first scope. A distinct policy treatment remains unselected. The historical `benchmarks/qa_resource_pilot.py` supervisor measures a Windows environment launcher rather than the actual worker; retain its source/raw record but do not reuse that supervisor for further training. Later fixed-registry owned-worker prototypes use separately validated suspended-launch Windows job inheritance; their controls do not establish production/Linux ownership, hard quotas or a hostile-code sandbox.

The pinned built-in BERT constructor already defaults to SDPA. Do not credit Atlas with that native-default saving or treat the explicit eager resource condition as the ordinary baseline. Logical-32 reference repeats differ in selected post-update parameters, even with matching initialized/input/RNG records. Preserve literal comparisons separately from transformation attribution and full-quality claims. Six reviewed official metric functions have a bounded invented-data coverage diagnostic; no full QA evaluation was run. See the latest recommendation, source-scoped result notes and prospective QA evaluation specification.

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
