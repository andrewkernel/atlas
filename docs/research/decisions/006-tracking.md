# ADR 006 — Portable evidence plus local SQLite/MLflow

**Status:** accepted for implementation.

**Context:** run provenance must survive service failure and remain inspectable without a dashboard. The coordinator selected SQLite and MLflow.

**Alternatives:** dashboard-only storage; flat files only; mandatory hosted tracking/database; immutable JSON artifacts with local control metadata and an MLflow adapter.

**Evidence:** [MLflow tracking](https://mlflow.org/docs/latest/ml/tracking/) provides parameters, metrics and artifacts, but does not define Atlas's evidence semantics. A local workflow does not require a distributed database or mandatory cloud service.

**Decision:** JSON/JSONL plus content hashes are portable evidence; Atlas SQLite owns control states; MLflow owns a separate tracking store. Keep binary data/weights local and ignored by Git. Use Parquet only when tables justify it.

**Tradeoffs:** some metadata duplication and explicit recovery/sync logic; stronger recoverability and less coupling to tracking internals. Hashes establish byte identity, not truth.

**Revisit when:** real concurrent/multi-user requirements exceed the single-supervisor model. Only then consider PostgreSQL or a service deployment; never add it preemptively.
