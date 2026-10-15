# Policy acquisition and validation lineage amendment

The [distinct-treatment audit](15-comparison-identifiability.md) leaves the adaptive algorithm unselected. Further primary-source inspection narrows two tempting contribution claims:

- [Constrained Bayesian optimization](papers/constrained-bo-gardner.md) already models costly objective/feasibility evaluations. [Decoupled constrained optimization](papers/constrained-bo-gelbart.md) also chooses which observation to buy under differing costs. A proposed Atlas evidence-acquisition policy needs a specific additional mechanism and comparison; cost-aware checking order is not new by itself.
- [Alive2](papers/alive2.md) supplies a much stronger and different kind of bounded semantic validation than finite differential tensor observations. “Translation validation” or “proof-carrying plan” must not be used to imply Atlas has an executable semantics, symbolic coverage or a verified checker.

These sources are related lineages, not drop-in PyTorch training replacements. The relevance inference is ours: their existence weakens broad novelty rhetoric without establishing that they solve Atlas's entire workload/intent problem. No implementation was installed or run, no study was replicated, and the search is not exhaustive.

## Architecture implications

Separate a **ranking belief** from an **acceptance obligation**. A policy may use a model to decide where to spend time; required unknown semantic evidence cannot become PASS merely because the predicted probability is high. The proposed evidence object must state whether it contains an observed finite comparison, an assumption-conditioned derivation, an external adjudication or a verified formal result. Those evidence kinds do not share a default coverage domain.

An abstention can reflect a missing observation, unsupported executor or undefined objective. A scalar constraint surrogate should not silently label all three as numerical infeasibility. Calibration, dependence assumptions and target population need their own validation. For a tiny native candidate set, the overhead and adaptation burden of a Bayesian optimizer may exceed any benefit; do not add it before a disjoint cost pilot and an explicit simpler comparator.

The near-term research artifact remains the reproducible benchmark/evaluation package under [ADR 007](decisions/007-benchmark-first.md). Formal verification infrastructure and a new general optimizer are outside the demonstrated scope. This is a literature-informed design amendment, not a positive policy result.

## Access and read-depth record

[Five source captures](comparison-lineage-sources.json) retain byte identities; [the initial attempt](comparison-lineage-initial-attempt.json) records a 4 MiB retrieval cap and TLS-chain failure. A HEAD response reports the publisher PDF at 8,278,316 bytes; a fixed 16 MiB cap then retrieves it. The alternate Alive2 author host succeeds without disabling certificate checks. Web PDF screenshots were unavailable, so local Poppler renders were inspected. Source text/method read scopes are explicit in each note. A successful download does not count as a full-paper read. The earlier 32-note metadata audit is historical and has not been rerun over these additions.
