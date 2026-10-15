# Research probes and future benchmarks

`semantic_probes.py` and `composition_probes.py` are original offline CPU research code; [actual results and limits](../docs/research/experiments/README.md) explain their scope. `snapshot_references.py` makes read-only authenticated GitHub/public Hugging Face metadata requests; it never executes imported research code. It is not run in CI and requires local GitHub authorization. `snapshot_dependencies.py` records primary PyPI metadata and installed package versions without installing anything. `audit_specification.py` checks internal consistency offline.

The [planned evaluation](../docs/research/07-experimental-design.md) is separate from these probes. No public dataset, model weight or competing framework was downloaded/executed in this session. Weights, datasets and large outputs belong in ignored local artifact stores.
