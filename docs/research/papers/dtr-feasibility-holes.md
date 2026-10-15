# When More Memory Fails: Hidden Feasibility Holes in Dynamic Tensor Rematerialization

- **Authors:** Mahesh Reddy Pagadala.
- **Year / venue:** 2026; arXiv preprint; peer review not established.
- **Verified primary URL:** [When More Memory Fails: Hidden Feasibility Holes in Dynamic Tensor Rematerialization](https://arxiv.org/abs/2610.14369).
- **Read depth:** Abstract only; exploratory context, not foundation of verdict.
- **Objective:** Examine monotonic feasibility under memory budgets.
- **Technical method:** Replay public DTR traces in a reference simulator with budget sweeps.
- **Main finding used:** Reports nonmonotone failure regions under some policies.
- **Limitations / Atlas assessment:** Simulation without allocator model; GPU behavior untested.
- **Relevance:** Do not assume budget sweep monotonicity without checking.
- **Code status:** Simulator and trace workflow described; not replicated here.
- **Open question:** Does this behavior occur on actual allocators?
- **Requires further verification:** Recent preprint claims require independent replication.

Retrieved in this research session; no independent replication is implied.
