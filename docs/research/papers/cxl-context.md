# CXL paper context for Atlas

Added 2026-10-17 at the owner's request. This is a source entry point for a prospective CXL direction, not an approved replacement for the existing research scope or evidence of a working CXL implementation.

## Read these sources

1. **Architectural and System Implications of CXL-enabled Tiered Memory** — Yujie Yang et al., including Jia Rao; arXiv:2503.17864v2, revised March 26, 2025.
   - Source: https://arxiv.org/abs/2503.17864v2
   - PDF: ../../../artifacts/cxl-papers/miku-2503.17864v2.pdf
   - Searchable full text: ../../../artifacts/cxl-papers/miku-2503.17864v2.txt
   - Relevance: investigates interference between DDR and CXL requests, including memory parallelism and queuing. MIKU controls CXL requests to protect DDR throughput. This is request control, not an Atlas placement algorithm.
2. **DSA-2LM: A CPU-Free Tiered Memory Architecture with Intel DSA** — Ruili Liu et al., including Jia Rao; USENIX ATC 2025.
   - Author-hosted PDF: https://ranger.uta.edu/~jrao/papers/atc25.pdf
   - PDF: ../../../artifacts/cxl-papers/dsa-2lm-atc25.pdf
   - Searchable full text: ../../../artifacts/cxl-papers/dsa-2lm-atc25.txt
   - Code referenced by paper: https://github.com/madsys-dev/DSA-2LM
   - Relevance: integrates DSA into tiered-memory page migration; migration concurrency, batching and copying costs matter alongside placement. Hardware and kernel requirements must be checked before proposing reproduction.

PDFs and extracted text are local reference artifacts, excluded from Git by the existing artifacts/ rule. manifest.json in that directory records page counts and PDF SHA-256 identities. Full text was extracted across all pages; extraction is not a substitute for checking diagrams, equations and tables in the PDFs. The notes above reflect abstracts/introduction inspection, not a completed technical audit or reproduction.

## Proposed connection to Atlas

Candidate question: can ML phase information improve placement and migration decisions on DDR–CXL memory compared with equally informed competent baselines, after accounting for monitoring and migration costs?

This question is an Atlas hypothesis, not a claim made by either paper. Reuse Atlas's reproducibility and scoped numerical checks. Define a distinct policy, assess phase-aware prior art, establish hardware access, and select a CPU-accessible memory workload before implementation. NUMA emulation must be labeled as emulation; existing RTX 3060 measurements are not CXL evidence. Do not assume GPU memory is directly expanded by attaching CXL memory.

Read both full papers before detailed design or quantitative claims. Separate reported paper findings, reproduced results and Atlas proposals. No CXL experiments have been run for this addition.
