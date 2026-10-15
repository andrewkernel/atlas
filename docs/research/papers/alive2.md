# Alive2: Bounded Translation Validation for LLVM

Nuno P. Lopes, Juneyoung Lee, Chung-Kil Hur, Zhengyang Liu and John Regehr; PLDI 2021. [Author-hosted paper](https://sf.snu.ac.kr/publications/alive2.pdf), [upstream](https://github.com/AliveToolkit/alive2).

Read scope: abstract and selected semantics/unsupported-feature/loop sections; PDF page 9 visually inspected. Current README limitations inspected. No LLVM build, reproduction or proof audit. [Capture record](../comparison-lineage-sources.json) retains the initially failing host; an alternate author host succeeds with TLS verification.

Alive2 validates LLVM refinement with symbolic execution/SMT and bounded loops, so longer behavior may escape the checked bound. The current README separately excludes inter-procedural transformations and warns of spurious counterexamples there. Atlas can learn explicit relation, support and trust boundaries; its finite tensor comparisons are not symbolic validation over an input domain. A report containing identities/predicates is not thereby a proof certificate or a general scientific-validity theorem.
