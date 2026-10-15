# Mixed Precision Training

- **Authors:** Paulius Micikevicius et al.; full author list at source.
- **Year / venue:** 2017 preprint; 2018 conference version; ICLR 2018.
- **Verified primary URL:** [Mixed Precision Training](https://arxiv.org/abs/1710.03740).
- **Read depth:** Abstract and current AMP guidance.
- **Objective:** Train with lower-precision arithmetic.
- **Technical method:** Use half precision with FP32 state and loss scaling.
- **Main finding used:** Resource benefits can coexist with task-quality preservation on studied models.
- **Limitations / Atlas assessment:** Empirical success is not bitwise or universal scientific equivalence.
- **Relevance:** Precision changes require a distinct empirical contract.
- **Code status:** Modern native AMP exists; historical code not audited.
- **Open question:** Which study conclusions are sensitive to rounding?
- **Requires further verification:** Do not apply historical memory factors to every current autocast setup.

Retrieved in this research session; no independent replication is implied.
