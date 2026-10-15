# Differential equality does not validate a shared reference objective

The [frozen token-objective protocol](token-reference-protocol.md), [authored cases](../../../benchmarks/token_reference_cases.py) and [launcher/oracle](../../../benchmarks/token_reference_control.py) compare an explicitly intended valid-token mean with a deliberately different per-sequence mean. Both models use the same tiny embedding/linear architecture, identical initial parameters, four fixed sequences and valid-token counts 4, 2, 1, 3. No language dataset, attention, pretrained model or full training is involved.

| Authored adapter | Existing checker accumulation result | Adapter loss | Declared token-oracle loss | Gradient relative L2 error versus token oracle |
|---|---|---:|---:|---:|
|Valid-token numerator/count|PASS|2.0390467720|2.0390467720|0|
|Mean of per-sequence means|PASS|2.0098987699|2.0390467720|0.4483878394|

The [corrected raw run](token-reference-control.json) retains both complete existing-checker results and independent full-batch objective observations. Two sequence microbatches are correctly accumulated **for each supplied adapter's own objective**. The deliberately wrong adapter is internally consistent, finite and reproducible; reference and candidate share its discrepancy from the stated token-weighted intent. Its maximum gradient absolute discrepancy is 0.2097564014. This is a correct differential PASS for a different relation, not a TrainParity defect, unsafe acceptance decision by Atlas or a new scientific checker.

The independent authored oracle uses full-batch log-softmax/gather and the ten valid target tokens. It shares the same trusted framework/initial parameters; this is not an independent human label or proof of framework correctness. Equality to a supplied reference and correctness of that reference relative to the scientific contract remain separate obligations. A model's declared objective prose is not machine-verifiable authority by itself.

## Preserved initial reporting failure

Source/protocol froze at `8e3e820`. The first attempt completed both checker calls with PASS, then the investigator oracle collector read nonexistent LossAccounting.loss rather than its documented value field. Both outer errors and saved checker reports remain in [the historical attempted run](token-reference-reporting-attempt.json). Amendment `9903439` changes only that field access and freezes the correction before rerunning identical cases/constants. Do not count the historical report errors as two checker failures or silently drop the failed investigator workflow.

## Consequence for the proposed contract

A denominator needs both a number and meaning: valid target tokens, samples, elements or declared weighted mass, plus masks/ignored targets and auxiliary-term normalization. A method contract can compare the observed ten tokens against that declared unit. Give the competent baseline the same semantics, masks, expected counts and adaptation opportunity. This observation does not establish Atlas alone can enforce them or improve held-out selection.

The reviewed checker revision and separate Python 3.11/Torch 2.13 CPU environment remain unchanged. All intended cases and source/review/helper identities are retained. The strongest missing evidence remains independently adjudicated workload/policy comparison; these two investigator-authored controls are discovery evidence.
