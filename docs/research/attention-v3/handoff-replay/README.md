# Local handoff replay

This is a same-author, same-machine operational check of an extracted researcher
kit. It is separate from the completed 18-case study and six-case diagnostic.
The known passing long-000 case was selected to test the product workflow; this
selection is not unseen confirmation or a timing-benefit estimate.

A new empty virtual environment was installed outside the source checkout.
All 51 package names and versions matched the frozen profile; dependency checks,
pinned asset restoration and preflight passed. The base Python installation was
reused and asset-cache reuse was not excluded. The offline wheel-cache attempt
failed before the online hash-required installation succeeded. The environment
inventory and receipt record these limits.

The browser launched `long-validation-e74d858cbe6c`, which completed in 821.1
seconds. Independent array reconstruction passed all three arms. Generation
totals were 49.76s native, 235.06s resident and 97.72s hybrid. The large resident
timing difference from the original study is retained, not pooled away. Desktop
and browser activity occurred during this operational test; no cause for the
difference was isolated and no additional performance claim is made.

`execution.json` contains only this one-case subset of the unchanged registry.
The verifier correctly reports 1/18, with 17 cases unattempted in this local
workflow check. That does not change the original study's completed 18/18.
Recompute the new case from its full arrays with Python and NumPy:

```powershell
python benchmarks/verify_attention_v3.py --study docs/research/attention-v3/handoff-replay --source-root .
```

The separately started `long-validation-4094f0bf2f79` was deliberately cancelled
through the browser. It remains a cancelled record with unknown numerical
outcome, not another successful model trial. Both observed worker processes
exited and the active-job state cleared. Its full raw folder is retained under
`runs/` and identified in `receipt.json`; it is not pooled into the numerical
subset or either original study. A source/archive hash cannot independently
prove that the author actually performed these actions.

The receipt records the archive used at initial model launch. Later UI
refinements expose generation totals and allocator peaks, label reference/local
runs and provide a direct JSON download. These browser checks, export equality,
failure blocking and cancellation were exercised with those refinements. The
frozen model execution source and pinned requirements remained identical.
The final outer archive identity is supplied separately after packaging.
