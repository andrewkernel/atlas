# QA study cost windows and accounting limits

[Independent aggregation](../../../benchmarks/derive_qa_workflow_costs.py) binds five saved study records by canonical identity. [Derived ledger](qa-workflow-costs.json).

| Development study | Owned worker windows | BERT model executions | BERT optimizer updates | Sum of supervised windows, seconds |
|---|---:|---:|---:|---:|
| Native resource segments | 16 | 16 | 128 | 408.630 |
| Reduced numerical/capture controls | 2 | 10 | 10 | 72.703 |
| Dispatch capture, no optimizer | 2 | 2 | 0 | 17.172 |
| Selected AdamW sensitivity | 1 | 2 | 2 | 9.046 |
| Fresh-process logical-32 references | 4 | 4 | 4 | 29.125 |
| Total | 25 | 34 | 144 | 536.676 |

These are heterogeneous development executions, not 34 independent workloads or 144 outcome replications. Scalar counterfactual optimizer replays are not BERT training updates. The same input features/initialization recur; full-model horizons are one or eight updates, with logical batch three or 32.

The 536.676 seconds is **not complete research or policy cost, exclusive GPU allocation or GPU-active time**. Timer boundaries omit some controller initialization/environment construction/job creation, parent-side array analysis, dependencies/downloads and engineering/review effort. The current records do not provide a complete search-policy cost estimand.

Reported resource-segment update timers sum 257.898 seconds inside their worker lifetimes; ten reported main capture timers sum 4.677 seconds inside numerical-worker lifetimes. These windows are nested in the supervised intervals. Adding them to the outer total double counts. The capture subtotal also excludes unreported comparison, negative-capture and analysis subdivisions; the remainder cannot be called verifier overhead.

Future evaluation must measure cold process lifetime and warm request latency explicitly, with preparation/reference repeats, all failed trials, observation acquisition and final confirmation charged consistently to every arm. The [distinct-policy gate](../15-comparison-identifiability.md) remains unmet; this descriptive ledger does not establish a faster Atlas workflow.
