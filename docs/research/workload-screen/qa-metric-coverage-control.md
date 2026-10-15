# Reviewed metric-function coverage diagnostic

After freeze `fab240a`, four invented-data cases execute the six reviewed functions named in the [protocol](qa-metric-coverage-protocol.md). [Raw record](qa-metric-coverage-control.json): the complete fixture scores 33.33 over three questions; omitting a predetermined wrong answer reports 50 over two. Supplying all IDs instead raises KeyError for that missing score. An unexpected ID is ignored by this metric path; an all-missing default aggregate raises ZeroDivisionError. The research coverage guard rejects all three incomplete/unexpected cases.

This confirms the [source-inferred wrapper precondition](qa-quality-evaluation-spec.md) on a bounded fixture. No full CLI, model evaluation, new metric, measured incidence or general upstream-defect claim follows. Function extraction preserves source and selected-AST identities; external module imports/threshold/plotting code are not executed.
