# BERT constructor-default attention control

The installed built-in `BertForQuestionAnswering` constructor selects **SDPA by default** with the pinned full model configuration. The config field is initially null and becomes `sdpa` after construction. Consequently, the explicit eager condition in the [resource pilot](qa-resource-pilot.md) is not this library's unmodified default. Its approximately 14.18 GiB allocated peak must not be used to claim Atlas discovers a saving over the ordinary native baseline; the approximately 8.70 GiB full-batch SDPA condition is already a strong existing baseline.

The [protocol](qa-default-attention-protocol.md) and source were frozen in commit `ee480ac` before the [recorded run](qa-default-attention-control.json). A suspended, owned Windows job constructed 108,893,186 random parameters on CPU using the same reviewed runtime/config identities. All parameters were on CPU. The supervised probe took 4.282 seconds and left zero job members after cleanup.

This checks constructor selection in the installed, pinned stack. It loads no checkpoint, performs no forward, optimizer update or training, and measures no CUDA memory or dispatch. The separate [dispatch control](qa-dispatch-control.md) observes actual device kernels in its own two configurations. This result does not establish defaults for older libraries or every device/shape. It is deliberately outside the five-study training workflow cost ledger.

The attention change is established native functionality, not an Atlas contribution. A future baseline must record actual defaults before enumerating additional strategies and count only changes beyond that reference.
