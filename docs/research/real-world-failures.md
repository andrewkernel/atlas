# Historical failures and provenance for authored cases

This audit connects authored fixtures to documented failure mechanisms. It is not a prevalence study or a reproduction of historical installations. Issue status alone does not prove either a current bug or a successful fix. No issue was filed, commented on or changed.

| Mechanism | Primary record and status checked2026-10-15 | Relation to Atlas evidence | Limits |
|---|---|---|---|
|Checkpointed BatchNorm updates running statistics twice | [PyTorch issue96136](https://github.com/pytorch/pytorch/issues/96136), opened6 March2023, API state open; names FairScale's mitigation | Original CPU fixture independently exhibits gradient/buffer disagreement on 2.14.1; [FairScale source read](competitors/fairscale.md) verifies specialized handling exists | Does not establish every wrapper/region fails; exact historical test was not executed |
|Clipping partial accumulated gradients | [Transformers issue1563](https://github.com/huggingface/transformers/issues/1563),2019, closed; [PR1580](https://github.com/huggingface/transformers/pull/1580) merged22 October2019 at `b8c9ea0010a09cca8173e5bdf4af855123aebfc7` | Original two-update [composition fixture](experiments/cpu-composition-probes.json) isolates this mechanism | Not a claim that current Transformers has this historical bug; no old-version execution |
|Averaging microbatch means with unequal target counts | [Hugging Face's2024 correction](https://huggingface.co/blog/gradient_accumulation) explains summed loss divided by actual valid-target count | Original unequal3+4 partition provides an analogous separable-loss counterexample | Sample-weighted scalar fixture is not a reproduction of a language-model training run |
|Zero versus absent unused gradients | [PyTorch zero_grad API](https://docs.pytorch.org/docs/2.14/generated/torch.optim.Optimizer.zero_grad.html) explicitly distinguishes optimizer behavior | [Update-clock fixture](experiments/update-clock-probes.md) has identical observed second loss and shared-parameter result but different unused parameter/state | Intentional API semantics; not a PyTorch defect or novel Atlas finding |
|Scaler skips optimizer while scheduler/EMA advance | [Native AMP examples](https://docs.pytorch.org/docs/2.14/notes/amp_examples.html) describe skip behavior and effective-batch granularity | Original CPU GradScaler control injects an infinite gradient and records successful-update events | Control-flow experiment, not an actual low-precision convergence failure or empirical bug prevalence |

## Case-selection safeguards

Use source provenance to diversify mechanisms, not to assign every report a confirmed-bug label. Search across multiple frameworks, include correctly handled cases, record fixed versions, and reserve whole source projects and mechanisms before writing final policies. Older high-profile issues are likely already encoded in existing tools; they are necessary controls, not sufficient novelty evidence.

For an issue-derived benchmark case, preserve: source URL/retrieval date, affected and fixed revisions if established, reporter claim versus maintainer confirmation, independent expected relation, minimal original adaptation, license/provenance, raw execution evidence and label uncertainty. Do not copy arbitrary issue scripts into a trusted runner. Review or independently author a minimal fixture first.

Human labels remain uncollected. This session's authorship and source searches create investigator bias; neither citation count nor number of mutations removes it. Future independent annotators must assess the required contract property rather than merely agree with the injected fault's name.
