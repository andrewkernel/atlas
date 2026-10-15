# Planned workload construction details

This is an implementation specification, **not executed benchmark code**. Resolve dataset terms and produce hashed input manifests before any data-dependent run. Freeze actual constructors and configs at a protocol commit; record deviations as amendments.

## CNN control

Fashion-MNIST canonical 60,000 training images: use a seeded, saved permutation to assign 50,000 training and 10,000 validation examples; keep the canonical 10,000 test examples sealed for a later declared final analysis. Inputs are float32 pixel/255, no augmentation or fitted normalization. Reference CNN: valid3×3 Conv 1→32, ReLU, valid3×3 Conv 32→64, ReLU, 2×2 max-pool, flatten, Linear 9216→10. Cross-entropy sum divided by actual example count; no dropout or normalization. SGD lr.01, momentum.9, dampening 0, weight decay0, no Nesterov, no clipping, no scheduler. Logical batch 128, actual denominator for final tail, five full epochs, validation each epoch with no updates.

The BatchNorm workload is separately declared: BatchNorm after each convolution, eps1e-5, momentum.1, tracked running statistics, training/evaluation modes explicit. It is a countercondition for accumulation/checkpoint region eligibility, never a claimed equivalent replacement for the no-BatchNorm architecture.

## Decoder workloads

Use pinned WikiText-2 raw files after rights/provenance resolution. Baseline tokenizer is deterministic Unicode-whitespace splitting, with `<eos>` after each nonempty line. Training vocabulary only, lexicographically ordered with reserved `<pad>`, `<unk>`, `<eos>` IDs fixed first; record Unicode/Python version and token-to-ID hash. Unknown validation tokens map to `<unk>`. Concatenate within each split; construct nonoverlapping context-length input blocks with next-token targets, preserve/mask the final partial block. No cross-split content leakage. Replay block IDs and target masks per logical batch.

Small: six pre-LayerNorm causal decoder blocks, width 256, four heads, FFN 1024. Large:12 blocks, width 512, eight heads, FFN 2048. GELU, LayerNorm eps1e-5, learned positional embeddings, tied token/output embedding, bias enabled in linear layers. Strict positive-control dropout 0; separate declared dropout.1 condition case. Context 256/small and 512/large. Native dense causal attention, record actual SDPA backend; no silent attention-backend switch in the unmodified reference.

Initialize linear weights with Xavier-uniform, biases 0, embeddings Normal(0,.02), LayerNorm scale 1/offset 0; serialize/hash initial state for paired replay. AdamW lr3e-4, betas(.9,.999), eps1e-8, weight decay.01 for matrix-valued parameters (including tied embeddings once),0 for vectors/biases; no scheduler/clipping/EMA. Logical batch 32,2,000 successful updates; data cycling and partial-window rules recorded. Validation every 200 successful updates. Report token-weighted validation NLL and perplexity as descriptive metrics; equivalence margins still require scientific justification.

## Pretrained adaptation

Use the pinned DistilBERT model/tokenizer revision from the asset snapshot, loaded as a masked-language model from safetensors with remote code disabled. Keep all original trainable parameters, dropout and optimizer semantics fixed across reference/candidates. Tokenize WikiText-2 at length 128 with an explicitly frozen chunk/attention-mask policy. Precompute/replay MLM selections with a dedicated mask seed:15% eligible non-special tokens; of selected targets,80% mask,10% random vocabulary token,10% unchanged; loss denominator is the actual selected target count. Freeze special-token handling and random replacement distribution in code/manifest before confirmation.

AdamW lr5e-5, betas(.9,.999), eps1e-8, matrix decay.01/vector decay0, logical batch 32,1,000 successful updates, no warmup/scheduler/clipping in the initial control; validation every 100 updates with frozen validation masks. Dropout means changed microbatch shapes may fail a pathwise numerical contract; report refusal/unknown rather than disable reference dropout to manufacture parity. Checkpoint-only candidates can still be tested on eligible regions.

## Selection and scale policy

The pilot may show these controls do not stress VRAM sufficiently. Use the declared large variant or a prespecified shape sweep to characterize the memory/time boundary; any change becomes a separately identified workload and is frozen before confirmation. Do not increase benchmark count with near-duplicates or imply that a small toy reproduces a resource-heavy published experiment. Full protocol defaults here are proposals; no original scientific conclusion has yet been attached to them.
