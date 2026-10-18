# Research assessment after implementation

Atlas now provides an executable, source-bound CPU–GPU attention research artifact. It does not provide a universally accurate or unbiased benchmark, a CXL runtime, or proof of a novel algorithm.

The amended study completed 51 registered cases across three fresh process blocks: nine precision-aware layer sweeps, 27 real-text Qwen3 model cases, three complete-step cache ablations and 12 real memory-pressure cases. Its 579 rows contain 531 numerical passes, 36 failures and 12 unknown full-pressure numerical relations. These counts mix different evidence units and are not a scientific-validity score. All failures and unknowns remain inspectable and blocked from recommendation. Original v1 records and interrupted protocol attempts are preserved.

## What the measurements support

- Qwen3 FP32 model comparisons passed the original elementwise relation at 512, 2,048 and 8,192 context tokens in all three blocks. FP16/BF16 native/resident controls passed their matched-precision relations, while custom dense/hybrid model arms failed. Small NLL or softmax differences do not override those failures.
- At 8,192 FP32 tokens, the dense GPU path had a median native/candidate complete-generation ratio of 1.043, with a descriptive three-block interval of 1.022–1.043. The fixed-window hybrid was slower, with a ratio of 0.541. These narrow cases do not establish a universal improvement.
- The 256-token GPU window reduced measured allocator peaks by 63.8% at 32k and 66.3% at 40k. The 40k feasibility runs had a median generation ratio of 1.265, while 32k was slower at 0.866. Full-context numerical replay is unknown, so neither pressure result is a correctness-qualified speedup. Both native and hybrid completed: no OOM rescue was demonstrated. WDDM residency/paging prevents a hard physical-VRAM inference.
- The ring removes window-sized append/eviction work and stores its GPU window once. Complete-step results were slower at 512/4,096 and faster at 16,384 tokens: O(1) append does not guarantee the fastest complete attention step. The ablation has a fixed-order limitation.
- Calibration is compared with a budget rule and a calibration-selected fixed plan. Temporary GPU KV copies and low-precision promotion are included in the corrected selection budget. Layer calibration never certifies full-model routing. Calibration and analysis costs are visible; no net adaptive advantage is established.

## Target points that still need research

1. **Full-pressure numerical validation.** Reproduce identical long token histories and compare full model logits/NLL before promoting the apparent 40k resource/time benefit.
2. **Low-precision error attribution.** Trace Q/K normalization, rotary transforms, partition arithmetic, fusion and projection propagation without changing bounds after viewing results. The legacy GPT-2 diagnosis already shows local oracle agreement alongside a literal final-logit failure at step 47.
3. **Generalization and measurement bias.** Add non-overlapping documents, other modern models, more separately scheduled sessions/machines, balanced cache-arm order and a separately declared steady-state timing regime. Current bootstrap intervals are descriptive; process freshness does not make shared-hardware noise or documents independent.
4. **Transfer/concurrency mechanisms.** Separate host bandwidth, materialization, synchronization and copying costs. Dense offloading is a baseline; sparse work selection and asynchronous execution require their own algorithms, controls and comparisons.
5. **Actual CXL questions.** A future study could ask whether measured bandwidth/interference should control KV admission and movement. This connects to the request-interference and migration-cost questions in the paper notes, but requires real CXL/DSA hardware evidence and competent published baselines. Consumer CPU–GPU results cannot substitute for it.

The present artifact is suitable for an engineering/research discussion with inspectable negative results. External human reproduction, broad downstream quality, energy savings and novelty remain unestablished. No message has been sent to a researcher.
