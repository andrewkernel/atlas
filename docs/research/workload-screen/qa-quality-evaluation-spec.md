# Prospective QA quality evaluation and decoder contract

This is an implementation specification, not a preregistered full-training experiment or an evaluation result. The existing BERT segments have no QA quality score. Their training-only arrays cannot supply this missing evidence. Read together with [claim applicability](../../architecture/claim-applicability.md), [study estimands](../evaluation-estimands.md) and [statistical limits](../statistical-claim-selection.md).

## Source and version boundary

The pinned [Google BERT decoder](https://github.com/google-research/bert/blob/eedf5716ce1268e56f0a50264a88cafad334ac61/run_squad.py) filters candidate endpoints to mapped context tokens, requires maximum context for the start, enforces ordered endpoints and a maximum length, and ranks candidates from all windows by summed start/end logits. It projects WordPiece predictions back to original text, with a fallback when alignment fails. These are meaningful decoder choices; endpoint argmax alone is not a substitute.

The inspected [official SQuAD v2 evaluator](https://github.com/rajpurkar/SQuAD-explorer/blob/eee5fdbf62f8613a7812b03419e6b29617b74fd1/evaluate-v2.0.py) defines normalization, exact match, token-overlap F1 and best score over annotated answers. Its raw-score routine skips missing predictions; the aggregate without an explicit ID list uses the number of scored IDs. **Inference from those sections:** complete prediction coverage is a necessary wrapper precondition for a fixed-denominator comparison. This is not a measured upstream failure. Do not silently use a smaller question denominator.

The guessed v1.1 evaluator path returns 404 at this revision; the actual repository file is v2.0. [Capture and read scope](qa-evaluation-source-screen.json) preserve that distinction. Neither evaluator was imported or executed. Reusing selected answerable-question metric definitions would be an explicitly versioned Atlas evaluation recipe, not a replicated official v1.1 protocol. The [dataset site](https://rajpurkar.github.io/SQuAD-explorer/) distinguishes the answerable v1.1 task from the v2 no-answer task.

## Freeze before acquisition or training

1. Use the answerable v1.1 task initially. Record source revision/URL, license, byte digest, all question IDs and article membership for separately acquired training/development data. No development data has been acquired for the current segment studies. Check overlap/disjointness explicitly. A public development split can be an exploratory evaluation set but is not an untouched policy holdout after tuning on its scores.
2. Keep the current 512-question/518-feature resource subset labeled as training-only development information. Do not relabel its outcomes as external quality or use its four CLS-target windows as unanswerable questions. CLS there means that a particular training window lacks the annotated span.
3. Freeze training horizon, order/augmentation, optimizer/clipping/scheduler, seed pairing, checkpoint-selection rule and complete candidate method. A new full-training study needs its own budget/power justification; eight updates provide neither endpoint quality nor a useful convergence estimate.
4. Freeze a decoder version before seeing arm scores. A reasonable initial compatibility target is BERT-style n-best 20, maximum answer length 30, uncased processing, explicit start maximum-context filtering, and ranking across all windows by unnormalized summed logits. These are proposed fixed choices, not tuned optima or an implemented decoder. Use original character offsets or a reviewed projection with explicit fallback records; do not mix alignment conventions across arms.

## Required evaluation artifact

Evaluation preprocessing must retain question ID, article ID, original context, window ID, context-token membership, original character offsets and maximum-context metadata. The present training NPZ does not provide a complete evaluation artifact. Hash the tokenizer and every decoder configuration/source; preserve window ordering and deterministic tie-breaking. Duplicate IDs, missing windows, invalid offsets and nonfinite logits are explicit errors.

Each checkpoint is evaluated using the same declared inference method and batching, with stochastic training behavior disabled and its previous mode restored afterwards. Record actual evaluation attention/backend and precision separately from training. Produce one final string per question plus bounded audit fields for selected window/span, score, alignment fallback and decoder status. Retain hashes and local per-question results; do not commit context text or full logits by default.

Validate exact coverage of the frozen ID set before aggregation: missing, unexpected or duplicate IDs prevent a complete metric result. A crash or undecodable question cannot silently disappear. Partial diagnostics name their coverage and cannot satisfy the complete-quality claim. Report per-question exact/F1 and the aggregate over all declared questions, with normalization and multi-answer behavior identified by version. No-answer thresholds or best-threshold searches belong to a separate prospective v2 protocol.

## Acceptance and interpretation

- Decoder controls cover question/padding/special-token exclusion, reversed/too-long spans, overlapping windows/max-context, ties, accent/whitespace offsets, fallback, multiple gold answers, and missing/duplicate IDs. A high invalid-token logit must not become an answer.
- A trusted tiny gold-logit fixture yields the intended original-text span across at least two windows. Positive and negative changes check the decoder and metric boundary independently of model parameter comparisons.
- Both training arms use exactly the same evaluation artifact and implementation. The baseline receives every correction and adaptation. Fixing a shared decoder cannot count as an Atlas policy advantage.
- Report paired full-training endpoint differences by seed, failures and total training/evaluation cost. Questions/windows/tokens are not independent training-seed replications. Distinguish variability across training seeds from sampling/generalization uncertainty; article dependence requires an appropriate design if making that further claim.
- A practical quality margin and outcome model require domain justification and a disjoint pilot. Preserve inconclusive results. Passing strict state checks does not imply a quality margin, and a favorable aggregate score does not waive a forbidden method change or establish preservation of a domain conclusion.

These obligations are a gate for a later quality study, not a reason to run full fine-tuning before the independently labeled benchmark and distinct-policy decisions are resolved.
