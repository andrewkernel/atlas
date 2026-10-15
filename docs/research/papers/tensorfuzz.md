# TensorFuzz

Augustus Odena and Ian Goodfellow. **TensorFuzz: Debugging Neural Networks with Coverage-Guided Fuzzing.** [arXiv1807.10875v1](https://arxiv.org/abs/1807.10875v1),28 July2018. This note refers to that preprint's authors/version; abstract only, no execution.

The abstract describes coverage-guided input mutation for user-specified constraints, including numerical errors, quantization disagreements and undesirable model behavior.

**Atlas assessment:** rare-input and differential numerical testing are not new. A few hand-chosen ordinary batches are especially weak evidence for universal preservation. Atlas should include adversarial masks/shapes/state cases and disclose the tested input domain, while recognizing that even fuzzing coverage does not prove absence of semantic failures. This source is context for test design, not evidence that TensorFuzz already implements the proposed memory-selection workflow.
