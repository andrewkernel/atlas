# BERT/SQuAD workload suitability screen

This is a prospective screen, not an OOM, fit, quality or Atlas-selection result. The [source capture](qa-source-screen.json) pins Google's original BERT implementation and the dataset repository; downloaded scripts are reference text and will not execute.

## Why this workload

The [canonical run_squad.py](https://github.com/google-research/bert/blob/eedf5716ce1268e56f0a50264a88cafad334ac61/run_squad.py) defaults to logical batch 32 and sequence length 384. The same repository's GPU example instead recommends batch 12; its historical TensorFlow 1.11/Titan-X table reports 12 at that length. Keep both facts. Selecting logical 32 tests preserving a documented larger-batch recipe; it is not the GPU recommendation, a replica of the original TensorFlow study or evidence that every modern native configuration OOMs.

The prospective model is the pinned public BERT-base encoder with an initialized question-answering head. Actual SQuAD training questions/context provide the inputs. Its batch-separable transformer/LayerNorm structure is more relevant than enlarging the previous toy. Dropout, sliding-window preprocessing, attention dispatch and AdamW state create meaningful semantic/resource obligations. Existing native accumulation, checkpointing and efficient attention must be competent baselines.

## Rights and acquisition boundary

The [dataset publisher](https://rajpurkar.github.io/SQuAD-explorer/) and [SQuAD 1.1 card](https://huggingface.co/datasets/rajpurkar/squad/blob/main/README.md) identify CC BY-SA 4.0. The pinned BERT model repository provides Apache-2.0 metadata and a LICENSE file, which must be read before training-asset acquisition. Preserve attribution/notices locally. Data, derived input features and weights stay ignored; no test/dev partition or evaluation script is downloaded in the resource screen.

Acquire only allowlisted config/tokenizer/license files and, after review, a safetensors weight file and official training JSON. Verify pinned sizes, Git blob identities and the advertised LFS SHA-256. Do not download pickle weights or remote model code. The [acquisition source](../../../benchmarks/prepare_qa_assets.py) checks each asset and maintains at least 1 GB free disk after its download; retained partial files are owned scratch, not successful assets.

## Prospective empirical scope

Use a separate Python 3.11/Torch 2.13 CUDA environment compatible with the existing checker, plus hash-pinned Transformers 5.19.0. Resolution/installation is not runtime support. Inspect the BERT forward/loss, native checkpoint wrapper, SDPA dispatch and local-only tensor/tokenizer loading paths before importing the model.

A first resource experiment should compare eager/SDPA, full/microbatched execution and checkpoint on/off under one fixed feature manifest. Run contemporary controls, include all OOM/error/unsupported outcomes and distinguish cold from initialized-optimizer peaks. Do not resize the model, change context, reduce logical samples or tune the recipe after seeing results merely to manufacture a rescue.

An eight-update training segment can establish only that segment's completion/cost. It does not complete SQuAD fine-tuning, establish accuracy or satisfy the project's useful full-experiment access gate. A longer run requires a separately frozen protocol justified by measured cost. A reduced-batch parity control cannot stand in for a missing full logical-batch reference. Stochastic pathwise and distributional claims remain distinct. Independent adjudication and the primary selection comparison are still open.
