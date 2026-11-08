# CS781 Capstone Project

This repository contains notebook experiments for legal-language modeling and downstream legal NLP tasks. It includes seven phases, from base-model pretraining through prior-case retrieval, along with selected submission archives.

## Phases

- [**Phase 1:** Language-model pretraining](Phase_1/README.md): Build and compare base language-model architectures trained on legal text. The notebooks cover decoder-only, encoder–decoder, MoE, dual-attention, MLA, GFNet, and FFT variants, with different token budgets and corpus mixes.
- [**Phase 2:** Legal named-entity recognition](phase_2/README.md): Adapt Phase 1 backbones to token-level legal NER using BIO labels. Experiments compare LoRA, LoRMA, DoRA, partial fine-tuning, distillation, encoder–decoder and MoE backbones, plus word-level Transformer and gated-adapter variants.
- [**Phase 3:** Rhetorical-role classification](phase_3/README.md): Train models for rhetorical-role (RR) classification on the CL and IT domains. The experiments include joint training, adapter and distillation methods, and sentence-context and pooling variants.
- [**Phase 4:** CJPE experiments](phase_4/README.md): Explore chunk-based CJPE experiments with long case documents. Notebooks compare parameter-efficient adaptation, pooling and chunk attention, partial fine-tuning, and smaller students trained with knowledge distillation.
- [**Phase 5:** Legal statute identification](phase_5/README.md): Train multi-label legal statute identification (LSI) models. Experiments compare LoRMA, AdaLoRA, QLoRA, partial fine-tuning, and distillation across decoder-only, MoE, dual-attention, and encoder–decoder backbones.
- [**Phase 6:** Legal summarization](phase_6/README.md): Generate summaries of legal documents with decoder-only, MoE, dual-attention, and encoder–decoder models. Experiments include LoRA, LoRMA, Houlsby adapters, distillation, partial fine-tuning, neighboring context, and learned sentence selection.
- [**Phase 7:** Prior case retrieval](phase_7/README.md): Rank candidate prior cases for a query case (PCR). The notebooks compare LoRA and LoRMA retrievers, distillation, partial fine-tuning, QKV rotation/scaling, and added adapters with MLP scoring.

## Repository layout

- `phase_1/`–`phase_7/` contain the phase notebooks and a README

## Running the notebooks

The notebooks are separate experiments. Open the notebook for the desired run and review its configuration cells before execution. Set the required dataset, checkpoint, tokenizer, output paths, and runtime options for that experiment. Many notebooks are designed for a Kaggle-style GPU runtime and may read data or checkpoints from Hugging Face or configured storage. Provide credentials through the runtime's secret manager when required.


