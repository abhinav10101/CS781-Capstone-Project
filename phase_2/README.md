# Phase 2: Legal named-entity recognition

Adapt Phase 1 backbones to token-level legal NER using BIO labels. Experiments compare LoRA, LoRMA, DoRA, partial fine-tuning, distillation, encoder–decoder and MoE backbones, plus word-level Transformer and gated-adapter variants.

## Scope

- NER training and validation notebooks.
- Submission ZIPs for several experiment variants.
- Most runs depend on a matching Phase 1 checkpoint and tokenizer.

## Notebooks

- `phase2_decoder_126M_lorma_dora_gated_pre_post.ipynb` : Tests gated pre- and post-branches in a LoRMA–DoRA decoder for legal NER.
- `phase2_decoder_126M_lorma_dora_word_transformer_2.ipynb` : Adds bidirectional word-level Transformer layers to a LoRMA–DoRA decoder NER model.
- `phase2_decoder_126M_lorma_dorma.ipynb` : Evaluates the decoder LoRMA/DoRMA adaptation variant for legal NER.
- `phase2_decoder_126M_partial_ft.ipynb` : Fine-tunes selected decoder layers and the BIO classifier for legal NER.
- `phase2_decoder_126M_student_distillation_lorma_unweighted.ipynb` : Trains a LoRMA student using unweighted teacher distillation for legal NER.
- `phase2_decoder_126M_teacher_dora_unweighted.ipynb` : Trains a DoRA teacher for legal NER with unweighted training loss.
- `phase2_decoder_126M_teacher_lora.ipynb` : Trains a decoder LoRA teacher for legal NER.
- `phase2_decoder_126M_teacher_lorma.ipynb` : Trains a decoder LoRMA teacher for legal NER.
- `phase2_decoder_126M_teacher_lorma_unweighted_partial_ft.ipynb` : Combines unweighted LoRMA teacher training with partial decoder fine-tuning.
- `phase2_dual_attention_lorma_dora.ipynb` : Adapts a dual local/global attention MoE model with LoRMA and DoRA for legal NER.
- `phase2_encoder_decoder_lorma_dora.ipynb` : Adapts an encoder–decoder model with LoRMA and DoRA for legal NER.

## Submission archives

- `dual_attention_moe_lorma_dora.zip` : Legal NER predictions from the dual-attention MoE with LoRMA and DoRA.
- `encdec_lorma_dora.zip` : Legal NER predictions from the encoder–decoder LoRMA–DoRA model.
- `gated_pre_post_lorma_dora.zip` : Legal NER predictions from the gated pre/post LoRMA–DoRA variant.
- `lorma_dora.zip` : Legal NER predictions from a decoder LoRMA–DoRA run.
- `lorma_dora_1.zip` : Alternate legal NER predictions from a decoder LoRMA–DoRA run.
- `lorma_dora_bilstm_crf.zip` : Legal NER predictions from the LoRMA–DoRA BiLSTM–CRF variant.
- `lorma_dora_partial_FFT.zip` : Legal NER predictions from the LoRMA–DoRA partial fine-tuning/FFT experiment.
- `lorma_dora_word_transformer_1.zip` : Legal NER predictions from the LoRMA–DoRA model with one word-Transformer setting.
- `lorma_dora_word_transformer_1_1.zip` : Alternate legal NER predictions from the LoRMA–DoRA word-Transformer run.
- `lorma_dora_word_transformer_2.zip` : Legal NER predictions from the LoRMA–DoRA model with the deeper word-Transformer setting.
- `partial_FFT.zip` : Legal NER predictions from the partial fine-tuning experiment.
- `student_lorma_unweighted.zip` : Legal NER predictions from the unweighted LoRMA distillation student.
- `teacher_dora_unweighted.zip` : Legal NER predictions from the unweighted DoRA teacher.
- `teacher_lora_1.zip` : Legal NER predictions from the LoRA teacher with the smaller rank setting.
- `teacher_lora_64.zip` : Legal NER predictions from the LoRA teacher with rank 64.
- `teacher_lora_heirarchial.zip` : Legal NER predictions from the hierarchical LoRA teacher.
- `teacher_lora_seq_512.zip` : Legal NER predictions from the sequential LoRA teacher using 512-token chunks.
- `teacher_lora_weighted.zip` : Legal NER predictions from the weighted-loss LoRA teacher.
- `teacher_lorma.zip` : Legal NER predictions from the LoRMA teacher.
- `teacher_lorma_unweighted.zip` : Legal NER predictions from the unweighted LoRMA teacher.
- `teacher_lorma_unweighted_partial_FF.zip` : Legal NER predictions from the unweighted LoRMA teacher with partial fine-tuning.

## Running this phase

Review each notebook's configuration before running it. Set the required data, checkpoint, tokenizer, output, and runtime paths; some experiments depend on checkpoints produced in earlier phases. Use the environment and credentials required by the notebook, and run from a fresh runtime unless its resume settings say otherwise.

See the [project README](../README.md) for repository-wide setup notes.
