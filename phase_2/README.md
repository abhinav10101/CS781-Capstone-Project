# Phase 2: Legal named-entity recognition

I adapted Phase 1 models to predict character-level legal entity spans. The metric is strict entity Macro-F1: both the entity type and full boundary must match.

## Model and preprocessing

The 12 types are `APP`, `RESP`, `A.COUNSEL`, `R.COUNSEL`, `JUDGE`, `WIT`, `AUTH`, `COURT`, `STAT`, `PREC`, `DATE`, and `CASENO`. BIO tagging produces 25 labels including `O`.

I retain original-text offsets, group byte-level subwords into words, remove leading whitespace from recovered spans, and align gold entities by character overlap. The first word receives `B-type`; subsequent words receive `I-type`. Word-boundary chunks retain global offsets for reconstruction. The classifier averages subword states for each word and predicts its BIO label. Compatible labels merge into entity spans; an invalid `I` starts a new entity. Multiword entities can cross chunks, and BIO cannot encode arbitrary overlaps.

The two labeled folds contain 70 documents in total. Dense reference runs use a seeded 99/1 split, giving 69 train and one validation document; gated and word-Transformer runs use 90/10. Test contains 35 documents. Dense inputs are generally 512 tokens; the dual-attention variant uses 2,048.

## Adaptation

I compared LoRA, LoRMA, DoRA, and my LoRMA/DoRA combination. The combination applies a multiplicative transformation and learned row magnitudes. Typical attention settings are rank 16, alpha 32, dropout 0.05, plus a trainable BIO head.

The gated variant mixes pre/post multiplicative branches with a learned sigmoid. The word-context variant adds two bidirectional word Transformer blocks. These use my Phase 1 backbone, not pretrained BERT weights.

The plain partial-tuning baseline randomly applies updates to half the backbone parameter elements, restores the others, and clears their optimizer momentum. Combined partial variants instead unfreeze final blocks. The KD student has six width-768 layers and combines gold CE with temperature-two teacher KL, excluding padded words.

AdamW reference settings are learning rate 2e-4, weight decay 0.01, batch four, accumulation four, and clipping 1.0. Explicit training calls below can override earlier epoch constants.

## Results

Combined LoRMA/DoRA plus partial tuning led at **0.2368**, followed by word context at 0.2358 and combined PEFT at 0.2338. The unweighted LoRMA teacher scored 0.2290, plain partial tuning 0.2271, and the student 0.1675. Adding partial tuning to unweighted LoRMA gave 0.2235, but also changed learning rate to 1e-5. Dual-attention and encoder-decoder transfer scored 0.2203 and 0.1863. Small validation splits and differing adaptation scope limit conclusions from small margins.

## Submission format

```json
{"document_id": [[10, 24, "COURT"], [41, 51, "DATE"]]}
```

Offsets refer to the original text. Each ZIP contains root `prediction.json` with 35 documents. Keep all IDs and check `0 <= start < end <= len(text)`.

## Notebook experiments

Run settings below show the requested epoch/token limit. Checkpoints can be selected earlier. Shared preprocessing and loss are described above; the table records the changes for each experiment.

| Notebook | What changes | Run settings |
| :-- | :-- | :-- |
| [`phase2_decoder_126M_lorma_dora_gated_pre_post.ipynb`](phase2_decoder_126M_lorma_dora_gated_pre_post.ipynb) | Learn a sigmoid blend of pre/post LoRMA branches with DoRA-style magnitudes; use a 90/10 split. | 512 tokens; LR 0.0002; up to 5 epochs |
| [`phase2_decoder_126M_lorma_dora_word_transformer_2.ipynb`](phase2_decoder_126M_lorma_dora_word_transformer_2.ipynb) | Add two bidirectional word Transformer blocks, with zero-initialized residual outputs; use a 90/10 split. | 512 tokens; LR 0.0002; up to 5 epochs |
| [`phase2_decoder_126M_lorma_dorma.ipynb`](phase2_decoder_126M_lorma_dorma.ipynb) | Combine LoRMA transformations and learned DoRA-style row magnitudes, with a word-level BIO head. | 512 tokens; LR 0.0002; up to 20 epochs |
| [`phase2_decoder_126M_partial_ft.ipynb`](phase2_decoder_126M_partial_ft.ipynb) | Randomly update half the backbone parameter elements; restore the rest and clear their momentum. Train the BIO head normally. | 512 tokens; LR 0.0002; up to 20 epochs |
| [`phase2_decoder_126M_student_distillation_lorma_unweighted.ipynb`](phase2_decoder_126M_student_distillation_lorma_unweighted.ipynb) | Distill the unweighted LoRMA teacher into six width-768 student layers using CE and temperature-two KL. | 512 tokens; LR 0.0002; up to 30 epochs |
| [`phase2_decoder_126M_teacher_dora_unweighted.ipynb`](phase2_decoder_126M_teacher_dora_unweighted.ipynb) | Train DoRA attention adaptation and the BIO head using unweighted CE. | 512 tokens; LR 0.0002; up to 20 epochs |
| [`phase2_decoder_126M_teacher_lora.ipynb`](phase2_decoder_126M_teacher_lora.ipynb) | LoRA teacher reference. The training tag is teacher_lora_seq_512 and the export tag is teacher_lora_weighted. | 512 tokens; LR 0.0002; up to 15 epochs |
| [`phase2_decoder_126M_teacher_lorma.ipynb`](phase2_decoder_126M_teacher_lorma.ipynb) | Unweighted LoRMA teacher; the training/export calls use teacher_lorma_unweighted. | 512 tokens; LR 0.0002; up to 20 epochs |
| [`phase2_decoder_126M_teacher_lorma_unweighted_partial_ft.ipynb`](phase2_decoder_126M_teacher_lorma_unweighted_partial_ft.ipynb) | Unfreeze final blocks alongside LoRMA; lower the learning rate to 1e-5 and resume the unweighted teacher. | 512 tokens; LR 1e-05; up to 30 epochs |
| [`phase2_dual_attention_lorma_dora.ipynb`](phase2_dual_attention_lorma_dora.ipynb) | Use the dual local/global attention MoE backbone with LoRMA/DoRA and a 2,048-token BIO representation. | 2048 tokens; LR 0.0002; up to 5 epochs |
| [`phase2_encoder_decoder_lorma_dora.ipynb`](phase2_encoder_decoder_lorma_dora.ipynb) | Use encoder-decoder representations with LoRMA/DoRA and offset-based word tagging. | 512 tokens; LR 0.0002; up to 5 epochs |

## Submission files

Scores are official primary-metric values. The ZIPs contain predictions in the schema above.

| Submission | Experiment | Score |
| :-- | :-- | --: |
| [`dual_attention_moe_lorma_dora.zip`](dual_attention_moe_lorma_dora.zip) | Dual-attention MoE with LoRMA and DoRA | 0.2203 |
| [`encdec_lorma_dora.zip`](encdec_lorma_dora.zip) | Encoder–decoder LoRMA–DoRA model | 0.1863 |
| [`gated_pre_post_lorma_dora.zip`](gated_pre_post_lorma_dora.zip) | Gated pre/post LoRMA–DoRA variant | 0.2267 |
| [`lorma_dora.zip`](lorma_dora.zip) | A decoder LoRMA–DoRA run | 0.2338 |
| [`lorma_dora_1.zip`](lorma_dora_1.zip) | Alternate legal NER predictions from a decoder LoRMA–DoRA run | 0.2191 |
| [`lorma_dora_bilstm_crf.zip`](lorma_dora_bilstm_crf.zip) | LoRMA–DoRA BiLSTM–CRF variant | - |
| [`lorma_dora_partial_FFT.zip`](lorma_dora_partial_FFT.zip) | LoRMA + DoRA with additional partial fine-tuning | 0.2368 |
| [`lorma_dora_word_transformer_1.zip`](lorma_dora_word_transformer_1.zip) | LoRMA–DoRA model with one word-Transformer setting | - |
| [`lorma_dora_word_transformer_1_1.zip`](lorma_dora_word_transformer_1_1.zip) | Alternate legal NER predictions from the LoRMA–DoRA word-Transformer run | - |
| [`lorma_dora_word_transformer_2.zip`](lorma_dora_word_transformer_2.zip) | LoRMA–DoRA model with the deeper word-Transformer setting | 0.2358 |
| [`partial_FFT.zip`](partial_FFT.zip) | Random-element partial backbone updates and BIO head | 0.2271 |
| [`student_lorma_unweighted.zip`](student_lorma_unweighted.zip) | Unweighted LoRMA distillation student | 0.1675 |
| [`teacher_dora_unweighted.zip`](teacher_dora_unweighted.zip) | Unweighted DoRA teacher | 0.2285 |
| [`teacher_lora_1.zip`](teacher_lora_1.zip) | LoRA teacher with the smaller rank setting | - |
| [`teacher_lora_64.zip`](teacher_lora_64.zip) | LoRA teacher with rank 64 | - |
| [`teacher_lora_seq_512.zip`](teacher_lora_seq_512.zip) | Sequential LoRA teacher using 512-token chunks | - |
| [`teacher_lora_weighted.zip`](teacher_lora_weighted.zip) | Weighted-loss LoRA teacher | - |
| [`teacher_lorma.zip`](teacher_lorma.zip) | LoRMA teacher | - |
| [`teacher_lorma_unweighted.zip`](teacher_lorma_unweighted.zip) | Unweighted LoRMA teacher | 0.2290 |
| [`teacher_lorma_unweighted_partial_FF.zip`](teacher_lorma_unweighted_partial_FF.zip) | Unweighted LoRMA teacher with partial fine-tuning | 0.2235 |

## Changes for additional submissions

| Submission | Base implementation | Changed setting |
| :-- | :-- | :-- |
| `teacher_lora_1.zip` | [`phase2_decoder_126M_teacher_lora.ipynb`](phase2_decoder_126M_teacher_lora.ipynb) | LoRA rank `r=1` |
| `teacher_lora_64.zip` | [`phase2_decoder_126M_teacher_lora.ipynb`](phase2_decoder_126M_teacher_lora.ipynb) | LoRA rank `r=64` |
| `lorma_dora_word_transformer_1.zip`, `lorma_dora_word_transformer_1_1.zip` | [`phase2_decoder_126M_lorma_dora_word_transformer_2.ipynb`](phase2_decoder_126M_lorma_dora_word_transformer_2.ipynb) | One added word-Transformer layer: `EXTRA_WORD_LAYERS=1` |
| `lorma_dora_partial_FFT.zip` | [`phase2_decoder_126M_lorma_dorma.ipynb`](phase2_decoder_126M_lorma_dorma.ipynb) | Add partial backbone fine-tuning alongside LoRMA/DoRA |
| `lorma_dora_bilstm_crf.zip` | [`phase2_decoder_126M_lorma_dorma.ipynb`](phase2_decoder_126M_lorma_dorma.ipynb) | Replace the tagging head with BiLSTM + CRF |

[Setup and checkpoint instructions](../README.md#data-and-setup) | [Project overview](../README.md)
