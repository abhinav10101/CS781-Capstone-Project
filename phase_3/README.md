# Phase 3: Rhetorical-role prediction

I classified each sentence into one of 13 rhetorical roles, comparing joint CL/IT training, adaptation methods, document context, and pooling. The primary metric is Macro-F1.

## Data and architecture

Each domain has 40/5/5 train/dev/test documents. CL contains 10,593/1,396/1,339 sentences and IT 6,252/746/858. I load the role order from `label_vocab.json`: Fact, Issue, ArgumentPetitioner, ArgumentRespondent, Statute, Dissent, PrecedentReliedUpon, PrecedentNotReliedUpon, PrecedentOverruled, RulingByLowerCourt, RatioOfTheDecision, RulingByPresentCourt, and None.

The baseline pools the final valid token and adds a classifier. Context-free MLP uses a 512-wide GELU layer and dropout. Contextual variants pack up to two preceding and two following sentences within 512 tokens, reserving room for the target. Neighbors keep their relative order; the target goes last so a causal decoder can use both sides. Pooling uses target-token positions only, and neighboring labels are excluded.

Pooling variants use scalar attention, a one-layer BiLSTM with 256 units per direction, or 13 label-specific attention distributions with a 128-wide scoring network. Each label-specific representation receives its own score, followed by single-label prediction. There is no document-level CRF in these variants.

## Training and compression

LoRA, LoRMA, and DoRA adapt attention projections; partial adaptation also unfreezes the final two blocks. Typical settings are rank 16, alpha 32, dropout 0.05, AdamW at 2e-4, weight decay 0.01, batch four, accumulation four, and up to 20 epochs. Joint-training checkpoint selection averages CL and IT validation Macro-F1.

Four-layer students copy teacher layers `[0,2,4,6]` and combine CE with temperature-two KL at alpha 0.5. Dense task parameters fall from approximately 129.46M to 81.48M; MoE parameters fall from 178.44M to 70.86M.

## Results

Context-conditioned LoRMA led at **0.4065**. Contextual BiLSTM scored 0.4017, label attention 0.3874, partial adaptation 0.3870, and context-free MLP 0.3735. Head capacity also changes in these comparisons. Dense LoRA/DoRA/LoRMA references scored 0.3596/0.3578/0.3529. Dense KD fell to 0.3429; MoE KD improved from 0.3382 to 0.3427. Rare roles and argument-versus-reasoning attribution remained difficult, with Dissent and PrecedentOverruled absent from development.

## Submission format

```json
{"document_id": ["Fact", "Issue", "RatioOfTheDecision"]}
```

Each ZIP contains ten test documents. Preserve sentence order and output exactly one role for each input sentence.

## Notebook experiments

Run settings below show the requested epoch/token limit. Checkpoints can be selected earlier. Shared preprocessing and loss are described above; the table records the changes for each experiment.

| Notebook | What changes | Run settings |
| :-- | :-- | :-- |
| [`phase3_decoder_126M_distillation_joint.ipynb`](phase3_decoder_126M_distillation_joint.ipynb) | Copy dense teacher layers [0,2,4,6]; combine gold CE and teacher KL at temperature two and alpha 0.5. | 512 tokens; LR 0.0002; up to 20 epochs |
| [`phase3_decoder_126M_dora_joint.ipynb`](phase3_decoder_126M_dora_joint.ipynb) | DoRA adaptation of the dense GQA decoder. Joint CL/IT training with a sentence-level role head. | 512 tokens; LR 0.0002; up to 20 epochs |
| [`phase3_decoder_126M_lora_joint.ipynb`](phase3_decoder_126M_lora_joint.ipynb) | LoRA adaptation of the dense GQA decoder. Joint CL/IT training with a sentence-level role head. | 512 tokens; LR 0.0002; up to 20 epochs |
| [`phase3_decoder_126M_lorma_ctx0_mlp.ipynb`](phase3_decoder_126M_lorma_ctx0_mlp.ipynb) | Classify the target sentence alone with a 512-wide GELU MLP; no neighboring context. | 512 tokens; LR 0.0002; up to 20 epochs |
| [`phase3_decoder_126M_lorma_ctx2_bilstm.ipynb`](phase3_decoder_126M_lorma_ctx2_bilstm.ipynb) | Use two neighbors per side and BiLSTM pooling over target tokens, with 256 units per direction. | 512 tokens; LR 0.0002; up to 20 epochs |
| [`phase3_decoder_126M_lorma_ctx2_label_attention.ipynb`](phase3_decoder_126M_lorma_ctx2_label_attention.ipynb) | Use two-neighbor context and one target-token attention distribution and classifier per role. | 512 tokens; LR 0.0002; up to 20 epochs |
| [`phase3_decoder_126M_lorma_ctx2_partial_ft.ipynb`](phase3_decoder_126M_lorma_ctx2_partial_ft.ipynb) | Use two-neighbor context and unfreeze the last two blocks alongside LoRMA. | 512 tokens; LR 0.0002; up to 20 epochs |
| [`phase3_decoder_126M_lorma_joint.ipynb`](phase3_decoder_126M_lorma_joint.ipynb) | LoRMA adaptation of the dense GQA decoder. Joint CL/IT training with a sentence-level role head. | 512 tokens; LR 0.0002; up to 20 epochs |
| [`phase3_decoder_moe_125M_distillation_joint.ipynb`](phase3_decoder_moe_125M_distillation_joint.ipynb) | Copy four MoE teacher layers and train with gold roles plus teacher distributions. | 512 tokens; LR 0.0002; up to 20 epochs |
| [`phase3_decoder_moe_125M_lora_joint.ipynb`](phase3_decoder_moe_125M_lora_joint.ipynb) | LoRA adaptation of the MoE decoder. Joint CL/IT training with a sentence-level role head. | 512 tokens; LR 0.0002; up to 20 epochs |
| [`phase3_decoder_moe_125M_lorma_joint.ipynb`](phase3_decoder_moe_125M_lorma_joint.ipynb) | LoRMA adaptation of the MoE decoder. Joint CL/IT training with a sentence-level role head. | 512 tokens; LR 0.0002; up to 20 epochs |
| [`phase3_dual_attention_200M_dora_joint.ipynb`](phase3_dual_attention_200M_dora_joint.ipynb) | DoRA adaptation of the dual local/global attention MoE. Joint CL/IT training with a sentence-level role head. | 512 tokens; LR 0.0002; up to 20 epochs |
| [`phase3_dual_attention_200M_lorma_joint.ipynb`](phase3_dual_attention_200M_lorma_joint.ipynb) | LoRMA adaptation of the dual local/global attention MoE. Joint CL/IT training with a sentence-level role head. | 512 tokens; LR 0.0002; up to 20 epochs |
| [`phase3_encoder_decoder_mixed_dora_joint.ipynb`](phase3_encoder_decoder_mixed_dora_joint.ipynb) | DoRA adaptation of the encoder-decoder. Joint CL/IT training with a sentence-level role head. | 512 tokens; LR 0.0002; up to 20 epochs |
| [`phase3_encoder_decoder_mixed_lora_joint.ipynb`](phase3_encoder_decoder_mixed_lora_joint.ipynb) | LoRA adaptation of the encoder-decoder. Joint CL/IT training with a sentence-level role head. | 512 tokens; LR 0.0002; up to 20 epochs |
| [`phase3_encoder_decoder_mixed_lorma_joint.ipynb`](phase3_encoder_decoder_mixed_lorma_joint.ipynb) | LoRMA adaptation of the encoder-decoder. Joint CL/IT training with a sentence-level role head. | 512 tokens; LR 0.0002; up to 20 epochs |

## Submission files

Scores are official primary-metric values. The ZIPs contain predictions in the schema above.

| Submission | Experiment | Score |
| :-- | :-- | --: |
| [`rr_decoder126m_dora_joint.zip`](rr_decoder126m_dora_joint.zip) | 126M decoder with DoRA | 0.3578 |
| [`rr_decoder126m_lora_joint.zip`](rr_decoder126m_lora_joint.zip) | 126M decoder with LoRA | 0.3596 |
| [`rr_decoder126m_lorma_joint.zip`](rr_decoder126m_lorma_joint.zip) | 126M decoder with LoRMA | 0.3529 |
| [`rr_decoder_125M_moe_joint_lora_333c.zip`](rr_decoder_125M_moe_joint_lora_333c.zip) | Decoder MoE with LoRA | 0.3250 |
| [`rr_decoder_126M_joint_distillation_.zip`](rr_decoder_126M_joint_distillation_.zip) | Decoder distillation run | 0.3429 |
| [`rr_decoder_126M_joint_lorma_ctx0.zip`](rr_decoder_126M_joint_lorma_ctx0.zip) | Context-free decoder LoRMA run | 0.3735 |
| [`rr_decoder_126M_joint_lorma_ctx0_ML.zip`](rr_decoder_126M_joint_lorma_ctx0_ML.zip) | Context-free LoRMA model with label-specific pooling | 0.3493 |
| [`rr_decoder_126M_joint_lorma_ctx1.zip`](rr_decoder_126M_joint_lorma_ctx1.zip) | Decoder LoRMA model with one-sentence context | - |
| [`rr_decoder_126M_joint_lorma_ctx2_at.zip`](rr_decoder_126M_joint_lorma_ctx2_at.zip) | Decoder LoRMA model with two-sentence attention context | 0.4059 / 0.4065 |
| [`rr_decoder_126M_joint_lorma_ctx2_bi.zip`](rr_decoder_126M_joint_lorma_ctx2_bi.zip) | Decoder LoRMA model with two-sentence BiLSTM context | 0.4017 |
| [`rr_decoder_126M_joint_lorma_ctx2_pa.zip`](rr_decoder_126M_joint_lorma_ctx2_pa.zip) | Decoder LoRMA model with two-sentence context and partial fine-tuning | 0.3870 |
| [`rr_decoder_MOE_joint_distillation_9.zip`](rr_decoder_MOE_joint_distillation_9.zip) | Decoder MoE distillation run | 0.3427 |
| [`rr_decoder_MOE_joint_lorma_09186db3.zip`](rr_decoder_MOE_joint_lorma_09186db3.zip) | Decoder MoE with LoRMA | 0.3382 |
| [`rr_decoder_lorma_separate.zip`](rr_decoder_lorma_separate.zip) | Separate-domain decoder LoRMA training | 0.3691 |
| [`rr_decoder_moe.zip`](rr_decoder_moe.zip) | Decoder + MoE rhetorical-role model | 0.3382 |
| [`rr_decoder_moe_125M_joint_dora_94f7.zip`](rr_decoder_moe_125M_joint_dora_94f7.zip) | Decoder MoE with DoRA | 0.3410 |
| [`rr_decoder_rectified.zip`](rr_decoder_rectified.zip) | Dense decoder rhetorical-role submission | 0.3537 |
| [`rr_dual_attention_200M_joint_distil.zip`](rr_dual_attention_200M_joint_distil.zip) | Dual-attention distillation run | - |
| [`rr_dual_attention_200M_joint_dora_9.zip`](rr_dual_attention_200M_joint_dora_9.zip) | Dual-attention model with DoRA | 0.2565 |
| [`rr_dual_attention_200M_joint_lora_9.zip`](rr_dual_attention_200M_joint_lora_9.zip) | Dual-attention model with LoRA | 0.2760 |
| [`rr_dual_attention_200M_joint_lorma_.zip`](rr_dual_attention_200M_joint_lorma_.zip) | Dual-attention model with LoRMA | 0.2269 |
| [`rr_encoder_decoder_mixed_joint_dist.zip`](rr_encoder_decoder_mixed_joint_dist.zip) | Encoder–decoder distillation run | 0.2608 |
| [`rr_encoder_decoder_mixed_joint_dora.zip`](rr_encoder_decoder_mixed_joint_dora.zip) | Encoder–decoder model with DoRA | 0.1955 |
| [`rr_encoder_decoder_mixed_joint_lora.zip`](rr_encoder_decoder_mixed_joint_lora.zip) | Encoder–decoder model with LoRA | 0.1933 |
| [`rr_encoder_decoder_mixed_lorma_join.zip`](rr_encoder_decoder_mixed_lorma_join.zip) | Encoder–decoder model with LoRMA | 0.1934 |

## Changes for additional submissions

| Submission | Base implementation | Changed setting |
| :-- | :-- | :-- |
| `rr_decoder_126M_joint_lorma_ctx0_ML.zip` | [`phase3_decoder_126M_lorma_ctx2_label_attention.ipynb`](phase3_decoder_126M_lorma_ctx2_label_attention.ipynb) | Label-specific attention with `CONTEXT_SENTENCES=0` |
| `rr_decoder_126M_joint_lorma_ctx2_at.zip` | [`phase3_decoder_126M_lorma_ctx2_bilstm.ipynb`](phase3_decoder_126M_lorma_ctx2_bilstm.ipynb) | Scalar target-token attention: `RR_POOLING="attention"`, context two |
| `rr_decoder_lorma_separate.zip` | [`phase3_decoder_126M_lorma_joint.ipynb`](phase3_decoder_126M_lorma_joint.ipynb) | Train a separate model for CL and IT |

[Setup and checkpoint instructions](../README.md#data-and-setup) | [Project overview](../README.md)
