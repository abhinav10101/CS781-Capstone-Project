# Phase 4: Court judgment prediction

I predicted binary case outcomes using Phase 1 backbones and different long-document representations. The primary metric is Macro-F1.

## Data and document representations

CJPE contains 37,387/3,505/1,517 train/dev/test records, with 32,305/2,511 unique train/dev case IDs. Training has 22,067 zeros and 15,320 ones; development is nearly balanced at 1,749/1,756. Main comparisons use 20% training data; confidence KD uses 60%, and the older sentence-adapter run uses 80%.

The binary head operates on pooled document views:

- **Chunk mean:** sample overlapping token windows, encode them, then average at case level. Common 2,048-token runs use overlap 64 and two training/four evaluation windows.
- **Chunk attention:** learn weights over chunk vectors. The matched LoRA/LoRMA pair uses 768-token views.
- **Head/tail:** encode first/last 1,024-token views and use mean/max aggregation.
- **Sentence context:** encode selected sentences with neighboring text under 512 tokens and learn their case-level weights.

`BATCH_SIZE` counts cases; `CHUNK_ENCODE_BATCH_SIZE` limits views per encoder call. Some reused variables say `SENTENCES` even when the representation is chunks.

## Adaptation and custom experiments

I compared LoRA, LoRMA, AdaLoRA, bottleneck adapters, and partial tuning. Low-rank references generally use rank 16, alpha 32, dropout 0.05. AdaLoRA starts at rank 16 and targets rank eight, with orthogonality regularization.

Fixed partial tuning updates the final block and classifier. My Bernoulli version chooses one or two final blocks once per accumulation group: two with probability 0.25, otherwise one. The frozen prefix runs without gradients. This changes backward update depth; it does not drop forward layers.

KD builds three-, four-, or five-layer copied students from a frozen LoRMA teacher. Standard loss mixes gold CE and temperature-two KL at alpha 0.5. My confidence variant uses KD weight `0.5 * max_teacher_probability` only when the teacher predicts the correct training label; otherwise it uses gold CE alone. The CE control sets alpha zero. Other variants change the trainable student suffix or anneal teacher supervision. Teacher logits are cached with checks against the teacher and training subset.

## Results

Confidence KD led at **0.6639**, standard four-layer KD scored 0.6530, and three-layer KD 0.6355. LoRA chunk attention reached 0.6182 versus LoRMA's 0.5636. Head/tail scored 0.5780, Bernoulli tuning 0.5641, fixed-block tuning 0.5566, AdaLoRA 0.5398, and sentence context 0.4518. MoE reached 0.5619; dual attention and encoder-decoder each scored 0.3323 and collapsed toward the zero class in validation.

Bernoulli versus fixed-block tuning is a close update-depth comparison. Head/tail changes coverage alongside pooling; confidence KD changes data fraction alongside the loss. The four-layer model has about 81.47M parameters versus its teacher's 128.86M.

## Submission format

```json
[{"id": "case_id", "pred": 1}]
```

Each ZIP contains root `prediction.json` with 1,517 records and one integer 0/1 prediction per test ID.

## Notebook experiments

Run settings below show the requested epoch/token limit. Checkpoints can be selected earlier. Shared preprocessing and loss are described above; the table records the changes for each experiment.

| Notebook | What changes | Run settings |
| :-- | :-- | :-- |
| [`phase-4-adaptors.ipynb`](phase-4-adaptors.ipynb) | Bottleneck-64 adapters with 512-token sentence context; select eight training/sixteen evaluation views. | 512 tokens; 80% data; LR 0.0002; up to 20 epochs |
| [`phase-4-ce-control.ipynb`](phase-4-ce-control.ipynb) | Four-layer copied student trained with gold CE only (KD alpha zero); keep teacher-head initialization. | 2048 tokens; 60% data; LR 0.0001; up to 5 epochs |
| [`phase-4-chunk-attn.ipynb`](phase-4-chunk-attn.ipynb) | Bottleneck adapters with learned weights over 768-token chunks; select two training/four evaluation views. | 768 tokens; 20% data; LR 0.0002; up to 5 epochs |
| [`phase-4-confidence (1).ipynb`](phase-4-confidence%20%281%29.ipynb) | Four-layer confidence/correctness-gated KD; train the final two student blocks, with an eleven-hour session budget. | 2048 tokens; 60% data; LR 0.0001; up to 7 epochs |
| [`phase-4-final.ipynb`](phase-4-final.ipynb) | Four-layer student with fixed KD and the final three student blocks trainable. | 2048 tokens; 60% data; LR 0.0001; up to 5 epochs |
| [`phase-4-kd-5.ipynb`](phase-4-kd-5.ipynb) | Five-layer student with two trainable final blocks and a 20-epoch budget. | 2048 tokens; 20% data; LR 0.0001; up to 20 epochs |
| [`phase-4-kd-student.ipynb`](phase-4-kd-student.ipynb) | Three-layer copied student with two trainable final blocks. | 2048 tokens; 20% data; LR 0.0001; up to 20 epochs |
| [`phase-4-kd.ipynb`](phase-4-kd.ipynb) | Four-layer student with annealed teacher supervision followed by label-focused training. | 2048 tokens; 20% data; LR 0.0001; up to 20 epochs |
| [`phase4_decoder_126M_adalora_len2048.ipynb`](phase4_decoder_126M_adalora_len2048.ipynb) | Dense AdaLoRA: initial rank 16, target eight, orthogonality weight 0.1, with rank warmup/final fractions 0.10/0.20. | 2048 tokens; 20% data; LR 0.0002; up to 7 epochs |
| [`phase4_decoder_126M_confidence_kd_student4_len2048_data60pct.ipynb`](phase4_decoder_126M_confidence_kd_student4_len2048_data60pct.ipynb) | Four-layer student with confidence/correctness-gated KD and two trainable final blocks. | 2048 tokens; 60% data; LR 0.0001; up to 7 epochs |
| [`phase4_decoder_126M_kd_student4_len2048_data20pct.ipynb`](phase4_decoder_126M_kd_student4_len2048_data20pct.ipynb) | Four-layer student with fixed teacher weight, serving as the standard KD comparison. | 2048 tokens; 20% data; LR 0.0001; up to 8 epochs |
| [`phase4_decoder_126M_lora_chunk_attention_len768.ipynb`](phase4_decoder_126M_lora_chunk_attention_len768.ipynb) | LoRA with learned chunk attention; two training/four evaluation windows. | 768 tokens; 20% data; LR 0.0002; up to 5 epochs |
| [`phase4_decoder_126M_lorma_chunk_attention_len768.ipynb`](phase4_decoder_126M_lorma_chunk_attention_len768.ipynb) | LoRMA with the matching 768-token chunk-attention setup. | 768 tokens; 20% data; LR 0.0002; up to 5 epochs |
| [`phase4_decoder_126M_lorma_head_tail_len1024.ipynb`](phase4_decoder_126M_lorma_head_tail_len1024.ipynb) | LoRMA over first/last 1,024-token views with mean/max aggregation. | 1024 tokens; 20% data; LR 0.0002; up to 5 epochs |
| [`phase4_decoder_126M_lorma_len2048.ipynb`](phase4_decoder_126M_lorma_len2048.ipynb) | LoRMA adaptation of the dense GQA decoder. Pool overlapping document chunks into a binary case head. | 2048 tokens; LR 0.0002; up to 5 epochs |
| [`phase4_decoder_126M_lorma_sentence_attention_len512.ipynb`](phase4_decoder_126M_lorma_sentence_attention_len512.ipynb) | LoRMA over sentence-context views with learned case-level attention. | 512 tokens; 20% data; LR 0.0002; up to 5 epochs |
| [`phase4_decoder_126M_partial_ft_bernoulli_last2_p025_len512.ipynb`](phase4_decoder_126M_partial_ft_bernoulli_last2_p025_len512.ipynb) | Choose two final update blocks with probability 0.25, otherwise one, once per accumulation group. | 512 tokens; 20% data; LR 2e-05; up to 5 epochs |
| [`phase4_decoder_126M_partial_ft_last_block_len512.ipynb`](phase4_decoder_126M_partial_ft_last_block_len512.ipynb) | Train the final dense block and case head while freezing the prefix. | 512 tokens; 20% data; LR 2e-05; up to 5 epochs |
| [`phase4_decoder_moe_lorma_len2048.ipynb`](phase4_decoder_moe_lorma_len2048.ipynb) | LoRMA adaptation of the MoE decoder. Pool overlapping document chunks into a binary case head. | 2048 tokens; LR 0.0002; up to 5 epochs |
| [`phase4_dual_attention_200M_lorma_len2048.ipynb`](phase4_dual_attention_200M_lorma_len2048.ipynb) | LoRMA adaptation of the dual local/global attention MoE. Pool overlapping document chunks into a binary case head. | 2048 tokens; LR 0.0002; up to 5 epochs |
| [`phase4_encoder_decoder_mixed_lorma_len512.ipynb`](phase4_encoder_decoder_mixed_lorma_len512.ipynb) | LoRMA adaptation of the encoder-decoder. Pool overlapping document chunks into a binary case head. | 512 tokens; LR 0.0002; up to 5 epochs |

## Submission files

Scores are official primary-metric values. The ZIPs contain predictions in the schema above.

| Submission | Experiment | Score |
| :-- | :-- | --: |
| [`cjpe_decoder_126M_01_lorma_chunk_at.zip`](cjpe_decoder_126M_01_lorma_chunk_at.zip) | Decoder LoRMA with learned chunk attention | 0.5636 |
| [`cjpe_decoder_126M_03_lorma_sentence.zip`](cjpe_decoder_126M_03_lorma_sentence.zip) | Decoder LoRMA with sentence-context attention | 0.4518 |
| [`cjpe_decoder_126M_05_partial_last_b.zip`](cjpe_decoder_126M_05_partial_last_b.zip) | Partial last-block fine-tuning run | 0.5566 |
| [`cjpe_decoder_126M_06_lora_chunk_att.zip`](cjpe_decoder_126M_06_lora_chunk_att.zip) | Decoder LoRA with chunk attention | 0.6182 |
| [`cjpe_decoder_126M_07_bernoulli_trai.zip`](cjpe_decoder_126M_07_bernoulli_trai.zip) | Bernoulli partial fine-tuning run | 0.5641 |
| [`cjpe_decoder_126M_10_student3_kd.zip`](cjpe_decoder_126M_10_student3_kd.zip) | Three-layer knowledge-distillation student | 0.6355 |
| [`cjpe_decoder_126M_12_student4_confi.zip`](cjpe_decoder_126M_12_student4_confi.zip) | Four-layer confidence-gated distillation student | 0.6639 |
| [`cjpe_decoder_126M_adalora.zip`](cjpe_decoder_126M_adalora.zip) | Decoder AdaLoRA run | 0.5398 |
| [`cjpe_decoder_126M_distillation_stud.zip`](cjpe_decoder_126M_distillation_stud.zip) | Decoder distillation student | 0.6530 |
| [`cjpe_decoder_MOE_lorma.zip`](cjpe_decoder_MOE_lorma.zip) | Decoder MoE with LoRMA | 0.5619 |
| [`cjpe_dual_attention_200M_lorma.zip`](cjpe_dual_attention_200M_lorma.zip) | Dual-attention model with LoRMA | 0.3323 |
| [`cjpe_encoder_decoder_mixed_lorma.zip`](cjpe_encoder_decoder_mixed_lorma.zip) | Encoder–decoder model with LoRMA | 0.3323 |

[Setup and checkpoint instructions](../README.md#data-and-setup) | [Project overview](../README.md)
