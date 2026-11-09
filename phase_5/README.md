# Phase 5: Legal statute identification

I predicted the relevant statute IDs for each document using a 100-output multi-label classifier. Each output has an independent sigmoid; the primary metric is Macro-F1.

## Data and model

LSI contains 42,750/10,181/13,019 train/dev/test documents. Labels become multi-hot vectors with a fixed statute-ID/output-index mapping.

Dense models encode overlapping 2,048-token chunks with overlap 64. Sampled runs use two training/four evaluation chunks; all-chunk partial tuning processes every chunk in bounded batches. Encoder-decoder variants use 512-token representations. A document-level classifier pools encoded views into 100 logits.

## Loss and adaptation

I use weighted BCE-with-logits. Positive weights are `sqrt((N - positives) / positives)`, clipped to `[1,10]`. Development threshold search spans 0.05-0.80. Labels with at least five positives get individual thresholds; others use a global fallback. I save thresholds and label order with the checkpoint. There is no explicit label-dependency graph.

LoRMA references use rank 16, alpha 32, dropout 0.05. AdaLoRA changes rank from 16 toward eight. QLoRA uses frozen NF4 linear weights with double quantization and low-rank updates in the final two blocks. Partial tuning unfreezes the final two blocks and classifier, using full data/all chunks for one epoch at 3e-5.

The four-layer KD student learns from the LoRMA teacher using weighted gold BCE and temperature-two Bernoulli teacher divergence. It uses 40% data, three epochs, and 5e-5 learning rate; reference LoRMA uses 20%, five epochs, and 2e-4. These differences matter when interpreting the KD gain.

## Results

The student led at **0.3350**, reducing parameters by about 36.7%. Partial tuning scored 0.3080, QLoRA 0.3121, AdaLoRA 0.2884, dense LoRMA 0.2782, MoE 0.2700, full-data encoder-decoder 0.1128, 20%-data encoder-decoder 0.1034, and dual attention 0.0927. Rare-statute validation Macro-F1 remained lower: 0.2311 versus 0.3871 for other labels in the student run. Threshold tuning and weighting did not eliminate the long tail.

## Submission format

```json
[{"id": "case_id", "pred": [5, 34, 97]}]
```

Each ZIP contains root `prediction.json` with 13,019 records. Return valid statute IDs without duplicates and retain every test ID.

## Notebook experiments

Run settings below show the requested epoch/token limit. Checkpoints can be selected earlier. Shared preprocessing and loss are described above; the table records the changes for each experiment.

| Notebook | What changes | Run settings |
| :-- | :-- | :-- |
| [`phase5_decoder_126M_adalora.ipynb`](phase5_decoder_126M_adalora.ipynb) | Adapt rank from 16 toward eight with orthogonality regularization; tune independent statute thresholds. | 2048 tokens; 40% data; LR 0.0002; up to 5 epochs |
| [`phase5_decoder_126M_distillation_student4.ipynb`](phase5_decoder_126M_distillation_student4.ipynb) | Four-layer student using weighted gold BCE and temperature-two Bernoulli teacher targets. | 2048 tokens; 40% data; LR 5e-05; up to 3 epochs |
| [`phase5_decoder_126M_lorma.ipynb`](phase5_decoder_126M_lorma.ipynb) | LoRMA adaptation of the dense GQA decoder. Pool document chunks into 100 independent statute logits. | 2048 tokens; 20% data; LR 0.0002; up to 5 epochs |
| [`phase5_decoder_126M_partial_ft_all_chunks.ipynb`](phase5_decoder_126M_partial_ft_all_chunks.ipynb) | Train the last two blocks and classifier on all chunks/full data for one epoch. | 2048 tokens; 100% data; LR 3e-05; up to 1 epochs |
| [`phase5_decoder_126M_qlora.ipynb`](phase5_decoder_126M_qlora.ipynb) | Use NF4/double-quantized base linear weights and low-rank updates in the final two blocks. | 2048 tokens; 40% data; LR 0.0002; up to 5 epochs |
| [`phase5_decoder_moe_lorma.ipynb`](phase5_decoder_moe_lorma.ipynb) | LoRMA adaptation of the MoE decoder. Pool document chunks into 100 independent statute logits. | 2048 tokens; 20% data; LR 0.0002; up to 5 epochs |
| [`phase5_dual_attention_lorma.ipynb`](phase5_dual_attention_lorma.ipynb) | LoRMA adaptation of the dual local/global attention MoE. Pool document chunks into 100 independent statute logits. | 2048 tokens; 20% data; LR 0.0002; up to 5 epochs |
| [`phase5_encoder_decoder_mixed_lorma_data100pct.ipynb`](phase5_encoder_decoder_mixed_lorma_data100pct.ipynb) | Encoder-decoder LoRMA using all training documents. | 512 tokens; 100% data; LR 0.0002; up to 5 epochs |
| [`phase5_encoder_decoder_mixed_lorma_data20pct.ipynb`](phase5_encoder_decoder_mixed_lorma_data20pct.ipynb) | Encoder-decoder LoRMA using a 20% training sample. | 512 tokens; 20% data; LR 0.0002; up to 5 epochs |

## Submission files

Scores are official primary-metric values. The ZIPs contain predictions in the schema above.

| Submission | Experiment | Score |
| :-- | :-- | --: |
| [`lsi_decoder_126M_adalora.zip`](lsi_decoder_126M_adalora.zip) | 126M decoder with AdaLoRA | 0.2884 |
| [`lsi_decoder_126M_distillation.zip`](lsi_decoder_126M_distillation.zip) | Decoder knowledge-distillation run | 0.3350 |
| [`lsi_decoder_126M_lorma.zip`](lsi_decoder_126M_lorma.zip) | 126M decoder with LoRMA | 0.2782 |
| [`lsi_decoder_126M_partial_fft.zip`](lsi_decoder_126M_partial_fft.zip) | Decoder partial fine-tuning run | 0.3080 |
| [`lsi_decoder_126M_qlora_partial.zip`](lsi_decoder_126M_qlora_partial.zip) | Decoder QLoRA partial-training run | 0.3121 |
| [`lsi_decoder_MOE.zip`](lsi_decoder_MOE.zip) | Decoder MoE with LoRMA | 0.2700 |
| [`lsi_dual_attention_lorma.zip`](lsi_dual_attention_lorma.zip) | Dual-attention MoE with LoRMA | 0.0927 |
| [`lsi_encoder_decoder_mixed_full.zip`](lsi_encoder_decoder_mixed_full.zip) | Full-data encoder–decoder run | 0.1128 |
| [`lsi_encoder_decoder_mixed_lorma.zip`](lsi_encoder_decoder_mixed_lorma.zip) | Encoder–decoder LoRMA run | 0.1034 |

[Setup and checkpoint instructions](../README.md#data-and-setup) | [Project overview](../README.md)
