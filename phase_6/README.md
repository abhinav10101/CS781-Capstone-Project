# Phase 6: Abstractive legal summarization

I compared decoder-only and encoder-decoder generation, adaptation methods, source selection, and distillation. The official metric is BERTScore F1; notebook validation also tracks ROUGE, length, and repetition.

## Data and source selection

I removed two blank/null references from 7,030 pairs, grouped identical source texts, and made a seeded 95/5 split. Training uses 6,676 pairs and validation is capped at 100; test contains 100 documents. Median source/reference lengths are about 3,675/756 tokens, so source selection and target truncation affect most examples.

- **Eight spans:** sample distributed passages and join them with gap markers within the source budget.
- **Learned sentence selection:** fit a CPU classifier on lexical, position, length, and legal-cue features. Training-reference overlap supplies oracle labels. At inference it sees only source text, with a redundancy penalty and token budget.
- **Neighboring context:** encode 256-token regions with 64 tokens per side, then pool into bounded main/context memories of 1,024/512 tokens. Chunk encoding and gradient replay bound memory use.

## Architecture and training

Decoder-only models place source text before the summary in a causal sequence, masking source positions from the summary loss. The source budget excludes delimiters and the generation allowance. Encoder-decoder references use 512 source/256 target tokens, teacher forcing, EOS-aware targets, and padding-masked loss.

I compared LoRA, LoRMA, Houlsby bottleneck adapters, and partial tuning of the final two encoder plus final two decoder blocks. Reference PEFT learning rate is 2e-4; partial tuning uses 2e-5. The KD student copies three encoder and three decoder layers from the six/six LoRMA teacher. Gold CE and temperature-two full-vocabulary KL use weight 0.5; token microbatches bound the KL memory. The student also uses bounded long-document memory, changing coverage alongside the loss.

I retained greedy decoding after a beam configuration generated very short outputs. Decoding settings, selector state, and model weights must be loaded together.

## Results

Distillation led at **0.7839**, followed by fixed-eight-span partial tuning at 0.7804 and learned-sentence-selection partial tuning at 0.7758. Encoder-decoder LoRA/LoRMA scored 0.7731/0.7721, Houlsby 0.7673, neighboring-context LoRMA 0.7656, dense decoder 0.7633, MoE 0.7585, and dual attention 0.7025.

Student parameters fell from 79.50M to 55.74M; a matched five-document timing check fell from 3.564 to 2.141 seconds/document. Student ROUGE-1/ROUGE-L were 0.2524/0.1906, but repeated-trigram fraction was 0.789. Reference similarity improved while repetition remained substantial; legal factuality needs separate review.

## Submission format

```json
[{"id": "case_id", "summary": "Generated legal summary."}]
```

Each ZIP contains root `prediction.json` with 100 records. Keep one cleaned summary per test ID and remove special tokens/task delimiters.

## Notebook experiments

Run settings below show the requested epoch/token limit. Checkpoints can be selected earlier. Shared preprocessing and loss are described above; the table records the changes for each experiment.

| Notebook | What changes | Run settings |
| :-- | :-- | :-- |
| [`phase6_decoder_126M_lorma.ipynb`](phase6_decoder_126M_lorma.ipynb) | LoRMA adaptation of the dense GQA decoder. Learn summary-token generation from eight distributed source spans. | 2048 tokens; 100% data; LR 0.0002; up to 5 epochs |
| [`phase6_decoder_moe_lorma.ipynb`](phase6_decoder_moe_lorma.ipynb) | LoRMA adaptation of the MoE decoder. Learn summary-token generation from eight distributed source spans. | 2048 tokens; 100% data; LR 0.0002; up to 5 epochs |
| [`phase6_dual_attention_200M_lorma.ipynb`](phase6_dual_attention_200M_lorma.ipynb) | LoRMA adaptation of the dual local/global attention MoE. Learn summary-token generation from eight distributed source spans. | 2048 tokens; 100% data; LR 0.0002; up to 5 epochs |
| [`phase6_encoder_decoder_mixed_distillation.ipynb`](phase6_encoder_decoder_mixed_distillation.ipynb) | Three-encoder/three-decoder student with gold CE, token-microbatched KL, and bounded long-document memory. | 256 tokens; 100% data; LR 5e-05; up to 5 epochs |
| [`phase6_encoder_decoder_mixed_houlsby.ipynb`](phase6_encoder_decoder_mixed_houlsby.ipynb) | Train residual bottleneck adapters around the frozen encoder-decoder. | 512 tokens; 100% data; LR 0.0002; up to 5 epochs |
| [`phase6_encoder_decoder_mixed_lora.ipynb`](phase6_encoder_decoder_mixed_lora.ipynb) | LoRA adaptation of the encoder-decoder. Learn summary-token generation from eight distributed source spans. | 512 tokens; 100% data; LR 0.0002; up to 5 epochs |
| [`phase6_encoder_decoder_mixed_lorma.ipynb`](phase6_encoder_decoder_mixed_lorma.ipynb) | LoRMA adaptation of the encoder-decoder. Learn summary-token generation from eight distributed source spans. | 512 tokens; 100% data; LR 0.0002; up to 5 epochs |
| [`phase6_encoder_decoder_mixed_lorma_neighboring_context.ipynb`](phase6_encoder_decoder_mixed_lorma_neighboring_context.ipynb) | Encode 256-token regions plus 64-token neighboring context per side; pool into 1,024/512-token main/context memories. | 256 tokens; 100% data; LR 0.0002; up to 5 epochs |
| [`phase6_encoder_decoder_mixed_partial_ft_enc2_dec2_sentence_selection.ipynb`](phase6_encoder_decoder_mixed_partial_ft_enc2_dec2_sentence_selection.ipynb) | Train final two encoder/decoder blocks with a learned lexical/position/legal-cue sentence selector and redundancy penalty. | 512 tokens; 100% data; LR 2e-05; up to 5 epochs |
| [`phase6_encoder_decoder_mixed_partial_ft_enc2_dec2_spans8.ipynb`](phase6_encoder_decoder_mixed_partial_ft_enc2_dec2_spans8.ipynb) | Train final two encoder/decoder blocks with eight distributed fixed source spans. | 512 tokens; 100% data; LR 2e-05; up to 5 epochs |

## Submission files

Scores are official primary-metric values. The ZIPs contain predictions in the schema above.

| Submission | Experiment | Score |
| :-- | :-- | --: |
| [`summ_decoder_126M_lorma.zip`](summ_decoder_126M_lorma.zip) | 126M decoder with LoRMA | 0.7633 |
| [`summ_decoder_MOE_lorma.zip`](summ_decoder_MOE_lorma.zip) | Decoder MoE with LoRMA | 0.7585 |
| [`summ_dual_attention_200M_lorma.zip`](summ_dual_attention_200M_lorma.zip) | Dual-attention model with LoRMA | 0.7025 |
| [`summ_encoder_decoder_mixed_houlsby.zip`](summ_encoder_decoder_mixed_houlsby.zip) | Encoder–decoder Houlsby-adapter run | 0.7673 |
| [`summ_encoder_decoder_mixed_kd.zip`](summ_encoder_decoder_mixed_kd.zip) | Encoder–decoder distillation student | 0.7839 |
| [`summ_encoder_decoder_mixed_lora.zip`](summ_encoder_decoder_mixed_lora.zip) | Encoder–decoder model with LoRA | 0.7731 |
| [`summ_encoder_decoder_mixed_lorma.zip`](summ_encoder_decoder_mixed_lorma.zip) | Encoder–decoder model with LoRMA | 0.7721 |
| [`summ_encoder_decoder_mixed_partial_.zip`](summ_encoder_decoder_mixed_partial_.zip) | Encoder-decoder final-two-encoder/final-two-decoder tuning with learned sentence selection | 0.7758 |

[Setup and checkpoint instructions](../README.md#data-and-setup) | [Project overview](../README.md)
