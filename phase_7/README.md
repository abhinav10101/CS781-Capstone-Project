# Phase 7: Prior-case retrieval

I ranked supplied prior cases for each query and returned seven candidate IDs. The metric is micro-F1@7.

## Data and representation

PCR has 827/118/237 train/dev/test queries and 4,320/1,023/1,727 candidates. I skip ten training queries without relevant candidates. Some checkpoint-selection runs use 100 development queries; grouped variants use all 118.

Eight distributed source passages form a 512- or 2,048-token case representation. A shared Phase 1 backbone encodes queries and candidates. Dense models use normalized masked-mean features; the encoder-decoder combines pooled encoder memory and a decoder representation. Inference encodes the candidate pool in batches and ranks similarities.

The basic loss uses temperature-0.07 dot products with one sampled positive and random negatives. Dense 512-token references request ten negatives; the MoE, dual-attention, encoder-decoder, and KD sources configure two. Candidate IDs remain strings, preserving leading zeros.

## Adaptation and ranking variants

LoRA/LoRMA adapt attention projections. Modified-sampling LoRMA groups queries and deduplicates document encodings, caches feature gradients, then replays encoder forwards with matching RNG state. Partial tuning unfreezes the final four blocks and adds a top-seven-boundary loss with weight 4.0. The adapter/pair-MLP variant starts from an adapted LoRMA checkpoint and adds bottleneck modules and a learned pair scorer.

My QKV method freezes backbone weights and learns identity-initialized vector controllers for bounded scale and pairwise rotation. Q/K transforms follow RoPE; V transforms follow projection. Parameters are shared across heads/tokens within each projection. Scale is bounded by ln(2) and angle by pi/4. It trains 6,192 parameters versus LoRA's 917,504, using learning rates 1e-3 and 2e-4 respectively.

The four-layer KD student copies teacher layers `[0,2,4,7]`, combining supervised retrieval with teacher score distributions and cosine representation alignment. Parameters fall from about 128.86M to 81.66M.

## Results

Dense LoRMA at 2,048 tokens led at **0.2481**, followed by dense LoRA 0.2404, MoE 0.2392, KD 0.2333, partial/boundary/GradCache 0.2315, grouped LoRMA 0.2191, 512-token LoRMA 0.2032, 512-token LoRA 0.1949, QKV 0.1778, encoder-decoder 0.0148, and dual attention 0.0130.

Longer-context dense configurations performed better. QKV traded some quality for far fewer trainable parameters; KD also lost quality. Changes in batch size, negative count, and learning rate prevent treating every pair as a context-only or adaptation-only ablation. Hard-negative mining remains a next experiment.

## Submission format

```json
{"query_id": ["candidate_1", "candidate_2", "candidate_3", "candidate_4", "candidate_5", "candidate_6", "candidate_7"]}
```

Each ZIP contains root `prediction.json` with 237 queries. Return seven unique valid candidates in rank order. Archive suffixes `len2` and `len5` denote the 2,048- and 512-token runs.

## Notebook experiments

Run settings below show the requested epoch/token limit. Checkpoints can be selected earlier. Shared preprocessing and loss are described above; the table records the changes for each experiment.

| Notebook | What changes | Run settings |
| :-- | :-- | :-- |
| [`phase7_decoder_126M_kd_student4_len2048.ipynb`](phase7_decoder_126M_kd_student4_len2048.ipynb) | Copy teacher layers [0,2,4,7]; transfer ranking distributions and align representations. Request two negatives per query. | 2048 tokens; 100% data; LR 0.0002; up to 5 epochs |
| [`phase7_decoder_126M_lorma_all_adapters_pair_mlp_len512.ipynb`](phase7_decoder_126M_lorma_all_adapters_pair_mlp_len512.ipynb) | Start from the adapted LoRMA checkpoint; add bottleneck-64 adapters and a width-128 pair scorer, with one head-only epoch. | 512 tokens; 100% data; LR 0.0002; up to 5 epochs |
| [`phase7_decoder_126M_lorma_len512.ipynb`](phase7_decoder_126M_lorma_len512.ipynb) | Dense LoRMA with eight source positions, batch one, and ten requested negatives. | 512 tokens; 100% data; LR 0.0002; up to 5 epochs |
| [`phase7_decoder_126M_partial_ft_last4_boundary7_gradcache_len512.ipynb`](phase7_decoder_126M_partial_ft_last4_boundary7_gradcache_len512.ipynb) | Train final four blocks with grouped gradient replay and a cutoff-seven boundary loss weighted 4.0. | 512 tokens; 100% data; LR 2e-05; up to 5 epochs |
| [`phase7_decoder_126M_qkv_rotation_scaling_len512.ipynb`](phase7_decoder_126M_qkv_rotation_scaling_len512.ipynb) | Freeze base weights and learn bounded vector scale/rotation controllers, initialized to identity. | 512 tokens; 100% data; LR 0.001; up to 5 epochs |
| [`phase7_decoder_126m_lora.ipynb`](phase7_decoder_126m_lora.ipynb) | Dense LoRA over all eight blocks; the source uses 512 tokens and ten requested negatives. | 512 tokens; 100% data; LR 0.0002; up to 5 epochs |
| [`phase7_decoder_126m_lorma_modified_sampling.ipynb`](phase7_decoder_126m_lorma_modified_sampling.ipynb) | All-layer LoRMA with modified negative sampling, deduplicated group encodings, and RNG-matched gradient replay. | 512 tokens; 100% data; LR 0.0002; up to 5 epochs |
| [`phase7_decoder_moe_lorma_len2048.ipynb`](phase7_decoder_moe_lorma_len2048.ipynb) | LoRMA adaptation of the MoE decoder. Shared query/candidate encoding; request 2 negatives per query. | 2048 tokens; 100% data; LR 0.0002; up to 5 epochs |
| [`phase7_dual_attention_lorma_len2048.ipynb`](phase7_dual_attention_lorma_len2048.ipynb) | LoRMA adaptation of the dual local/global attention MoE. Shared query/candidate encoding; request 2 negatives per query. | 2048 tokens; 100% data; LR 0.0002; up to 5 epochs |
| [`phase7_encoder_decoder_mixed_lorma_len512.ipynb`](phase7_encoder_decoder_mixed_lorma_len512.ipynb) | LoRMA adaptation of the encoder-decoder. Shared query/candidate encoding; request 2 negatives per query. | 512 tokens; 100% data; LR 0.0002; up to 5 epochs |

## Submission files

Scores are official primary-metric values. The ZIPs contain predictions in the schema above.

| Submission | Experiment | Score |
| :-- | :-- | --: |
| [`pcr_kd_decoder_126M_student4_len204.zip`](pcr_kd_decoder_126M_student4_len204.zip) | Four-layer decoder distillation student | 0.2333 |
| [`pcr_lora_decoder_126M_len2048.zip`](pcr_lora_decoder_126M_len2048.zip) | Decoder LoRA model using 2048-token passages | 0.2404 |
| [`pcr_lora_decoder_126M_len512.zip`](pcr_lora_decoder_126M_len512.zip) | Decoder LoRA model using 512-token passages | 0.1949 |
| [`pcr_qkv_rotation_scaling_decoder_12.zip`](pcr_qkv_rotation_scaling_decoder_12.zip) | Decoder QKV rotation-and-scaling control | 0.1778 |
| [`pcr_rebuilt_decoder_126M_lorma_all_.zip`](pcr_rebuilt_decoder_126M_lorma_all_.zip) | Rebuilt decoder with all-layer LoRMA | 0.2191 |
| [`pcr_rebuilt_decoder_126M_lorma_len2.zip`](pcr_rebuilt_decoder_126M_lorma_len2.zip) | Rebuilt decoder LoRMA 2048-token run | 0.2481 |
| [`pcr_rebuilt_decoder_126M_lorma_len5.zip`](pcr_rebuilt_decoder_126M_lorma_len5.zip) | Rebuilt decoder LoRMA 512-token run | 0.2032 |
| [`pcr_rebuilt_decoder_MOE_lorma_len20.zip`](pcr_rebuilt_decoder_MOE_lorma_len20.zip) | Rebuilt decoder MoE with LoRMA | 0.2392 |
| [`pcr_rebuilt_dual_attention_lorma_le.zip`](pcr_rebuilt_dual_attention_lorma_le.zip) | Rebuilt dual-attention model with LoRMA | 0.0130 |
| [`pcr_rebuilt_encoder_decoder_mixed_l.zip`](pcr_rebuilt_encoder_decoder_mixed_l.zip) | Rebuilt encoder–decoder LoRMA model | 0.0148 |
| [`pcr_rebuilt_decoder_126M_lorma_all_gradcache_v1_len512.zip`](pcr_rebuilt_decoder_126M_lorma_all_gradcache_v1_len512.zip) | Partial fine-tuning + top-7 boundary loss + GradCache | 0.2191 |
| [`pcr_rebuilt_decoder_lorma_adapters_pair_mlp_v1_len512.zip`](pcr_rebuilt_decoder_lorma_adapters_pair_mlp_v1_len512.zip) | Decoder LoRMA with adaptors | 0.2215 |
## Longer-context dense runs

| Submission | Base notebook | Changed settings |
| :-- | :-- | :-- |
| `pcr_lora_decoder_126M_len2048.zip` | [`phase7_decoder_126m_lora.ipynb`](phase7_decoder_126m_lora.ipynb) | `MAX_SEQ_LEN=2048`, `NEGATIVES_PER_QUERY=10` |
| `pcr_rebuilt_decoder_126M_lorma_len2.zip` | [`phase7_decoder_126M_lorma_len512.ipynb`](phase7_decoder_126M_lorma_len512.ipynb) | `MAX_SEQ_LEN=2048`, `NEGATIVES_PER_QUERY=10` |

[Setup and checkpoint instructions](../README.md#data-and-setup) | [Project overview](../README.md)
