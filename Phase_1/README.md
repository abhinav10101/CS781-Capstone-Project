# Phase 1: Legal language-model pretraining

I trained foundation models from scratch to produce a 128-word continuation of a legal prompt. I compared Supreme Court-only training with mixtures of Supreme Court, High Court, Tribunal, and Acts text. The official metric is BERTScore F1.

## Architecture

| Model | Layers | Width | Main components | Parameters |
| :-- | :-- | --: | :-- | --: |
| Small GQA decoder | 12 | 512 | 16 query / 4 KV heads, FFN 1,408 | 50.82M-51.18M |
| Large GQA decoder | 8 | 1,024 | 8 query / 4 KV heads, FFN 2,816 | 128.66M |
| MLA | 12 | 640 | KV latent 160, rotary component 32, FFN 1,728 | 73.84M |
| MoE | 12 | 512 | GQA 8/4, eight experts, top-two routing | 177.84M |
| Dual attention | 12 | 512 | Local window 64, global stride 128, GQA and MoE | 177.84M |
| Dual decoder | 14 routed | 512 | Two decoder branches per routed layer, FFN 1,408 | 96.99M |
| Encoder-decoder | 6 + 6 | 640 | Bidirectional encoder, causal decoder, cross-attention | 79.50M |
| FNet/FFT encoder-decoder | 6 + 6 | 640 | Real 2D FFT encoder mixing, attentive decoder | 73.54M |
| GFNet encoder-decoder | 8 + 8 | 768 | Learned complex frequency filters, attentive decoder | 126.40M |

Dense blocks use RMSNorm, rotary positions, SwiGLU, residual connections, and tied token/output embeddings. MLA and MoE adapt DeepSeek-V2 ideas. My dual decoder routes a batch through one of two branches using pooled per-example preferences and majority vote. My dual-attention model combines local and periodically global causal access while retaining GQA projections and MoE computation.

FNet replaces encoder attention with the real component of an FFT over sequence and hidden dimensions. GFNet uses a one-dimensional real FFT, learned complex filters, and inverse FFT. Both retain decoder self-attention and cross-attention.

## Tokenization and training

I trained byte-level BPE tokenizers with approximately 33,178-33,887 tokens. Decoder models minimize shifted next-token cross-entropy in 2,048-token sequences. Encoder-decoder models learn source-prefix-to-continuation generation with teacher forcing and 512/256 source/target limits.

AdamW uses learning rate 3e-4. Decoder runs generally use batch four and weight decay 0.1; encoder-decoder runs use batch eight. Gradient clipping is 1.0 for the large decoder and 5.0 in several other variants. Checkpoints are commonly saved every million training tokens. The tables give the final training-history token counts; the strongest large-decoder submission used the earlier 175M-token checkpoint.

Generation uses temperature 0.8, top-p 0.9, and at most 256 new tokens for the 128-word target. Downstream phases load the matching model state and tokenizer together.

## Results

| Model / checkpoint | Final training tokens | Official score |
| :-- | :-- | --: |
| Large GQA | 200,007,680 | **0.8267** at 175M; 0.8217 at the final checkpoint |
| Small Supreme Court decoder | 50,003,968 | 0.8261 |
| Mixed small decoder | 100,007,936 | 0.8253 |
| MLA | 100,007,936 | 0.8244 |
| MoE | 125,001,728 | 0.8247 |
| Dual attention | 100,007,936 | 0.8238 |
| Dual decoder | 100,000,044 | 0.8183 |
| Encoder-decoder | 100,001,280 | 0.8043 |
| FNet/FFT | 124,241,920 | 0.7487 |
| GFNet | 100,001,280 | 0.7481 |

The small Supreme Court decoder nearly matched the larger model despite a smaller training budget. Sparse attention also remained competitive. The later 200M-token dense checkpoint scored below the 175M checkpoint, so longer training did not automatically improve continuation quality. Data, size, tokenizer, and budgets differ across architectures; these results are not an attention-only ablation.

## Notebook experiments

Run settings below show the requested epoch/token limit. Checkpoints can be selected earlier. Shared preprocessing and loss are described above; the table records the changes for each experiment.

| Notebook | What changes | Run settings |
| :-- | :-- | :-- |
| [`Phase1_decoder_126M.ipynb`](Phase1_decoder_126M.ipynb) | Eight-layer width-1,024 dense GQA. Resume training toward 200M tokens; I selected the 175M checkpoint for the leading submission. | 2048 tokens; LR 3e-4; batch 4; final tokens 200,007,680 |
| [`Phase1_decoder_moe_125M_tokens.ipynb`](Phase1_decoder_moe_125M_tokens.ipynb) | Replace dense feed-forward computation with eight experts and top-two routing; keep GQA. Tokenizer fitting uses Supreme Court text. | 2048 tokens; LR 3e-4; batch 4; final tokens 125,001,728 |
| [`Phase1_decoder_supreme_court_50M_tokens.ipynb`](Phase1_decoder_supreme_court_50M_tokens.ipynb) | Twelve-layer width-512 dense decoder trained on Supreme Court text. | 2048 tokens; LR 3e-4; batch 4; final tokens 50,003,968 |
| [`Phase1_dual_attention.ipynb`](Phase1_dual_attention.ipynb) | Combine causal local/global access with GQA and MoE. Final logged training reached 100,007,936 tokens. | 2048 tokens; LR 3e-4; batch 4; final tokens 100,007,936 |
| [`Phase1_dual_decoder_mixed_100M_tokens.ipynb`](Phase1_dual_decoder_mixed_100M_tokens.ipynb) | Route between two decoder branches per layer using batch-majority preferences. The save tag says dual_decoder_200M; the call requests 100M tokens. | 2048 tokens; LR 3e-4; batch 4; final tokens 100,000,044 |
| [`Phase1_encoder_decoder_100M_mixed_tokens.ipynb`](Phase1_encoder_decoder_100M_mixed_tokens.ipynb) | Six-layer encoder and six-layer decoder with source-prefix-to-continuation training. | source/target 512/256; LR 3e-4; batch 8; final tokens 100,001,280 |
| [`Phase1_encoder_decoder_fft_mixed_125M_tokens.ipynb`](Phase1_encoder_decoder_fft_mixed_125M_tokens.ipynb) | Replace encoder attention with real 2D FFT mixing; retain decoder attention. Final logged training reached 124,241,920 tokens. | source/target 512/256; LR 3e-4; batch 8; final tokens 124,241,920 |
| [`Phase1_encoder_decoder_gfnet_mixed_100M_tokens.ipynb`](Phase1_encoder_decoder_gfnet_mixed_100M_tokens.ipynb) | Use learned complex frequency filters in an eight-layer width-768 encoder with an eight-layer decoder. | source/target 512/256; LR 3e-4; batch 8; final tokens 100,001,280 |
| [`Phase1_mla_mixed_100M_tokens.ipynb`](Phase1_mla_mixed_100M_tokens.ipynb) | Use compressed KV attention in twelve width-640 layers; latent width 160 and rotary component 32. | 2048 tokens; LR 3e-4; batch 4; final tokens 100,007,936 |

[Setup and checkpoint instructions](../README.md#data-and-setup) | [Project overview](../README.md)
