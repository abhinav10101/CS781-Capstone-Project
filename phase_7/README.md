# Phase 7: Prior case retrieval

Rank candidate prior cases for a query case (PCR). The notebooks compare LoRA and LoRMA retrievers, distillation, partial fine-tuning, QKV rotation/scaling, and added adapters with MLP scoring.

## Scope

- Decoder-only, MoE, dual-attention, and encoder–decoder retrievers.
- Retrieval notebooks and PCR submission ZIPs.
- Some runs use bounded chunks or grouped gradient caching for long documents.

## Notebooks

- `phase7_decoder_126M_kd_student4_len2048.ipynb` : Distills a decoder LoRMA retriever into a four-layer student using 2048-token passages.
- `phase7_decoder_126m_lora.ipynb` : Trains a decoder LoRA model for prior-case retrieval.
- `phase7_decoder_126M_lorma_all_adapters_pair_mlp_len512.ipynb` : Extends a trained LoRMA retriever with adapters and an MLP pair scorer.
- `phase7_decoder_126M_lorma_len512.ipynb` : Trains a decoder LoRMA model for prior-case retrieval using 512-token passages.
- `phase7_decoder_126m_lorma_modified_sampling.ipynb` : Tests modified sampling and grouped gradient caching with all-layer decoder LoRMA.
- `phase7_decoder_126M_partial_ft_last4_boundary7_gradcache_len512.ipynb` : Uses partial fine-tuning, a top-7 boundary loss, and gradient caching for retrieval.
- `phase7_decoder_126M_qkv_rotation_scaling_len512.ipynb` : Tests QKV rotation and scaling as a control for prior-case retrieval.
- `phase7_decoder_moe_lorma_len2048.ipynb` : Trains a decoder MoE with LoRMA for retrieval using 2048-token passages.
- `phase7_dual_attention_lorma_len2048.ipynb` : Trains a dual-attention MoE with LoRMA for retrieval using 2048-token passages.
- `phase7_encoder_decoder_mixed_lorma_len512.ipynb` : Trains an encoder–decoder LoRMA model for retrieval using 512-token passages.

## Submission archives

- `pcr_kd_decoder_126M_student4_len204.zip` : Prior-case rankings from the four-layer decoder distillation student.
- `pcr_lora_decoder_126M_len2048.zip` : Prior-case rankings from the decoder LoRA model using 2048-token passages.
- `pcr_lora_decoder_126M_len512.zip` : Prior-case rankings from the decoder LoRA model using 512-token passages.
- `pcr_qkv_rotation_scaling_decoder_12.zip` : Prior-case rankings from the decoder QKV rotation-and-scaling control.
- `pcr_rebuilt_decoder_126M_lorma_all_.zip` : Prior-case rankings from the rebuilt decoder with all-layer LoRMA.
- `pcr_rebuilt_decoder_126M_lorma_len2.zip` : Prior-case rankings from the rebuilt decoder LoRMA 2048-token run.
- `pcr_rebuilt_decoder_126M_lorma_len5.zip` : Prior-case rankings from the rebuilt decoder LoRMA 512-token run.
- `pcr_rebuilt_decoder_MOE_lorma_len20.zip` : Prior-case rankings from the rebuilt decoder MoE with LoRMA.
- `pcr_rebuilt_dual_attention_lorma_le.zip` : Prior-case rankings from the rebuilt dual-attention model with LoRMA.
- `pcr_rebuilt_encoder_decoder_mixed_l.zip` : Prior-case rankings from the rebuilt encoder–decoder LoRMA model.

## Running this phase

Review each notebook's configuration before running it. Set the required data, checkpoint, tokenizer, output, and runtime paths; some experiments depend on checkpoints produced in earlier phases. Use the environment and credentials required by the notebook, and run from a fresh runtime unless its resume settings say otherwise.

See the [project README](../README.md) for repository-wide setup notes.
