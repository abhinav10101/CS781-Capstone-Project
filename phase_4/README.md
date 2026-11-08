# Phase 4: CJPE experiments

Explore chunk-based CJPE experiments with long case documents. Notebooks compare parameter-efficient adaptation, pooling and chunk attention, partial fine-tuning, and smaller students trained with knowledge distillation.

## Scope

- Decoder-only, MoE, dual-attention, and encoder–decoder variants.
- Chunk sampling, sentence or chunk attention, head/tail pooling, and distillation controls.
- CJPE submission ZIPs for selected experiments.

## Notebooks

- `phase-4-adaptors.ipynb` : Tests bottleneck adapters on the dense decoder for CJPE.
- `phase-4-ce-control.ipynb` : Provides a cross-entropy-only control for the four-layer distillation student.
- `phase-4-chunk-attn.ipynb` : Combines bottleneck adapters with learned chunk attention for CJPE.
- `phase-4-confidence (1).ipynb` : Tests confidence-gated knowledge distillation for the four-layer student.
- `phase-4-final.ipynb` : Trains a four-layer distillation student while fine-tuning its final three blocks.
- `phase-4-kd-5.ipynb` : Tests a five-layer student for the CJPE knowledge-distillation setup.
- `phase-4-kd-student.ipynb` : Tests a three-layer student for the CJPE knowledge-distillation setup.
- `phase-4-kd.ipynb` : Trains a four-layer student with distillation followed by label-focused training.
- `phase4_decoder_126M_adalora_len2048.ipynb` : Applies AdaLoRA to the 126M decoder using 2048-token document chunks.
- `phase4_decoder_126M_confidence_kd_student4_len2048_data60pct.ipynb` : Trains a four-layer confidence-gated distillation student on 60% of the data.
- `phase4_decoder_126M_kd_student4_len2048_data20pct.ipynb` : Distills a LoRMA CJPE teacher into a four-layer decoder student using a 20% data sample.
- `phase4_decoder_126M_lora_chunk_attention_len768.ipynb` : Combines decoder LoRA with chunk attention over 768-token windows.
- `phase4_decoder_126M_lorma_chunk_attention_len768.ipynb` : Combines decoder LoRMA with learned chunk weighting over 768-token windows.
- `phase4_decoder_126M_lorma_head_tail_len1024.ipynb` : Tests LoRMA with head/tail mean–max pooling over 1024-token chunks.
- `phase4_decoder_126M_lorma_len2048.ipynb` : Trains a decoder LoRMA CJPE model with sampled 2048-token chunks.
- `phase4_decoder_126M_lorma_sentence_attention_len512.ipynb` : Combines decoder LoRMA with sentence-context attention using 512-token chunks.
- `phase4_decoder_126M_partial_ft_bernoulli_last2_p025_len512.ipynb` : Randomly selects partial decoder fine-tuning depth, including the last two blocks, for CJPE.
- `phase4_decoder_126M_partial_ft_last_block_len512.ipynb` : Fine-tunes the decoder’s last block and adds chunk attention for CJPE.
- `phase4_decoder_moe_lorma_len2048.ipynb` : Trains a decoder MoE with LoRMA on sampled 2048-token chunks.
- `phase4_dual_attention_200M_lorma_len2048.ipynb` : Trains a dual-attention model with LoRMA on sampled 2048-token chunks.
- `phase4_encoder_decoder_mixed_lorma_len512.ipynb` : Trains an encoder–decoder model with LoRMA using 512-token chunks.

## Submission archives

- `cjpe_decoder_126M_01_lorma_chunk_at.zip` : CJPE predictions from decoder LoRMA with learned chunk attention.
- `cjpe_decoder_126M_03_lorma_sentence.zip` : CJPE predictions from decoder LoRMA with sentence-context attention.
- `cjpe_decoder_126M_05_partial_last_b.zip` : CJPE predictions from the partial last-block fine-tuning run.
- `cjpe_decoder_126M_06_lora_chunk_att.zip` : CJPE predictions from decoder LoRA with chunk attention.
- `cjpe_decoder_126M_07_bernoulli_trai.zip` : CJPE predictions from the Bernoulli partial fine-tuning run.
- `cjpe_decoder_126M_10_student3_kd.zip` : CJPE predictions from the three-layer knowledge-distillation student.
- `cjpe_decoder_126M_12_student4_confi.zip` : CJPE predictions from the four-layer confidence-gated distillation student.
- `cjpe_decoder_126M_adalora.zip` : CJPE predictions from the decoder AdaLoRA run.
- `cjpe_decoder_126M_distillation_stud.zip` : CJPE predictions from the decoder distillation student.
- `cjpe_decoder_MOE_lorma.zip` : CJPE predictions from the decoder MoE with LoRMA.
- `cjpe_dual_attention_200M_lorma.zip` : CJPE predictions from the dual-attention model with LoRMA.
- `cjpe_encoder_decoder_mixed_lorma.zip` : CJPE predictions from the encoder–decoder model with LoRMA.

## Running this phase

Review each notebook's configuration before running it. Set the required data, checkpoint, tokenizer, output, and runtime paths; some experiments depend on checkpoints produced in earlier phases. Use the environment and credentials required by the notebook, and run from a fresh runtime unless its resume settings say otherwise.

See the [project README](../README.md) for repository-wide setup notes.
