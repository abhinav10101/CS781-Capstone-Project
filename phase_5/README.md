# Phase 5: Legal statute identification

Train multi-label legal statute identification (LSI) models. Experiments compare LoRMA, AdaLoRA, QLoRA, partial fine-tuning, and distillation across decoder-only, MoE, dual-attention, and encoder–decoder backbones.

## Scope

- LSI training and validation notebooks.
- Data-coverage variants for the encoder–decoder LoRMA experiments.
- Submission ZIPs for selected runs.

## Notebooks

- `phase5_decoder_126M_adalora.ipynb` : Applies AdaLoRA to the 126M decoder for multi-label statute identification.
- `phase5_decoder_126M_distillation_student4.ipynb` : Distills the statute predictions of a decoder teacher into a four-layer student.
- `phase5_decoder_126M_lorma.ipynb` : Trains a decoder-only LoRMA model for multi-label statute identification.
- `phase5_decoder_126M_partial_ft_all_chunks.ipynb` : Partially fine-tunes the decoder using all document chunks for statute identification.
- `phase5_decoder_126M_qlora.ipynb` : Applies QLoRA to the 126M decoder for multi-label statute identification.
- `phase5_decoder_moe_lorma.ipynb` : Trains a decoder MoE with LoRMA for multi-label statute identification.
- `phase5_dual_attention_lorma.ipynb` : Trains a dual-attention MoE with LoRMA for multi-label statute identification.
- `phase5_encoder_decoder_mixed_lorma_data100pct.ipynb` : Trains an encoder–decoder LoRMA statute model using the full training set.
- `phase5_encoder_decoder_mixed_lorma_data20pct.ipynb` : Trains an encoder–decoder LoRMA statute model using a 20% training sample.

## Submission archives

- `lsi_decoder_126M_adalora.zip` : Statute predictions from the 126M decoder with AdaLoRA.
- `lsi_decoder_126M_distillation.zip` : Statute predictions from the decoder knowledge-distillation run.
- `lsi_decoder_126M_lorma.zip` : Statute predictions from the 126M decoder with LoRMA.
- `lsi_decoder_126M_partial_fft.zip` : Statute predictions from the decoder partial fine-tuning run.
- `lsi_decoder_126M_qlora_partial.zip` : Statute predictions from the decoder QLoRA partial-training run.
- `lsi_decoder_MOE.zip` : Statute predictions from the decoder MoE with LoRMA.
- `lsi_dual_attention_lorma.zip` : Statute predictions from the dual-attention MoE with LoRMA.
- `lsi_encoder_decoder_mixed_full.zip` : Statute predictions from the full-data encoder–decoder run.
- `lsi_encoder_decoder_mixed_lorma.zip` : Statute predictions from the encoder–decoder LoRMA run.

## Running this phase

Review each notebook's configuration before running it. Set the required data, checkpoint, tokenizer, output, and runtime paths; some experiments depend on checkpoints produced in earlier phases. Use the environment and credentials required by the notebook, and run from a fresh runtime unless its resume settings say otherwise.

See the [project README](../README.md) for repository-wide setup notes.
