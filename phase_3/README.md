# Phase 3: Rhetorical-role classification

Train models for rhetorical-role (RR) classification on the CL and IT domains. The experiments include joint training, adapter and distillation methods, and sentence-context and pooling variants.

## Scope

- Decoder-only, MoE, dual-attention, and encoder–decoder backbones.
- LoRA, LoRMA, DoRA, and distillation comparisons.
- RR submission ZIPs for multiple runs.

## Notebooks

- `phase3_decoder_126M_distillation_joint.ipynb` : Distills a decoder model for joint CL and IT rhetorical-role classification.
- `phase3_decoder_126M_dora_joint.ipynb` : Trains a DoRA decoder jointly on the CL and IT rhetorical-role datasets.
- `phase3_decoder_126M_lora_joint.ipynb` : Trains a LoRA decoder jointly on the CL and IT rhetorical-role datasets.
- `phase3_decoder_126M_lorma_ctx0_mlp.ipynb` : Tests a context-free LoRMA decoder with MLP pooling for rhetorical-role classification.
- `phase3_decoder_126M_lorma_ctx2_bilstm.ipynb` : Adds a target-token BiLSTM and two-sentence context to a LoRMA decoder classifier.
- `phase3_decoder_126M_lorma_ctx2_label_attention.ipynb` : Uses label-specific target attention and two-sentence context with decoder LoRMA.
- `phase3_decoder_126M_lorma_ctx2_partial_ft.ipynb` : Combines joint CL/IT training, two-sentence context, LoRMA, and partial fine-tuning.
- `phase3_decoder_126M_lorma_joint.ipynb` : Trains a LoRMA decoder jointly on the CL and IT rhetorical-role datasets.
- `phase3_decoder_moe_125M_distillation_joint.ipynb` : Distills a decoder MoE for joint CL and IT rhetorical-role classification.
- `phase3_decoder_moe_125M_lora_joint.ipynb` : Trains a LoRA decoder MoE jointly on the CL and IT rhetorical-role datasets.
- `phase3_decoder_moe_125M_lorma_joint.ipynb` : Trains a LoRMA decoder MoE jointly on the CL and IT rhetorical-role datasets.
- `phase3_dual_attention_200M_dora_joint.ipynb` : Trains a dual-attention model with DoRA on joint CL/IT rhetorical-role data.
- `phase3_dual_attention_200M_lorma_joint.ipynb` : Trains a dual-attention model with LoRMA on joint CL/IT rhetorical-role data.
- `phase3_encoder_decoder_mixed_dora_joint.ipynb` : Trains an encoder–decoder model with DoRA on joint CL/IT rhetorical-role data.
- `phase3_encoder_decoder_mixed_lora_joint.ipynb` : Trains an encoder–decoder model with LoRA on joint CL/IT rhetorical-role data.
- `phase3_encoder_decoder_mixed_lorma_joint.ipynb` : Trains an encoder–decoder model with LoRMA on joint CL/IT rhetorical-role data.

## Submission archives

- `rr_decoder126m_dora_joint.zip` : Joint CL/IT rhetorical-role predictions from the 126M decoder with DoRA.
- `rr_decoder126m_lora_joint.zip` : Joint CL/IT rhetorical-role predictions from the 126M decoder with LoRA.
- `rr_decoder126m_lorma_joint.zip` : Joint CL/IT rhetorical-role predictions from the 126M decoder with LoRMA.
- `rr_decoder_125M_moe_joint_lora_333c.zip` : Joint CL/IT rhetorical-role predictions from the decoder MoE with LoRA.
- `rr_decoder_126M_joint_distillation_.zip` : Joint CL/IT rhetorical-role predictions from the decoder distillation run.
- `rr_decoder_126M_joint_lorma_ctx0.zip` : Rhetorical-role predictions from the context-free decoder LoRMA run.
- `rr_decoder_126M_joint_lorma_ctx0_ML.zip` : Rhetorical-role predictions from the context-free LoRMA model with label-specific pooling.
- `rr_decoder_126M_joint_lorma_ctx1.zip` : Rhetorical-role predictions from the decoder LoRMA model with one-sentence context.
- `rr_decoder_126M_joint_lorma_ctx2_at.zip` : Rhetorical-role predictions from the decoder LoRMA model with two-sentence attention context.
- `rr_decoder_126M_joint_lorma_ctx2_bi.zip` : Rhetorical-role predictions from the decoder LoRMA model with two-sentence BiLSTM context.
- `rr_decoder_126M_joint_lorma_ctx2_pa.zip` : Rhetorical-role predictions from the decoder LoRMA model with two-sentence context and partial fine-tuning.
- `rr_decoder_lorma_separate.zip` : Rhetorical-role predictions from separate-domain decoder LoRMA training.
- `rr_decoder_moe.zip` : Rhetorical-role predictions from a decoder MoE run.
- `rr_decoder_moe_125M_joint_dora_94f7.zip` : Joint CL/IT rhetorical-role predictions from the decoder MoE with DoRA.
- `rr_decoder_MOE_joint_distillation_9.zip` : Joint CL/IT rhetorical-role predictions from the decoder MoE distillation run.
- `rr_decoder_MOE_joint_lorma_09186db3.zip` : Joint CL/IT rhetorical-role predictions from the decoder MoE with LoRMA.
- `rr_decoder_rectified.zip` : Rhetorical-role predictions from the rectified decoder experiment.
- `rr_dual_attention_200M_joint_distil.zip` : Joint CL/IT rhetorical-role predictions from the dual-attention distillation run.
- `rr_dual_attention_200M_joint_dora_9.zip` : Joint CL/IT rhetorical-role predictions from the dual-attention model with DoRA.
- `rr_dual_attention_200M_joint_lora_9.zip` : Joint CL/IT rhetorical-role predictions from the dual-attention model with LoRA.
- `rr_dual_attention_200M_joint_lorma_.zip` : Joint CL/IT rhetorical-role predictions from the dual-attention model with LoRMA.
- `rr_encoder_decoder_mixed_joint_dist.zip` : Joint CL/IT rhetorical-role predictions from the encoder–decoder distillation run.
- `rr_encoder_decoder_mixed_joint_dora.zip` : Joint CL/IT rhetorical-role predictions from the encoder–decoder model with DoRA.
- `rr_encoder_decoder_mixed_joint_lora.zip` : Joint CL/IT rhetorical-role predictions from the encoder–decoder model with LoRA.
- `rr_encoder_decoder_mixed_lorma_join.zip` : Joint CL/IT rhetorical-role predictions from the encoder–decoder model with LoRMA.

## Running this phase

Review each notebook's configuration before running it. Set the required data, checkpoint, tokenizer, output, and runtime paths; some experiments depend on checkpoints produced in earlier phases. Use the environment and credentials required by the notebook, and run from a fresh runtime unless its resume settings say otherwise.

See the [project README](../README.md) for repository-wide setup notes.
