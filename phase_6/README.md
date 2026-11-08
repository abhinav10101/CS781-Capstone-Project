# Phase 6: Legal summarization

Generate summaries of legal documents with decoder-only, MoE, dual-attention, and encoder–decoder models. Experiments include LoRA, LoRMA, Houlsby adapters, distillation, partial fine-tuning, neighboring context, and learned sentence selection.

## Scope

- Summary generation and validation notebooks.
- Adapter, distillation, context, and decoding experiments.
- Summary submission ZIPs for selected runs.

## Notebooks

- `phase6_decoder_126M_lorma.ipynb` : Fine-tunes the 126M decoder with LoRMA to generate legal-document summaries.
- `phase6_decoder_moe_lorma.ipynb` : Fine-tunes the decoder MoE with LoRMA to generate legal-document summaries.
- `phase6_dual_attention_200M_lorma.ipynb` : Fine-tunes the dual-attention MoE with LoRMA for legal summarization.
- `phase6_encoder_decoder_mixed_distillation.ipynb` : Distills an encoder–decoder summarization model into a smaller student.
- `phase6_encoder_decoder_mixed_houlsby.ipynb` : Uses Houlsby adapters in an encoder–decoder model for legal summarization.
- `phase6_encoder_decoder_mixed_lora.ipynb` : Uses LoRA in an encoder–decoder model for legal summarization.
- `phase6_encoder_decoder_mixed_lorma.ipynb` : Uses LoRMA in an encoder–decoder model for legal summarization.
- `phase6_encoder_decoder_mixed_lorma_neighboring_context.ipynb` : Adds neighboring-chunk context to encoder–decoder LoRMA summarization.
- `phase6_encoder_decoder_mixed_partial_ft_enc2_dec2_sentence_selection.ipynb` : Partially fine-tunes the encoder and decoder with learned sentence selection and tuned decoding.
- `phase6_encoder_decoder_mixed_partial_ft_enc2_dec2_spans8.ipynb` : Partially fine-tunes encoder and decoder blocks using eight selected spans for summarization.

## Submission archives

- `summ_decoder_126M_lorma.zip` : Generated legal summaries from the 126M decoder with LoRMA.
- `summ_decoder_MOE_lorma.zip` : Generated legal summaries from the decoder MoE with LoRMA.
- `summ_dual_attention_200M_lorma.zip` : Generated legal summaries from the dual-attention model with LoRMA.
- `summ_encoder_decoder_mixed_houlsby.zip` : Generated legal summaries from the encoder–decoder Houlsby-adapter run.
- `summ_encoder_decoder_mixed_kd.zip` : Generated legal summaries from the encoder–decoder distillation student.
- `summ_encoder_decoder_mixed_lora.zip` : Generated legal summaries from the encoder–decoder model with LoRA.
- `summ_encoder_decoder_mixed_lorma.zip` : Generated legal summaries from the encoder–decoder model with LoRMA.
- `summ_encoder_decoder_mixed_partial_.zip` : Generated legal summaries from the encoder–decoder partial fine-tuning run.

## Running this phase

Review each notebook's configuration before running it. Set the required data, checkpoint, tokenizer, output, and runtime paths; some experiments depend on checkpoints produced in earlier phases. Use the environment and credentials required by the notebook, and run from a fresh runtime unless its resume settings say otherwise.

See the [project README](../README.md) for repository-wide setup notes.
