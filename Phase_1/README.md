# Phase 1: Language-model pretraining

Build and compare base language-model architectures trained on legal text. The notebooks cover decoder-only, encoder–decoder, MoE, dual-attention, MLA, GFNet, and FFT variants, with different token budgets and corpus mixes.

## Scope

- Base-model architecture experiments and pretraining configurations.
- Checkpoint and tokenizer creation for downstream experiments.
- A starter training notebook and project-level notebook are also in the repository root.

## Notebooks

- `Phase1_decoder_126M.ipynb` : Pretrains the 126M-parameter decoder-only language model used as a Phase 1 baseline.
- `Phase1_decoder_moe_125M_tokens.ipynb` : Trains the decoder MoE variant on a 125M-token budget.
- `Phase1_decoder_supreme_court_50M_tokens.ipynb` : Trains a decoder-only model on Supreme Court text with a 50M-token budget.
- `Phase1_dual_attention.ipynb` : Trains the dual-attention language-model architecture.
- `Phase1_dual_decoder_mixed_100M_tokens.ipynb` : Trains a dual-decoder model on mixed legal text with a 100M-token budget.
- `Phase1_encoder_decoder_100M_mixed_tokens.ipynb` : Trains an encoder–decoder model on mixed legal text with a 100M-token budget.
- `Phase1_encoder_decoder_fft_mixed_125M_tokens.ipynb` : Trains an FFT-based encoder–decoder variant on mixed legal text with a 125M-token budget.
- `Phase1_encoder_decoder_gfnet_mixed_100M_tokens.ipynb` : Trains a GFNet encoder–decoder variant on mixed legal text with a 100M-token budget.
- `Phase1_mla_mixed_100M_tokens.ipynb` : Trains an MLA-based language model on mixed legal text with a 100M-token budget.

## Running this phase

Review each notebook's configuration before running it. Set the required data, checkpoint, tokenizer, output, and runtime paths; some experiments depend on checkpoints produced in earlier phases. Use the environment and credentials required by the notebook, and run from a fresh runtime unless its resume settings say otherwise.

See the [project README](../README.md) for repository-wide setup notes.
