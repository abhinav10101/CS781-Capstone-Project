# CS781 Capstone Project: In-House Legal Intelligence

I trained compact legal language models from scratch and adapted them to six downstream tasks: entity recognition, rhetorical roles, judgment prediction, statute identification, summarization, and prior-case retrieval. This repository contains the training and inference notebooks, along with the submitted predictions.

Each downstream task uses its own adapted Phase 1 checkpoint. The tasks are trained independently; their predictions are not passed through a sequential pipeline.

## Phases

| Phase | Task | Primary metric | Documentation |
| :-- | :-- | :-- | :-- |
| 1 | Legal text continuation | BERTScore F1 | [Foundation models](Phase_1/README.md) |
| 2 | Legal named-entity recognition | Strict entity Macro-F1 | [NER](phase_2/README.md) |
| 3 | Rhetorical-role prediction | Macro-F1 | [RR](phase_3/README.md) |
| 4 | Court judgment prediction | Macro-F1 | [CJPE](phase_4/README.md) |
| 5 | Legal statute identification | Macro-F1 | [LSI](phase_5/README.md) |
| 6 | Abstractive legal summarization | BERTScore F1 | [Summarization](phase_6/README.md) |
| 7 | Prior-case retrieval | micro-F1@7 | [PCR](phase_7/README.md) |

The phase folders contain 86 notebooks and 85 prediction ZIPs. Each phase README explains the shared implementation, then lists what changes in each notebook and submission. The two root notebooks and root ZIP/JSON files are excluded from this documentation.

## Architecture and transfer

```text
Official legal corpus -> byte-level BPE -> Phase 1 model
  -> word-level BIO tagging                         (P2)
  -> sentence representation + 13-role classifier  (P3)
  -> document representation + binary classifier  (P4)
  -> document representation + 100 sigmoid labels  (P5)
  -> autoregressive summary generation             (P6)
  -> query/candidate embeddings + ranking          (P7)
```

I compared dense GQA decoders, MLA, MoE, a routed dual decoder, local/global dual attention, encoder-decoder models, and FNet/GFNet encoder variants. Measured foundation sizes ranged from about 50.82M to 177.84M parameters. Names such as `126M` and `dual_attention_200M` are experiment tags: parameter count and training-token budget are separate quantities.

Downstream experiments use LoRA, LoRMA, DoRA, bottleneck/Houlsby adapters, AdaLoRA, QLoRA, partial tuning, and distillation. My additional experiments include LoRMA combined with DoRA, RR context packing, Bernoulli partial tuning, confidence-weighted distillation, summary sentence selection, and QKV rotation/scaling. The phase documents explain their implementations.

## Data and setup

I used the official [`Exploration-Lab/CS-781-Capstone`](https://huggingface.co/datasets/Exploration-Lab/CS-781-Capstone) data. Phase 1 uses Supreme Court, High Court, Tribunal, and Acts text; later phases use their task datasets. Foundation models were initialized from scratch. Downstream models load my matching Phase 1 checkpoint and tokenizer from the configured storage paths.

1. Open the required notebook in a fresh GPU runtime and run its dependency cells. The code targets Kaggle-style runtimes.
2. Supply Hugging Face credentials through runtime secrets or environment variables. Set the dataset, checkpoint, tokenizer, and output paths.
3. Check the model family, context limit, data fraction, adaptation scope, and final training call. Some explicit calls override the earlier epoch/token-budget constants.
4. Use a new run tag for a new experiment. Resume only from a compatible checkpoint, restoring optimizer/scheduler state where supported.
5. Run preprocessing, model construction, training, best-checkpoint loading, and prediction export. KD runs also require their adapted teacher checkpoint.

Common dependencies are PyTorch, `tokenizers`, `huggingface_hub`, NumPy, and scikit-learn. QLoRA requires compatible CUDA/`bitsandbytes`; learned sentence selection also uses SciPy. There is no single repository-wide environment lockfile or training CLI.

Keep checkpoints with their tokenizer, model configuration, label vocabulary, and task settings. LSI needs its thresholds; summarization needs its decoding/selector settings; PCR pair-scoring variants need their learned scorer. Prediction ZIPs contain a root `prediction.json`, not model weights.

## Results

| Phase | Best configuration | Official score |
| :-- | :-- | --: |
| 1 | Large GQA, 175M training tokens | 0.8267 |
| 2 | LoRMA + DoRA + partial tuning | 0.2368 |
| 3 | Context-conditioned LoRMA | 0.4065 |
| 4 | Confidence-weighted four-layer student | 0.6639 |
| 5 | Distilled statute classifier | 0.3350 |
| 6 | Distilled encoder-decoder summarizer | 0.7839 |
| 7 | LoRMA retriever, 2,048-token input | 0.2481 |

The 50M-class decoder scored 0.8261, close to the larger winner, and dual sparse attention scored 0.8238. Distillation helped the strongest P4-P6 systems but hurt the compared NER and retrieval systems. Full details and negative results are in the phase READMEs.

I used notebook validation for training decisions and Codabench for submission comparisons. I did not run a separate independent local benchmark or repeated-seed study. Several comparisons change data coverage, learning rate, or architecture together, so their score differences do not isolate one cause. BERTScore/ROUGE similarity also does not establish legal factuality.

## Methods and sources

- [Transformer](https://arxiv.org/abs/1706.03762) and CS781 slides: baseline architectures; GQA followed the slides.
- [DeepSeek-V2](https://arxiv.org/abs/2405.04434): compact MLA/MoE adaptations.
- [FNet](https://aclanthology.org/2022.naacl-main.319/) and [GFNet](https://arxiv.org/abs/2107.00645): spectral encoder ideas.
- [LoRA](https://arxiv.org/abs/2106.09685), [LoRMA](https://aclanthology.org/2025.findings-acl.527/), [DoRA](https://arxiv.org/abs/2402.09353), [AdaLoRA](https://arxiv.org/abs/2303.10512), and [QLoRA](https://arxiv.org/abs/2305.14314): adaptation methods.
- [Distillation](https://arxiv.org/abs/1503.02531), [Houlsby adapters](https://proceedings.mlr.press/v97/houlsby19a.html), and [GradCache](https://aclanthology.org/2021.repl4nlp-1.31/).
- [BiLSTM](https://www.cs.toronto.edu/~graves/nn_2005.pdf), [label attention](https://aclanthology.org/N18-1100/), [hierarchical attention](https://aclanthology.org/N16-1174/), and [head/tail selection](https://arxiv.org/abs/1905.05583): pooling and document-representation components.
