# English → Welsh Neural Machine Translation

A Transformer-based sequence-to-sequence model for English to Welsh translation, implemented from scratch in PyTorch.

## Overview

This project trains an encoder-decoder Transformer (as described in *Attention Is All You Need*) on a bilingual English–Welsh corpus. The model, tokenizer, and training loop are all implemented from scratch without relying on pre-built sequence model libraries.

**Training data:** [`techiaith/cardiff-university-tm-en-cy`](https://huggingface.co/datasets/techiaith/cardiff-university-tm-en-cy) (Cardiff University Translation Memory)  
**Benchmark:** [`openlanguagedata/flores_plus`](https://huggingface.co/datasets/openlanguagedata/flores_plus)  
**Evaluation metric:** BLEU (via sacrebleu)

## Architecture

| Component | Detail |
|---|---|
| Model | Encoder-Decoder Transformer |
| Embedding dim (`d_model`) | 512 |
| Attention heads | 8 |
| Encoder / Decoder layers | 6 / 6 |
| Feed-forward dim (`d_ff`) | 2048 |
| Vocabulary size | 10,000 |
| Max sequence length | 1,024 tokens |
| Tokenizer | WordPiece (shared bilingual vocabulary) |

Training uses mixed-precision (AMP), gradient accumulation, and a warmup inverse square-root learning rate schedule.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Training

```bash
python scripts/train.py base
```

Pass any config name from [configs/](configs/) as the argument. A `test` config is provided for quick smoke-test runs:

```bash
python scripts/train.py test
```

Checkpoints are saved to `./runs/<run-name>/checkpoints/` every `checkpoint_steps` steps. Training automatically resumes from the latest checkpoint if one exists.

## Configuration

Configs live in [configs/](configs/) as YAML files. Key options:

```yaml
model:
  d_model: 512          # embedding dimension
  num_heads: 8          # attention heads
  num_enc_layers: 6     # encoder depth
  num_dec_layers: 6     # decoder depth
  d_ff: 2048            # feed-forward hidden dim
  vocab_size: 10000
  max_length: 1024

train:
  effective_batch_token_size: 20000   # tokens per gradient step (via accumulation)
  minibatch_token_size: 512           # tokens per forward pass
  learning_rate: 1.0e-4
  num_steps: 3000
  warm_up_steps: 50
  checkpoint_steps: 5
  validation_steps: 10
  device: cuda
```

## Project Structure

```
├── configs/          # YAML training configs
├── scripts/
│   └── train.py      # Training entry point
└── src/
    ├── model.py      # Transformer implementation
    ├── tokenizer.py  # WordPiece tokenizer
    ├── datasets.py   # Dataset loading
    ├── dataloader.py # DataLoader creation
    ├── train.py      # Training loop, checkpointing, LR schedule
    └── generation.py # Autoregressive decoding
```
