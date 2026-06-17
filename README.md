---
language:
  - en
  - cy
tags:
  - translation
  - en-cy
license: cc-by-4.0
---

# English → Welsh Neural Machine Translation

A Transformer-based sequence-to-sequence model for English to Welsh translation, implemented from scratch in PyTorch.

## Usage

```python
from transformers import pipeline
pipe = pipeline("translation", model="mdpead/en-cy-translation")
pipe("Hello, how are you?")
```

Or with the model and tokenizer directly:

```python
from src.hf_wrapper import EnCyForTranslation
from transformers import PreTrainedTokenizerFast

model = EnCyForTranslation.from_pretrained("mdpead/en-cy-translation")
tokenizer = PreTrainedTokenizerFast.from_pretrained("mdpead/en-cy-translation")

inputs = tokenizer("Hello, how are you?", return_tensors="pt")
output_ids = model.generate(**inputs, max_length=256)
print(tokenizer.decode(output_ids[0], skip_special_tokens=True))
```

## Architecture

| Component | Detail |
|---|---|
| Model | Encoder-Decoder Transformer |
| Embedding dim (`d_model`) | 512 |
| Attention heads | 8 |
| Encoder / Decoder layers | 6 / 6 |
| Feed-forward dim (`d_ff`) | 2048 |
| Vocabulary size | 16,000 |
| Max sequence length | 256 tokens |
| Tokenizer | WordPiece (shared bilingual vocabulary) |

Training uses mixed-precision (AMP), gradient accumulation, and a warmup inverse square-root learning rate schedule.

## Training

- **Dataset**: [techiaith/cardiff-university-tm-en-cy](https://huggingface.co/datasets/techiaith/cardiff-university-tm-en-cy) (~1.3M sentence pairs)
- **Steps**: 50,000
- **Effective batch size**: 25,000 tokens
- **Optimiser**: AdamW (β₁=0.9, β₂=0.98, ε=1e-9)
- **Learning rate**: 1e-3 with 2,000 warmup steps, inverse square root decay

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

```bash
python scripts/train.py --config base
```

Checkpoints are saved to `runs/<run-name>/checkpoints/` every `checkpoint_steps` steps. Training resumes automatically from the latest checkpoint.

## Project Structure

```
├── configs/          # YAML training configs
├── scripts/
│   ├── train.py      # Training entry point
│   └── push_to_hub.py
└── src/
    ├── model.py      # Transformer implementation
    ├── tokenizer.py  # WordPiece tokenizer
    ├── datasets.py   # Dataset loading
    ├── dataloader.py # Token-bucketed DataLoader
    ├── train.py      # Training loop and checkpointing
    ├── generation.py # Autoregressive decoding
    └── hf_wrapper.py # HuggingFace PreTrainedModel wrapper
```

## License

CC BY 4.0 — derived from the [Cardiff University Translation Memory](https://huggingface.co/datasets/techiaith/cardiff-university-tm-en-cy) dataset, also licensed CC BY 4.0. Attribution to Cardiff University Language Technologies Unit.
