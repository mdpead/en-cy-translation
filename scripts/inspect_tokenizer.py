"""Train a tokenizer on the full corpus and inspect it."""
import sys
sys.path.insert(0, ".")

import yaml
from src import datasets, tokenizer as tok_module

with open("configs/base.yaml") as f:
    config = yaml.safe_load(f)

# Load full corpus, no sample limit
config["data"]["train"].pop("sample_size", None)
config["tokenizer"]["training_size"] = None

print("Loading dataset...")
ds = datasets.get_dataset("train", config)
print(f"Train size: {len(ds['train'])}, Test size: {len(ds['test'])}")

print(f"\nTraining tokenizer (vocab_size={config['tokenizer']['vocab_size']})...")
tok = tok_module.create_tokenizer(ds, config["tokenizer"])
print("Done.")

# --- Basic stats ---
print(f"\nVocab size: {tok.vocab_size}")

# --- Apostrophe handling ---
print("\n--- Apostrophe handling ---")
examples = [
    "it's a test",
    "i'r dŵr",
    "Mae'r plant yn yr ysgol.",
    "don't can't won't",
    "o'r blaen",
]
for ex in examples:
    print(f"  {ex!r:40s} -> {tok.tokenize(ex)}")

# --- Punctuation attachment ---
print("\n--- Punctuation attachment ---")
punct_examples = [
    "finally.",
    "hello, world!",
    "yes; no.",
]
for ex in punct_examples:
    print(f"  {ex!r:40s} -> {tok.tokenize(ex)}")

# --- Check for key continuation tokens in vocab ---
print("\n--- Continuation tokens (##) sample ---")
vocab = tok.get_vocab()
continuations = sorted(k for k in vocab if k.startswith("##"))
print(f"  Total ## tokens: {len(continuations)}")
punct_cont = [k for k in continuations if not k[2:].isalpha()]
print(f"  Punct/mixed ## tokens ({len(punct_cont)}): {punct_cont[:60]}")

# --- UNK rate on a sample ---
print("\n--- UNK rate (first 1000 sentences) ---")
unk_id = tok.unk_token_id
total_tokens = 0
unk_tokens = 0
for text in list(ds["train"]["text_cy"])[:1000] + list(ds["train"]["text_en"])[:1000]:
    ids = tok(text)["input_ids"]
    total_tokens += len(ids)
    unk_tokens += ids.count(unk_id)
print(f"  {unk_tokens}/{total_tokens} tokens are UNK ({100*unk_tokens/total_tokens:.2f}%)")
