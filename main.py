from src import datasets, tokenize, dataloaders, model
import torch

D_MODEL = 512
NUM_HEADS = 4
VOCAB_SIZE = 10000
MAX_LENGTH = 1024

ds = datasets.load_dataset()
ds["train"] = ds["train"].select(range(10000))
ds["test"] = ds["test"].select(range(10000))
tokenizers = tokenize.create_tokenizers(ds)

ds_tokenized = tokenize.tokenize_dataset(ds, tokenizers)

dataloader = dataloaders.create_dataloaders(ds_tokenized, tokenizers, token_batch_size=1024)


for batch in dataloader["train"]:
    x = model.Embedding(VOCAB_SIZE, D_MODEL)(batch["src_input_ids"])
    x = model.PositionalEncoding(D_MODEL, MAX_LENGTH)(x)
    break

print(x)
