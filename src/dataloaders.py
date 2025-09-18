from torch.utils.data import DataLoader
import torch
from torch.utils.data.sampler import BatchSampler
import random

SEED = 0


class TokenSampler(BatchSampler):
    def __init__(self, ds, token_batch_size):
        self.token_batch_size = token_batch_size
        self.batches = self.generate_batches(ds)

    def generate_batches(self, ds):
        ds = ds.map(
            lambda row, idx: {"en_token_length": len(row["text_en_tokenized"]), "idx": idx},
            with_indices=True,
        )

        # Create batches based on token counts
        lengths = list(zip(ds["idx"], ds["en_token_length"]))
        lengths_sorted = sorted(lengths, key=lambda x: x[1])
        batches = []
        batch = []
        batch_token_count = 0
        for idx, token_count in lengths_sorted:
            if batch_token_count + token_count > self.token_batch_size and batch:
                batches.append(batch)
                batch = []
                batch_token_count = 0
            batch.append(idx)
            batch_token_count += token_count
        if batch:
            batches.append(batch)

        return batches

    def __iter__(self):
        # Shuffle batches to introduce randomness
        rng = random.Random(SEED)
        batches = rng.sample(self.batches, len(self.batches))
        for batch in batches:
            yield batch

    def __len__(self):
        return len(self.batches)


def collate_batch(batch, pad_token_id=3):

    output = {}
    for type in ["src", "tgt"]:
        lang = "en" if type == "src" else "cy"
        input_tokens = [item[f"text_{lang}_tokenized"] for item in batch]
        max_len = max(len(ids) for ids in input_tokens)
        input_ids = torch.tensor(
            [ids + [pad_token_id] * (max_len - len(ids)) for ids in input_tokens], dtype=torch.long
        )
        padding_mask = (input_ids != pad_token_id).bool()
        output[f"{type}_input_ids"] = input_ids
        output[f"{type}_padding_mask"] = padding_mask

    output["tgt_output_ids"] = output["tgt_input_ids"][:, 1:].contiguous()
    output["tgt_input_ids"] = output["tgt_input_ids"][:, :-1].contiguous()

    return output


def create_dataloaders(
    ds,
    tokenizers,
    token_batch_size,
):
    pad_token_id = tokenizers["en"].pad_token_id
    dataloaders = {}
    for split in ["train", "test"]:
        dataloaders[split] = DataLoader(
            ds[split],
            batch_sampler=TokenSampler(ds[split], token_batch_size),
            collate_fn=lambda x: collate_batch(x, pad_token_id=pad_token_id),
        )
    return dataloaders
