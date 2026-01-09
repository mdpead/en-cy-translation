from torch.utils.data import DataLoader
import torch
from torch.utils.data.sampler import BatchSampler
import random
import pickle, json
import os
from src import utils
from functools import partial

SEED = 0


class TokenSampler(BatchSampler):
    def __init__(self, ds, token_batch_size):
        self.token_batch_size = token_batch_size
        self.batches = self.generate_batches(ds)

    def generate_batches(self, ds):

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
        while True:
            batches = rng.sample(self.batches, len(self.batches))
            for batch in batches:
                yield batch

    def __len__(self):
        return len(self.batches)


def collate_batch(batch, pad_token_id):

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
    output["tgt_padding_mask"] = output["tgt_padding_mask"][:, :-1].contiguous()

    output["src_text"] = [item["text_en"] for item in batch]
    output["tgt_text"] = [item["text_cy"] for item in batch]

    return output


def make_collate_fn(pad_token_id):
    def collate_fn(x):
        return collate_batch(x, pad_token_id)

    return collate_fn


def create_dataloaders(
    ds_tokenized,
    config,
):

    dataloaders = {}
    for split in ["train", "test"]:
        dataloaders[split] = DataLoader(
            ds_tokenized[split],
            batch_sampler=TokenSampler(
                ds_tokenized[split], config["train"]["minibatch_token_size"]
            ),
            collate_fn=partial(collate_batch, pad_token_id=config["tokenizers"]["pad_token_id"]),
            pin_memory=True,
        )
    return dataloaders


def load_dataloaders(dataloaders_path):
    with open(dataloaders_path + "/dataloaders.pkl", "rb") as f:
        dataloaders = pickle.load(f)
        return dataloaders


def save_dataloaders(dataloaders, dataloaders_path):
    os.makedirs(dataloaders_path, exist_ok=True)
    with open(dataloaders_path + "/dataloaders.pkl", "wb") as f:
        pickle.dump(dataloaders, f)
    return None


def get_dataloaders(ds_tokenized, ds_tokenized_hash, config):
    dataloaders_config = config["dataloaders"]
    dataloaders_config_resolved = {
        **dataloaders_config,
        "datasets_tokenized_hash": ds_tokenized_hash,
    }
    dataloaders_hash = utils.fingerprint(dataloaders_config_resolved)
    dataloaders_path = f"{config["locations"]["dataloaders_data_dir"]}/{dataloaders_hash}"

    if os.path.exists(dataloaders_path):
        dataloaders = load_dataloaders(dataloaders_path)

    else:
        dataloaders = create_dataloaders(ds_tokenized, config)
        save_dataloaders(dataloaders, dataloaders_path)
        with open(dataloaders_path + "/config.json", "w") as f:
            json.dump(dataloaders_config_resolved, f, indent=2)

    return dataloaders, dataloaders_hash
