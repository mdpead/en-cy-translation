import datasets as hf_datasets
from src import token, utils
import os
import json


def download_dataset(dataset_name):
    ds = hf_datasets.load_dataset(dataset_name)
    return ds


def preprocess_train_ds(ds, test_split_ratio, seed):
    ds = ds["train"].train_test_split(test_size=test_split_ratio, seed=seed)
    return ds


def get_raw_dataset(config):
    data_config = config["data"]
    ds_hash = utils.fingerprint(data_config)
    dataset_path = config["locations"]["raw_data_dir"] + f"/{ds_hash}"

    if os.path.exists(dataset_path):
        ds = hf_datasets.load_from_disk(dataset_path)
    else:
        ds = download_dataset(data_config["train_ds"])
        ds = preprocess_train_ds(ds, data_config["test_split_ratio"], data_config["seed"])
        ds.save_to_disk(dataset_path)
        with open(dataset_path + "/config.json", "w") as f:
            json.dump(data_config, f, indent=2)

    return ds, ds_hash


def _tokenize_text(text, tokenizers):
    texts_tokenized = {}
    for lang in ["en", "cy"]:
        text_tokenized = tokenizers[lang](
            text[f"text_{lang}"],
        )
        texts_tokenized[f"text_{lang}_tokenized"] = text_tokenized["input_ids"]
    return texts_tokenized


def tokenize_dataset(ds, tokenizers, max_length):
    ds = ds.map(lambda x: _tokenize_text(x, tokenizers), batched=True, batch_size=100000)
    ds = ds.map(
        lambda row: {
            "en_token_length": len(row["text_en_tokenized"]),
            "cy_token_length": len(row["text_cy_tokenized"]),
        },
    )
    ds = ds.filter(
        lambda x: (x["en_token_length"] <= max_length) and (x["cy_token_length"] <= max_length)
    )
    ds = ds.map(
        lambda _, idx: {
            "idx": idx,
        },
        with_indices=True,
    )
    return ds


def get_tokenized_dataset(ds, tokenizers, ds_raw_hash, ds_tokenizers_hash, config):
    data_config = config["data"]
    data_config_resolved = {
        **data_config,
        "datasets_raw_hash": ds_raw_hash,
        "tokenizers_hash": ds_tokenizers_hash,
    }
    data_hash = utils.fingerprint(data_config_resolved)
    dataset_path = f"{config["locations"]["tokenized_data_dir"]}/{data_hash}"

    if os.path.exists(dataset_path):
        ds_tokenized = hf_datasets.load_from_disk(dataset_path)

    else:
        ds_tokenized = tokenize_dataset(ds, tokenizers, data_config["max_length"])
        ds_tokenized.save_to_disk(dataset_path)
        with open(dataset_path + "/config.json", "w") as f:
            json.dump(data_config_resolved, f, indent=2)

    return ds_tokenized, data_hash
