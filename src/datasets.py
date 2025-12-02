import datasets as hf_datasets
from src import token

SEED = 0


def download_dataset(dataset_name):
    ds = hf_datasets.load_dataset(dataset_name)
    return ds


def preprocess_train_ds(ds, test_split_ratio):
    ds = ds["train"].train_test_split(test_size=test_split_ratio, seed=SEED)
    return ds


def save_dataset(ds, ds_name, config):
    path = f"{config['train']['data_dir']}/{ds_name}"
    ds.save_to_disk(path)
    return path


def get_train_dataset(config):
    data_config = config["data"]
    if data_config["build"]:
        ds = download_dataset(data_config["train_ds"])
        ds = preprocess_train_ds(ds, data_config["test_split_ratio"])
        ds.save_to_disk(f"{data_config["data_dir"]}/ds_raw")
    else:
        ds = hf_datasets.load_from_disk(data_config["data_dir"] + "/ds_raw")
    return ds


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


def get_tokenized_dataset(ds, tokenizers, config):
    data_config = config["data"]
    if data_config["build"]:
        ds_tokenized = tokenize_dataset(ds, tokenizers, data_config["max_length"])
        ds_tokenized.save_to_disk(f"{data_config["data_dir"]}/ds_tokenized")
    else:
        ds_tokenized = hf_datasets.load_from_disk(data_config["data_dir"] + "/ds_tokenized")
    return ds_tokenized
