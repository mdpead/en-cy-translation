import datasets as hf_datasets


def _load_cardiff(ds_config, seed):
    ds = hf_datasets.load_dataset("techiaith/cardiff-university-tm-en-cy")
    train = ds["train"]
    if "sample_size" in ds_config:
        train = train.select(range(ds_config["sample_size"]))
    return train.train_test_split(test_size=ds_config["test_split_ratio"], seed=seed)


def _load_flores(ds_config, _seed):
    ds = hf_datasets.load_dataset("openlanguagedata/flores_plus")
    if "sample_size" in ds_config:
        ds = {split: ds[split].select(range(ds_config["sample_size"])) for split in ds}
    return ds


LOADERS = {
    "techiaith/cardiff-university-tm-en-cy": _load_cardiff,
    "openlanguagedata/flores_plus": _load_flores,
}


def get_dataset(key, config):
    ds_config = config["data"][key]
    name = ds_config["ds"]
    loader = LOADERS.get(name)
    if loader is None:
        raise ValueError(f"Unknown dataset: {name}")
    return loader(ds_config, config["seed"])
