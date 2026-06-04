import datasets as hf_datasets


def preprocess_train_ds(ds, test_split_ratio, seed):
    return ds["train"].train_test_split(test_size=test_split_ratio, seed=seed)


def get_raw_dataset(config):
    data_config = config["data"]
    ds = hf_datasets.load_dataset(dataset_name)
    ds_preprocessed = preprocess_train_ds(ds, data_config["test_split_ratio"], data_config["seed"])
    return ds_preprocessed
