import datasets


TRAINING_DS = "techiaith/cardiff-university-tm-en-cy"
BENCHMARK_DS = "openlanguagedata/flores_plus"
SEED = 0
TEST_PERC = 0.2


def download_dataset():
    ds = datasets.load_dataset(TRAINING_DS)
    return ds


def preprocess_ds(ds):
    ds = ds["train"].train_test_split(test_size=TEST_PERC, seed=SEED)
    return ds


def load_dataset():
    ds = download_dataset()
    ds = preprocess_ds(ds)
    return ds
