from src import datasets, token, dataloaders, model, train, utils, generation
import logging

logging.basicConfig(level=logging.INFO)

config = {
    "model": {
        "d_model": 512,
        "num_heads": 8,
        "vocab_size": 10000,
        "max_length": 1024,
        "d_ff": 2048,
        "num_enc_layers": 6,
        "num_dec_layers": 6,
        "dropout": 0.1,
    },
    "train": {
        "effective_batch_token_size": 20000,
        "minibatch_token_size": 512,
        "learning_rate": 1e-4,  # 1 / (D_MODEL * WARM_UP_STEPS) ** 0.5
        "adam_betas": (0.9, 0.98),
        "adam_eps": 1e-9,
        "num_steps": 3000,
        "label_smoothing": 0.1,
        "device": "cuda",
        "checkpoint_steps": 5,
        "warm_up_steps": 50,
        "models_dir": "./models",
        "resume": False,
        "model_dir": None,
        "validation_steps": 10,
        "validation_accum_steps": 1,
    },
    "data": {
        "train_ds": "techiaith/cardiff-university-tm-en-cy",
        "benchmark_ds": "openlanguagedata/flores_plus",
        "max_length": 1024,
        "test_split_ratio": 0.1,
        "seed": 0,
    },
    "tokenizers": {
        "vocab_size": 10000,
        "special_tokens": {
            "pad_token": "<pad>",
            "unk_token": "<unk>",
            "bos_token": "<s>",
            "eos_token": "</s>",
        },
        "pad_token_id": 2,
        "training_size": 100000,
    },
    "dataloaders": {
        "minibatch_token_size": 512,
    },
    "locations": {
        "raw_data_dir": "./data/datasets/raw",
        "tokenized_data_dir": "./data/datasets/tokenized",
        "models_dir": "./data/models",
        "tokenizers_dir": "./data/tokenizers",
        "dataloaders_data_dir": "./data/dataloaders",
        "run_dir": "./runs",
    },
}

ds_raw, ds_raw_hash = datasets.get_raw_dataset(config)

tokenizers, tokenizers_hash = token.get_tokenizers(ds_raw, ds_raw_hash, config)

ds_tokenized, ds_tokenized_hash = datasets.get_tokenized_dataset(
    ds_raw, tokenizers, ds_raw_hash, tokenizers_hash, config
)

dataloader, dataloader_hash = dataloaders.get_dataloaders(ds_tokenized, ds_tokenized_hash, config)

transformer = model.build_transformer(config)

results = train.train(transformer, dataloader, tokenizers, dataloader_hash, config)

test = generation.generate_texts(
    transformer, tokenizers, input_texts=["This is a test sentence."], max_length=50, device="cuda"
)


print(results)
