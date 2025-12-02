from src import datasets, token, dataloaders, model, train, utils, generation
import torch
import logging
import datasets as hf_datasets

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
        "checkpoint_steps": 500,
        "warm_up_steps": 50,
        "models_dir": "./models",
        "resume": True,
        "model_dir": None,
    },
    "data": {
        "train_ds": "techiaith/cardiff-university-tm-en-cy",
        "benchmark_ds": "openlanguagedata/flores_plus",
        "max_length": 1024,
        "test_split_ratio": 0.1,
        "build": False,
        "data_dir": "./data",
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
        "build": False,
        "tokenizers_dir": "./tokenizers",
    },
}

# Set up saving directory and config
model_dir = utils.set_up_run(config)

ds = datasets.get_train_dataset(config)

tokenizers = token.get_tokenizers(ds, config)

ds_tokenized = datasets.get_tokenized_dataset(ds, tokenizers, config)

dataloader = dataloaders.create_dataloaders(ds_tokenized, config)

transformer = model.build_transformer(config)


results = train.train(transformer, dataloader, model_dir, config)

test = generation.generate_texts(
    transformer, tokenizers, input_texts=["This is a test sentence."], max_length=50, device="cuda"
)


print(results)
