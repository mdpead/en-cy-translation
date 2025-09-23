from src import datasets, token, dataloaders, model, train
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
        "pad_token_id": 2,
        "device": "cuda",
        "checkpoint_steps": 500,
        "warm_up_steps": 50,
        "data_dir": "./data",
        "models_dir": "./models",
    },
}


# ds = datasets.load_dataset()

# tokenizers = token.create_tokenizers(ds)

# ds_tokenized = token.tokenize_dataset(ds, tokenizers, MAX_LENGTH)

# ds_tokenized.save_to_disk(f"{DATA_DIR}/ds_tokenized")
# token.save_tokenizers(tokenizers, f"{MODEL_DIR}/tokenizers")

ds_tokenized = hf_datasets.load_from_disk(f"{DATA_DIR}/ds_tokenized")
tokenizers = token.load_tokenizers(f"{MODELS_DIR}/tokenizers")

dataloader = dataloaders.create_dataloaders(ds_tokenized, tokenizers, MINIBATCH_TOKEN_SIZE)

transformer = model.build_transformer(config["model"])

results = train.train(transformer, dataloader, config)
