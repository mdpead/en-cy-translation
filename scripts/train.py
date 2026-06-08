from src import datasets, tokenizer, model, train, utils, generation, dataloader
import logging
import argparse
import yaml

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument("config", help="Config name (e.g. base, test)")
args = parser.parse_args()

with open(f"configs/{args.config}.yaml") as f:
    config = yaml.safe_load(f)

utils.init_run(config)

ds_raw = datasets.get_dataset("train", config)

token = tokenizer.get_tokenizer(ds_raw, config)

ds_tokenized = tokenizer.tokenize_dataset(ds_raw, token, config)

dataloaders = dataloader.create_dataloaders(ds_tokenized, config)

transformer = model.build_transformer(config)

results = train.train(transformer, dataloaders, token, config)

logging.info(results)
