import os
import datetime
import json
import torch
import re
import glob
import logging


def create_model_dir(models_dir):
    model_dir = f"{models_dir}/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    return model_dir


def save_config(models_dir, model_dir, config):
    filepath = f"{models_dir}/{model_dir}/config.json"
    json.dump(config, open(filepath, "w"))
    return filepath


def load_config(models_dir, model_dir):
    filepath = f"{models_dir}/{model_dir}/config.json"
    with open(filepath) as f:
        config = json.load(f)
    return config


def save_results(models_dir, model_dir, config):
    filepath = f"{models_dir}/{model_dir}/results.json"
    json.dump(config, open(filepath, "w"))
    return filepath


def load_results(models_dir, model_dir):
    filepath = f"{models_dir}/{model_dir}/results.json"
    with open(filepath) as f:
        config = json.load(f)
    return config


def create_run(config):
    models_dir = config["train"]["models_dir"]
    model_dir = create_model_dir(config["train"]["models_dir"])
    config["train"]["model_dir"] = model_dir
    save_config(models_dir, model_dir, config)
    return config


def load_config(models_dir, model_dir):
    filepath = f"{models_dir}/{model_dir}/config.json"
    with open(filepath) as f:
        config = json.load(f)
    return config


def load_run(config):
    models_dir = config["train"]["models_dir"]
    model_dir = config["train"].get("resume_model")
    if not model_dir:
        dirs = os.listdir(models_dir)
        model_dirs = [d for d in dirs if re.fullmatch(r"^\d{8}_\d{6}$", d)]
        model_dirs.sort()
        model_dir = model_dirs[-1]

    config_loaded = load_config(models_dir, model_dir)
    if config != config_loaded:
        logging.warning("Warning: Loaded config differs from current config, using loaded config.")
        config = config_loaded

    return config


def set_up_run(config):
    if config["train"].get("resume", False):
        config = load_run(config)
    else:
        model_dir = create_run(config)
    return model_dir, config


def load_model_checkpoint(models_dir, model_dir, step=None, device="cuda"):
    if step is None:
        # Find latest checkpoint
        checkpoints = glob.glob(os.path.join(f"{models_dir}/{model_dir}/", "model_step_*.pt"))
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints found in {model_dir}")
        steps = [int(os.path.basename(cp).split("_")[-1].split(".")[0]) for cp in checkpoints]
        step = max(steps)

    checkpoint_path = os.path.join(model_dir, f"model_step_{step}.pt")

    print(f"Loading checkpoint from step {step}")
    checkpoint = torch.load(f"{models_dir}/{checkpoint_path}", map_location=device)

    return checkpoint, step
