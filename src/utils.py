import os
import json
import logging
import shutil


def get_run_path(config):
    return f"{config['locations']['runs_dir']}/{config['name']}"


def init_run(config):
    run_path = get_run_path(config)
    os.makedirs(run_path, exist_ok=True)
    config_path = f"{run_path}/config.json"
    if os.path.exists(config_path):
        existing = load_config(run_path)
        if existing != config:
            logging.warning("Config differs from saved config — clearing run directory.")
            shutil.rmtree(run_path)
            os.makedirs(run_path)
    save_config(run_path, config)
    return run_path


def save_config(run_path, config):
    filepath = f"{run_path}/config.json"
    json.dump(config, open(filepath, "w"))
    return filepath


def load_config(run_path):
    filepath = f"{run_path}/config.json"
    with open(filepath) as f:
        config = json.load(f)
    return config
