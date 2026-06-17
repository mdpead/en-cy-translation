import sys
import argparse
import yaml

sys.path.insert(0, ".")

from src.hf_wrapper import EnCyForTranslation
from src.tokenizer import load_tokenizer
from src.utils import get_run_path

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True, help="Config name (e.g. base)")
parser.add_argument("--repo", required=True, help="HuggingFace repo (e.g. username/en-cy)")
args = parser.parse_args()

with open(f"configs/{args.config}.yaml") as f:
    config = yaml.safe_load(f)

run_path = get_run_path(config)

print(f"Loading model from {run_path}...")
model = EnCyForTranslation.from_run(run_path)

print("Loading tokenizer...")
tokenizer = load_tokenizer(run_path)

print(f"Pushing to {args.repo}...")
model.push_to_hub(args.repo)
tokenizer.push_to_hub(args.repo)
print(f"Done — https://huggingface.co/{args.repo}")
