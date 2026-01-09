from tokenizers import Tokenizer
from tokenizers import models, pre_tokenizers, trainers, processors
from tokenizers import normalizers
from tokenizers import decoders
from transformers import PreTrainedTokenizerFast
import itertools
import os
from src import utils
import json


def create_tokenizer(text, tokenizers_config):
    tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.Sequence(
        [normalizers.NFD(), normalizers.Lowercase(), normalizers.StripAccents()]
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [pre_tokenizers.WhitespaceSplit(), pre_tokenizers.Punctuation()]
    )
    special_tokens = ["[BOS]", "[EOS]", "[PAD]", "[MASK]", "[UNK]"]
    tokenizer.model = models.WordPiece(unk_token="[UNK]")
    trainer = trainers.WordPieceTrainer(
        vocab_size=tokenizers_config["vocab_size"], special_tokens=special_tokens
    )
    train_iter = itertools.islice(text, tokenizers_config["training_size"])
    tokenizer.train_from_iterator(train_iter, trainer)
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[BOS] $A [EOS]",
        special_tokens=[
            ("[BOS]", tokenizer.token_to_id("[BOS]")),
            ("[EOS]", tokenizer.token_to_id("[EOS]")),
        ],
    )
    tokenizer.decoder = decoders.WordPiece(prefix="##")

    pretrained_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="[BOS]",
        eos_token="[EOS]",
        pad_token="[PAD]",
        mask_token="[MASK]",
        unk_token="[UNK]",
    )
    return pretrained_tokenizer


def create_tokenizers(ds, tokenizers_config):
    english_tokenizer = create_tokenizer(ds["train"]["text_en"], tokenizers_config)
    welsh_tokenizer = create_tokenizer(ds["train"]["text_cy"], tokenizers_config)
    return {"en": english_tokenizer, "cy": welsh_tokenizer}


def save_tokenizers(tokenizers, tokenizers_dir):
    tokenizers["en"].save_pretrained(f"{tokenizers_dir}/tokenizer_en")
    tokenizers["cy"].save_pretrained(f"{tokenizers_dir}/tokenizer_cy")
    return None


def load_tokenizers(tokenizers_dir):
    tokenizer_en = PreTrainedTokenizerFast.from_pretrained(f"{tokenizers_dir}/tokenizer_en")
    tokenizer_cy = PreTrainedTokenizerFast.from_pretrained(f"{tokenizers_dir}/tokenizer_cy")
    return {"en": tokenizer_en, "cy": tokenizer_cy}


def get_tokenizers(ds, ds_hash, config):
    tokenizers_config = config["tokenizers"]
    tokenizers_config_resolved = {**tokenizers_config, "datasets_raw_hash": ds_hash}
    tokenizers_hash = utils.fingerprint(tokenizers_config_resolved)
    tokenizers_path = f"{config['locations']['tokenizers_dir']}/{tokenizers_hash}"
    if os.path.exists(tokenizers_path):
        tokenizers = load_tokenizers(tokenizers_path)
    else:
        tokenizers = create_tokenizers(ds, tokenizers_config)
        save_tokenizers(tokenizers, tokenizers_path)
        with open(tokenizers_path + "/config.json", "w") as f:
            json.dump(tokenizers_config_resolved, f, indent=2)

    return tokenizers, tokenizers_hash
