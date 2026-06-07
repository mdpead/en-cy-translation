from tokenizers import Tokenizer
from tokenizers import models, pre_tokenizers, trainers, processors
from tokenizers import normalizers
from tokenizers import decoders
from transformers import PreTrainedTokenizerFast
import itertools
from src import utils


def create_tokenizer(ds, tokenizer_config):
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
        vocab_size=tokenizer_config["vocab_size"], special_tokens=special_tokens
    )
    combined_text = itertools.chain(ds["train"]["text_en"], ds["train"]["text_cy"])
    train_iter = itertools.islice(combined_text, tokenizer_config["training_size"])
    tokenizer.train_from_iterator(train_iter, trainer)
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[BOS] $A [EOS]",
        special_tokens=[
            ("[BOS]", tokenizer.token_to_id("[BOS]")),
            ("[EOS]", tokenizer.token_to_id("[EOS]")),
        ],
    )
    tokenizer.decoder = decoders.WordPiece(prefix="##")

    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="[BOS]",
        eos_token="[EOS]",
        pad_token="[PAD]",
        mask_token="[MASK]",
        unk_token="[UNK]",
    )


def save_tokenizer(tokenizer, dir):
    tokenizer.save_pretrained(f"{dir}/tokenizer")


def load_tokenizer(dir):
    return PreTrainedTokenizerFast.from_pretrained(f"{dir}/tokenizer")


def _tokenize_text(text, tokenizer):
    return {
        "text_en_tokenized": tokenizer(text["text_en"])["input_ids"],
        "text_cy_tokenized": tokenizer(text["text_cy"])["input_ids"],
    }


def tokenize_dataset(ds, tokenizer, config):
    ds = ds.map(lambda x: _tokenize_text(x, tokenizer), batched=True, batch_size=100000)
    ds = ds.map(
        lambda row: {
            "en_token_length": len(row["text_en_tokenized"]),
            "cy_token_length": len(row["text_cy_tokenized"]),
        },
    )
    max_length = config["model"]["max_length"]
    ds = ds.filter(
        lambda x: (x["en_token_length"] <= max_length) and (x["cy_token_length"] <= max_length)
    )
    ds = ds.map(lambda _, idx: {"idx": idx}, with_indices=True)
    return ds


def get_tokenizer(ds, config):

    tokenizer_config = config["tokenizer"]
    model_path = utils.get_model_path(config)

    tokenizer = create_tokenizer(ds, tokenizer_config)
    save_tokenizer(tokenizer, model_path)

    return tokenizer
