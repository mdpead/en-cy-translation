from tokenizers import Tokenizer
from tokenizers import models, pre_tokenizers, trainers, processors
from tokenizers import normalizers
from tokenizers import decoders
from transformers import PreTrainedTokenizerFast
import itertools


VOCAB_SIZE = 10000
TOKENIZER_TRAINING_SIZE = 100000


def create_tokenizer(text):
    tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.Sequence(
        [normalizers.NFD(), normalizers.Lowercase(), normalizers.StripAccents()]
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [pre_tokenizers.WhitespaceSplit(), pre_tokenizers.Punctuation()]
    )
    special_tokens = ["[BOS]", "[EOS]", "[PAD]", "[MASK]", "[UNK]"]
    tokenizer.model = models.WordPiece(unk_token="[UNK]")
    trainer = trainers.WordPieceTrainer(vocab_size=VOCAB_SIZE, special_tokens=special_tokens)
    train_iter = itertools.islice(text, TOKENIZER_TRAINING_SIZE)
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


def create_tokenizers(ds):
    english_tokenizer = create_tokenizer(text=ds["train"]["text_en"])
    welsh_tokenizer = create_tokenizer(text=ds["train"]["text_cy"])
    return {"en": english_tokenizer, "cy": welsh_tokenizer}


def tokenize_text(text, tokenizers):
    texts_tokenized = {}
    for lang in ["en", "cy"]:
        text_tokenized = tokenizers[lang](
            text[f"text_{lang}"],
        )
        texts_tokenized[f"text_{lang}_tokenized"] = text_tokenized["input_ids"]
    return texts_tokenized


def tokenize_dataset(ds, tokenizers):
    ds = ds.map(lambda x: tokenize_text(x, tokenizers), batched=True, batch_size=100000)
    return ds
