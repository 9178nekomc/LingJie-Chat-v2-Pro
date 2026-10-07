"""Train the LingJie tokenizer (spec section 3).

- ByteLevel BPE, vocab_size 8000, trained on TinyStories sample.
- Digits forced to single chars via preprocess_for_bpe (spec 3.4): each digit becomes
  its own space-delimited pretoken, so BPE merges can never span digit characters.
- 13 special tokens + 4 dialog tokens added afterwards -> 8017 total ids.
- Decode never re-joins spaces (spec 3.4.4); eval uses normalize_for_eval.
"""
import argparse
import os
import re

from tokenizers import Tokenizer, decoders, pre_tokenizers, trainers
from tokenizers.models import BPE
from tokenizers.normalizers import NFKC, Sequence as NormalizerSeq
from tokenizers.pre_tokenizers import ByteLevel

SPECIAL_TOKENS = [
    "<CALC>", "</CALC>", "<EXPR>", "</EXPR>", "<RESULT>", "</RESULT>",
    "<UNK_EXPR>", "<UNK_NUM>", "<NEED_INFO>", "<CANNOT>", "<PAD>", "<BOS>", "<EOS>",
]
DIALOG_TOKENS = ["<|user|>", "<|assistant|>", "<|tool|>", "<|system|>"]

DIGIT_SPLIT = re.compile(r"(\d)")


def preprocess_for_bpe(text: str) -> str:
    """Spec 3.4.2: split every digit into its own space-delimited chunk,
    then collapse runs of spaces so decode round-trips with single spaces."""
    t = DIGIT_SPLIT.sub(r" \1 ", text)
    return re.sub(r"\s+", " ", t).strip()


def normalize_for_eval(text: str) -> str:
    """Spec 3.4.4: collapse spaces inserted by digit-splitting before comparing.
    Three rules: (1) between digits/operators/punctuation, (2) before '</' closers,
    (3) after '>' openers. Word spacing is left untouched."""
    text = re.sub(r"(?<=[0-9+\-*/=(),.:;?])[ ]+(?=[0-9+\-*/=(),.:;?])", "", text)
    text = re.sub(r"(?<=[0-9+\-*/=()])[ ]+(?=</)", "", text)
    text = re.sub(r"(?<=>)[ ]+(?=[0-9+\-*/=()])", "", text)
    return text


def train(corpus_glob: str, out_dir: str, vocab_size: int = 8000):
    from tokenizers import Tokenizer
    tok = Tokenizer(BPE(unk_token=None))
    tok.normalizer = NormalizerSeq([NFKC()])
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Digits(individual_digits=True),  # belt: force single-digit pretokens
        ByteLevel(add_prefix_space=False, use_regex=True),
    ])
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        show_progress=True,
        initial_alphabet=ByteLevel.alphabet(),
    )
    tok.train([corpus_glob], trainer)  # accepts glob

    # add specials + dialog tokens
    tok.add_special_tokens([t for t in SPECIAL_TOKENS])
    tok.add_tokens(DIALOG_TOKENS)

    os.makedirs(out_dir, exist_ok=True)
    tok.save(os.path.join(out_dir, "lingjie_tokenizer.json"))

    # self-test: digits must be single tokens, markup must round-trip
    id_of = {t: tok.token_to_id(t) for t in SPECIAL_TOKENS + DIALOG_TOKENS}
    assert all(v is not None for v in id_of.values())
    test = "What is 4827 plus 13? <CALC><EXPR>4827+13</EXPR></CALC><RESULT>4840</RESULT>"
    enc = tok.encode(preprocess_for_bpe(test)).ids
    dec = tok.decode(enc, skip_special_tokens=False)
    assert normalize_for_eval(dec) == test, (dec, test)
    digit_ids = [tok.token_to_id(str(d)) for d in "0123456789"]
    toks4827 = [tok.id_to_token(i) for i in tok.encode("4827").ids]
    assert toks4827 == ["4", "8", "2", "7"], toks4827
    print("vocab size:", tok.get_vocab_size())
    print("digit ids:", digit_ids)
    print("roundtrip OK; 4827 ->", toks4827)
    return tok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", default="tokenizer")
    ap.add_argument("--vocab", type=int, default=8000)
    args = ap.parse_args()
    train(args.corpus, args.out, args.vocab)
