"""Tokenize and pack documents into fixed-length uint16 sequences (memmap .bin).

Documents: <BOS> text <EOS>, concatenated and chunked at max_seq_len (no padding waste).
"""
import argparse
import glob
import json
import os
import sys
import numpy as np
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_tokenizer import preprocess_for_bpe  # noqa: E402

_TOK = None
_BOS = None
_EOS = None


def _init(tok_path):
    global _TOK, _BOS, _EOS
    from tokenizers import Tokenizer
    _TOK = Tokenizer.from_file(tok_path)
    _TOK.no_truncation()
    _BOS = _TOK.token_to_id("<BOS>")
    _EOS = _TOK.token_to_id("<EOS>")


def _encode_file(path):
    ids = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            enc = _TOK.encode(preprocess_for_bpe(line)).ids
            ids.extend([_BOS] + enc + [_EOS])
    return np.asarray(ids, dtype=np.uint16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="glob of text files, one doc per line")
    ap.add_argument("--tokenizer", default="tokenizer/lingjie_tokenizer.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--procs", type=int, default=32)
    args = ap.parse_args()

    files = sorted(glob.glob(args.input))
    assert files, f"no files match {args.input}"
    chunks = []
    with Pool(args.procs, initializer=_init, initargs=(args.tokenizer,)) as p:
        for i, arr in enumerate(p.imap_unordered(_encode_file, files)):
            chunks.append(arr)
            if (i + 1) % 16 == 0:
                print(f"tokenized {i+1}/{len(files)} files", flush=True)
    all_ids = np.concatenate(chunks)
    n_seq = len(all_ids) // args.seq_len
    all_ids = all_ids[: n_seq * args.seq_len].reshape(n_seq, args.seq_len)
    meta = {"tokens": int(all_ids.size), "seq_len": args.seq_len, "sequences": n_seq}
    np.save(args.out, all_ids)
    with open(args.out + ".meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("packed:", json.dumps(meta))


if __name__ == "__main__":
    main()
