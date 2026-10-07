"""LingJie SFT (spec 5.1/5.4): LR 1e-5, LM loss on assistant segments only,
plus weighted multi-head losses: 0.3 intent, 0.2 integrity, 0.2 parser,
0.1 critic, 0.1 template, 0.1 verb."""
import argparse
import json
import math
import os
import re
import time

import numpy as np
import torch

from model import LingJie, LingJieConfig
from build_tokenizer import preprocess_for_bpe

OPS = set("+-*/=()")


def tokenize_sample(tok, s, max_len=256):
    if "turns" in s:
        return tokenize_multiturn(tok, s, max_len)
    return _unused_single(tok, s, max_len)


def tokenize_multiturn(tok, s, max_len=256):
    """Multi-turn: concatenate turns; user parts masked, all assistant
    segments trained; pooled head labels come from the final turn."""
    bos = tok.token_to_id("<BOS>")
    eos = tok.token_to_id("<EOS>")
    pad = tok.token_to_id("<PAD>")

    def enc(t):
        return tok.encode(preprocess_for_bpe(t), add_special_tokens=False).ids

    ids, labels, parser = [bos], [-100], [-100]
    for turn in s["turns"]:
        u = enc("<|user|>" + turn["q"])
        ids += u
        labels += [-100] * len(u)
        parser += [-100] * len(u)
        for seg in turn["assistant_segments"]:
            if "<|tool|>" in seg:
                head_p, result = seg.split("<|tool|>", 1)
                part = enc("<|assistant|>" + head_p + "<|tool|>")
                ids += part
                labels += part
                res = enc(result)
                ids += res
                labels += [-100] * len(res)
            else:
                part = enc("<|assistant|>" + seg)
                ids += part
                labels += part
            parser += [-100] * (len(ids) - len(parser))
    part = enc("<EOS>")
    ids += part
    labels += part
    parser += [-100] * len(part)

    ids = ids[:max_len]
    labels = labels[:max_len]
    parser = parser[:max_len]
    attn = [1] * len(ids)
    pad_n = max_len - len(ids)
    ids += [pad] * pad_n
    labels += [-100] * pad_n
    parser += [-100] * pad_n
    attn += [0] * pad_n
    return ids, labels, parser, attn, s["turns"][-1]["labels"]


def _unused_single(tok, s, max_len=256):
    bos = tok.token_to_id("<BOS>")
    eos = tok.token_to_id("<EOS>")
    pad = tok.token_to_id("<PAD>")

    def enc(t):
        return tok.encode(preprocess_for_bpe(t), add_special_tokens=False).ids

    ids = [bos]
    labels = [-100]
    parser = [-100]
    ids += enc("<|user|>" + s["q"])
    labels += [-100] * (len(ids) - len(labels))
    # parser tags over the user segment
    utoks = [tok.id_to_token(i) for i in ids]
    prev_digit = False
    for i in range(len(parser), len(ids)):
        t = utoks[i].replace("Ġ", "").strip()
        if t.isdigit() and len(t) == 1:
            parser.append(1 if not prev_digit else 2)
            prev_digit = True
        elif t in OPS:
            parser.append(3)
            prev_digit = False
        else:
            parser.append(0)
            prev_digit = False
    for seg in s["assistant_segments"]:
        if "<|tool|>" in seg:
            head, result = seg.split("<|tool|>", 1)
            part = enc("<|assistant|>" + head + "<|tool|>")
            ids += part
            labels += part  # assistant action incl. tool request: trainable
            res = enc(result)
            ids += res
            labels += [-100] * len(res)  # environment output: masked
        else:
            part = enc("<|assistant|>" + seg)
            ids += part
            labels += part
        parser += [-100] * (len(ids) - len(parser))
    part = enc("<EOS>")
    ids += part
    labels += part
    parser += [-100] * len(part)

    ids = ids[:max_len]
    labels = labels[:max_len]
    parser = parser[:max_len]
    attn = [1] * len(ids)
    pad_n = max_len - len(ids)
    ids += [pad] * pad_n
    labels += [-100] * pad_n
    parser += [-100] * pad_n
    attn += [0] * pad_n
    return ids, labels, parser, attn, s["labels"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/sft.jsonl")
    ap.add_argument("--tokenizer", default="tokenizer/lingjie_tokenizer.json")
    ap.add_argument("--init", default="ckpt/pretrain/last.pt")
    ap.add_argument("--out", default="ckpt/sft")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--micro-bs", type=int, default=64)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--log-every", type=int, default=20)
    args = ap.parse_args()

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(args.tokenizer)

    samples = [json.loads(l) for l in open(args.data)]
    print(f"{len(samples)} sft samples", flush=True)
    encoded = [tokenize_sample(tok, s, args.max_len) for s in samples]
    n_steps = args.epochs * math.ceil(len(encoded) / (args.micro_bs * args.accum))
    print(f"steps: {n_steps} (effective batch {args.micro_bs*args.accum})", flush=True)

    dev = "cuda"
    cfg = LingJieConfig()
    model = LingJie(cfg).to(dev)
    state = torch.load(args.init, map_location=dev, weights_only=False)
    model.load_state_dict(state["model"])
    model.train()

    decay, no_decay = [], []
    for n, p in model.named_parameters():
        (no_decay if (p.ndim < 2 or "norm" in n) else decay).append(p)
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.1},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95), eps=1e-8, fused=True)
    scaler = torch.cuda.amp.GradScaler()

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(7)
    step = 0
    t0 = time.time()
    running = {}
    for ep in range(args.epochs):
        order = rng.permutation(len(encoded))
        for bstart in range(0, len(order), args.micro_bs * args.accum):
            opt.zero_grad(set_to_none=True)
            mb = order[bstart: bstart + args.micro_bs * args.accum]
            for mstart in range(0, len(mb), args.micro_bs):
                idxs = mb[mstart: mstart + args.micro_bs]
                batch = [encoded[i] for i in idxs]
                input_ids = torch.tensor([b[0] for b in batch], device=dev)
                lm_labels = torch.tensor([b[1] for b in batch], device=dev)
                # shift: logits[t] must predict token t+1 (model CE is unshifted;
                # pretrain.py shifts externally, here we shift the label tensor)
                lm_targets = torch.full_like(lm_labels, -100)
                lm_targets[:, :-1] = lm_labels[:, 1:]
                parser = torch.tensor([b[2] for b in batch], device=dev)
                attn = torch.tensor([b[3] for b in batch], device=dev)
                pooled_labels = {
                    "intent": torch.tensor([b[4]["intent"] for b in batch], device=dev),
                    "integrity": torch.tensor([b[4]["integrity"] for b in batch], device=dev),
                    "critic": torch.tensor([b[4]["critic"] for b in batch], device=dev),
                    "template": torch.tensor([b[4]["template"] for b in batch], device=dev),
                    "verb": torch.tensor([b[4]["verb"] for b in batch], device=dev),
                    "parser": parser,
                }
                with torch.autocast("cuda", dtype=torch.float16):
                    out = model(input_ids, targets=lm_targets, head_labels=pooled_labels,
                                attn_mask=attn)
                # LM loss with ignore_index + attn mask already handled via -100
                loss = (out["loss_lm"] + out["loss_heads"])
                scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            step += 1
            if step % args.log_every == 0:
                msg = {"step": step, "of": n_steps, "loss_lm": out["loss_lm"].item(),
                       "loss_heads": out["loss_heads"].item(),
                       "tok_s": step * args.micro_bs * args.accum * args.max_len / (time.time() - t0)}
                print(json.dumps(msg), flush=True)
        torch.save({"model": model.state_dict(), "config": cfg.__dict__, "epoch": ep},
                   os.path.join(args.out, "last.pt"))
    print("sft done, steps:", step)


if __name__ == "__main__":
    main()
