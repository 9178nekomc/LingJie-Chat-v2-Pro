"""LingJie pretraining (spec section 5).

- AdamW, LR 3e-4, cosine decay with warmup, effective batch 256 (grad accum).
- fp16 autocast + GradScaler (Turing GPU), grad clip 1.0.
- Curriculum (spec 5.2): last 10% of steps mixes in dialog samples (5%) and
  protocol-heavy labeled-style samples (8%).
- Data: memmap uint16 .bin pools; sampler switches mix in the curriculum phase.
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from model import LingJie, LingJieConfig


class BinDataset(Dataset):
    def __init__(self, path):
        self.meta = json.load(open(path + ".meta.json"))
        self.data = np.load(path, mmap_mode="r")
        assert self.data.shape[1] >= 2

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, i):
        seq = torch.from_numpy(self.data[i].astype(np.int64))
        return {"input": seq[:-1], "target": seq[1:]}


class MixSampler:
    """Yields micro-batches as (pool, [idx...]) with pool 0=main, 1=curriculum.
    Last `cur_frac` of steps mix curriculum batches in with prob (1-main_frac)."""

    def __init__(self, lens, batch_size, total_micro, cur_frac, main_frac, seed=1234):
        self.lens = lens
        self.batch_size = batch_size
        self.total_micro = total_micro
        self.cur_start = int(total_micro * (1 - cur_frac))
        self.main_frac = main_frac
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.total_micro

    def __iter__(self):
        idxs = [self.rng.permutation(n) for n in self.lens]
        pos = [0, 0]
        for b in range(self.total_micro):
            micro_step = b
            use_cur = (micro_step >= self.cur_start and
                       self.rng.random() > self.main_frac)
            pool = 1 if use_cur else 0
            need = self.batch_size
            out = []
            while need > 0:
                if pos[pool] + need <= len(idxs[pool]):
                    take = idxs[pool][pos[pool]: pos[pool] + need]
                    pos[pool] += need
                    out.extend(take.tolist())
                    need = 0
                else:
                    take = idxs[pool][pos[pool]:]
                    pos[pool] = 0
                    idxs[pool] = self.rng.permutation(self.lens[pool])
                    out.extend(take.tolist())
                    need -= len(take)
            yield pool, out


def cosine_lr(step, total, warmup, base_lr, min_lr_ratio=0.1):
    if step < warmup:
        return base_lr * step / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return base_lr * (min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * p)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main-bin", required=True)
    ap.add_argument("--cur-bin", required=True, help="curriculum pool (dialog/protocol samples)")
    ap.add_argument("--out", default="ckpt/pretrain")
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--micro-bs", type=int, default=32)
    ap.add_argument("--accum", type=int, default=8)  # 32*8 = 256 effective
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    torch.manual_seed(1234)
    dev = "cuda"
    cfg = LingJieConfig()
    model = LingJie(cfg).to(dev)
    model.train()

    main_ds = BinDataset(args.main_bin)
    cur_ds = BinDataset(args.cur_bin)
    print(f"main pool: {main_ds.meta['tokens']/1e6:.1f}M tokens; "
          f"curriculum pool: {cur_ds.meta['tokens']/1e6:.1f}M tokens", flush=True)

    sampler = MixSampler([len(main_ds), len(cur_ds)], args.micro_bs,
                         total_micro=args.steps * args.accum,
                         cur_frac=0.10,
                         main_frac=0.6)  # curriculum step: 40% curriculum batches
    import queue
    import threading

    def make_batch(pool, idx_list):
        ds = main_ds if pool == 0 else cur_ds
        rows = [ds[i] for i in idx_list]
        return (torch.stack([r["input"] for r in rows]),
                torch.stack([r["target"] for r in rows]))

    def data_stream():
        q = queue.Queue(maxsize=8)

        def producer():
            for pool, idx_list in sampler:
                q.put(make_batch(pool, idx_list))
            q.put(None)

        threading.Thread(target=producer, daemon=True).start()
        while True:
            item = q.get()
            if item is None:
                return
            yield item

    decay, no_decay = [], []
    for n, p in model.named_parameters():
        (no_decay if (p.ndim < 2 or "norm" in n) else decay).append(p)
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95), eps=1e-8, fused=True)
    scaler = torch.cuda.amp.GradScaler()

    net = torch.compile(model) if args.compile else model

    start_step = 0
    if args.resume and os.path.exists(args.resume):
        state = torch.load(args.resume, map_location=dev, weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        scaler.load_state_dict(state["scaler"])
        start_step = state["step"]
        print(f"resumed from {args.resume} at step {start_step}", flush=True)

    os.makedirs(args.out, exist_ok=True)
    log_path = os.path.join(args.out, "train_log.jsonl")
    t0 = time.time()
    tokens_per_step = args.micro_bs * args.accum * (cfg.max_seq_len - 1)
    step = start_step
    micro_iter = data_stream()
    running = 0.0
    running_n = 0
    while step < args.steps:
        lr = cosine_lr(step, args.steps, args.warmup, args.lr)
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        for _ in range(args.accum):
            batch = next(micro_iter)
            idx = batch[0].to(dev, non_blocking=True)
            tgt = batch[1].to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                out = net(idx, targets=tgt)
                loss = out["loss_lm"] / args.accum
            scaler.scale(loss).backward()
            running += out["loss_lm"].item()
            running_n += 1
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        step += 1
        if step % args.log_every == 0:
            dt = time.time() - t0
            tok_s = step * tokens_per_step / dt
            msg = {"step": step, "loss": running / max(1, running_n),
                   "lr": lr, "tok_per_s": tok_s,
                   "eta_hours": (args.steps - step) * tokens_per_step / max(tok_s, 1) / 3600}
            print(json.dumps(msg), flush=True)
            with open(log_path, "a") as f:
                f.write(json.dumps(msg) + "\n")
            running, running_n = 0.0, 0
        if step % args.save_every == 0 or step == args.steps:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "scaler": scaler.state_dict(), "step": step,
                        "config": cfg.__dict__},
                       os.path.join(args.out, "last.pt"))
    print("pretraining done:", step, "steps")


if __name__ == "__main__":
    main()
