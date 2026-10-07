"""Generate the curriculum pool (spec 5.2): dialog-formatted samples mixed into
the last 10% of pretraining steps to warm up dialog-token embeddings."""
import argparse
import os
import random
from multiprocessing import Pool

import gen_sft

U, A, T = "<|user|>", "<|assistant|>", "<|tool|>"


def render(s):
    parts = [U + s["q"]]
    segs = s["assistant_segments"]
    for seg in segs[:-1]:
        parts.append(A + seg)
    parts.append(A + segs[-1])
    return "".join(parts)


def one(seed):
    rng = random.Random(seed)
    x = rng.random()
    if x < 0.60:
        s = gen_sft.gen_arith(rng)
    elif x < 0.70:
        s = gen_sft.gen_clarify(rng)
    elif x < 0.80:
        s = gen_sft.gen_cannot(rng)
    elif x < 0.85:
        s = gen_sft.gen_negative(rng)
    else:
        s = gen_sft.gen_correction(rng)
    return render(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--total", type=int, default=1_500_000)
    ap.add_argument("--out", default="data/curriculum")
    ap.add_argument("--shards", type=int, default=32)
    ap.add_argument("--procs", type=int, default=32)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    per = args.total // args.shards
    jobs = [(500_000 + i, per) for i in range(args.shards)]
    with Pool(args.procs) as p:
        outs = p.map(one_parallel, jobs)
    for i, lines in enumerate(outs):
        with open(os.path.join(args.out, f"cur_{i:03d}.txt"), "w") as f:
            f.write("\n".join(lines))
    print("done")


def one_parallel(job):
    seed, n = job
    return [one(seed * 1000 + k) for k in range(n)]


if __name__ == "__main__":
    main()
