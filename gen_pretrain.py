"""Program-generated pretraining corpus for LingJie.

Mix per spec 4.3: arithmetic 40%, negative 30%, boundary 10%,
clarification 10%, idk 5%, correction 5%.
Outputs plain-text docs (one per line, <BOS>/<EOS> added at pack time).
"""
import argparse
import json
import os
import random
import re
import sys
from multiprocessing import Pool

OPS = ["+", "-", "*", "/"]
OP_WORDS = {
    "+": ["plus", "added to", "and"],
    "-": ["minus", "subtracted from"],  # note: "a subtracted from b" = b-a handled at call site
    "*": ["times", "multiplied by"],
    "/": ["divided by", "over"],
}
ANSWER_TPL = "{a} {w} {b} equals {r}."
QUESTION_TPL = [
    "What is {a} {w} {b}?",
    "How much is {a} {w} {b}?",
    "Calculate {a} {w} {b}.",
    "Compute {a} {w} {b}.",
    "{a} {w} {b}?",
    "Please compute {a} {w} {b} for me.",
    "Can you work out {a} {w} {b}?",
    "Tell me: {a} {w} {b}.",
]
NEG_TPL = [
    "I have {n} apples in my basket.",
    "The train arrives at platform {n}.",
    "She read {n} pages of her book yesterday.",
    "The city was founded in {n}.",
    "My phone number ends with {n}.",
    "Room {n} is on the third floor.",
    "He scored {n} points in the final quarter.",
    "The recipe calls for {n} cups of flour.",
    "There were {n} birds on the wire this morning.",
    "The bus leaves in {n} minutes.",
    "Ticket number {n} won the raffle.",
    "The store closes at {n} o'clock.",
    "They walked {n} blocks before it started to rain.",
    "Her garden has {n} rose bushes.",
    "The movie is {n} minutes long.",
]
BOUNDARY_TPL = [
    "What is {a} {w} {b}?",
    "Calculate {a} {w} {b}.",
    "How much is {a} {w} {b}?",
]
CANNOT_LONG = "<CANNOT> I can only handle numbers with up to 6 digits."
CANNOT_ZERO = "<CANNOT> Division by zero is undefined."
CANNOT_OP = "<CANNOT> I can only help with +, -, * and /."
CLARIFY_TPL = [
    ("Add them.", "<NEED_INFO> Which numbers would you like me to add?"),
    ("What is plus {b}?", "<NEED_INFO> Please provide the first number."),
    ("Subtract {a} from it.", "<NEED_INFO> What number should I subtract {a} from?"),
    ("Multiply it by that thing.", "<NEED_INFO> Please provide the number to multiply by."),
    ("How much is {a} plus?", "<NEED_INFO> Please provide the second number."),
    ("Do the math for those two.", "<NEED_INFO> Which two numbers do you mean?"),
]
IDK_QA = [
    ("What is the capital of France?", "<CANNOT> I can only help with arithmetic."),
    ("Who won the game last night?", "<CANNOT> I don't have that information."),
    ("What is the weather like today?", "<CANNOT> I don't have that information."),
    ("Tell me a joke about cats.", "<CANNOT> I can only help with arithmetic."),
    ("Why is the sky blue?", "<CANNOT> I can only help with arithmetic."),
    ("What did she say to him at the party?", "<CANNOT> I don't have that information."),
    ("Is it going to rain tomorrow?", "<CANNOT> I don't have that information."),
]
CORRECTION_TPL = [
    ("Someone said {a} {w} {b} is {wrong}. That is incorrect.",
     "<CALC><EXPR>{expr}</EXPR></CALC><RESULT>{r}</RESULT> Actually, {a} {w} {b} equals {r}."),
    ("{a} {w} {b} is {wrong}, right?",
     "<CALC><EXPR>{expr}</EXPR></CALC><RESULT>{r}</RESULT> No. {a} {w} {b} equals {r}."),
    ("I think {a} {w} {b} equals {wrong}.",
     "<CALC><EXPR>{expr}</EXPR></CALC><RESULT>{r}</RESULT> Let me check. {a} {w} {b} equals {r}, not {wrong}."),
]

CALC = "<CALC><EXPR>{a}{op}{b}</EXPR></CALC><RESULT>{r}</RESULT>"


def fmt_num(n):
    return str(n)


def calc_text(a, op, b, r):
    return CALC.format(a=a, op=op, b=b, r=r)


def gen_arithmetic(rng):
    op = rng.choice(OPS)
    scale = rng.random()
    if scale < 0.45:
        a, b = rng.randint(0, 99), rng.randint(0, 99)
    elif scale < 0.85:
        a, b = rng.randint(0, 9999), rng.randint(0, 999)
    else:
        a, b = rng.randint(10, 999999), rng.randint(10, 999999)
    if op == "/":
        if rng.random() < 0.7:  # exact integer division
            b = rng.randint(1, 99)
            a = b * rng.randint(0, 9999)
            r = a // b
        else:  # terminating decimal
            a, b = rng.randint(1, 999), rng.choice([2, 4, 5, 8, 10, 20, 25, 40, 50])
            r = round(a / b, 4)
            if r == int(r):
                r = int(r)
    elif op == "-":
        if rng.random() < 0.8 and b > a:
            a, b = b, a  # keep most answers non-negative
        r = a - b
    elif op == "+":
        r = a + b
    else:
        a, b = min(a, 9999), rng.randint(0, 999)
        r = a * b
    w = rng.choice(OP_WORDS[op])
    if op == "-" and w == "subtracted from":
        # "a subtracted from b" means b - a; keep it consistent by re-stating
        q = rng.choice(QUESTION_TPL).format(a=a, w="minus", b=b)
    else:
        q = rng.choice(QUESTION_TPL).format(a=a, w=w, b=b)
    if rng.random() < 0.5:
        body = f"{q} {calc_text(a, op, b, r)} {ANSWER_TPL.format(a=a, w=w, b=b, r=r)}"
    else:
        body = f"{q} {calc_text(a, op, b, r)} The answer is {r}."
    return body


def gen_negative(rng):
    t = rng.choice(NEG_TPL)
    if rng.random() < 0.6:
        s = t.format(n=rng.randint(0, 99999))
    else:
        s = t.format(n=rng.randint(0, 20))
    extra = rng.choice([
        "", " It was a normal day.", " Nobody paid much attention.",
        " That is all I remember.", " Everyone was happy about it.",
    ])
    return s + extra


def gen_boundary(rng):
    kind = rng.random()
    if kind < 0.6:  # too-long numbers
        a = rng.randint(1000000, 99999999)
        b = rng.randint(2, 99)
        op = rng.choice(["+", "*", "-"])
        w = rng.choice(OP_WORDS[op])
        q = rng.choice(BOUNDARY_TPL).format(a=fmt_num(a), w=w, b=b)
        return f"{q} {CANNOT_LONG}"
    elif kind < 0.8:  # division by zero
        a = rng.randint(1, 999)
        q = rng.choice(BOUNDARY_TPL).format(a=a, w="divided by", b=0)
        return f"{q} {CANNOT_ZERO}"
    else:  # unsupported operation
        a = rng.randint(1, 999)
        fn = rng.choice(["sqrt", "sin", "log", "the square root of"])
        q = f"What is {fn} {a}?" if fn != "the square root of" else f"What is the square root of {a}?"
        return f"{q} {CANNOT_OP}"


def gen_clarify(rng):
    q, resp = rng.choice(CLARIFY_TPL)
    return f"{q.format(b=rng.randint(1, 99), a=rng.randint(1, 99))} {resp}"


def gen_idk(rng):
    q, resp = rng.choice(IDK_QA)
    return f"{q} {resp}"


def gen_correction(rng):
    q, resp = rng.choice(CORRECTION_TPL)
    op = rng.choice(["+", "*", "-"])
    a, b = rng.randint(2, 99), rng.randint(2, 99)
    if op == "-" and b > a:
        a, b = b, a
    r = {"+": a + b, "*": a * b, "-": a - b}[op]
    w = rng.choice(OP_WORDS[op])
    wrong = r + rng.choice([-3, -2, -1, 1, 2, 3, 10]) * rng.choice([1, 10])
    while wrong == r:
        wrong = r + rng.choice([-3, -2, -1, 1, 2, 3]) * 10
    expr = f"{a}{op}{b}"
    return (q.format(a=a, w=w, b=b, wrong=wrong) + " " +
            resp.format(a=a, w=w, b=b, r=r, expr=expr, wrong=wrong))


GENS = [
    (gen_arithmetic, 0.40),
    (gen_negative, 0.30),
    (gen_boundary, 0.10),
    (gen_clarify, 0.10),
    (gen_idk, 0.05),
    (gen_correction, 0.05),
]


def worker(args):
    seed, n, out_path = args
    rng = random.Random(seed)
    lines = []
    for _ in range(n):
        x = rng.random()
        acc = 0.0
        for fn, w in GENS:
            acc += w
            if x < acc:
                lines.append(fn(rng))
                break
        else:
            lines.append(gen_arithmetic(rng))
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--total", type=int, default=4_000_000, help="number of docs")
    ap.add_argument("--out", default="data/pretrain_gen")
    ap.add_argument("--shards", type=int, default=32)
    ap.add_argument("--procs", type=int, default=32)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    per = args.total // args.shards
    jobs = [(1000 + i, per, os.path.join(args.out, f"shard_{i:03d}.txt")) for i in range(args.shards)]
    with Pool(args.procs) as p:
        for path in p.imap_unordered(worker, jobs):
            print("wrote", path, flush=True)
    print("done")


if __name__ == "__main__":
    main()
