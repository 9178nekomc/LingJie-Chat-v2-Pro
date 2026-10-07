"""Generate LingJie SFT dataset (~100k dialog samples with multi-head labels).

Canonical dialog format:
  <|user|>{q}<|assistant|><CALC><EXPR>{e}</EXPR></CALC><|tool|><RESULT>{r}</RESULT><|assistant|>{answer}<EOS>

Head labels (spec 2.4 / 5.4):
  intent/integrity/critic: pooled binary; template: 100-way; verb: 20-way;
  parser: per-token 4-way tags (0 other, 1 number-start, 2 number-cont, 3 operator).
LM loss is masked to assistant segments only (tool output is environment text).
"""
import argparse
import json
import os
import random
import re
from multiprocessing import Pool

VERBS = {
    "+": ["plus", "and", "added to", "sum of", "total of"],
    "-": ["minus", "less", "subtracted by", "take away", "reduced by"],
    "*": ["times", "multiplied by", "product of", "multiplied with", "twice by"],
    "/": ["divided by", "over", "quotient of", "split by", "shared by"],
}
ANSWERS = [
    "{a} {w} {b} equals {r}.",
    "The answer is {r}.",
    "That is {r}.",
    "So, {a} {w} {b} is {r}.",
    "The result is {r}.",
    "It equals {r}.",
    "You get {r}.",
    "{r} is the answer.",
    "After computing, {a} {w} {b} is {r}.",
    "In short, the answer is {r}.",
    "The total is {r}.",
    "The final result is {r}.",
    "That works out to {r}.",
    "It comes to {r}.",
    "The value is {r}.",
    "We get {r} as the result.",
    "The computation gives {r}.",
    "Therefore, the answer is {r}.",
    "So the answer is {r}.",
    "The solution is {r}.",
]
QUESTION_TPL = [
    "What is {a} {w} {b}?",
    "How much is {a} {w} {b}?",
    "Calculate {a} {w} {b}.",
    "Compute {a} {w} {b}.",
    "{a} {w} {b}?",
    "Please compute {a} {w} {b} for me.",
    "Can you work out {a} {w} {b}?",
    "Tell me: {a} {w} {b}.",
    "I need the result of {a} {w} {b}.",
    "Do the math: {a} {w} {b}.",
]
CLARIFY = [
    ("Add them.", "<NEED_INFO> Which numbers would you like me to add?"),
    ("What is plus {b}?", "<NEED_INFO> Please provide the first number."),
    ("Subtract them.", "<NEED_INFO> Which numbers should I use, and in what order?"),
    ("Multiply it by that number.", "<NEED_INFO> Please provide the two numbers."),
    ("How much is {a} plus?", "<NEED_INFO> Please provide the second number."),
    ("Divide it.", "<NEED_INFO> Please provide the two numbers."),
]
IDK = [
    ("What is the capital of France?", "<CANNOT> I can only help with arithmetic."),
    ("Who won the game last night?", "<CANNOT> I don't have that information."),
    ("Tell me a story about dragons.", "<CANNOT> I can only help with arithmetic."),
    ("What is the weather today?", "<CANNOT> I don't have that information."),
    ("Why do people like music?", "<CANNOT> I can only help with arithmetic."),
    ("What is the capital of France, and what is 2 plus 2?",
     "<CANNOT> I can only help with arithmetic. I cannot answer questions about capitals."),
]
NEG_QA = [
    "I have {n} apples in my basket.",
    "The train arrives at platform {n}.",
    "She read {n} pages of her book yesterday.",
    "The city was founded in {n}.",
    "He scored {n} points in the final quarter.",
    "Room {n} is on the third floor.",
]


def calc(a, op, b):
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    if op == "*":
        return a * b
    assert b != 0
    r = a / b
    return int(r) if r == int(r) else round(r, 4)


def gen_arith(rng):
    two_step = rng.random() < 0.08
    critic_ok = rng.random() > 0.05
    op = rng.choice(list(VERBS))
    w = rng.choice(VERBS[op])
    scale = rng.random()
    if scale < 0.5:
        a, b = rng.randint(0, 99), rng.randint(0, 99)
    elif scale < 0.9:
        a, b = rng.randint(0, 9999), rng.randint(0, 999)
    else:
        a, b = rng.randint(10, 999999), rng.randint(10, 99999)
    if op == "/":
        b = max(1, b)
        if rng.random() >= 0.7:
            b = rng.choice([2, 4, 5, 8, 10, 20, 25, 40, 50])
    if op == "-" and b > a and rng.random() < 0.8:
        a, b = b, a
    r = calc(a, op, b)
    expr = f"{a}{op}{b}"
    if not critic_ok:
        wrong_a = a + rng.choice([-1, 1, 10]) if rng.random() < 0.5 else a
        expr = f"{wrong_a}{op}{b}"
    q = rng.choice(QUESTION_TPL).format(a=a, w=w, b=b)
    seg1 = f"<CALC><EXPR>{expr}</EXPR></CALC><|tool|><RESULT>{r}</RESULT>"
    ans = rng.choice(ANSWERS).format(a=a, w=w, b=b, r=r)
    if two_step:
        c = rng.randint(2, 12)
        op2 = rng.choice(list(VERBS))
        r2 = calc(r, op2, c)
        expr2 = f"{r}{op2}{c}"
        w2 = rng.choice(VERBS[op2])
        seg1 = (f"<CALC><EXPR>{expr}</EXPR></CALC><|tool|><RESULT>{r}</RESULT>"
                f"<|assistant|><CALC><EXPR>{expr2}</EXPR></CALC><|tool|><RESULT>{r2}</RESULT>")
        ans = rng.choice(ANSWERS).format(a=f"({a} {w} {b})", w=w2, b=c, r=r2)
    return {
        "q": q, "assistant_segments": [seg1, ans],
        "labels": {
            "intent": 1, "integrity": 1,
            "critic": 1 if critic_ok else 0,
            "template": rng.randrange(100),
            "verb": VERBS[op].index(w) * 4 + list(VERBS).index(op),
        },
    }


def gen_clarify(rng):
    q, a = rng.choice(CLARIFY)
    q = q.format(a=rng.randint(1, 99), b=rng.randint(1, 99))
    return {"q": q, "assistant_segments": [a],
            "labels": {"intent": 1, "integrity": 0, "critic": 0,
                       "template": rng.randrange(100), "verb": 0}}


def gen_cannot(rng):
    if rng.random() < 0.5:
        a = rng.randint(1000000, 99999999)
        b = rng.randint(2, 99)
        op = rng.choice(["+", "*"])
        w = rng.choice(VERBS[op])
        q = rng.choice(QUESTION_TPL).format(a=a, w=w, b=b)
        a = f"<CANNOT> I can only handle numbers with up to 6 digits."
        return {"q": q, "assistant_segments": [a],
                "labels": {"intent": 1, "integrity": 1, "critic": 0,
                           "template": rng.randrange(100), "verb": 0}}
    q, a = rng.choice(IDK)
    return {"q": q, "assistant_segments": [a],
            "labels": {"intent": 0, "integrity": 1, "critic": 0,
                       "template": rng.randrange(100), "verb": 0}}


def gen_negative(rng):
    q = rng.choice(NEG_QA).format(n=rng.randint(0, 99999))
    a = rng.choice(["That is a nice fact.", "Good to know.", "Thanks for sharing that."])
    return {"q": q, "assistant_segments": [a],
            "labels": {"intent": 0, "integrity": 1, "critic": 0,
                       "template": rng.randrange(100), "verb": 0}}


def gen_correction(rng):
    op = rng.choice(list(VERBS))
    a, b = rng.randint(2, 99), rng.randint(2, 99)
    if op == "-" and b > a:
        a, b = b, a
    r = calc(a, op, b)
    w = rng.choice(VERBS[op])
    wrong = r + rng.choice([-2, -1, 1, 2, 10]) * rng.choice([1, 10])
    while wrong == r:
        wrong = r + 10
    q = f"I think {a} {w} {b} is {wrong}."
    seg = f"<CALC><EXPR>{a}{op}{b}</EXPR></CALC><|tool|><RESULT>{r}</RESULT>"
    ans = rng.choice(ANSWERS).format(a=a, w=w, b=b, r=r) + f" Not {wrong}."
    return {"q": q, "assistant_segments": [seg, ans],
            "labels": {"intent": 1, "integrity": 1,
                       "critic": 1,
                       "template": rng.randrange(100),
                       "verb": VERBS[op].index(w) * 4 + list(VERBS).index(op)}}


def gen_one(seed):
    rng = random.Random(seed)
    x = rng.random()
    if x < 0.70:
        s = gen_arith(rng)
    elif x < 0.80:
        s = gen_clarify(rng)
    elif x < 0.86:
        s = gen_cannot(rng)
    elif x < 0.91:
        s = gen_negative(rng)
    else:
        s = gen_correction(rng)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--total", type=int, default=100_000)
    ap.add_argument("--out", default="data/sft.jsonl")
    ap.add_argument("--procs", type=int, default=32)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with Pool(args.procs) as p:
        samples = p.map(gen_one, range(10_000, 10_000 + args.total), chunksize=256)
    with open(args.out, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    print("wrote", len(samples), "samples to", args.out)


if __name__ == "__main__":
    main()
