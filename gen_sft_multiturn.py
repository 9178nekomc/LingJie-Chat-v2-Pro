"""Multi-turn SFT data for LingJie: teaches dialog-state handling.

Archetypes:
  A. number setup (ack only, NO eager computation) -> pronoun follow-ups
     (their sum / product / difference / quotient)
  B. chained anaphora: "that times c" / "it minus c" / "half of that"
  C. ellipsis follow-up: "And 82?" (reuse previous template with new operand)
  D. substitution: "Now use 100 instead of 55" then re-ask
  E. interference: unrelated CANNOT question mid-dialogue, then back to task
  F. clarification in context: pronoun with missing referent -> NEED_INFO

Sample format: {"turns": [ {q, assistant_segments, labels}, ... ]}
The final turn's labels are the pooled head labels for the sample.
"""
import argparse
import json
import os
import random
from multiprocessing import Pool

ACK_TPL = [
    "Got it, the numbers are {a} and {b}.",
    "Noted: {a} and {b}.",
    "Sure — {a} and {b}.",
    "OK, I have {a} and {b}.",
    "The numbers are {a} and {b}. Understood.",
]
ANSWERS = [
    "{x} equals {r}.",
    "The answer is {r}.",
    "That is {r}.",
    "The result is {r}.",
    "So the answer is {r}.",
    "It comes to {r}.",
    "That works out to {r}.",
    "The final result is {r}.",
]

VERBS4 = {"+": ("sum", "plus"), "-": ("difference", "minus"),
          "*": ("product", "times"), "/": ("quotient", "divided by")}


def calc(a, op, b):
    if op == "+":
        r = a + b
    elif op == "-":
        return a - b
    elif op == "*":
        return a * b
    else:
        r = a / b
        return int(r) if r == int(r) else round(r, 4)
    return r


def calc_turn(a, op, b, rng, phrase=None):
    """Assistant turns for one arithmetic request."""
    r = calc(a, op, b)
    expr = f"{a}{op}{b}"
    seg = f"<CALC><EXPR>{expr}</EXPR></CALC><|tool|><RESULT>{r}</RESULT>"
    if phrase == "pronoun":
        q_word = {"+": "sum", "-": "difference", "*": "product", "/": "quotient"}[op]
        ans = rng.choice(ANSWERS).format(x=f"The {q_word} of {a} and {b}", r=r)
    else:
        w = VERBS4[op][1]
        ans = rng.choice(ANSWERS).format(x=f"{a} {w} {b}", r=r)
    labels = {"intent": 1, "integrity": 1, "critic": 1,
              "template": rng.randrange(100),
              "verb": list(VERBS4).index(op) % 20}
    return {"q": None, "assistant_segments": [seg, ans], "labels": labels, "r": r}


def ack_turn(a, b, rng):
    return {"q": None, "assistant_segments": [rng.choice(ACK_TPL).format(a=a, b=b)],
            "labels": {"intent": 0, "integrity": 1, "critic": 0,
                       "template": rng.randrange(100), "verb": 0}}


def archetype_ab(rng):
    """A+B: setup (ack only) -> their X -> that op c."""
    a, b = rng.randint(2, 99), rng.randint(2, 99)
    op = rng.choice(list(VERBS4))
    c = rng.randint(2, 9)
    turns = [{**ack_turn(a, b, rng), "q": f"My two numbers are {a} and {b}."}]
    t2 = calc_turn(a, op, b, rng, phrase="pronoun")
    t2["q"] = f"What is their {VERBS4[op][0]}?"
    r2 = t2["r"]
    turns.append(t2)
    op2 = rng.choice(["*", "-", "+"])
    w2 = {"*": f"times {c}", "-": f"minus {c}", "+": f"plus {c}"}[op2]
    t3 = calc_turn(r2, op2, c, rng)
    t3["q"] = f"How much is that {w2}?"
    turns.append(t3)
    return turns


def archetype_c(rng):
    """C: ellipsis — full question then bare 'And X?' reusing the template."""
    a = rng.randint(2, 99)
    pairs = rng.sample(range(2, 99), 2)
    b, c = pairs
    op = rng.choice(list(VERBS4))
    w = VERBS4[op][1]
    t1 = calc_turn(a, op, b, rng)
    t1["q"] = f"What is {a} {w} {b}?"
    turns = [t1]
    t2 = calc_turn(a, op, c, rng)
    t2["q"] = f"And {c}?"
    turns.append(t2)
    return turns


def archetype_d(rng):
    """D: substitution — swap one number, re-ask."""
    a, b = rng.randint(20, 99), rng.randint(2, 19)
    op = rng.choice(["+", "*"])
    w = {"+": "plus", "*": "times"}[op]
    t1 = calc_turn(a, op, b, rng)
    t1["q"] = f"What is {a} {w} {b}?"
    turns = [t1]
    a2 = rng.randint(20, 99)
    while a2 == a:
        a2 = rng.randint(20, 99)
    t2 = calc_turn(a2, op, b, rng)
    t2["q"] = f"Now use {a2} instead of {a}. What is the result?"
    turns.append(t2)
    return turns


def archetype_e(rng):
    """E: interference — unrelated CANNOT question mid-dialogue, then back."""
    a, b = rng.randint(2, 99), rng.randint(2, 99)
    op = rng.choice(["+", "*"])
    w = {"+": "plus", "*": "times"}[op]
    t1 = {**ack_turn(a, b, rng), "q": f"My two numbers are {a} and {b}."}
    t2 = {"q": "What is the capital of France?",
          "assistant_segments": ["<CANNOT> I can only help with arithmetic."],
          "labels": {"intent": 0, "integrity": 1, "critic": 0,
                     "template": rng.randrange(100), "verb": 0}}
    t3 = calc_turn(a, op, b, rng, phrase="pronoun")
    t3["q"] = f"What is their {VERBS4[op][0]}?"
    return [t1, t2, t3]


def archetype_f(rng):
    """F: pronoun with no referent -> NEED_INFO, then with referent -> answer."""
    b = rng.randint(2, 99)
    t1 = {"q": "What is their sum?",
          "assistant_segments": ["<NEED_INFO> Which numbers would you like me to add?"],
          "labels": {"intent": 1, "integrity": 0, "critic": 0,
                     "template": rng.randrange(100), "verb": 0}}
    a = rng.randint(2, 99)
    t2 = {**ack_turn(a, b, rng), "q": f"Oh sorry — {a} and {b}."}
    t3 = calc_turn(a, "+", b, rng, phrase="pronoun")
    t3["q"] = "What is their sum?"
    return [t1, t2, t3]


ARCHETYPES = [archetype_ab, archetype_ab, archetype_ab,  # weight A+B heaviest
              archetype_c, archetype_c, archetype_d, archetype_e, archetype_f]


def gen_one(seed):
    rng = random.Random(seed)
    return {"turns": rng.choice(ARCHETYPES)(rng)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--total", type=int, default=40_000)
    ap.add_argument("--out", default="data/sft_multiturn.jsonl")
    ap.add_argument("--procs", type=int, default=32)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with Pool(args.procs) as p:
        samples = p.map(gen_one, range(900_000, 900_000 + args.total), chunksize=256)
    with open(args.out, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    print("wrote", len(samples), "multi-turn samples to", args.out)


if __name__ == "__main__":
    main()
