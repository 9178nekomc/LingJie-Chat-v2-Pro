"""Held-out evaluation for LingJie: arithmetic exact-answer accuracy,
protocol validity, and boundary behavior (clarify/cannot/idk)."""
import argparse
import json
import re
import random

from infer import Pipeline, SpellDict, EXPR_RE, safe_eval, normalize_for_eval
from gen_sft import gen_arith, gen_clarify, gen_cannot, gen_negative, gen_correction


def extract_answer(text):
    """Pull the final numeric answer from the model reply."""
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    return nums[-1] if nums else None


def expected_answer(a, op, b):
    if op == "+":
        r = a + b
    elif op == "-":
        r = a - b
    elif op == "*":
        r = a * b
    else:
        r = a / b
        return str(int(r)) if r == int(r) else str(round(r, 4))
    return str(int(r)) if isinstance(r, float) and r.is_integer() else str(r)


def expected_from_sample(s):
    """Ground truth = the FIRST <EXPR> in the sample (that is what the question
    asks); two-step samples have their final answer in a later step, which is
    not what the question asked, and correction answers end with 'not X.'."""
    m = EXPR_RE.search(s["assistant_segments"][0])
    if not m:
        return None
    val = safe_eval(m.group(1))
    return val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpt/sft/last.pt")
    ap.add_argument("--tokenizer", default="tokenizer/lingjie_tokenizer.json")
    ap.add_argument("--dict", default="data/spell_dict.txt")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=999)
    args = ap.parse_args()

    words = [w.strip() for w in open(args.dict)]
    spell = SpellDict(words)
    pipe = Pipeline(args.ckpt, args.tokenizer, spell)

    rng = random.Random(args.seed)
    stats = {"arith_total": 0, "arith_correct": 0, "protocol_valid": 0,
             "clarify_total": 0, "clarify_ok": 0,
             "cannot_total": 0, "cannot_ok": 0,
             "idk_total": 0, "idk_ok": 0,
             "neg_total": 0, "neg_ok": 0,
             "corr_total": 0, "corr_ok": 0,
             "multi_total": 0, "multi_ok": 0}
    fails = []

    for i in range(args.n):
        x = (i % 10) / 10
        if x < 0.5:
            s = gen_arith(rng)
            if "<|tool|><RESULT>" in s["assistant_segments"][0] and s["assistant_segments"][0].count("<CALC>") >= 2:
                stats["multi_total"] += 1
            stats["arith_total"] += 1
            ans = pipe.answer(s["q"])
            got = extract_answer(ans)
            want = expected_from_sample(s)
            if "<CALC>" in ans or "<RESULT>" in ans:
                pass  # markup leaked into final reply
            else:
                stats["protocol_valid"] += 1
            if got is not None and want is not None and abs(float(got) - float(want)) < 1e-6:
                stats["arith_correct"] += 1
            elif len(fails) < 15:
                fails.append(("arith", s["q"], ans, want))
            if s["assistant_segments"][0].count("<CALC>") >= 2:
                if ans.count("<CALC>") + ans.count("CALC") == 0:
                    stats["multi_ok"] += 1
        elif x < 0.6:
            s = gen_clarify(rng)
            stats["clarify_total"] += 1
            ans = pipe.answer(s["q"])
            if "<NEED_INFO>" in ans:
                stats["clarify_ok"] += 1
            elif len(fails) < 15:
                fails.append(("clarify", s["q"], ans, "NEED_INFO"))
        elif x < 0.7:
            s = gen_cannot(rng)
            stats["cannot_total"] += 1
            ans = pipe.answer(s["q"])
            if "<CANNOT>" in ans:
                stats["cannot_ok"] += 1
            elif len(fails) < 15:
                fails.append(("cannot", s["q"], ans, "CANNOT"))
        elif x < 0.8:
            s = gen_negative(rng)
            stats["neg_total"] += 1
            ans = pipe.answer(s["q"])
            if "<CALC>" not in ans and "<NEED_INFO>" not in ans and "<CANNOT>" not in ans:
                stats["neg_ok"] += 1
            elif len(fails) < 15:
                fails.append(("neg", s["q"], ans, "plain"))
        else:
            s = gen_correction(rng)
            stats["corr_total"] += 1
            ans = pipe.answer(s["q"])
            want = expected_from_sample(s)
            got = extract_answer(ans.split(" not ")[0])
            if got is not None and want is not None and abs(float(got) - float(want)) < 1e-6:
                stats["corr_ok"] += 1
            elif len(fails) < 15:
                fails.append(("corr", s["q"], ans, want))

    print(json.dumps(stats, indent=2))
    arith_acc = stats["arith_correct"] / max(1, stats["arith_total"])
    print(f"\nARITHMETIC EXACT ACCURACY: {arith_acc:.3f}")
    if fails:
        print("\nSample failures:")
        for kind, q, ans, want in fails[:10]:
            print(f"  [{kind}] {q}\n      got: {ans!r}\n      want: {want}")


if __name__ == "__main__":
    main()
