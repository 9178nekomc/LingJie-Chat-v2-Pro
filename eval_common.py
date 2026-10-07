"""Model-agnostic 3-way evaluation: LingJie v2 (new) vs Nova/LingJie v1 (old)
vs EleutherAI/pythia-70m (size-matched English baseline).

Same test distribution as eval.py (gen_sft generators, fixed seeds), scored
without assuming tool-call markup:
  arith      : last number in reply equals expected (first <EXPR> of sample)
  multi      : two-step sample, final value correct
  clarify    : reply asks for the missing information
  cannot     : reply refuses / states limitation
  neg        : no fabricated (or wrong) equation for a non-arithmetic query
  correction : reply contains the recomputed correct value
"""
import re
import json
import random
import re as _re
import importlib.util as _ilu


def _load(name, path):
    spec = _ilu.spec_from_file_location(name, path)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_gs = _load("lingjie_gen_sft", "/data/coding/lingjie/gen_sft.py")
gen_arith, gen_clarify, gen_cannot = _gs.gen_arith, _gs.gen_clarify, _gs.gen_cannot
gen_negative, gen_correction = _gs.gen_negative, _gs.gen_correction

EXPR_RE = _re.compile(r"<CALC><EXPR>(.*?)</EXPR></CALC>")
NUM_RE = _re.compile(r"-?\d+(?:\.\d+)?")


def normalize_for_eval(text):
    text = _re.sub(r"(?<=[0-9+\-*/=(),.:;?])[ ]+(?=[0-9+\-*/=(),.:;?])", "", text)
    text = _re.sub(r"(?<=[0-9+\-*/=()])[ ]+(?=</)", "", text)
    text = _re.sub(r"(?<=>)[ ]+(?=[0-9+\-*/=()])", "", text)
    return text


def safe_eval(expr):
    expr = normalize_for_eval(expr.strip())
    if not expr or _re.search(r"[^0-9+\-*/(). ]", expr):
        return None
    compact = expr.replace(" ", "")
    if not _re.fullmatch(r"[\d.+\-*/()]+", compact):
        return None
    if compact.count("(") != compact.count(")"):
        return None
    try:
        val = eval(compact, {"__builtins__": {}}, {})
    except ZeroDivisionError:
        return "undefined"
    except Exception:
        return None
    if isinstance(val, float):
        if abs(val) > 1e15:
            return None
        return str(int(val)) if val.is_integer() else str(round(val, 4))
    return str(val)

NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")
CLARIFY_MARK = re.compile(
    r"(which|provide|specify|tell me|what number|how many|confirm|"
    r"missing|need more|need to know|please give|could you give|\?)", re.I)
CANNOT_MARK = re.compile(
    r"(cannot|can't|can not|unable|only (help|handle|support)|sorry|"
    r"not able|up to six|too (big|large)|don't have)", re.I)
EQ_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*([+\-*/x×÷])\s*(-?\d+(?:\.\d+)?)\s*=\s*(-?\d+(?:\.\d+)?)")


def last_number(text):
    nums = NUM_RE.findall(text)
    return nums[-1] if nums else None


def close(a, b):
    try:
        return abs(float(a) - float(b)) < 1e-4
    except (TypeError, ValueError):
        return False


def score(kind, s, reply):
    r = reply or ""
    if kind == "arith":
        m = EXPR_RE.search(s["assistant_segments"][0])
        want = safe_eval(m.group(1)) if m else None
        got = last_number(r.split(" not ")[0]) if want is not None else None
        return got is not None and want is not None and close(got, want)
    if kind == "multi":
        exprs = EXPR_RE.findall(s["assistant_segments"][0])
        if len(exprs) < 2:
            return False
        want = safe_eval(exprs[-1])
        got = last_number(r)
        return got is not None and want is not None and close(got, want)
    if kind == "clarify":
        return bool(CLARIFY_MARK.search(r))
    if kind == "cannot":
        return bool(CANNOT_MARK.search(r))
    if kind == "neg":
        for m in EQ_RE.finditer(r):
            a, op, b, c = m.groups()
            ops = {"+": float(a) + float(b), "-": float(a) - float(b),
                   "*": float(a) * float(b), "x": float(a) * float(b),
                   "×": float(a) * float(b), "/": float(a) / float(b) if float(b) else None,
                   "÷": float(a) / float(b) if float(b) else None}
            want = ops[op]
            if want is None or not close(c, want):
                return False  # fabricated/wrong equation
        return True
    if kind == "correction":
        m = EXPR_RE.search(s["assistant_segments"][0])
        want = safe_eval(m.group(1)) if m else None
        if want is None:
            return False
        return any(close(n, want) for n in NUM_RE.findall(r))
    raise ValueError(kind)


def make_testset(n=400, seed=999):
    """Same distribution as eval.py; returns list of (kind, sample).
    Two-step samples get their question rewritten as a proper compound
    expression so all models are judged on the same chained task."""
    rng = random.Random(seed)
    out = []
    for i in range(n):
        x = (i % 10) / 10
        if x < 0.5:
            s = gen_arith(rng)
            if s["assistant_segments"][0].count("<CALC>") >= 2:
                exprs = EXPR_RE.findall(s["assistant_segments"][0])
                e1, e2 = exprs[0], exprs[-1]
                m2 = re.match(r"(-?\d+(?:\.\d+)?)([+\-*/])(.+)", e2)
                assert m2, e2
                _, op2, c2 = m2.groups()
                s["q"] = f"What is ({e1}){op2}{c2}?"
                kind = "multi"
            else:
                kind = "arith"
        elif x < 0.6:
            s, kind = gen_clarify(rng), "clarify"
        elif x < 0.7:
            s, kind = gen_cannot(rng), "cannot"
        elif x < 0.8:
            s, kind = gen_negative(rng), "neg"
        else:
            s, kind = gen_correction(rng), "correction"
        out.append((kind, s))
    return out


def run_eval(name, answer_fn, testset):
    stats, fails = {}, []
    for kind, s in testset:
        stats[kind] = stats.get(kind, [0, 0])
        stats[kind][1] += 1
        try:
            reply = answer_fn(s["q"])
        except Exception as e:
            reply = ""
            print(f"[{name}] ERROR on {s['q']!r}: {e}", flush=True)
        ok = score(kind, s, reply)
        stats[kind][0] += int(ok)
        if not ok and len(fails) < 12:
            fails.append((kind, s["q"], (reply or "")[:110]))
    print(f"\n===== {name} =====")
    for k, (c, t) in stats.items():
        print(f"  {k:11s} {c}/{t} = {c/t:.3f}")
    tot_c = sum(c for c, _ in stats.values())
    tot_t = sum(t for _, t in stats.values())
    print(f"  {'TOTAL':11s} {tot_c}/{tot_t} = {tot_c/tot_t:.3f}")
    for kind, q, r in fails:
        print(f"  FAIL[{kind}] {q}\n      -> {r!r}")
    return {k: [c, t] for k, (c, t) in stats.items()}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--who", required=True, choices=["new", "old", "pythia"])
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    testset = make_testset(args.n)

    if args.who == "new":
        from infer import Pipeline, SpellDict
        words = [w.strip() for w in open("/data/coding/lingjie/data/spell_dict.txt")]
        pipe = Pipeline("/data/coding/lingjie/ckpt/sft/last.pt",
                        "/data/coding/lingjie/tokenizer/lingjie_tokenizer.json",
                        SpellDict(words))
        fn = lambda q: pipe.answer(q, minimal=True)
    elif args.who == "old":
        sys.path.insert(0, "/data/lingjie_old")
        from harness.engine import Engine
        from minimal_harness import (math_assist, profanity_check,
                                     PROFANITY_REPLY, extract_v1_tool)
        from infer import safe_eval, normalize_for_eval
        eng = Engine()

        def fn(q, _max_tok=96):
            if profanity_check(q):
                return PROFANITY_REPLY
            m = math_assist(q, safe_eval, normalize_for_eval)
            if m:
                return m
            reply, calls = "", 0
            while calls < 3:
                ids = eng.build_context([], q if not reply else q + " " + reply)
                gen = "".join(eng.stream_generate(ids, max_new_tokens=_max_tok,
                                                  temperature=0.05, top_k=None))
                reply = (reply + " " + gen).strip()
                hit = extract_v1_tool(gen)
                if hit:
                    reply = reply.replace(hit[0].group(0), " " + hit[1] + " ")
                    calls += 1
                    continue
                break
            return reply
    else:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import os
        path = None
        for root, dirs, files in os.walk("/data/coding/lingjie/hf_cache"):
            if "model.safetensors" in files and "pythia" in root:
                path = root
        assert path, "pythia not downloaded"
        tok = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(path).cuda().eval()

        def fn(q):
            inputs = tok(f"Question: {q}\nAnswer:", return_tensors="pt").to("cuda")
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=48, do_sample=False,
                                     pad_token_id=tok.eos_token_id)
            return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    res = run_eval("LingJie-Chat-v2-Pro (56M)" if args.who == "new" else
                   "LingJie-Chat-v1-Flash (8M)" if args.who == "old" else
                   "pythia-70m (bare)", fn, testset)
    if args.out:
        json.dump(res, open(args.out, "w"), indent=2)


# ---------------- multi-turn (pronoun / follow-up) benchmark ----------------

def make_multiturn(n=50, seed=4242):
    """Dialogues where later turns refer to earlier results via pronouns:
    turn1 sets two numbers, turn2 asks 'their sum', turn3 chains 'that times c'."""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        a, b = rng.randint(2, 99), rng.randint(2, 99)
        c = rng.randint(2, 9)
        turns = [f"My two numbers are {a} and {b}.",
                 "What is their sum?",
                 f"How much is that times {c}?"]
        exp = [a + b, round((a + b) * c, 4)]
        out.append({"turns": turns, "expected": exp})
    return out


def run_multiturn(conv_fn, dialogues):
    """conv_fn(history, q) -> reply; history = [{'q':..,'a':..}]"""
    total = ok = 0
    for d in dialogues:
        hist = []
        for j, q in enumerate(d["turns"]):
            reply = conv_fn(hist, q) or ""
            if j >= 1:
                total += 1
                want = d["expected"][j - 1]
                nums = NUM_RE.findall((reply or "").split(" not ")[0])
                if nums and close(nums[-1], want):
                    ok += 1
            hist.append({"q": q, "a": reply.strip()})
    return ok, total


def measure_speed(gen64_fn, repeats=3):
    """gen64_fn() -> (n_new_tokens, seconds). Returns best tok/s."""
    best = 0.0
    for _ in range(repeats):
        n, t = gen64_fn()
        if t > 0:
            best = max(best, n / t)
    return best
