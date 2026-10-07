"""LingJie inference pipeline (spec section 6).

user input
  -> rules layer (deterministic): long numbers, missing numbers, spelling,
     arithmetic-correction, unsupported ops
  -> model generation (only for what rules pass through), with a tool-call loop:
     on <|tool|>, evaluate the pending <CALC><EXPR>...</EXPR>, splice
     <RESULT>...</RESULT><|assistant|> and continue; max 3 tool calls.
"""
import argparse
import math
import os
import re
from collections import Counter, defaultdict

import torch

from build_tokenizer import preprocess_for_bpe, normalize_for_eval

U, A, T = "<|user|>", "<|assistant|>", "<|tool|>"
NUM_RE = re.compile(r"\d[\d,]*")
ARITH_WORDS = re.compile(
    r"\b(plus|minus|times|divided|multiply|add|subtract|calculate|compute|sum|"
    r"product|quotient|over|equals?)\b", re.I)
CORRECTION_RE = re.compile(
    r"(i think|is it|am i right|right\?|someone said|actually,? i|didn'?t you say|"
    r"isn'?t it)\b.*?\b(\d[\d,]*)\b.*?\b(plus|minus|times|multiplied by|multiplied with|"
    r"divided by|over|quotient of|split by|shared by|added to|sum of|total of|and|"
    r"subtracted by|take away|reduced by|less|product of|twice by)\b.*?\b(\d[\d,]*)\b",
    re.I)
OP_MAP = {
    "plus": "+", "and": "+", "added to": "+", "sum of": "+", "total of": "+",
    "minus": "-", "less": "-", "subtracted by": "-", "take away": "-",
    "reduced by": "-",
    "times": "*", "multiplied by": "*", "multiplied with": "*",
    "product of": "*", "twice by": "*",
    "divided by": "/", "over": "/", "quotient of": "/", "split by": "/",
    "shared by": "/",
}
EXPR_FORBIDDEN = re.compile(r"[^0-9+\-*/(). ]")


def to_int(s):
    return int(s.replace(",", ""))


# ---------------------------------------------------------------- spell dict
def edit_distance(a, b, cap=3):
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        if min(cur) > cap:
            return cap + 1
        prev = cur
    return prev[-1]


class SpellDict:
    def __init__(self, words):
        self.by_len = defaultdict(list)
        for w in words:
            self.by_len[len(w)].append(w)

    def known(self, w):
        return w in self.by_len.get(len(w), [])

    def suggest(self, w, max_d=2):
        # length-bucketed scan (spec 6.3): O(~5k) instead of O(100k)
        best = (max_d + 1, None)
        for L in range(max(1, len(w) - max_d), len(w) + max_d + 1):
            for cand in self.by_len.get(L, ()):  # buckets
                d = edit_distance(w.lower(), cand, max_d)
                if d < best[0]:
                    best = (d, cand)
                    if d == 0:
                        return cand
        return best[1]


def build_dict(story_files, top_n=100_000):
    cnt = Counter()
    for path in story_files:
        with open(path) as f:
            for line in f:
                cnt.update(re.findall(r"[a-zA-Z']+", line.lower()))
    return SpellDict([w for w, _ in cnt.most_common(top_n)])


# ---------------------------------------------------------------- rules layer
def rules_check(q, spell: SpellDict):
    """Return (handled, response) — deterministic replies per spec 6.2."""
    nums = [to_int(n) for n in NUM_RE.findall(q)]
    for n in nums:
        if n > 999_999:
            return True, "<CANNOT> I can only handle numbers with up to 6 digits."
    if re.search(r"\b(sqrt|sin|cos|tan|log)\b", q, re.I) or "%" in q or "^" in q:
        return True, "<CANNOT> I can only help with +, -, * and /."
    # compound arithmetic expression: strip question filler; if the rest is a
    # pure arithmetic expression, compute it deterministically (spec 6.2:
    # what can be decided by rules is decided by rules)
    filler = r"(what is|how much is|calculate|compute|work out|please|for me|[?:])"
    expr_cand = re.sub(filler, " ", q, flags=re.I).strip()
    if re.fullmatch(r"[\d.,+\-*/() ]+", expr_cand) and re.search(r"[+\-*/]", expr_cand) \
            and len(nums) >= 2:
        expr = normalize_for_eval(expr_cand.replace(",", ""))
        r = safe_eval(expr)
        if r is not None and r != "undefined":
            return True, (f"<CALC><EXPR>{expr}</EXPR></CALC><RESULT>{r}</RESULT> "
                          f"{expr} equals {r}.")
    has_arith = bool(ARITH_WORDS.search(q)) and not re.search(r"\b(story|joke|poem)\b", q, re.I)
    if has_arith and not nums:
        return True, "<NEED_INFO> Please provide the numbers for the calculation."
    if has_arith and len(nums) < 2 and "plus" not in q.lower():
        # e.g. "subtract 5" — missing second operand
        if not re.search(r"\b(what is|how much|calculate|compute)\b.*\b\d+\b\s*(plus|times)", q, re.I):
            pass  # model may handle "x2" style; leave to model
    m = CORRECTION_RE.search(q)
    if m and len(nums) >= 3:
        op_word = m.group(3).lower()
        op = OP_MAP[op_word]
        a, b, claimed = nums[0], nums[1], nums[2]
        try:
            r = {"+": a + b, "-": a - b, "*": a * b,
                 "/": a / b if b else None}[op]
        except Exception:
            r = None
        if r is not None:
            if isinstance(r, float) and not r.is_integer():
                r_s = str(round(r, 4))
            else:
                r = int(r)
                r_s = str(r)
            return True, (f"<CALC><EXPR>{a}{op}{b}</EXPR></CALC><RESULT>{r_s}</RESULT> "
                          f"Let me check. {a} {op_word} {b} equals {r_s}, not {claimed}.")
    # spelling: first long non-dictionary word triggers a confirm.
    # Command words are whitelisted: the TinyStories dictionary is a children's
    # vocabulary and lacks adult words like "compute".
    command_words = {
        "compute", "calculate", "multiply", "subtract", "divide", "division",
        "quotient", "product", "result", "answer", "number", "numbers",
        "equals", "math", "arithmetic", "please", "provide", "confirm",
        "second", "first", "order", "instead", "amount", "total", "value",
        "operator", "missing", "secondly",
    }
    for w in re.findall(r"[a-zA-Z']+", q):
        lw = w.lower()
        if len(lw) >= 5 and lw not in command_words and not spell.known(lw):
            s = spell.suggest(w.lower())
            if s:
                return True, (f"<NEED_INFO> Did you mean '{s}' instead of '{w}'? "
                              f"Please confirm so I can help.")
    return False, None


# ---------------------------------------------------------------- tool loop
EXPR_RE = re.compile(r"<CALC><EXPR>(.*?)</EXPR></CALC>")


def safe_eval(expr):
    expr = normalize_for_eval(expr.strip())
    if not expr or EXPR_FORBIDDEN.search(expr):
        return None
    compact = expr.replace(" ", "")
    # allow parentheses (compound expressions): charset is digits/ops/dots/parens
    # only — no letters means no names, so eval cannot reach anything else
    if not re.fullmatch(r"[\d.+\-*/()]+", compact):
        return None
    if compact.count("(") != compact.count(")"):
        return None
    try:
        val = eval(compact, {"__builtins__": {}}, {})  # gated by charset above
    except ZeroDivisionError:
        return "undefined"
    except Exception:
        return None
    if isinstance(val, float):
        if abs(val) > 1e15:
            return None
        return str(int(val)) if val.is_integer() else str(round(val, 4))
    return str(val)


class Pipeline:
    def __init__(self, ckpt, tokenizer_path, spell, device="cuda"):
        from model import LingJie, LingJieConfig
        from tokenizers import Tokenizer
        state = torch.load(ckpt, map_location=device, weights_only=False)
        self.cfg = LingJieConfig(**state["config"])
        self.model = LingJie(self.cfg).to(device).eval()
        self.model.load_state_dict(state["model"])
        self.tok = Tokenizer.from_file(tokenizer_path)
        self.bos = self.tok.token_to_id("<BOS>")
        self.eos = self.tok.token_to_id("<EOS>")
        self.device = device
        self.spell = spell

    def encode(self, t):
        return self.tok.encode(preprocess_for_bpe(t), add_special_tokens=False).ids

    def decode(self, ids):
        return normalize_for_eval(self.tok.decode(ids, skip_special_tokens=False))

    @torch.no_grad()
    def answer(self, q, max_new_tokens=160, temperature=0.0, minimal=False, history=None):
        from minimal_harness import math_assist, profanity_check, PROFANITY_REPLY
        if minimal:
            # shared minimal harness: math assist + profanity only
            if profanity_check(q):
                return PROFANITY_REPLY
            m = math_assist(q, safe_eval, normalize_for_eval)
            if m:
                return m
        else:
            handled, resp = rules_check(q, self.spell)
            if handled:
                return resp
        # prompt must end with <|assistant|>, matching the SFT format
        ids = [self.bos]
        if history:
            for turn in history:
                ids += self.encode(U + turn["q"]) + self.encode(A + turn["a"])
        ids += self.encode(U + q) + self.encode(A)
        tool_calls = 0
        while tool_calls <= 3:
            ctx = torch.tensor([ids], device=self.device)
            out_ids = self.model.generate(
                ctx, max_new_tokens=max_new_tokens, temperature=temperature,
                eos_id=self.eos)[0].tolist()[len(ids):]
            text = self.decode(out_ids)
            # find an unclosed <CALC><EXPR>..</EXPR> awaiting a result
            m = None
            for m_ in EXPR_RE.finditer(text):
                m = m_
            if m is not None and tool_calls < 3 and "<RESULT>" not in text[m.end():m.end() + 12]:
                expr = m.group(1)
                r = safe_eval(expr)
                if r is None:
                    return f"<UNK_EXPR> I could not parse the expression '{expr}'."
                if r == "undefined":
                    return "<CANNOT> Division by zero is undefined."
                # cut the generated tokens at the </CALC> token (char offsets
                # from the decoded text must not be used to slice token lists)
                close_id = self.tok.token_to_id("</CALC>")
                cut = (i for i in range(len(out_ids) - 1, -1, -1) if out_ids[i] == close_id)
                try:
                    n_cut = next(cut) + 1
                except StopIteration:
                    n_cut = len(out_ids)
                ids = ids + out_ids[:n_cut] + self.encode(f"<RESULT>{r}</RESULT>" + A)
                tool_calls += 1
                continue
            return self._final(text)
        return "<CANNOT> Too many calculation steps."

    @staticmethod
    def _final(text):
        # strip protocol markup; the user-facing answer is plain text
        text = re.sub(r"<CALC>.*?</CALC>", " ", text, flags=re.S)
        text = re.sub(r"<RESULT>.*?</RESULT>", " ", text, flags=re.S)
        text = re.sub(r"<CALC>.*?(?=<|$)", " ", text, flags=re.S)
        for tag in ("<|tool|>", "<|assistant|>", "<|user|>", "<|system|>", "<|endoftext|>", "<EOS>"):
            text = text.replace(tag, " ")
        return re.sub(r"\s+", " ", text).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpt/sft/last.pt")
    ap.add_argument("--tokenizer", default="tokenizer/lingjie_tokenizer.json")
    ap.add_argument("--dict", default="data/spell_dict.txt")
    ap.add_argument("--interactive", action="store_true")
    ap.add_argument("--questions", nargs="*", default=[
        "What is 1234 plus 567?",
        "How much is 999 times 999?",
        "Calculate 48 divided by 6.",
        "What is 12345678 plus 1?",
        "Add them.",
        "What is the capital of France?",
        "I think 5 plus 3 is 9.",
        "What is 12 sqruared?",
    ])
    args = ap.parse_args()
    words = [w.strip() for w in open(args.dict)]
    spell = SpellDict(words)
    pipe = Pipeline(args.ckpt, args.tokenizer, spell)
    if args.interactive:
        while True:
            try:
                q = input("\n> ").strip()
            except EOFError:
                break
            if not q:
                continue
            print("LingJie:", pipe.answer(q))
    else:
        for q in args.questions:
            print(f"Q: {q}\nA: {pipe.answer(q)}\n", flush=True)


if __name__ == "__main__":
    main()
