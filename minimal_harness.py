"""Minimal shared harness for the LingJie v1/v2 comparison.

Only two assist mechanisms (per eval protocol):
  1. math assist: deterministic computation of explicit arithmetic
     (natural-language ops normalized to symbols; compound expressions OK),
     plus executing tool calls the model itself emits
  2. profanity handling: refuse and redirect
Everything else is raw model ability.
"""
import re

PROFANITY_RE = re.compile(
    r"\b(damn|hell|shit|fuck|fucking|bitch|asshole|bastard|crap|dick|"
    r"ass\b|stupid|idiot|dumb)\b", re.I)

FILLER = r"(what is|what's|how much is|calculate|compute|work out|tell me|i need the result of|do the math|can you|please|for me|[?:])"
OP_WORDS = {
    "plus": "+", "and": "+", "added to": "+", "sum of": "+", "total of": "+",
    "minus": "-", "less": "-", "subtracted by": "-", "take away": "-",
    "reduced by": "-",
    "times": "*", "multiplied by": "*", "multiplied with": "*",
    "product of": "*", "twice by": "*",
    "divided by": "/", "over": "/", "quotient of": "/", "split by": "/",
    "shared by": "/",
}

PROFANITY_REPLY = ("I won't respond to that. Please keep the conversation "
                   "friendly and I'll be happy to help with math.")


def profanity_check(q):
    return PROFANITY_RE.search(q) is not None


def math_assist(q, safe_eval, normalize):
    """If the question is an explicit arithmetic request, compute it.
    Returns the reply text or None (not a math question).
    Tool contract (shared by both LingJie models): numbers up to 6 digits,
    + - * / only, no division by zero."""
    if profanity_check(q):
        return PROFANITY_REPLY
    nums = re.findall(r"\d[\d,]*", q)
    if any(int(n.replace(",", "")) > 999_999 for n in nums):
        return "<CANNOT> I can only handle numbers with up to 6 digits."
    if re.search(r"\b(sqrt|sin|cos|tan|log)\b", q, re.I) or "%" in q or "^" in q:
        return "<CANNOT> I can only help with +, -, * and /."
    expr_cand = re.sub(FILLER, " ", q, flags=re.I).strip()
    for word, sym in sorted(OP_WORDS.items(), key=lambda kv: -len(kv[0])):
        expr_cand = re.sub(rf"\b{word}\b", sym, expr_cand, flags=re.I)
    expr_cand = re.sub(r"\bby\b", " ", expr_cand, flags=re.I)
    nums = re.findall(r"-?\d[\d,]*", expr_cand)
    if re.fullmatch(r"[\d.,+\-*/() ]+", expr_cand) and re.search(r"[+\-*/]", expr_cand) \
            and len(nums) >= 2:
        expr = normalize(expr_cand.replace(",", ""))
        r = safe_eval(expr)
        if r is not None and r != "undefined":
            return f"{expr} equals {r}."
        if r == "undefined":
            return "<CANNOT> Division by zero is undefined."
    return None


def extract_v1_tool(text):
    m = re.search(r"<tool>\s*calc\s*</tool>\s*<args>\s*(.*?)\s*</args>", text, re.S)
    if not m:
        return None
    try:
        a_s, op, b_s = [x.strip() for x in m.group(1).split("|")]
        a = float(a_s) if "." in a_s else int(a_s)
        b = float(b_s) if "." in b_s else int(b_s)
        r = {"+": a + b, "-": a - b, "*": a * b, "/": a / b if b else None}[op]
        if isinstance(r, float):
            if r == int(r):
                r = int(r)
            else:
                r = round(r, 4)
        return m, f"{a_s} {OP_WORDS.get(op, op)} {b_s} equals {r}."
    except Exception:
        return None
