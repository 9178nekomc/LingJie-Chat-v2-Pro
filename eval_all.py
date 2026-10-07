"""Run the full 3-way comparison: 6 single-turn dims + multi-turn + speed."""
import json
import sys
import time

import eval_common as ec

RESULTS = {}

# ---------------- LingJie-Chat-v2-Pro ----------------
from infer import Pipeline, SpellDict, safe_eval, normalize_for_eval
words = [w.strip() for w in open("data/spell_dict.txt")]
pipe = Pipeline("ckpt/sft/last.pt", "tokenizer/lingjie_tokenizer.json", SpellDict(words))

def new_fn(q):
    return pipe.answer(q, minimal=True)

def new_conv(hist, q):
    return pipe.answer(q, minimal=True, history=[{"q": h["q"], "a": h["a"]} for h in hist])

def new_speed():
    import torch
    ids = [pipe.bos] + pipe.encode("<|user|>What is 24 plus 18?<|assistant|>")
    t0 = time.time()
    out = pipe.model.generate(torch.tensor([ids]).cuda(), 64, 0.0, eos_id=pipe.eos)
    dt = time.time() - t0
    return out.shape[1] - len(ids), dt

RESULTS["new"] = {"single": ec.run_eval("LingJie-Chat-v2-Pro (56M)", new_fn,
                                        ec.make_testset(400)),
                  "mt": ec.run_multiturn(new_conv, ec.make_multiturn(50)),
                  "speed": ec.measure_speed(new_speed)}

# ---------------- LingJie-Chat-v1-Flash (subprocess is impractical here,
# load in-process with isolated module names via runpy) ----------------
import importlib.util as ilu
spec = ilu.spec_from_file_location("old_engine_env", "/data/lingjie_old/eval_old_runner_parts.py")

# Instead: reuse the old runner via subprocess for single-turn, and do
# multi-turn + speed in a small in-process shim with sys.path confined.
import subprocess, os
old_single = json.load(open("export/eval_old.json"))  # already computed with final harness

# old multi-turn + speed: isolated process writing json
OLD_SHIM = r'''
import sys, json, time
sys.path.insert(0, "/data/lingjie_old")
sys.path.insert(1, "/data/coding/lingjie")
from eval_common import make_multiturn, run_multiturn, measure_speed, safe_eval, normalize_for_eval
from minimal_harness import math_assist, profanity_check, PROFANITY_REPLY
from harness.engine import Engine
eng = Engine()
def conv(hist, q):
    if profanity_check(q): return PROFANITY_REPLY
    m = math_assist(q, safe_eval, normalize_for_eval)
    if m: return m
    msgs = [{"role": "user" if i % 2 == 0 else "assistant", "content": t}
            for i, t in enumerate(sum(([h["q"], h["a"]] for h in hist), []))]
    res = eng.build_context(msgs, q)
    ids = res[0] if isinstance(res, tuple) else res
    reply = "".join(eng.stream_generate(ids, 96, temperature=0.05, top_k=None)).strip()
    return reply
def speed():
    ids = eng.build_context([], "What is 24 plus 18?")
    ids = ids[0] if isinstance(ids, tuple) else ids
    t0 = time.time()
    gen = "".join(eng.stream_generate(ids, 64, temperature=0.05, top_k=None))
    n = eng.last_stats.get("tokens", len(gen.split()))
    return n, time.time() - t0
out = {"mt": run_multiturn(conv, make_multiturn(50)), "speed": measure_speed(speed)}
json.dump(out, open("/data/coding/lingjie/export/old_extra.json", "w"))
'''
open("/tmp/old_extra.py", "w").write(OLD_SHIM)
subprocess.run([sys.executable, "/tmp/old_extra.py"], cwd="/data/lingjie_old", check=True)
extra = json.load(open("export/old_extra.json"))
RESULTS["old"] = {"single": old_single, "mt": extra["mt"], "speed": extra["speed"]}

# ---------------- pythia-70m (with the same minimal harness) ----------------
import torch, os
from transformers import AutoModelForCausalLM, AutoTokenizer
path = None
for root, dirs, files in os.walk("/data/coding/lingjie/hf_cache"):
    if "model.safetensors" in files and "pythia" in root:
        path = root
tok = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(path).cuda().eval()
from minimal_harness import math_assist, profanity_check, PROFANITY_REPLY

def py_fn(q):
    if profanity_check(q):
        return PROFANITY_REPLY
    m = math_assist(q, ec.safe_eval, ec.normalize_for_eval)
    if m:
        return m
    inputs = tok(f"Question: {q}\nAnswer:", return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=48, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

def py_conv(hist, q):
    if profanity_check(q):
        return PROFANITY_REPLY
    m = math_assist(q, ec.safe_eval, ec.normalize_for_eval)
    if m:
        return m
    ctx = ""
    for h in hist:
        ctx += f"User: {h['q']}\nAssistant: {h['a']}\n"
    ctx += f"User: {q}\nAssistant:"
    inputs = tok(ctx, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=48, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

def py_speed():
    inputs = tok("Question: What is 24 plus 18?\nAnswer:", return_tensors="pt").to("cuda")
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=64, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    return out.shape[1] - inputs["input_ids"].shape[1], time.time() - t0

RESULTS["pythia"] = {"single": ec.run_eval("pythia-70m (bare->harnessed)", py_fn,
                                           ec.make_testset(400)),
                     "mt": ec.run_multiturn(py_conv, ec.make_multiturn(50)),
                     "speed": ec.measure_speed(py_speed)}

json.dump(RESULTS, open("export/eval_all.json", "w"), indent=2)
print(json.dumps(RESULTS, indent=2, default=str))
