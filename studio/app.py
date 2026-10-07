import os
import sys

import gradio as gr
import torch
from modelscope import snapshot_download

REPO_ID = "LingJie2026/LingJie-Chat-v2-Pro"
MODEL_DIR = snapshot_download(REPO_ID)

sys.path.insert(0, os.path.join(MODEL_DIR, "code"))
from infer import Pipeline, SpellDict  # noqa: E402

_words_path = os.path.join(MODEL_DIR, "data", "spell_dict.txt")
_words = [w.strip() for w in open(_words_path)]
PIPE = Pipeline(
    os.path.join(MODEL_DIR, "weights", "sft_last.pt"),
    os.path.join(MODEL_DIR, "tokenizer", "lingjie_tokenizer.json"),
    SpellDict(_words),
    device="cpu",
)

TITLE = "LingJie-Chat-v2-Pro"

LOGO_SVG = open(os.path.join(MODEL_DIR, "lingjie_logo.svg")).read() \
    if os.path.exists(os.path.join(MODEL_DIR, "lingjie_logo.svg")) else ""

WELCOME = """
<div id="hero">
  <div class="hero-logo">{logo}</div>
  <div class="hero-title">LingJie-Chat-v2-Pro</div>
  <div class="hero-sub">56.4M &middot; trained from scratch &middot; arithmetic &amp; tool protocol</div>
  <div class="hero-hint">
    Try: <em>What is 1234 plus 567?</em> &nbsp;&middot;&nbsp;
    <em>My two numbers are 40 and 2. What is their sum?</em> &nbsp;&middot;&nbsp;
    <em>What is the capital of France?</em>
  </div>
</div>
""".format(logo=LOGO_SVG)

CSS = """
#hero {
  max-height: 560px;
  opacity: 1;
  transform: scale(1);
  transition: max-height .8s cubic-bezier(.6,0,.3,1), opacity .55s ease,
              transform .8s cubic-bezier(.6,0,.3,1), margin .8s ease;
  overflow: hidden;
  text-align: center;
  padding: 18px 0 6px;
}
#hero.hide {
  max-height: 0;
  opacity: 0;
  transform: scale(.92);
  padding: 0;
  pointer-events: none;
}
.hero-logo svg { width: 168px; height: 168px; filter: drop-shadow(0 10px 26px rgba(30,50,160,.28)); }
.hero-title { font-size: 1.9em; font-weight: 700; margin-top: 4px; letter-spacing: .5px; }
.hero-sub { color: #6b7280; font-size: .95em; margin-top: 6px; }
.hero-hint { color: #9ca3af; font-size: .85em; margin-top: 14px; }
"""


def respond(message, chat_history):
    hist = []
    for turn in chat_history:
        if turn["role"] == "user":
            hist.append({"q": turn["content"], "a": ""})
        elif hist:
            hist[-1]["a"] = turn["content"]
    reply = PIPE.answer(message, minimal=True, history=hist)
    chat_history = chat_history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": reply},
    ]
    return "", chat_history


JS = """
() => {
  const obs = new MutationObserver(() => {
    const hero = document.getElementById('hero');
    const chat = document.getElementById('chatbot');
    if (!hero || !chat) return;
    const started = chat.querySelectorAll('[class*="message"], .bubble-wrap, [data-testid="user"]').length > 0;
    hero.classList.toggle('hide', started);
  });
  const mount = () => {
    const chat = document.getElementById('chatbot');
    if (chat) { obs.observe(chat, {childList: true, subtree: true}); return true; }
    return false;
  };
  if (!mount()) {
    const mo = new MutationObserver(() => { if (mount()) mo.disconnect(); });
    mo.observe(document.body, {childList: true, subtree: true});
  }
  return '';
}
"""

with gr.Blocks(title=TITLE, css=CSS, js=JS) as demo:
    gr.HTML(WELCOME)
    chatbot = gr.Chatbot(elem_id="chatbot", height=430)
    msg = gr.Textbox(placeholder="Ask me arithmetic... e.g. What is 48 divided by 6?",
                     show_label=False, autofocus=True)
    with gr.Row():
        clear = gr.Button("New conversation")
    clear.click(lambda: [], None, chatbot)

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860)
