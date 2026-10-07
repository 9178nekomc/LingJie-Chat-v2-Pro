"""DeepSeek-R1-style grouped bar chart comparing the three models."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ALL = json.load(open("export/eval_all.json"))
new, old, pyt = ALL["new"], ALL["old"], ALL["pythia"]
SPEED = [ALL["new"]["speed"], ALL["old"]["speed"], ALL["pythia"]["speed"]]

DIMS = ["arith", "multi", "clarify", "cannot", "neg", "correction", "mt"]
LABELS = ["Arithmetic\n(Pass@1)", "Multi-step\nArithmetic", "Clarify\nBehavior",
          "Boundary\nRefusal", "Negative\nControl", "Correction\n(Recompute)",
          "Multi-turn\n(Pronoun)"]

def pct(d, k):
    c, t = d["mt"] if k == "mt" else d["single"][k]
    return c / t * 100

series = [
    ("LingJie-Chat-v2-Pro (56.4M)", [pct(new, k) for k in DIMS], "#3b5bfd", "//", "white"),
    ("LingJie-Chat-v1-Flash (8.1M)", [pct(old, k) for k in DIMS], "#9a9a9a", "", None),
    ("EleutherAI/pythia-70m (70.4M; non-emb. 44.7M)", [pct(pyt, k) for k in DIMS], "#a8c4ff", "", None),
]

fig = plt.figure(figsize=(15.5, 7.5), dpi=160)
gs = fig.add_gridspec(1, 4, wspace=0.28)
ax = fig.add_subplot(gs[0, :3])
ax2 = fig.add_subplot(gs[0, 3])
x = np.arange(len(DIMS))
w = 0.26
for i, (name, vals, color, hatch, edge) in enumerate(series):
    bars = ax.bar(x + (i - 1) * w, vals, w, label=name, color=color,
                  hatch=hatch, edgecolor=edge if edge else "white", linewidth=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.1f}",
                ha="center", va="bottom", fontsize=9.5, fontweight="bold",
                color="#1a1a1a", rotation=90)
sb = ax2.bar(np.arange(3), SPEED, 0.6,
             color=[s[2] for s in series], hatch=[s[3] for s in series],
             edgecolor=[s[4] if s[4] else "white" for s in series], linewidth=0.6)
for b, v in zip(sb, SPEED):
    ax2.text(b.get_x() + b.get_width() / 2, v + 3, f"{v:.0f}",
             ha="center", va="bottom", fontsize=11, fontweight="bold")
ax2.set_title("Inference Speed\n(tokens/s, GPU, batch=1)", fontsize=12, fontweight="bold")
ax2.set_ylim(0, max(SPEED) * 1.18)
ax2.set_xticks(np.arange(3))
ax2.set_xticklabels(["v2-Pro", "v1-Flash", "pythia-70m"], fontsize=10.5)
ax2.text(0.5, -0.22, "no KV cache,\nfull recompute", transform=ax2.transAxes,
         ha="center", fontsize=9, color="#555555")
ax2.annotate("KV-cache\nincremental decoding", xy=(2, SPEED[2]), xytext=(1.15, SPEED[2]*0.78),
             fontsize=9, color="#555555",
             arrowprops=dict(arrowstyle="->", color="#888888", lw=1))
ax2.yaxis.grid(True, linestyle="--", alpha=0.35)
ax2.set_axisbelow(True)
for spine in ["top", "right"]:
    ax2.spines[spine].set_visible(False)

ax.set_ylim(0, 112)
ax.set_xticks(x)
ax.set_xticklabels(LABELS, fontsize=11.5)
ax.set_ylabel("Accuracy (%)", fontsize=13)
ax.set_title("LingJie-Chat: v2-Pro vs v1-Flash vs size-matched baseline",
             fontsize=15, fontweight="bold", pad=26)
ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.01), ncol=3, fontsize=10.5,
          frameon=False, columnspacing=1.6, handletextpad=0.5)
fig.text(0.5, 0.005, "All three models use the same minimal harness (math assist + profanity handling); everything else is raw model ability",
         ha="center", fontsize=10.5, color="#555555")
ax.yaxis.grid(True, linestyle="--", alpha=0.35)
ax.set_axisbelow(True)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

plt.tight_layout()
plt.savefig("export/lingjie_benchmark.png", bbox_inches="tight", facecolor="white")
print("saved export/lingjie_benchmark.png")
for name, vals, *_ in series:
    print(f"{name}: " + "  ".join(f"{d}={v:.1f}" for d, v in zip(DIMS, vals)))
