"""LingJie (GCA-60) model implementation.

Spec v3.0 with the v3.1 fixes decided at review:
- Vocab: 8017 real tokens (8000 BPE + 13 special + 4 dialog), model vocab padded to 8064.
- Gated FFN: SwiGLU-style 3-matrix gate/up/down with d_ff=1365 (matches the 2.62M/layer budget).
- Embedding tied with LM head (implied by the 56.88M budget).
- RoPE positional encoding (spec was silent; LLaMA-family default).
- Depthwise causal conv k=5 as its own pre-normed sublayer per the architecture diagram.
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LingJieConfig:
    vocab_size: int = 8064          # 8017 real + padding to multiple of 64
    d_model: int = 640
    n_layers: int = 12
    n_heads: int = 8
    d_ff: int = 1365                # 3 * 640 * 1365 ~= 2.62M per layer
    conv_kernel: int = 5
    max_seq_len: int = 1024
    rope_theta: float = 10000.0
    dropout: float = 0.0
    # multi-module heads (SFT only)
    n_templates: int = 100
    n_verbs: int = 20
    n_parser_tags: int = 4          # 0 other, 1 number-start, 2 number-cont, 3 operator


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


def rope_cache(seq_len, head_dim, theta, device, dtype):
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv)
    emb = torch.cat((freqs, freqs), dim=-1)
    return torch.cos(emb).to(dtype), torch.sin(emb).to(dtype)


def apply_rope(x, cos, sin):
    # x: (b, h, t, hd); cos/sin: (t, hd)
    x1, x2 = x.float().chunk(2, dim=-1)
    rot = torch.cat((-x2, x1), dim=-1)
    out = x.float() * cos + rot * sin
    return out.to(x.dtype)


class CausalDepthwiseConv(nn.Module):
    """Causal depthwise conv, kernel k, zero-padded on the left."""

    def __init__(self, dim, k):
        super().__init__()
        self.k = k
        self.conv = nn.Conv1d(dim, dim, k, groups=dim, bias=True)

    def forward(self, x):
        # x: (b, t, d) -> causal: pad k-1 on the left only
        w = x.transpose(1, 2)
        w = F.pad(w, (self.k - 1, 0))
        w = self.conv(w)
        return w.transpose(1, 2)


class Attention(nn.Module):
    def __init__(self, cfg: LingJieConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.o = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x, cos, sin, attn_mask=None):
        b, t, d = x.shape
        qkv = self.qkv(x).view(b, t, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(b, t, d)
        return self.o(y)


class FFN(nn.Module):
    def __init__(self, cfg: LingJieConfig):
        super().__init__()
        self.gate = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.up = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.down = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, cfg: LingJieConfig):
        super().__init__()
        self.n1 = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.n2 = RMSNorm(cfg.d_model)
        self.conv = CausalDepthwiseConv(cfg.d_model, cfg.conv_kernel)
        self.n3 = RMSNorm(cfg.d_model)
        self.ffn = FFN(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.n1(x), cos, sin)
        x = x + self.conv(self.n2(x))
        x = x + self.ffn(self.n3(x))
        return x


class MultiHeads(nn.Module):
    """Auxiliary heads, activated during SFT only (no params charged to pretraining FLOPs)."""

    def __init__(self, cfg: LingJieConfig):
        super().__init__()
        d = cfg.d_model
        self.intent = nn.Linear(d, 2, bias=False)      # binary: is arithmetic
        self.integrity = nn.Linear(d, 2, bias=False)   # binary: complete
        self.parser = nn.Linear(d, cfg.n_parser_tags, bias=False)  # per-token tagging
        self.critic = nn.Linear(d, 2, bias=False)      # binary: expr consistent
        self.template = nn.Linear(d, cfg.n_templates, bias=False)
        self.verb = nn.Linear(d, cfg.n_verbs, bias=False)

    def forward(self, h, pooled):
        return {
            "intent": self.intent(pooled),
            "integrity": self.integrity(pooled),
            "parser": self.parser(h),                  # (b, t, tags)
            "critic": self.critic(pooled),
            "template": self.template(pooled),
            "verb": self.verb(pooled),
        }


class LingJie(nn.Module):
    def __init__(self, cfg: LingJieConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight  # tied
        self.heads = MultiHeads(cfg)
        cos, sin = rope_cache(cfg.max_seq_len + 1, cfg.d_model // cfg.n_heads,
                              cfg.rope_theta, "cpu", torch.float32)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init)
        # scale residual-output inits
        for name, p in self.named_parameters():
            if name.endswith("o.weight") or name.endswith("down.weight") or name.endswith("conv.conv.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layers))

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.Conv1d):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, idx, targets=None, head_labels=None, attn_mask=None):
        b, t = idx.shape
        cos = self.rope_cos[:t].to(idx.device)
        sin = self.rope_sin[:t].to(idx.device)
        x = self.embed(idx)
        for blk in self.blocks:
            x = blk(x, cos, sin)
        x = self.norm(x)
        logits = self.lm_head(x) if (self.training or targets is not None) else None
        out = {}
        if targets is not None:
            out["loss_lm"] = F.cross_entropy(
                logits.view(-1, logits.size(-1)).float(), targets.reshape(-1), ignore_index=-100
            )
            out["logits"] = logits
        if head_labels is not None and self.heads is not None:
            if attn_mask is not None:
                m = attn_mask.to(x.dtype).unsqueeze(-1)
                pooled = (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
            else:
                pooled = x.mean(dim=1)
            hd = self.heads(x, pooled)
            lw = {"intent": 0.3, "integrity": 0.2, "parser": 0.2,
                  "critic": 0.1, "template": 0.1, "verb": 0.1}
            loss_h = x.new_zeros(())
            for k, w in lw.items():
                lab = head_labels[k]
                if lab is not None:
                    loss_h = loss_h + w * F.cross_entropy(
                        hd[k].reshape(-1, hd[k].size(-1)).float(), lab.reshape(-1), ignore_index=-100
                    )
            out["loss_heads"] = loss_h
            out["head_logits"] = hd
        return out

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.0, top_p=0.9, eos_id=None, stop_strings=None, tok=None):
        """Greedy (temperature=0) or top-p sampling, incremental KV-free (recompute) for simplicity.

        For a 60M model, recompute-per-step is cheap enough and avoids KV-cache bugs.
        """
        was_training = self.training
        self.eval()
        for _ in range(max_new_tokens):
            idx_c = idx if idx.size(1) <= self.cfg.max_seq_len else idx[:, -self.cfg.max_seq_len:]
            cos = self.rope_cos[: idx_c.size(1)].to(idx.device)
            sin = self.rope_sin[: idx_c.size(1)].to(idx.device)
            x = self.embed(idx_c)
            for blk in self.blocks:
                x = blk(x, cos, sin)
            x = self.norm(x)
            next_logits = self.lm_head(x[:, -1]).float()
            if temperature and temperature > 0:
                probs = F.softmax(next_logits / temperature, dim=-1)
                sp, si = torch.sort(probs, descending=True, dim=-1)
                cum = sp.cumsum(-1)
                keep = cum - sp < top_p
                sp = sp * keep
                sp = sp / sp.sum(-1, keepdim=True)
                nx = torch.multinomial(sp, 1)
                nxt = si.gather(-1, nx)
            else:
                nxt = next_logits.argmax(dim=-1, keepdim=True)
            idx = torch.cat([idx, nxt], dim=1)
            if eos_id is not None and (nxt == eos_id).all():
                break
            if stop_strings and tok is not None:
                cur = tok.decode(idx[0].tolist())
                if any(s in cur for s in stop_strings):
                    break
        if was_training:
            self.train()
        return idx


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    heads = sum(p.numel() for p in model.heads.parameters())
    return total, total - heads, heads


if __name__ == "__main__":
    cfg = LingJieConfig()
    m = LingJie(cfg)
    total, trunk, hd = count_params(m)
    print(f"total={total/1e6:.2f}M trunk={trunk/1e6:.2f}M heads={hd/1e6:.3f}M")
    idx = torch.randint(0, cfg.vocab_size, (2, 64))
    tgt = torch.randint(0, cfg.vocab_size, (2, 64))
    m.train()
    out = m(idx, tgt)
    print("lm loss:", out["loss_lm"].item())
    idx_out = m.generate(idx[:1, :8], 5)
    print("gen ok:", idx_out.shape)
