---
license: apache-2.0
language:
  - en
tags:
  - arithmetic
  - tool-use
  - tiny-model
  - from-scratch
  - GCA-60
pipeline_tag: text-generation
---

<p align="center"><img src="lingjie_logo.svg" width="180" alt="LingJie logo"></p>

# LingJie-Chat-v2-Pro（56.4M）

从零训练的 56.4M 参数窄域对话模型（GCA-60 架构），专注**算术识别 + 工具调用协议 + 行为规范**。
不依赖任何预训练权重：Tokenizer、底模、SFT 全部自行完成。

**LingJie-Chat** 系列：
- **v2-Pro（本仓库，56.4M）**：12 层 / d_model 640 / 多模块 Head，多轮代词指代增强 SFT
- v1-Flash（8.1M，即旧版 Nova LLM）：对照模型

## 架构

| 组件 | 配置 |
|------|------|
| 层数 / 隐藏维度 | 12 层，d_model=640，8 头（RoPE 因果注意力） |
| 子层 | 注意力 → 深度因果卷积 k=5 → SwiGLU FFN(d_ff=1365)，Pre-norm RMSNorm |
| 词表 | 8017（ByteLevel BPE 8000 + 13 特殊 + 4 对话；数字强制单字切分） |
| 上下文 | 1024 |
| 参数 | 56.42M（主干 56.34M + 6 个辅助 Head 0.083M） |
| 训练 | 预训练 1.2B tokens（LR 3e-4）→ SFT 14 万条（LR 1e-5，含 4 万多轮对话） |

## 评测（400 条留出样本 + 50 组多轮对话，三方均接入同一极简 harness）

| 维度 | LingJie-Chat-v2-Pro | LingJie-Chat-v1-Flash (8M) | pythia-70m (non-emb 44.7M) |
|------|------|------|------|
| 算术（Pass@1） | 99.4 | 99.4 | 99.4 |
| 多步复合算术 | 100.0 | 100.0 | 100.0 |
| 澄清行为 | **100.0** | 0 | 0 |
| 边界拒绝 | **100.0** | 47.5 | 47.5 |
| 负样本控制 | 100.0 | 100.0 | 100.0 |
| 纠错重算 | **83.8** | 0 | 2.5 |
| 多轮上下文（代词/追问） | **76.0** | 0 | 0 |
| 推理速度 (tok/s, GPU, batch=1) | 92* | 26 | 192 |

*两个 LingJie 模型推理为无 KV cache 全量重算实现；pythia 为 HF generate() KV-cache 增量解码。
算术类打平是因为三方共享同一个确定性计算 harness——验证"能力分离：模型做识别、工具做计算"的设计。
![benchmark](lingjie_benchmark.png)

## 能力边界（诚实声明）

- ✅ 算术识别与计算（+ - * /，≤6 位数字，最多 3 步工具调用）
- ✅ 信息不足时澄清（NEED_INFO）、越界拒绝（CANNOT）、纠错重算
- ✅ 多轮对话中的代词/省略式指代（"their sum"、"that times 2"）
- ❌ 语义纠错、开放域问答、自由生成——超出范围会明确告知 CANNOT
- ❌ 中文（训练语料为英文）

## 快速使用

```python
# 依赖: pip install torch tokenizers
import sys
sys.path.insert(0, "code")
from infer import Pipeline, SpellDict

words = [w.strip() for w in open("data/spell_dict.txt")]
pipe = Pipeline("weights/sft_last.pt", "tokenizer/lingjie_tokenizer.json", SpellDict(words))

print(pipe.answer("What is 1234 plus 567?", minimal=True))
# -> '1234+567 equals 1801.'

# 多轮（minimal=True 使用极简 harness；False 使用完整规则层）
hist = [{"q": "My two numbers are 55 and 18.", "a": "Got it."}]
print(pipe.answer("What is their sum?", minimal=True, history=hist))
```

命令行交互：`python code/infer.py --interactive`

## 文件说明

| 路径 | 说明 |
|------|------|
| `weights/sft_last.pt` | SFT 最终权重（含模型 config） |
| `tokenizer/lingjie_tokenizer.json` | ByteLevel BPE 分词器（8017 词） |
| `data/spell_dict.txt` | 拼写检查词典（TinyStories 词频 Top-100k） |
| `code/` | 模型定义 + 推理管线 + 评测脚本 |
| `training_args.json` | 完整训练超参数 |


## 仓库结构与下载

| 内容 | 位置 |
|------|------|
| 代码（训练/推理/评测/数据生成） | 本仓库 `code/` 及根目录 |
| SFT 数据（14 万条，含多轮） | `data/sft.jsonl` + `data/sft_multiturn.jsonl` |
| Tokenizer / 拼写词典 | `tokenizer/` `data/spell_dict.txt` |
| **模型权重** | **[Releases](../../releases)**：`sft_last.pt`（SFT 最终版）+ `pretrain_weights.pt`（预训练底模） |

权重文件不进 git（大文件），请从 Release 下载后放到 `weights/` 目录使用。

数据集（TinyStories + 程序生成语料）的完整重建流程见 `gen_pretrain.py` / `gen_curriculum.py` / `gen_sft*.py` / `pack_data.py` 与上文步骤。

## 在线体验

- 魔搭模型页：https://www.modelscope.cn/models/LingJie2026/LingJie-Chat-v2-Pro
- 创空间在线对话：https://www.modelscope.cn/studios/LingJie2026/lingjie-chat-demo
