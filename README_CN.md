[English](./README.md) | [中文](./README_CN.md)

<p align="center">
  <h1 align="center">🧩 MemeBench</h1>
  <p align="center">
    面向开放式梗图理解的 benchmark 工具箱。
  </p>
  <p align="center">
    <em>看见图片还不够，模型得真的懂这个梗。</em>
  </p>
  <p align="center">
    <a href="https://star-history.com/#whwangovo/memebench&Date">
      <img src="https://img.shields.io/github/stars/whwangovo/memebench?style=social" alt="GitHub stars" />
    </a>
  </p>
</p>

---

## 🧠 这是什么？

MemeBench 用来评测多模态模型是否能用开放式文本解释一张梗图：图里有什么、指向了谁、需要什么文化背景，以及这个梗到底为什么成立。

我们把梗图理解拆成 **VIKR** 四层：

| 层级 | 要回答的问题 |
|---|---|
| **V — Visual** | 图里看到了什么？有哪些文字？ |
| **I — Identity** | 出现了哪些具体人物、角色、作品或文化引用？ |
| **K — Knowledge** | 理解这个梗需要哪些背景知识？ |
| **R — Reasoning** | 视觉线索和文化语境如何组合成笑点或表达意图？ |

这个仓库提供 MemeBench 的 KAR 推理路径和评测工具。数据、prompt 模板和实验输出作为独立 artifact 管理。

---

## ✨ 亮点

- **开放式评测** — 模型写解释，不做选择题。
- **分层诊断** — VIKR 能看出模型到底卡在视觉、身份、知识还是推理。
- **KAR 推理** — Knowledge Anatomy-informed Retrieval，面向文化背景很重的梗图。
- **评测 prompt 解耦** — judge prompt 从外部文件加载，方便替换和复现实验。
- **artifact 分离** — 代码、数据、prompt、预测结果分开版本管理。

---

## 🚀 快速开始

### 环境要求

- Python 3.10+
- 一个 OpenAI 兼容的模型接口
- 如果运行 KAR web search，需要 Tavily API key

### 安装

```bash
git clone https://github.com/whwangovo/memebench.git
cd memebench

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
```

填写：

```bash
OPENAI_API_KEY=...
TAVILY_API_KEY=...
```

如果使用自定义 OpenAI 兼容接口：

```bash
OPENAI_BASE_URL=https://your-endpoint/v1
```

---

## 🔎 KAR 推理

KAR 全称 **Knowledge Anatomy-informed Retrieval**。

```text
image
  └─ VLM extraction：OCR、视觉线索、实体猜测、搜索 query
       └─ CultureBase retrieval：候选文化实体
            └─ web search：背景证据
                 └─ grounded VLM reasoning：最终梗图解释
```

prompt 从你的本地文件加载：

```python
import asyncio

from memebench.culture_base import CultureBaseRetriever
from pipelines.inference.kar import KarConfig, load_prompt, run_kar


async def main():
    config = KarConfig(
        model="gpt-5",
        extract_prompt=load_prompt("prompts/kar_extract.txt"),
        reason_prompt_template=load_prompt("prompts/kar_reason.txt"),
    )
    retriever = CultureBaseRetriever(cb_dir="data/culture_base")

    result = await run_kar(
        image_path="data/images/example.png",
        config=config,
        retriever=retriever,
    )

    print(result["response"])
    print(result["kar_trace"])


asyncio.run(main())
```

reasoning prompt 模板需要包含：

```text
{knowledge_source}
{knowledge}
```

---

## 📏 评测

MemeBench 使用 checklist-based evaluation。Judge 模型读取候选回答和参考标注，输出 VIKR 各维度得分。

运行 judge：

```bash
python -m pipelines.evaluation.judge \
  --bench data/annotations/annotations_v8.json \
  --candidate output/model_predictions.json \
  --output output/model_judge.json \
  --prompt prompts/judge.txt \
  --v3
```

judge prompt 需要包含：

```text
{reference_answer}
{generated_answer_to_eval}
```

聚合两个 judge：

```bash
python -m pipelines.evaluation.aggregate_dual_judge \
  --judge-a output/model_judge_a.json \
  --judge-b output/model_judge_b.json \
  --output output/model_dual_judge.json
```

打印分数：

```bash
python -m pipelines.evaluation.score output/model_dual_judge.json
```

---

## 📦 数据格式

Benchmark item：

```json
{
  "id": 1,
  "image_path": "meme_images/image_0001.png",
  "visual": {},
  "identity": {},
  "knowledge": {},
  "reasoning": {},
  "eval_checklist": {
    "visual": [],
    "identity": [],
    "knowledge": [],
    "reasoning": []
  }
}
```

Prediction item：

```json
{
  "id": 1,
  "image_path": "meme_images/image_0001.png",
  "response": "这个梗成立是因为……"
}
```

---

## 📁 仓库结构

```text
memebench/
├── memebench/
│   ├── culture_base/      # 本地 CultureBase 检索器
│   ├── search/            # Tavily 文本搜索封装
│   ├── llm_client.py      # OpenAI 兼容异步客户端 helper
│   └── utils/retry.py     # 重试工具
└── pipelines/
    ├── inference/kar.py   # KAR 实现
    └── evaluation/        # judge、score、aggregation
```

---

## 🧾 Artifact 管理

| Artifact | 管理位置 |
|---|---|
| 代码 | 当前仓库 |
| 数据集 | 单独的数据集发布 |
| Prompt 模板 | 单独的 prompt/config artifact |
| 预测与 judge 输出 | 实验 artifact 存储 |

---

## 📚 引用

```bibtex
@misc{memebench2026,
  title = {MemeBench},
  author = {MemeBench Authors},
  year = {2026},
  note = {Citation information coming soon}
}
```

---

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=whwangovo/memebench&type=Date)](https://star-history.com/#whwangovo/memebench&Date)
