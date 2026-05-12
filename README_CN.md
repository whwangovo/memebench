[English](./README.md) | [中文](./README_CN.md)

<h1 align="center">MemeBench：面向开放式梗图理解的多模态模型评测</h1>

<p align="center">
  <a href="">论文</a> •
  <a href="">数据集</a> •
  <a href="">排行榜</a> •
  <a href="https://github.com/whwangovo/memebench">代码</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Task-Meme%20Understanding-blue" alt="task">
  <img src="https://img.shields.io/badge/Evaluation-Open--Ended-green" alt="evaluation">
  <img src="https://img.shields.io/badge/Method-KAR-orange" alt="method">
</p>

## 📢 最新动态

- **[2026/05]** 发布 KAR 推理与 MemeBench 评测代码。
- 论文、数据集和排行榜链接将在 benchmark 正式发布时更新。

## 🌟 亮点

- **开放式梗图解释。** 模型需要生成自由文本解释，而不是从候选项中选择答案。
- **细粒度诊断协议。** MemeBench 从 Visual、Identity、Knowledge、Reasoning 四个层次评测模型能力。
- **知识密集型评测。** 重点考察文化引用、命名实体、互联网语境和幽默机制。
- **KAR 推理。** 提供 Knowledge Anatomy-informed Retrieval 的官方实现，用于检索并利用文化背景信息。
- **Artifact 分离。** 代码、数据、prompt 和实验输出分别管理，便于复现和版本控制。

## 📖 简介

梗图是一种高度压缩的多模态文化表达。模型可能能读出图片文字，也能描述画面，但如果无法识别被引用的实体、补全必要的文化背景，或把引用和表达意图联系起来，它仍然没有真正理解这个梗。

MemeBench 旨在评测这种分层理解能力。每个样本按照 **VIKR** schema 进行评估：

| 维度 | 描述 |
|---|---|
| **V — Visual** | 图像内容、OCR、场景布局和视觉证据。 |
| **I — Identity** | 命名实体、角色、来源作品、公众人物和文化引用。 |
| **K — Knowledge** | 背景事实、互联网文化、梗图惯例和领域知识。 |
| **R — Reasoning** | 视觉证据、文化语境、幽默、反讽或表达意图之间的连接。 |

本仓库包含 MemeBench 的官方评测工具和 KAR 推理实现。

## 🧩 仓库结构

```text
memebench/
├── memebench/
│   ├── culture_base/      # CultureBase 检索器接口
│   ├── search/            # Tavily 文本搜索封装
│   ├── llm_client.py      # OpenAI 兼容异步客户端 helper
│   └── utils/retry.py     # 重试工具
└── pipelines/
    ├── inference/kar.py   # KAR 推理实现
    └── evaluation/        # judge、score 与双 judge 聚合
```

## 🔎 KAR 推理

KAR 全称 **Knowledge Anatomy-informed Retrieval**，将梗图解释拆解为带检索的多阶段流程：

```text
Image
  → VLM extraction：OCR、视觉线索、实体假设、搜索 query
  → CultureBase retrieval：候选文化实体
  → Web search：文化背景证据
  → Grounded VLM reasoning：最终梗图解释
```

使用示例：

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


asyncio.run(main())
```

Prompt 模板作为外部 artifact 管理。Reasoning prompt 需要包含：

```text
{knowledge_source}
{knowledge}
```

## 📏 评测

MemeBench 使用 checklist-based evaluation。给定候选回答和参考标注，judge 会输出 VIKR 维度分数和整体正确性判断。

运行 judge：

```bash
python -m pipelines.evaluation.judge \
  --bench data/annotations/annotations_v8.json \
  --candidate output/model_predictions.json \
  --output output/model_judge.json \
  --prompt prompts/judge.txt \
  --v3
```

Judge prompt 需要包含：

```text
{reference_answer}
{generated_answer_to_eval}
```

聚合两个 judge 文件：

```bash
python -m pipelines.evaluation.aggregate_dual_judge \
  --judge-a output/model_judge_a.json \
  --judge-b output/model_judge_b.json \
  --output output/model_dual_judge.json
```

输出分数：

```bash
python -m pipelines.evaluation.score output/model_dual_judge.json
```

## 📦 数据格式

数据集单独发布。Benchmark item 需要包含：

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

Prediction 文件需要包含：

```json
{
  "id": 1,
  "image_path": "meme_images/image_0001.png",
  "response": "这个梗成立是因为……"
}
```

## 📌 TODO

- 更新论文链接。
- 更新数据集卡片和下载链接。
- 更新排行榜提交说明。
- 补充官方 benchmark 统计与实验结果表。

## 📚 引用

```bibtex
@misc{memebench2026,
  title = {MemeBench: Evaluating Open-Ended Meme Understanding in Multimodal Models},
  author = {MemeBench Authors},
  year = {2026},
  note = {Citation information coming soon}
}
```

## 📮 联系

如有问题，欢迎在本仓库提交 issue。
