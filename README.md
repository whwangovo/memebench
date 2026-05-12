[English](./README.md) | [中文](./README_CN.md)

<h1 align="center">MemeBench: Evaluating Open-Ended Meme Understanding in Multimodal Models</h1>

<p align="center">
  <a href="">Paper</a> •
  <a href="https://huggingface.co/datasets/anonymous-neurips-2026/memebench">Dataset</a> •
  <a href="">Leaderboard</a> •
  <a href="https://github.com/whwangovo/memebench">Code</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Task-Meme%20Understanding-blue" alt="task">
  <img src="https://img.shields.io/badge/Evaluation-Open--Ended-green" alt="evaluation">
  <img src="https://img.shields.io/badge/Method-KAR-orange" alt="method">
</p>

## 📢 News

- **[2026/05]** MemeBench dataset is available on [Hugging Face](https://huggingface.co/datasets/anonymous-neurips-2026/memebench).
- **[2026/05]** We release the official code for KAR inference and MemeBench evaluation.
- Paper and leaderboard links will be updated with the public benchmark release.

## 🌟 Highlights

- **Open-ended meme interpretation.** Models generate free-form explanations rather than selecting from predefined options.
- **Fine-grained diagnostic protocol.** MemeBench evaluates four layers of meme understanding: Visual, Identity, Knowledge, and Reasoning.
- **Knowledge-intensive evaluation.** The benchmark emphasizes cultural references, named entities, internet context, and humor mechanisms.
- **KAR inference.** We provide the implementation of Knowledge Anatomy-informed Retrieval for retrieving and using cultural background information.
- **Artifact separation.** Code, dataset, prompts, and experiment outputs are versioned separately for clearer reproducibility.

## 📖 Introduction

Memes are compact multimodal artifacts. A model may correctly read the text and describe the image, yet still miss the joke because it fails to identify the referenced entity, retrieve the necessary cultural background, or connect the reference to the intended communicative effect.

MemeBench is designed to evaluate this kind of layered understanding. Each example is assessed under the **VIKR** schema:

| Dimension | Description |
|---|---|
| **V — Visual** | Visual content, OCR, scene layout, and image-level evidence. |
| **I — Identity** | Named entities, characters, source works, public figures, and cultural references. |
| **K — Knowledge** | Background facts, internet culture, meme conventions, and domain knowledge. |
| **R — Reasoning** | The connection between visual evidence, cultural context, humor, irony, or communicative intent. |

This repository contains the official evaluation utilities and the KAR inference implementation used with MemeBench.

## 🧩 Repository Structure

```text
memebench/
├── memebench/
│   ├── culture_base/      # CultureBase retriever interface
│   ├── search/            # Tavily text search wrapper
│   ├── llm_client.py      # OpenAI-compatible async client helper
│   └── utils/retry.py     # retry utilities
└── pipelines/
    ├── inference/kar.py   # KAR inference implementation
    └── evaluation/        # judging, scoring, and dual-judge aggregation
```

## 🔎 KAR Inference

KAR, short for **Knowledge Anatomy-informed Retrieval**, decomposes meme interpretation into retrieval-aware stages:

```text
Image
  → VLM extraction: OCR, visual handle, entity hypotheses, search queries
  → CultureBase retrieval: candidate cultural entities
  → Web search: cultural background evidence
  → Grounded VLM reasoning: final meme explanation
```

Example usage:

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

Prompt templates are provided as external artifacts. The reasoning template should contain:

```text
{knowledge_source}
{knowledge}
```

## 📏 Evaluation

MemeBench uses checklist-based evaluation. Given a candidate response and a reference annotation, the judge produces VIKR scores and an overall correctness decision.

Run a judge:

```bash
python -m pipelines.evaluation.judge \
  --bench data/annotations/annotations_v8.json \
  --candidate output/model_predictions.json \
  --output output/model_judge.json \
  --prompt prompts/judge.txt \
  --v3
```

The judge prompt must contain:

```text
{reference_answer}
{generated_answer_to_eval}
```

Aggregate two judge files:

```bash
python -m pipelines.evaluation.aggregate_dual_judge \
  --judge-a output/model_judge_a.json \
  --judge-b output/model_judge_b.json \
  --output output/model_dual_judge.json
```

Report scores:

```bash
python -m pipelines.evaluation.score output/model_dual_judge.json
```

## 📦 Data Format

The dataset is hosted on Hugging Face: [anonymous-neurips-2026/memebench](https://huggingface.co/datasets/anonymous-neurips-2026/memebench).

The Hub dataset viewer uses an automatically converted Parquet view, while the canonical release also includes the JSON annotations and image files. A benchmark item is expected to contain:

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

Prediction files should contain:

```json
{
  "id": 1,
  "image_path": "meme_images/image_0001.png",
  "response": "The meme works because ..."
}
```

## 📌 TODO

- Release paper link.
- Release leaderboard submission instructions.
- Add official benchmark statistics and result table.

## 📚 Citation

```bibtex
@misc{memebench2026,
  title = {MemeBench: Evaluating Open-Ended Meme Understanding in Multimodal Models},
  author = {MemeBench Authors},
  year = {2026},
  note = {Citation information coming soon}
}
```

## 📮 Contact

For questions about MemeBench, please open an issue in this repository.
