[English](./README.md) | [中文](./README_CN.md)

<p align="center">
  <h1 align="center">🧩 MemeBench</h1>
  <p align="center">
    A benchmark toolkit for open-ended meme understanding.
  </p>
  <p align="center">
    <em>Seeing the image is not enough. The model has to get the joke.</em>
  </p>
  <p align="center">
    <a href="https://star-history.com/#whwangovo/memebench&Date">
      <img src="https://img.shields.io/github/stars/whwangovo/memebench?style=social" alt="GitHub stars" />
    </a>
  </p>
</p>

---

## 🧠 What is MemeBench?

MemeBench evaluates whether multimodal models can explain memes in open-ended language: what is visible, who or what is referenced, what cultural knowledge is needed, and why the meme works.

The benchmark follows the **VIKR** decomposition:

| Layer | Question |
|---|---|
| **V — Visual** | What is visible in the image? What text appears? |
| **I — Identity** | Which named entities, characters, works, or cultural references appear? |
| **K — Knowledge** | What background knowledge is needed to understand the meme? |
| **R — Reasoning** | How do the visual cue and cultural context produce the joke or message? |

This repository contains the official KAR inference path and evaluation utilities. Dataset files, prompt templates, and experiment outputs are versioned as separate artifacts.

---

## ✨ Highlights

- **Open-ended evaluation** — models write explanations, not multiple-choice labels.
- **Layer-wise diagnosis** — VIKR makes failures easier to locate.
- **KAR inference** — Knowledge Anatomy-informed Retrieval for culture-heavy memes.
- **Prompt-agnostic evaluation** — bring your own judge prompt template.
- **Clean artifact split** — code, data, prompts, and experiment outputs stay decoupled.

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- An OpenAI-compatible model endpoint
- Tavily API key if you run KAR with web search

### Installation

```bash
git clone https://github.com/whwangovo/memebench.git
cd memebench

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
```

Fill in:

```bash
OPENAI_API_KEY=...
TAVILY_API_KEY=...
```

If you use a custom OpenAI-compatible endpoint:

```bash
OPENAI_BASE_URL=https://your-endpoint/v1
```

---

## 🔎 KAR Inference

KAR stands for **Knowledge Anatomy-informed Retrieval**.

```
image
  └─ VLM extraction: OCR, visual handle, entity guesses, search queries
       └─ CultureBase retrieval: candidate cultural entities
            └─ web search: background evidence
                 └─ grounded VLM reasoning: final meme explanation
```

Prompts are loaded from your local files:

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

The reasoning prompt template should accept:

```text
{knowledge_source}
{knowledge}
```

---

## 📏 Evaluation

MemeBench evaluation is checklist-based. A judge model reads the candidate response and the reference annotation, then outputs VIKR scores.

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

Aggregate two judges:

```bash
python -m pipelines.evaluation.aggregate_dual_judge \
  --judge-a output/model_judge_a.json \
  --judge-b output/model_judge_b.json \
  --output output/model_dual_judge.json
```

Print scores:

```bash
python -m pipelines.evaluation.score output/model_dual_judge.json
```

---

## 📦 Data Format

Benchmark item:

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

Prediction item:

```json
{
  "id": 1,
  "image_path": "meme_images/image_0001.png",
  "response": "The meme works because ..."
}
```

---

## 📁 Repository Structure

```text
memebench/
├── memebench/
│   ├── culture_base/      # local CultureBase retriever
│   ├── search/            # Tavily text search wrapper
│   ├── llm_client.py      # OpenAI-compatible async client helper
│   └── utils/retry.py     # retry helpers
└── pipelines/
    ├── inference/kar.py   # KAR implementation
    └── evaluation/        # judging, scoring, aggregation
```

---

## 🧾 Artifact Layout

| Artifact | Where it lives |
|---|---|
| Code | This repository |
| Dataset | Separate dataset release |
| Prompt templates | Separate prompt/config artifact |
| Predictions and judge outputs | Experiment artifact storage |

---

## 📚 Citation

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
