<p align="center">
  <h1 align="center">MemeBench</h1>
  <p align="center">
    A lean benchmark toolkit for open-ended meme understanding.
  </p>
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> |
  <a href="#kar-inference">KAR</a> |
  <a href="#evaluation">Evaluation</a> |
  <a href="#data-format">Data Format</a> |
  <a href="#citation">Citation</a>
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-blue">
  <img alt="Release" src="https://img.shields.io/badge/release-minimal-lightgrey">
  <img alt="Prompts" src="https://img.shields.io/badge/prompts-external-orange">
</p>

MemeBench asks a deceptively simple question: can a multimodal model explain why a meme is funny, pointed, ironic, or culturally loaded?

This repo is the **minimal public code release**. It includes the KAR inference core, a small search wrapper, and evaluation utilities. It does not include the dataset, prompts, data construction pipeline, annotation tools, scripts, tests, or studio UI.

## What Is Inside

```text
memebench/
  culture_base/       Local CultureBase retriever interface
  search/             Minimal Tavily text search provider
  llm_client.py       OpenAI-compatible async client helper
  utils/retry.py      Small retry helpers

pipelines/
  inference/kar.py    KAR, packaged as a library API
  evaluation/         Judge parsing, scoring, dual-judge aggregation
```

## Quick Start

```bash
git clone https://github.com/whwangovo/memebench.git
cd memebench

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
```

Fill in the keys you need:

```bash
OPENAI_API_KEY=...
TAVILY_API_KEY=...
```

Use `OPENAI_BASE_URL` if your model endpoint is OpenAI-compatible but not hosted by OpenAI.

## KAR Inference

KAR stands for **Knowledge Anatomy-informed Retrieval**. It is a four-stage pipeline:

1. Extract OCR, visual clues, entity guesses, and search queries from the image.
2. Retrieve candidate entities from a local CultureBase.
3. Fuse VLM-generated queries with high-confidence KB hits, then search the web.
4. Ask the model for a grounded meme explanation.

Prompts are deliberately external. Bring your own prompt files and pass them in:

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

## Evaluation

The evaluation code is prompt-agnostic. It can parse judge outputs, call a judge with your own prompt template, aggregate two judges, and report scores.

Run a judge:

```bash
python -m pipelines.evaluation.judge \
  --bench data/annotations/annotations_v8.json \
  --candidate output/model_predictions.json \
  --output output/model_judge.json \
  --prompt prompts/judge.txt \
  --v3
```

The judge prompt file must contain:

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

Score a result file:

```bash
python -m pipelines.evaluation.score output/model_dual_judge.json
```

## Data Format

The dataset is distributed separately. A benchmark item should contain at least:

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

Prediction files are just as simple:

```json
{
  "id": 1,
  "image_path": "meme_images/image_0001.png",
  "response": "The meme works because ..."
}
```

## Release Scope

This repository is intentionally small. The following are kept out of the public code release:

- dataset files and images
- prompt text
- crawlers and annotation pipelines
- data generation scripts
- internal experiments, logs, tests, and UI tooling

That separation keeps the code release tidy and makes the benchmark artifacts easier to version independently.

## Citation

```bibtex
@misc{memebench2026,
  title = {MemeBench},
  author = {MemeBench Authors},
  year = {2026},
  note = {Citation information coming soon}
}
```
