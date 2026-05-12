# MemeBench

MemeBench is a diagnostic benchmark for open-ended meme interpretation. This
repository is an intentionally minimal public release: it contains the KAR
inference implementation, evaluation utilities, and the smallest support code
needed to run them.

Data, prompts, data construction pipelines, annotation tools, scripts, tests,
and the local review studio are not included in this release.

## Contents

```text
memebench/
  culture_base/       Local CultureBase retriever interface
  search/             Minimal Tavily text search provider
  llm_client.py       Minimal OpenAI-compatible client helper
  utils/retry.py      Retry helpers
pipelines/
  inference/kar.py    KAR implementation
  evaluation/         Judge parsing, scoring, and dual-judge aggregation
```

## KAR Inference

`pipelines/inference/kar.py` exposes a small library API. Prompt text is loaded
by the caller and is not bundled in this repository.

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

Required environment variables:

```bash
OPENAI_API_KEY=...
TAVILY_API_KEY=...
```

Optional:

```bash
OPENAI_BASE_URL=...
```

## Evaluation

The evaluation code supports:

- parsing judge outputs
- running a judge with an externally supplied prompt template
- aggregating two judges by item-level intersection
- reporting aggregate scores

Run a judge:

```bash
python -m pipelines.evaluation.judge \
  --bench data/annotations/annotations_v8.json \
  --candidate output/model_predictions.json \
  --output output/model_judge.json \
  --prompt prompts/judge.txt \
  --v3
```

The judge prompt file must contain these placeholders:

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

Score a judge file:

```bash
python -m pipelines.evaluation.score output/model_dual_judge.json
```

## Dataset

The dataset is distributed separately. Evaluation expects a JSON list whose
items contain at least:

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

Prediction files are JSON lists with:

```json
{
  "id": 1,
  "image_path": "meme_images/image_0001.png",
  "response": "..."
}
```

## Citation

```bibtex
@misc{memebench2026,
  title = {MemeBench},
  author = {MemeBench Authors},
  year = {2026},
  note = {Citation information coming soon}
}
```
