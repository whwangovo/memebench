"""Minimal KAR inference utilities.

KAR (Knowledge Anatomy-informed Retrieval) is a four-stage meme-understanding
pipeline:

1. Ask a vision-language model to extract OCR, visual clues, entity guesses, and
   search queries from an image.
2. Retrieve candidate cultural entities from a local CultureBase.
3. Fuse VLM queries and high-confidence KB matches, then search the web.
4. Ask the VLM to produce the final grounded meme explanation.

Prompt text is intentionally not included in this repository. Pass prompt strings
from your own files or configuration when calling `run_kar`.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from memebench.llm_client import build_async_client, encode_image
from memebench.search.search_tools import TavilySearchProvider
from memebench.utils.retry import retry_api_call, retry_sync

KB_MEDIUM_CONFIDENCE = 0.5
CONFIDENCE_RANK = {"low": 0, "medium": 1, "high": 2}
KAR_REASON_KNOWLEDGE_SOURCE = (
    "cultural background information based on entities and text identified in this meme"
)


@dataclass
class KarConfig:
    """Runtime settings for a KAR call."""

    model: str
    extract_prompt: str
    reason_prompt_template: str
    search_top_k: int = 5
    kb_top_k: int = 5
    query_budget: int = 5


def load_prompt(path: str | Path) -> str:
    """Load prompt text from a local file."""
    return Path(path).read_text(encoding="utf-8")


def parse_vlm_extraction(text: str) -> dict[str, Any]:
    """Parse the Stage-1 VLM extraction response.

    Expected sections:
      - **OCR Text**: ...
      - **Visual Handle**: ...
      - **Entities**: 1. Name: ... | Source: ... | Confidence: ...
      - **Suggested Queries**: 1. ...
    """
    result: dict[str, Any] = {
        "ocr": "",
        "visual_handle": "",
        "entities": [],
        "visual_desc": "",
        "vlm_queries": [],
        "raw": text,
    }

    ocr_match = re.search(r"\*\*OCR Text\*\*:\s*(.+?)(?=\n-\s*\*\*|\Z)", text, re.DOTALL)
    if ocr_match:
        ocr = ocr_match.group(1).strip().strip('"')
        if ocr.lower() not in {"none", "n/a", "无"}:
            result["ocr"] = ocr

    visual_match = re.search(r"\*\*Visual Handle\*\*:\s*(.+?)(?=\n-\s*\*\*|\Z)", text, re.DOTALL)
    if visual_match:
        visual_handle = visual_match.group(1).strip().strip('"')
        if visual_handle.lower() not in {"none", "n/a", "无"}:
            result["visual_handle"] = visual_handle
            result["visual_desc"] = visual_handle

    entities_match = re.search(r"\*\*Entities\*\*:\s*(.+?)(?=\n-\s*\*\*|\Z)", text, re.DOTALL)
    if entities_match:
        entity_pattern = re.compile(
            r"^\s*\d+\.\s*Name:\s*(.*?)\s*\|\s*Source:\s*(.*?)\s*\|\s*Confidence:\s*(high|medium|low)\s*\|",
            re.IGNORECASE,
        )
        for line in entities_match.group(1).splitlines():
            match = entity_pattern.match(line.strip())
            if match and match.group(1).strip():
                result["entities"].append(
                    {
                        "name": match.group(1).strip(),
                        "source": match.group(2).strip(),
                        "confidence": match.group(3).lower(),
                    }
                )

    in_queries = False
    query_pattern = re.compile(r"^\s*\d+\.\s*(.+)", re.MULTILINE)
    for line in text.splitlines():
        if "suggested quer" in line.lower() or "search quer" in line.lower():
            in_queries = True
            continue
        if not in_queries:
            continue
        match = query_pattern.match(line)
        if match:
            query = match.group(1).strip()
            query = re.split(r"\s*[—–-]+\s*targeting", query, maxsplit=1)[0].strip()
            if 5 < len(query) < 400:
                result["vlm_queries"].append(query)
        elif line.strip() and result["vlm_queries"]:
            break

    return result


def _is_chinese(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def _entity_is_confident(entity: dict[str, Any], min_confidence: str = "medium") -> bool:
    return CONFIDENCE_RANK.get(entity.get("confidence", "low"), 0) >= CONFIDENCE_RANK[min_confidence]


def _build_kb_queries(kb_results: list[dict[str, Any]]) -> list[str]:
    queries: list[str] = []
    for entry in kb_results:
        if entry.get("similarity", 0) < KB_MEDIUM_CONFIDENCE:
            continue
        name = str(entry.get("name", "")).strip()
        source = str(entry.get("source", "")).strip()
        if not name:
            continue
        if len(name) > 30:
            name = name.split("/")[0].split("-")[0].strip()
        if _is_chinese(name) or _is_chinese(source):
            queries.append(f"{name} {source} 梗 文化背景 出处".strip())
        else:
            queries.append(f"{name} {source} meme origin meaning".strip())
    return queries


def build_hybrid_queries(
    vlm_queries: list[str],
    kb_results: list[dict[str, Any]],
    extraction: dict[str, Any],
    budget: int = 5,
) -> list[str]:
    """Fuse VLM-generated queries with KB-augmented queries."""
    final: list[str] = []
    seen: set[str] = set()

    def add(query: str) -> None:
        query = query.strip()
        if query and query not in seen and len(final) < budget:
            seen.add(query)
            final.append(query)

    for query in vlm_queries:
        add(query)
    for query in _build_kb_queries(kb_results):
        add(query)
    for entity in extraction.get("entities", []):
        name = str(entity.get("name", "")).strip()
        if name and _entity_is_confident(entity):
            add(f"{name} 梗 文化背景 出处" if _is_chinese(name) else f"{name} meme origin meaning")

    if not final:
        ocr = str(extraction.get("ocr", "")).strip()
        if ocr:
            add(f"{ocr} 梗 含义 出处" if _is_chinese(ocr) else f"{ocr} meme meaning origin")
    return final


async def stage_extract(
    *,
    async_client: Any,
    model: str,
    base64_image: str,
    extract_prompt: str,
) -> dict[str, Any]:
    """Run Stage 1 extraction and parse the result."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
                {"type": "text", "text": extract_prompt},
            ],
        }
    ]
    response = await retry_api_call(
        lambda: async_client.chat.completions.create(model=model, messages=messages)
    )
    if response is None:
        raise RuntimeError("KAR Stage 1 extraction failed")
    return parse_vlm_extraction(response.choices[0].message.content)


def stage_kb_retrieve(extraction: dict[str, Any], retriever: Any, top_k: int = 5) -> tuple[list[dict], dict]:
    """Run Stage 2 retrieval using a CultureBase-like retriever."""
    entity_names = [
        item["name"]
        for item in extraction.get("entities", [])
        if item.get("name") and _entity_is_confident(item)
    ]
    results = retriever.retrieve_for_meme(
        ocr=extraction.get("ocr", ""),
        visual_handle=extraction.get("visual_handle", ""),
        entity_names=entity_names,
        top_k=top_k,
    )
    return results, {
        "n_results": len(results),
        "entities": [
            {"name": item.get("name"), "similarity": item.get("similarity")}
            for item in results
        ],
    }


def stage_search(queries: list[str], search_provider: Any) -> tuple[str, dict]:
    """Run Stage 3 web search for each query."""
    summaries: list[str] = []
    meta = {"source": "tavily", "queries": queries, "results": 0, "per_query": []}
    for query in queries:
        result = retry_sync(lambda q=query: search_provider.text_search(q))
        query_meta = {"query": query, "success": False, "results": 0}
        if result is not None and "error" not in result.text_summary.lower():
            summaries.append(f"[Search: {query}]\n{result.text_summary}")
            query_meta["success"] = True
            query_meta["results"] = int(result.metadata.get("results", 0))
            meta["results"] += query_meta["results"]
        meta["per_query"].append(query_meta)
    if not summaries:
        return "No relevant cultural background information found.", meta
    return "\n\n".join(summaries), meta


async def stage_reason(
    *,
    async_client: Any,
    model: str,
    base64_image: str,
    reason_prompt_template: str,
    cultural_knowledge: str,
) -> str:
    """Run Stage 4 grounded reasoning."""
    prompt = reason_prompt_template.format(
        knowledge_source=KAR_REASON_KNOWLEDGE_SOURCE,
        knowledge=cultural_knowledge,
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    response = await retry_api_call(
        lambda: async_client.chat.completions.create(model=model, messages=messages)
    )
    if response is None:
        raise RuntimeError("KAR Stage 4 reasoning failed")
    return response.choices[0].message.content


async def run_kar(
    *,
    image_path: str | Path,
    config: KarConfig,
    retriever: Any,
    search_provider: Any | None = None,
    async_client: Any | None = None,
) -> dict[str, Any]:
    """Run KAR for one image and return response plus trace metadata."""
    client = async_client or build_async_client()
    search = search_provider or TavilySearchProvider(topk=config.search_top_k)
    base64_image = encode_image(image_path)

    extraction = await stage_extract(
        async_client=client,
        model=config.model,
        base64_image=base64_image,
        extract_prompt=config.extract_prompt,
    )
    kb_results, kb_meta = await asyncio.to_thread(
        stage_kb_retrieve, extraction, retriever, config.kb_top_k
    )
    queries = build_hybrid_queries(
        extraction.get("vlm_queries", []),
        kb_results,
        extraction,
        budget=config.query_budget,
    )
    knowledge, search_meta = await asyncio.to_thread(stage_search, queries, search)
    response = await stage_reason(
        async_client=client,
        model=config.model,
        base64_image=base64_image,
        reason_prompt_template=config.reason_prompt_template,
        cultural_knowledge=knowledge,
    )
    return {
        "response": response,
        "kar_trace": {
            "stage1_extraction": extraction,
            "stage2_kb": kb_meta,
            "stage3_queries": queries,
            "stage3_knowledge": knowledge,
            "stage3_meta": search_meta,
        },
    }


def dumps_result(result: dict[str, Any]) -> str:
    """Serialize a KAR result with stable formatting."""
    return json.dumps(result, ensure_ascii=False, indent=2)
