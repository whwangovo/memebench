"""Minimal search provider used by the public KAR implementation."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import requests


@dataclass
class TextSearchResult:
    """Text search result summary plus lightweight metadata."""

    text_summary: str
    metadata: dict[str, Any] = field(default_factory=dict)


class TavilySearchProvider:
    """Small Tavily wrapper for cultural background retrieval.

    Set `TAVILY_API_KEY` in the environment or pass `api_key` explicitly.
    """

    def __init__(
        self,
        api_key: str | None = None,
        topk: int = 5,
        timeout: int = 30,
        endpoint: str = "https://api.tavily.com/search",
    ) -> None:
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        self.topk = topk
        self.timeout = timeout
        self.endpoint = endpoint
        if not self.api_key:
            raise ValueError("TAVILY_API_KEY is required for TavilySearchProvider")

    def text_search(self, query: str, **kwargs: Any) -> TextSearchResult:
        """Run a text search and return formatted snippets."""
        max_results = int(kwargs.get("topk", self.topk))
        payload = {
            "api_key": self.api_key,
            "query": query,
            "search_depth": kwargs.get("search_depth", "basic"),
            "max_results": max_results,
            "include_answer": False,
            "include_raw_content": False,
        }
        try:
            response = requests.post(self.endpoint, json=payload, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            return TextSearchResult(
                text_summary=f"[Search error] {exc}",
                metadata={"source": "tavily", "query": query, "error": str(exc)},
            )

        rows = data.get("results") or []
        if not rows:
            return TextSearchResult(
                text_summary="[Search results] No results found.",
                metadata={"source": "tavily", "query": query, "results": 0},
            )

        lines = ["[Search results]"]
        for index, row in enumerate(rows[:max_results], start=1):
            title = row.get("title") or "Untitled"
            url = row.get("url") or ""
            content = row.get("content") or row.get("snippet") or ""
            lines.append(f"{index}. {title} ({url})\n{content}".strip())

        return TextSearchResult(
            text_summary="\n\n".join(lines),
            metadata={"source": "tavily", "query": query, "results": len(rows[:max_results])},
        )
