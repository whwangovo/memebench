"""Minimal OpenAI-compatible LLM client helpers for MemeBench."""

from __future__ import annotations

import base64
import os
from pathlib import Path


def encode_image(image_path: str | Path) -> str:
    """Read a local image and return a base64 string."""
    with Path(image_path).open("rb") as handle:
        return base64.b64encode(handle.read()).decode("utf-8")


def build_async_client(
    *,
    api_key: str | None = None,
    base_url: str | None = None,
) -> AsyncOpenAI:
    """Build an OpenAI-compatible async client.

    Environment fallbacks:
      - `OPENAI_API_KEY`
      - `OPENAI_BASE_URL` (optional)
    """
    from openai import AsyncOpenAI

    return AsyncOpenAI(
        api_key=api_key or os.getenv("OPENAI_API_KEY"),
        base_url=base_url or os.getenv("OPENAI_BASE_URL") or None,
    )
