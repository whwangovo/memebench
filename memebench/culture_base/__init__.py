"""CultureBase retrieval utilities for MemeBench.

KB Schema is designed to be compatible with web data sources (萌娘百科, 小鸡词典):
  - name: entity canonical name (web: page title)
  - aliases: alternative names (web: redirects, parenthetical variants)
  - source: origin work/franchise (web: infobox)
  - description: entity traits/identity (web: page summary)
  - cultural_usage: meme catchphrase or cultural code (web: meme section)
  - visual_desc: visual appearance description (web: infobox image + VLM)
  - meme_ids: which memes reference this entity

Background is NOT stored — it comes from web search at query time.

Embedding text = "{name} | {aliases} | {source} | {description} | {visual_desc} | {cultural_usage}"
"""

import json
import os
import threading

import numpy as np
from sentence_transformers import SentenceTransformer

CB_DIR = "data/culture_base"
CB_ENTRIES_FILE = os.path.join(CB_DIR, "kb_entries.json")
CB_EMBEDDINGS_FILE = os.path.join(CB_DIR, "kb_embeddings.npy")
CB_META_FILE = os.path.join(CB_DIR, "kb_meta.json")

EMBEDDING_MODEL = "BAAI/bge-m3"


class CultureBaseRetriever:
    """Multi-query retriever for CultureBase.

    3-route query strategy:
      Q1: OCR text → matches name, aliases, cultural_usage
      Q2: VLM visual handle → matches visual_desc, description
      Q3: Confident entity names → matches name, aliases, source

    Each CultureBase entry score = max(similarity across all queries).
    Results are deduplicated by entity name and returned as top-K.
    """

    def __init__(self, top_k: int = 5, threshold: float = 0.5, cb_dir: str | None = None):
        self.top_k = top_k
        self.threshold = threshold
        self._cb_dir = cb_dir or CB_DIR
        self._model = None
        self._entries = None
        self._embeddings = None
        self._load_lock = threading.Lock()

    def _load(self):
        if self._model is not None:
            return

        with self._load_lock:
            if self._model is not None:
                return

            entries_file = os.path.join(self._cb_dir, "kb_entries.json")
            embeddings_file = os.path.join(self._cb_dir, "kb_embeddings.npy")

            with open(entries_file, "r", encoding="utf-8") as f:
                self._entries = json.load(f)

            self._embeddings = np.load(embeddings_file)
            self._model = SentenceTransformer(EMBEDDING_MODEL)

            print(f"CultureBase loaded: {len(self._entries)} entries, "
                  f"dim={self._embeddings.shape[1]}")

    def _encode_queries(self, texts: list[str]) -> np.ndarray:
        """Batch-encode multiple query texts."""
        emb = self._model.encode(texts, normalize_embeddings=True)
        return np.array(emb, dtype=np.float32)

    def retrieve(
        self,
        queries: list[str],
        top_k: int | None = None,
        threshold: float | None = None,
    ) -> list[dict]:
        """Retrieve entities using multiple query texts.

        Each KB entry is scored by max similarity across all queries.
        Results are deduplicated by entity name (top score wins).

        Returns:
            List of dicts with KB entry fields + similarity score, sorted desc.
        """
        self._load()

        k = top_k or self.top_k
        thresh = threshold or self.threshold

        # Filter out empty/short queries
        valid_queries = [q.strip() for q in queries if q and len(q.strip()) >= 2]
        if not valid_queries:
            return []

        # Batch encode all queries at once
        q_embs = self._encode_queries(valid_queries)

        # Score: max similarity across all queries for each KB entry
        # q_embs: (n_queries, dim), embeddings: (n_entries, dim)
        all_scores = q_embs @ self._embeddings.T  # (n_queries, n_entries)
        best_scores = all_scores.max(axis=0)  # (n_entries,)

        # Rank and deduplicate by entity name
        ranked_indices = np.argsort(-best_scores)
        results = []
        seen_names = set()

        for idx in ranked_indices:
            score = float(best_scores[idx])
            if score < thresh:
                break

            entry = self._entries[idx]
            name = entry["name"]
            if name in seen_names:
                continue
            seen_names.add(name)

            results.append({**entry, "similarity": score})
            if len(results) >= k:
                break

        return results

    def retrieve_for_meme(
        self,
        ocr: str = "",
        visual_handle: str = "",
        entity_names: list[str] | None = None,
        top_k: int | None = None,
    ) -> list[dict]:
        """Build 3-route multi-query from meme features and retrieve.

        Q1: OCR text (matches name, aliases, cultural_usage)
        Q2: Clean visual handle (matches visual_desc, description)
        Q3: Confident entity names (matches name, aliases, source)
        """
        queries = []

        # Q1: OCR text
        if ocr and len(ocr.strip()) > 2:
            queries.append(ocr.strip())

        # Q2: Visual handle (kept short to stay orthogonal to OCR)
        if visual_handle and len(visual_handle.strip()) > 5:
            queries.append(visual_handle.strip()[:160])

        # Q3: Entity names
        if entity_names:
            for entity_name in entity_names:
                if entity_name and len(entity_name.strip()) > 1:
                    queries.append(entity_name.strip())

        if not queries:
            return []

        return self.retrieve(queries, top_k=top_k)
