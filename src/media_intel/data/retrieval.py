"""Retrieval-augmented generation building blocks.

- :class:`EmbeddingProvider` protocol with a deterministic mock and an
  OpenAI-compatible implementation.
- :class:`VectorSearchBackend` protocol with a local cosine-similarity
  simulator (Delta-like persistence) and a Databricks Vector Search adapter.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any, Protocol

import httpx


# --------------------------------------------------------------------- embeddings


class EmbeddingProvider(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class MockEmbeddings:
    """Feature-hashing embeddings: deterministic, dependency-free, decent relevance.

    Each token hashes to multiple positions (reduces collision artifacts) and
    bigrams add local word-order signal - good enough for demo RAG quality.
    """

    def __init__(self, dim: int = 128):
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._one(t) for t in texts]

    def _one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        tokens = [_stem(t) for t in _tokenize(text)]
        for token in tokens:
            weight = _weight(token)
            for seed in (token, "b:" + token, "c:" + token):
                idx = _stable_hash(seed) % self.dim
                sign = 1.0 if _stable_hash("s:" + seed) % 2 == 0 else -1.0
                vec[idx] += weight * sign
        for a, b in zip(tokens, tokens[1:]):
            bg = a + "_" + b
            idx = _stable_hash(bg) % self.dim
            sign = 1.0 if _stable_hash("s:" + bg) % 2 == 0 else -1.0
            vec[idx] += 0.7 * sign
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [round(v / norm, 6) for v in vec]


class OpenAIEmbeddings:
    """Embeddings via any OpenAI-compatible ``/embeddings`` endpoint."""

    def __init__(self, model: str = "text-embedding-3-small", base_url: str | None = None, api_key: str | None = None):
        self.model = model
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")

    def embed(self, texts: list[str]) -> list[list[float]]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        with httpx.Client(timeout=30) as client:
            resp = client.post(
                f"{self.base_url}/embeddings",
                json={"model": self.model, "input": texts},
                headers=headers,
            )
            resp.raise_for_status()
        data = sorted(resp.json()["data"], key=lambda d: d["index"])
        return [item["embedding"] for item in data]


def make_embeddings(provider: str, model: str, dim: int) -> EmbeddingProvider:
    if provider == "openai":
        return OpenAIEmbeddings(model=model)
    return MockEmbeddings(dim=dim)


# ------------------------------------------------------------------ vector search


class VectorSearchBackend(Protocol):
    def upsert(self, index_id: str, records: list[dict[str, Any]]) -> int: ...

    def query(self, index_id: str, vector: list[float], k: int = 5, filter: dict[str, Any] | None = None) -> list[dict[str, Any]]: ...


class LocalVectorSearch:
    """In-memory cosine-similarity index persisted as JSON (Delta-like).

    Interface mirrors Databricks Vector Search: ``upsert`` + ``query`` with
    metadata filters, so swapping backends is a one-line factory change.
    """

    def __init__(self, root: str = ".lakehouse/vectors", embedding_dim: int = 128):
        self.root = root
        self.embedding_dim = embedding_dim
        self._indexes: dict[str, list[dict[str, Any]]] = {}
        self._load()

    def upsert(self, index_id: str, records: list[dict[str, Any]]) -> int:
        table = self._indexes.setdefault(index_id, [])
        by_id = {r["id"]: r for r in table}
        for rec in records:
            if len(rec.get("vector", [])) != self.embedding_dim:
                raise ValueError(f"vector dim mismatch: expected {self.embedding_dim}")
            by_id[rec["id"]] = rec
        self._indexes[index_id] = list(by_id.values())
        self._persist(index_id)
        return len(records)

    def query(
        self,
        index_id: str,
        vector: list[float],
        k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        rows = self._indexes.get(index_id, [])
        if filter:
            rows = [r for r in rows if all(r.get("metadata", {}).get(key) == val for key, val in filter.items())]
        scored = []
        for row in rows:
            sim = _cosine(vector, row["vector"])
            scored.append({**row, "score": round(sim, 6)})
        scored.sort(key=lambda r: r["score"], reverse=True)
        return scored[:k]

    def _persist(self, index_id: str) -> None:
        os.makedirs(self.root, exist_ok=True)
        with open(os.path.join(self.root, f"{index_id}.json"), "w", encoding="utf-8") as fh:
            json.dump(self._indexes[index_id], fh)

    def _load(self) -> None:
        os.makedirs(self.root, exist_ok=True)
        for fname in os.listdir(self.root):
            if fname.endswith(".json"):
                with open(os.path.join(self.root, fname), encoding="utf-8") as fh:
                    self._indexes[fname[:-5]] = json.load(fh)


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


def _tokenize(text: str) -> list[str]:
    return [t for t in "".join(c if c.isalnum() else " " for c in text.lower()).split() if len(t) > 1]


def _stem(token: str) -> str:
    """Light suffix normalization so 'scored'/'scores' align."""
    if len(token) > 5 and token.endswith("ing"):
        token = token[:-3]
    elif len(token) > 4 and token.endswith("ed"):
        token = token[:-2]
    elif len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        token = token[:-1]
    if len(token) > 4 and token.endswith("e"):
        token = token[:-1]
    return token


_STOPWORDS = {
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "and", "or", "so",
    "far", "during", "with", "by", "is", "are", "was", "were", "who", "that",
    "this", "it", "its", "as", "be", "been", "has", "have", "had", "their",
}


def _weight(token: str) -> float:
    """Down-weight stopwords/digits; boost longer content words (cheap IDF)."""
    if token in _STOPWORDS:
        return 0.15
    if token.isdigit():
        return 0.4
    if len(token) >= 7:
        return 1.25
    return 1.0


def _stable_hash(token: str) -> int:
    h = 2166136261
    for ch in token:
        h = (h ^ ord(ch)) * 16777619 & 0xFFFFFFFF
    return h


# --------------------------------------------------------------- RAG pipeline glue


class Retriever:
    """Embeds a query and pulls top-k chunks from a vector index."""

    def __init__(self, embeddings: EmbeddingProvider, backend: VectorSearchBackend, default_k: int = 4):
        self.embeddings = embeddings
        self.backend = backend
        self.default_k = default_k

    def retrieve(self, index_id: str, query: str, k: int | None = None, filter: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        vector = self.embeddings.embed([query])[0]
        return self.backend.query(index_id, vector, k=k or self.default_k, filter=filter)
