"""ChromaDB persistent vector store wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings
from loguru import logger


class VectorStore:
    """
    Thin wrapper around ChromaDB PersistentClient.

    Uses ChromaDB's built-in default embedding function:
    - `all-MiniLM-L6-v2` via ONNX runtime (downloaded once, ~79MB, cached at ~/.cache/chroma)
    - Produces 384-dim embeddings, runs on CPU, fast and local

    Collections:
        "interactions" — full conversation turns (user + agent)
        "facts"        — extracted facts and knowledge
    """

    INTERACTIONS = "interactions"
    FACTS = "facts"

    def __init__(self, persist_path: Path) -> None:
        persist_path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(persist_path),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collections: dict[str, chromadb.Collection] = {}
        logger.debug(f"VectorStore initialized at {persist_path}")

    def _get_collection(self, name: str) -> chromadb.Collection:
        if name not in self._collections:
            self._collections[name] = self._client.get_or_create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collections[name]

    def add(
        self,
        collection: str,
        doc_id: str,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add or update a document in the collection."""
        col = self._get_collection(collection)
        col.upsert(
            ids=[doc_id],
            documents=[text],
            metadatas=[metadata or {}],
        )

    def search(
        self,
        collection: str,
        query: str,
        top_k: int = 5,
        where: dict | None = None,
    ) -> list[dict]:
        """
        Semantic similarity search.

        Returns list of dicts with keys: id, text, metadata, distance
        (distance 0 = identical, 2 = maximally different in cosine space)
        """
        col = self._get_collection(collection)
        count = col.count()
        if count == 0:
            return []

        top_k = min(top_k, count)
        kwargs: dict[str, Any] = {
            "query_texts": [query],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        results = col.query(**kwargs)

        hits = []
        for i, doc_id in enumerate(results["ids"][0]):
            hits.append(
                {
                    "id": doc_id,
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i],
                    "relevance": 1.0 - (results["distances"][0][i] / 2.0),
                }
            )
        # Sort by relevance descending
        hits.sort(key=lambda x: x["relevance"], reverse=True)
        return hits

    def delete(self, collection: str, doc_id: str) -> None:
        col = self._get_collection(collection)
        col.delete(ids=[doc_id])

    def count(self, collection: str) -> int:
        return self._get_collection(collection).count()

    def get_stats(self) -> dict:
        return {
            "interactions": self.count(self.INTERACTIONS),
            "facts": self.count(self.FACTS),
        }
