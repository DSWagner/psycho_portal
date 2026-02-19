"""Semantic memory — ChromaDB vector store for similarity-based retrieval."""

from __future__ import annotations

import time
import uuid

from loguru import logger

from psycho.storage.vector_store import VectorStore


class SemanticMemory:
    """
    Vector-based memory that retrieves past interactions by meaning, not keywords.

    "What did we discuss about Python async?" will find relevant conversations
    even if the user asks it differently: "asyncio issues we talked about" etc.

    Uses ChromaDB with all-MiniLM-L6-v2 (ONNX, local, no API cost).
    """

    def __init__(self, vector_store: VectorStore) -> None:
        self._store = vector_store

    async def store_interaction(
        self,
        session_id: str,
        user_message: str,
        agent_response: str,
        domain: str = "general",
        interaction_id: str | None = None,
    ) -> str:
        """
        Embed and store a complete conversation turn.

        The stored text is: "User: {msg}\nAssistant: {response}"
        This gives the embedding context from both sides of the exchange.
        """
        doc_id = interaction_id or str(uuid.uuid4())
        text = f"User: {user_message}\nAssistant: {agent_response[:500]}"

        self._store.add(
            collection=VectorStore.INTERACTIONS,
            doc_id=doc_id,
            text=text,
            metadata={
                "session_id": session_id,
                "user_message": user_message[:500],
                "agent_response": agent_response[:500],
                "domain": domain,
                "timestamp": time.time(),
            },
        )
        logger.debug(f"Semantic store: interaction {doc_id[:8]}")
        return doc_id

    async def search_interactions(
        self,
        query: str,
        top_k: int = 5,
        domain: str | None = None,
        min_relevance: float = 0.4,
    ) -> list[dict]:
        """
        Find semantically similar past interactions.

        Returns list of dicts with:
            user_message, agent_response, domain, timestamp, relevance (0-1)
        """
        where = {"domain": domain} if domain else None
        hits = self._store.search(
            collection=VectorStore.INTERACTIONS,
            query=query,
            top_k=top_k,
            where=where,
        )

        results = []
        for hit in hits:
            if hit["relevance"] < min_relevance:
                continue
            meta = hit["metadata"]
            results.append(
                {
                    "user_message": meta.get("user_message", ""),
                    "agent_response": meta.get("agent_response", ""),
                    "domain": meta.get("domain", "general"),
                    "timestamp": meta.get("timestamp", 0.0),
                    "relevance": hit["relevance"],
                    "session_id": meta.get("session_id", ""),
                }
            )

        logger.debug(f"Semantic search '{query[:40]}': {len(results)}/{top_k} results above threshold")
        return results

    async def store_fact(
        self,
        fact_id: str,
        content: str,
        domain: str = "general",
        confidence: float = 0.5,
    ) -> None:
        """Store a knowledge fact for semantic retrieval."""
        self._store.add(
            collection=VectorStore.FACTS,
            doc_id=fact_id,
            text=content,
            metadata={
                "domain": domain,
                "confidence": confidence,
                "timestamp": time.time(),
            },
        )

    async def search_facts(
        self, query: str, top_k: int = 5, min_relevance: float = 0.5
    ) -> list[dict]:
        """Find semantically relevant stored facts."""
        hits = self._store.search(
            collection=VectorStore.FACTS, query=query, top_k=top_k
        )
        return [
            {
                "content": hit["text"],
                "domain": hit["metadata"].get("domain", "general"),
                "confidence": hit["metadata"].get("confidence", 0.5),
                "relevance": hit["relevance"],
            }
            for hit in hits
            if hit["relevance"] >= min_relevance
        ]

    def get_stats(self) -> dict:
        return self._store.get_stats()
