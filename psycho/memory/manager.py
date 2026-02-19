"""MemoryManager — unified facade over all memory systems."""

from __future__ import annotations

import asyncio

from loguru import logger

from psycho.config import get_settings
from psycho.storage.database import Database
from psycho.storage.vector_store import VectorStore

from .episodic import EpisodicMemory
from .long_term import LongTermMemory
from .semantic import SemanticMemory
from .short_term import ShortTermMemory


class MemoryManager:
    """
    Central memory coordinator.

    Four layers:
        short_term  — in-process deque (last N conversation turns)
        long_term   — SQLite (sessions, interactions, facts, preferences)
        semantic    — ChromaDB (vector similarity over all past interactions)
        episodic    — SQLite event log (ordered timeline of what happened)

    The agent always talks to this class, never to individual stores directly.
    """

    def __init__(self, db: Database, vector_store: VectorStore) -> None:
        settings = get_settings()
        self.short_term = ShortTermMemory(max_turns=settings.max_short_term_messages)
        self.long_term = LongTermMemory(db)
        self.semantic = SemanticMemory(vector_store)
        self.episodic = EpisodicMemory(db)
        self._settings = settings

    async def initialize(self) -> None:
        """Run async initialization (create tables etc.)."""
        await self.episodic.ensure_table()

    async def add_interaction(
        self,
        session_id: str,
        user_message: str,
        agent_response: str,
        domain: str = "general",
        tokens_used: int = 0,
    ) -> None:
        """Record a completed interaction across all relevant memory stores."""
        # 1. Short-term (in-process, immediate context window)
        self.short_term.add(user_message, agent_response)

        # 2. Long-term SQL persistence + get interaction ID
        interaction_id = await self.long_term.save_interaction(
            session_id=session_id,
            user_message=user_message,
            agent_response=agent_response,
            domain=domain,
            tokens_used=tokens_used,
        )

        # 3. Semantic (ChromaDB) + episodic (event log) — run in parallel
        await asyncio.gather(
            self.semantic.store_interaction(
                session_id=session_id,
                user_message=user_message,
                agent_response=agent_response,
                domain=domain,
                interaction_id=interaction_id,
            ),
            self.episodic.log_event(
                session_id=session_id,
                event_type="interaction",
                domain=domain,
                content={
                    "user": user_message[:300],
                    "agent": agent_response[:300],
                    "tokens": tokens_used,
                },
                importance=0.4,
            ),
        )
        logger.debug(
            f"Interaction recorded across 4 memory stores: {len(user_message)}→{len(agent_response)} chars"
        )

    async def retrieve_context(
        self, query: str, domain: str | None = None
    ) -> list[dict]:
        """
        Retrieve relevant past context for a query.

        Hybrid retrieval:
            1. Semantic search (ChromaDB) — finds meaning-based matches
            2. Keyword fallback (SQLite) — ensures we don't miss exact matches

        Returns a deduplicated, ranked list of relevant past exchanges.
        """
        # Semantic search (primary — finds related topics even with different wording)
        semantic_hits = await self.semantic.search_interactions(
            query=query,
            top_k=self._settings.max_context_memories,
            domain=domain,
            min_relevance=0.35,
        )

        if semantic_hits:
            logger.debug(f"Semantic retrieval: {len(semantic_hits)} hits")
            return semantic_hits

        # Keyword fallback (secondary — for very specific terms or when semantic cache is cold)
        keyword_hits = await self.long_term.search_interactions(
            query=query, limit=self._settings.max_context_memories
        )
        if keyword_hits:
            logger.debug(f"Keyword fallback retrieval: {len(keyword_hits)} hits")
        return keyword_hits

    async def get_recent_history(self, limit: int = 5) -> list[dict]:
        """Return last N interactions from long-term memory."""
        return await self.long_term.get_recent_interactions(limit=limit)

    async def get_stats(self) -> dict:
        """Aggregate stats across all memory systems."""
        stats = await self.long_term.get_stats()
        stats["short_term_turns"] = len(self.short_term)

        # Vector store stats
        vec_stats = self.semantic.get_stats()
        stats["semantic_interactions"] = vec_stats.get("interactions", 0)
        stats["semantic_facts"] = vec_stats.get("facts", 0)

        # Event log stats
        ep_stats = await self.episodic.get_stats()
        stats["total_events"] = ep_stats.get("total_events", 0)

        return stats
