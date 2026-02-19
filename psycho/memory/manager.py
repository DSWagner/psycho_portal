"""MemoryManager — unified facade over all memory systems."""

from __future__ import annotations

from loguru import logger

from psycho.config import get_settings
from psycho.storage.database import Database

from .long_term import LongTermMemory
from .short_term import ShortTermMemory


class MemoryManager:
    """
    Central memory coordinator.

    Phase 1: short-term + long-term (SQLite)
    Phase 2: + semantic (ChromaDB) + episodic event log

    The agent always talks to this class, never to individual stores directly.
    """

    def __init__(self, db: Database) -> None:
        settings = get_settings()
        self.short_term = ShortTermMemory(
            max_turns=settings.max_short_term_messages
        )
        self.long_term = LongTermMemory(db)
        self._settings = settings

    async def add_interaction(
        self,
        session_id: str,
        user_message: str,
        agent_response: str,
        domain: str = "general",
        tokens_used: int = 0,
    ) -> None:
        """Record a completed interaction in all relevant memory stores."""
        # Short-term (in-process, immediate context)
        self.short_term.add(user_message, agent_response)

        # Long-term (persisted to SQLite)
        await self.long_term.save_interaction(
            session_id=session_id,
            user_message=user_message,
            agent_response=agent_response,
            domain=domain,
            tokens_used=tokens_used,
        )
        logger.debug(f"Interaction recorded: {len(user_message)} chars → {len(agent_response)} chars")

    async def retrieve_context(
        self, query: str, domain: str | None = None
    ) -> list[dict]:
        """
        Retrieve relevant past context for a query.

        Phase 1: keyword search on long-term memory
        Phase 2: semantic vector search via ChromaDB
        """
        memories = await self.long_term.search_interactions(
            query=query, limit=self._settings.max_context_memories
        )
        return memories

    async def get_recent_history(self, limit: int = 5) -> list[dict]:
        """Return last N interactions from long-term memory (for display/stats)."""
        return await self.long_term.get_recent_interactions(limit=limit)

    async def get_stats(self) -> dict:
        """Aggregate stats across all memory systems."""
        stats = await self.long_term.get_stats()
        stats["short_term_turns"] = len(self.short_term)
        return stats
