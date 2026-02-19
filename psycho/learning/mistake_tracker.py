"""
Mistake tracker — records agent errors, detects patterns, and injects
warnings into future system prompts so the agent never repeats mistakes.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from loguru import logger

from psycho.storage.database import Database
from psycho.storage.vector_store import VectorStore

if TYPE_CHECKING:
    pass

MISTAKES_COLLECTION = "mistakes"
MAX_WARNINGS_IN_PROMPT = 3
MIN_SIMILARITY_FOR_WARNING = 0.55


@dataclass
class MistakeRecord:
    id: str
    session_id: str
    user_input: str
    agent_response: str
    correction: str
    domain: str
    error_pattern: str = ""
    timestamp: float = field(default_factory=time.time)
    similar_count: int = 0

    def warning_text(self) -> str:
        return (
            f"• When asked '{self.user_input[:100]}', "
            f"you incorrectly said: '{self.agent_response[:120]}'. "
            f"Correct: '{self.correction[:150]}'"
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "user_input": self.user_input,
            "agent_response": self.agent_response,
            "correction": self.correction,
            "domain": self.domain,
            "error_pattern": self.error_pattern,
            "timestamp": self.timestamp,
            "similar_count": self.similar_count,
        }


class MistakeTracker:
    """
    Tracks agent mistakes and injects "known failure pattern" warnings
    into the system prompt before similar questions.

    Two-layer storage:
        SQLite     — full mistake records, queryable by domain / timestamp
        ChromaDB   — embeddings of user_input for semantic similarity search
    """

    def __init__(self, db: Database, vector_store: VectorStore) -> None:
        self._db = db
        self._vs = vector_store

    # ── Record a mistake ──────────────────────────────────────────

    async def record_mistake(
        self,
        session_id: str,
        user_input: str,
        agent_response: str,
        correction: str,
        domain: str = "general",
        error_pattern: str = "",
    ) -> str:
        """Record a confirmed agent mistake."""
        mistake_id = str(uuid.uuid4())
        record = MistakeRecord(
            id=mistake_id,
            session_id=session_id,
            user_input=user_input,
            agent_response=agent_response[:500],
            correction=correction[:300],
            domain=domain,
            error_pattern=error_pattern,
        )

        # Persist to SQLite
        await self._db.execute(
            """INSERT INTO mistakes
               (id, session_id, user_input, agent_response, correction,
                domain, error_pattern, timestamp, similar_count)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                record.id, record.session_id, record.user_input,
                record.agent_response, record.correction,
                record.domain, record.error_pattern,
                record.timestamp, 0,
            ),
        )

        # Index user_input in ChromaDB for semantic similarity
        self._vs.add(
            collection=MISTAKES_COLLECTION,
            doc_id=mistake_id,
            text=user_input,
            metadata={
                "mistake_id": mistake_id,
                "correction": correction[:200],
                "agent_response": agent_response[:200],
                "domain": domain,
                "error_pattern": error_pattern,
                "timestamp": record.timestamp,
            },
        )

        logger.info(
            f"Mistake recorded [{mistake_id[:8]}]: "
            f"'{user_input[:60]}' → corrected to '{correction[:60]}'"
        )
        return mistake_id

    # ── Retrieve warnings ─────────────────────────────────────────

    async def get_warnings_for_prompt(
        self, user_message: str, top_k: int = MAX_WARNINGS_IN_PROMPT
    ) -> list[str]:
        """
        Semantic search for past mistakes similar to the current question.
        Returns formatted warning strings to inject into the system prompt.
        """
        try:
            hits = self._vs.search(
                collection=MISTAKES_COLLECTION,
                query=user_message,
                top_k=top_k,
            )
        except Exception as e:
            logger.debug(f"Mistake similarity search failed: {e}")
            return []

        warnings = []
        for hit in hits:
            if hit["relevance"] < MIN_SIMILARITY_FOR_WARNING:
                continue
            meta = hit["metadata"]
            warnings.append(
                f"• Previously, when asked '{hit['text'][:80]}', "
                f"you said something incorrect. "
                f"The correct answer is: '{meta.get('correction', '?')[:120]}'"
            )
            # Increment similar_count for analytics
            mid = meta.get("mistake_id", "")
            if mid:
                await self._db.execute(
                    "UPDATE mistakes SET similar_count = similar_count + 1 WHERE id = ?",
                    (mid,),
                )

        if warnings:
            logger.debug(f"Found {len(warnings)} relevant mistake warnings")
        return warnings

    def build_warning_block(self, warnings: list[str]) -> str:
        """Format warnings as a system prompt block."""
        if not warnings:
            return ""
        lines = ["\n─── KNOWN FAILURE PATTERNS — AVOID THESE MISTAKES ───"]
        lines.extend(warnings)
        lines.append(
            "These are documented past errors. Think carefully before "
            "responding to similar questions.\n───────────────────────────────────"
        )
        return "\n".join(lines)

    # ── Analytics ─────────────────────────────────────────────────

    async def get_all_mistakes(
        self, domain: str | None = None, limit: int = 50
    ) -> list[dict]:
        if domain:
            rows = await self._db.fetch_all(
                "SELECT * FROM mistakes WHERE domain=? ORDER BY timestamp DESC LIMIT ?",
                (domain, limit),
            )
        else:
            rows = await self._db.fetch_all(
                "SELECT * FROM mistakes ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            )
        return [dict(row) for row in rows]

    async def get_stats(self) -> dict:
        row = await self._db.fetch_one("SELECT COUNT(*) FROM mistakes")
        total = row[0] if row else 0
        row2 = await self._db.fetch_one(
            "SELECT domain, COUNT(*) as cnt FROM mistakes "
            "GROUP BY domain ORDER BY cnt DESC LIMIT 1"
        )
        most_common_domain = row2[0] if row2 else "—"
        return {
            "total_mistakes": total,
            "most_common_domain": most_common_domain,
            "semantic_indexed": self._vs.count(MISTAKES_COLLECTION),
        }
