"""Long-term memory — SQLite-backed persistent storage for interactions and facts."""

from __future__ import annotations

import json
import time
import uuid

from loguru import logger

from psycho.storage.database import Database


class LongTermMemory:
    """
    Persistent memory backed by SQLite.

    Stores:
        - All user↔agent interactions (full history)
        - Explicit facts the agent decides are worth keeping
        - User preferences
    """

    def __init__(self, db: Database) -> None:
        self._db = db

    # ── Sessions ──────────────────────────────────────────────────

    async def create_session(self, session_id: str, domain: str = "general") -> None:
        await self._db.execute(
            "INSERT INTO sessions (id, started_at, domain) VALUES (?, ?, ?)",
            (session_id, time.time(), domain),
        )
        logger.debug(f"Session created: {session_id}")

    async def end_session(self, session_id: str, summary: str | None = None) -> None:
        await self._db.execute(
            "UPDATE sessions SET ended_at=?, summary=? WHERE id=?",
            (time.time(), summary, session_id),
        )

    async def update_session_message_count(self, session_id: str) -> None:
        await self._db.execute(
            "UPDATE sessions SET message_count = message_count + 1 WHERE id=?",
            (session_id,),
        )

    # ── Interactions ──────────────────────────────────────────────

    async def save_interaction(
        self,
        session_id: str,
        user_message: str,
        agent_response: str,
        domain: str = "general",
        tokens_used: int = 0,
    ) -> str:
        interaction_id = str(uuid.uuid4())
        await self._db.execute(
            """INSERT INTO interactions
               (id, session_id, user_message, agent_response, domain, timestamp, tokens_used)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                interaction_id,
                session_id,
                user_message,
                agent_response,
                domain,
                time.time(),
                tokens_used,
            ),
        )
        await self.update_session_message_count(session_id)
        return interaction_id

    async def get_recent_interactions(
        self, limit: int = 10, domain: str | None = None
    ) -> list[dict]:
        if domain:
            rows = await self._db.fetch_all(
                """SELECT user_message, agent_response, domain, timestamp
                   FROM interactions WHERE domain=?
                   ORDER BY timestamp DESC LIMIT ?""",
                (domain, limit),
            )
        else:
            rows = await self._db.fetch_all(
                """SELECT user_message, agent_response, domain, timestamp
                   FROM interactions ORDER BY timestamp DESC LIMIT ?""",
                (limit,),
            )
        return [dict(row) for row in rows]

    async def search_interactions(self, query: str, limit: int = 5) -> list[dict]:
        """Simple keyword search (Phase 2 upgrades this to semantic search)."""
        rows = await self._db.fetch_all(
            """SELECT user_message, agent_response, domain, timestamp
               FROM interactions
               WHERE user_message LIKE ? OR agent_response LIKE ?
               ORDER BY timestamp DESC LIMIT ?""",
            (f"%{query}%", f"%{query}%", limit),
        )
        return [dict(row) for row in rows]

    # ── Facts ─────────────────────────────────────────────────────

    async def save_fact(
        self,
        content: str,
        domain: str = "general",
        confidence: float = 0.5,
        source_session: str | None = None,
        tags: list[str] | None = None,
    ) -> str:
        fact_id = str(uuid.uuid4())
        now = time.time()
        await self._db.execute(
            """INSERT INTO facts (id, content, domain, confidence, created_at, updated_at, source_session, tags)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                fact_id,
                content,
                domain,
                confidence,
                now,
                now,
                source_session,
                json.dumps(tags or []),
            ),
        )
        logger.debug(f"Fact saved: {content[:60]}...")
        return fact_id

    async def get_facts(
        self, domain: str | None = None, min_confidence: float = 0.2, limit: int = 20
    ) -> list[dict]:
        if domain:
            rows = await self._db.fetch_all(
                """SELECT * FROM facts WHERE domain=? AND confidence >= ?
                   ORDER BY confidence DESC, updated_at DESC LIMIT ?""",
                (domain, min_confidence, limit),
            )
        else:
            rows = await self._db.fetch_all(
                """SELECT * FROM facts WHERE confidence >= ?
                   ORDER BY confidence DESC, updated_at DESC LIMIT ?""",
                (min_confidence, limit),
            )
        return [dict(row) for row in rows]

    async def update_fact_confidence(self, fact_id: str, delta: float) -> None:
        await self._db.execute(
            """UPDATE facts
               SET confidence = MAX(0.05, MIN(0.95, confidence + ?)), updated_at = ?
               WHERE id = ?""",
            (delta, time.time(), fact_id),
        )

    # ── Preferences ───────────────────────────────────────────────

    async def set_preference(
        self, key: str, value: str, domain: str = "general"
    ) -> None:
        await self._db.execute(
            """INSERT INTO preferences (key, value, domain, updated_at)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at""",
            (key, value, domain, time.time()),
        )

    async def get_preference(self, key: str) -> str | None:
        row = await self._db.fetch_one(
            "SELECT value FROM preferences WHERE key=?", (key,)
        )
        return row["value"] if row else None

    async def get_all_preferences(self) -> dict[str, str]:
        rows = await self._db.fetch_all("SELECT key, value FROM preferences")
        return {row["key"]: row["value"] for row in rows}

    # ── Stats ─────────────────────────────────────────────────────

    async def get_stats(self) -> dict:
        stats = await self._db.get_stats()

        # Most recent session info
        row = await self._db.fetch_one(
            "SELECT id, started_at FROM sessions ORDER BY started_at DESC LIMIT 1"
        )
        if row:
            stats["last_session_id"] = row["id"]
            stats["last_session_at"] = row["started_at"]

        return stats
