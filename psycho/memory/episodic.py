"""Episodic memory — timestamped event log, persisted to SQLite."""

from __future__ import annotations

import json
import time
import uuid

from loguru import logger

from psycho.storage.database import Database


class EpisodicMemory:
    """
    Ordered event log for time-based recall.

    Answers questions like:
        "What happened in session X?"
        "What did we talk about yesterday?"
        "When did the agent last make a mistake about topic Y?"

    Unlike semantic memory (similarity-based), episodic memory is
    sequence and causality-based. Events have a timeline.
    """

    def __init__(self, db: Database) -> None:
        self._db = db

    async def ensure_table(self) -> None:
        """Create events table if not exists (called at startup)."""
        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                id          TEXT    PRIMARY KEY,
                session_id  TEXT    NOT NULL,
                event_type  TEXT    NOT NULL,
                domain      TEXT    DEFAULT 'general',
                content     TEXT    NOT NULL,
                timestamp   REAL    NOT NULL,
                importance  REAL    DEFAULT 0.5
            )
            """
        )
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id)"
        )
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp)"
        )

    async def log_event(
        self,
        session_id: str,
        event_type: str,
        content: dict,
        domain: str = "general",
        importance: float = 0.5,
    ) -> str:
        """
        Log a new event.

        event_type examples:
            "interaction"   — normal conversation turn
            "correction"    — user corrected the agent
            "session_start" — beginning of a session
            "session_end"   — end of session with summary
            "insight"       — agent derived an insight
            "mistake"       — agent made a recorded mistake
        """
        event_id = str(uuid.uuid4())
        await self._db.execute(
            """INSERT INTO events (id, session_id, event_type, domain, content, timestamp, importance)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                event_id,
                session_id,
                event_type,
                domain,
                json.dumps(content),
                time.time(),
                importance,
            ),
        )
        logger.debug(f"Event logged: {event_type} [{event_id[:8]}]")
        return event_id

    async def get_session_events(self, session_id: str) -> list[dict]:
        """Get all events from a specific session, ordered by time."""
        rows = await self._db.fetch_all(
            "SELECT * FROM events WHERE session_id=? ORDER BY timestamp ASC",
            (session_id,),
        )
        return [self._deserialize(row) for row in rows]

    async def get_recent_events(
        self, limit: int = 20, event_type: str | None = None
    ) -> list[dict]:
        """Get most recent events across all sessions."""
        if event_type:
            rows = await self._db.fetch_all(
                "SELECT * FROM events WHERE event_type=? ORDER BY timestamp DESC LIMIT ?",
                (event_type, limit),
            )
        else:
            rows = await self._db.fetch_all(
                "SELECT * FROM events ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            )
        return [self._deserialize(row) for row in rows]

    async def get_important_events(
        self, min_importance: float = 0.7, limit: int = 10
    ) -> list[dict]:
        """Get high-importance events — useful for reflection."""
        rows = await self._db.fetch_all(
            "SELECT * FROM events WHERE importance >= ? ORDER BY timestamp DESC LIMIT ?",
            (min_importance, limit),
        )
        return [self._deserialize(row) for row in rows]

    async def get_stats(self) -> dict:
        row = await self._db.fetch_one("SELECT COUNT(*) FROM events")
        return {"total_events": row[0] if row else 0}

    @staticmethod
    def _deserialize(row) -> dict:
        d = dict(row)
        try:
            d["content"] = json.loads(d["content"])
        except (json.JSONDecodeError, KeyError):
            pass
        return d
