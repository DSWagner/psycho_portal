"""SQLite database — async connection, schema migrations."""

from __future__ import annotations

import asyncio
from pathlib import Path

import aiosqlite
from loguru import logger


SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    id          TEXT    PRIMARY KEY,
    started_at  REAL    NOT NULL,
    ended_at    REAL,
    message_count INTEGER DEFAULT 0,
    domain      TEXT    DEFAULT 'general',
    summary     TEXT
);

CREATE TABLE IF NOT EXISTS interactions (
    id              TEXT    PRIMARY KEY,
    session_id      TEXT    NOT NULL,
    user_message    TEXT    NOT NULL,
    agent_response  TEXT    NOT NULL,
    domain          TEXT    DEFAULT 'general',
    timestamp       REAL    NOT NULL,
    tokens_used     INTEGER DEFAULT 0,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

CREATE INDEX IF NOT EXISTS idx_interactions_session ON interactions(session_id);
CREATE INDEX IF NOT EXISTS idx_interactions_timestamp ON interactions(timestamp);

CREATE TABLE IF NOT EXISTS facts (
    id              TEXT    PRIMARY KEY,
    content         TEXT    NOT NULL,
    domain          TEXT    DEFAULT 'general',
    confidence      REAL    DEFAULT 0.5,
    created_at      REAL    NOT NULL,
    updated_at      REAL    NOT NULL,
    source_session  TEXT,
    tags            TEXT    DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS mistakes (
    id              TEXT    PRIMARY KEY,
    session_id      TEXT,
    user_input      TEXT    NOT NULL,
    agent_response  TEXT    NOT NULL,
    correction      TEXT    NOT NULL,
    domain          TEXT    DEFAULT 'general',
    error_pattern   TEXT,
    timestamp       REAL    NOT NULL,
    similar_count   INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS preferences (
    key         TEXT    PRIMARY KEY,
    value       TEXT    NOT NULL,
    domain      TEXT    DEFAULT 'general',
    updated_at  REAL    NOT NULL
);

CREATE TABLE IF NOT EXISTS health_metrics (
    id          TEXT    PRIMARY KEY,
    metric_type TEXT    NOT NULL,
    value       REAL    NOT NULL,
    unit        TEXT    NOT NULL,
    notes       TEXT    DEFAULT '',
    timestamp   REAL    NOT NULL,
    session_id  TEXT
);

CREATE INDEX IF NOT EXISTS idx_health_type ON health_metrics(metric_type, timestamp);

CREATE TABLE IF NOT EXISTS tasks (
    id          TEXT    PRIMARY KEY,
    title       TEXT    NOT NULL,
    description TEXT    DEFAULT '',
    priority    TEXT    DEFAULT 'normal',
    status      TEXT    DEFAULT 'pending',
    due_date    TEXT,
    tags        TEXT    DEFAULT '[]',
    created_at  REAL    NOT NULL,
    updated_at  REAL    NOT NULL,
    completed_at REAL,
    session_id  TEXT
);

CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status, priority);

CREATE TABLE IF NOT EXISTS reminders (
    id              TEXT    PRIMARY KEY,
    title           TEXT    NOT NULL,
    notes           TEXT    DEFAULT '',
    due_timestamp   REAL    NOT NULL,
    recurrence      TEXT    DEFAULT 'none',
    priority        TEXT    DEFAULT 'normal',
    completed       INTEGER DEFAULT 0,
    snoozed_until   REAL    DEFAULT 0,
    created_at      REAL    NOT NULL,
    session_id      TEXT    DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_reminders_due ON reminders(due_timestamp, completed);

CREATE TABLE IF NOT EXISTS calendar_events (
    id                  TEXT    PRIMARY KEY,
    title               TEXT    NOT NULL,
    start_timestamp     REAL    NOT NULL,
    end_timestamp       REAL    NOT NULL,
    location            TEXT    DEFAULT '',
    notes               TEXT    DEFAULT '',
    recurrence          TEXT    DEFAULT 'none',
    google_event_id     TEXT    DEFAULT '',
    all_day             INTEGER DEFAULT 0,
    reminder_minutes    INTEGER DEFAULT 15,
    created_at          REAL    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_calendar_start ON calendar_events(start_timestamp);
"""

CURRENT_VERSION = 3


class Database:
    """Async SQLite database manager with schema migrations."""

    def __init__(self, db_path: Path) -> None:
        self._path = db_path
        self._conn: aiosqlite.Connection | None = None
        self._lock = asyncio.Lock()

    async def connect(self) -> None:
        """Open connection and run migrations."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = await aiosqlite.connect(self._path)
        self._conn.row_factory = aiosqlite.Row
        await self._run_migrations()
        logger.debug(f"Database connected: {self._path}")

    async def close(self) -> None:
        if self._conn:
            await self._conn.close()
            self._conn = None
            logger.debug("Database connection closed")

    async def _run_migrations(self) -> None:
        async with self._lock:
            await self._conn.executescript(SCHEMA_SQL)
            await self._conn.commit()

            # Check/set schema version
            cursor = await self._conn.execute(
                "SELECT MAX(version) FROM schema_version"
            )
            row = await cursor.fetchone()
            current = row[0] if row and row[0] else 0

            if current < CURRENT_VERSION:
                import time
                await self._conn.execute(
                    "INSERT OR REPLACE INTO schema_version (version, applied_at) VALUES (?, ?)",
                    (CURRENT_VERSION, time.time()),
                )
                await self._conn.commit()
                logger.info(f"Schema migrated to version {CURRENT_VERSION}")

    @property
    def conn(self) -> aiosqlite.Connection:
        if not self._conn:
            raise RuntimeError("Database not connected. Call connect() first.")
        return self._conn

    async def execute(self, sql: str, params: tuple = ()) -> aiosqlite.Cursor:
        async with self._lock:
            cursor = await self._conn.execute(sql, params)
            await self._conn.commit()
            return cursor

    async def fetch_all(self, sql: str, params: tuple = ()) -> list[aiosqlite.Row]:
        cursor = await self._conn.execute(sql, params)
        return await cursor.fetchall()

    async def fetch_one(
        self, sql: str, params: tuple = ()
    ) -> aiosqlite.Row | None:
        cursor = await self._conn.execute(sql, params)
        return await cursor.fetchone()

    async def get_stats(self) -> dict:
        """Return high-level database statistics."""
        stats = {}
        for table in ("sessions", "interactions", "facts", "mistakes"):
            row = await self.fetch_one(f"SELECT COUNT(*) FROM {table}")
            stats[table] = row[0] if row else 0
        return stats
