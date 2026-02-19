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
"""

CURRENT_VERSION = 1


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
