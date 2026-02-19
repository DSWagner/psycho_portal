"""PsychoAgent — initializes and wires all subsystems."""

from __future__ import annotations

import uuid
from pathlib import Path

from loguru import logger

from psycho.config import get_settings
from psycho.config.constants import AGENT_NAME, SYSTEM_PROMPT_BASE
from psycho.llm import create_provider
from psycho.llm.base import LLMProvider
from psycho.memory import MemoryManager
from psycho.storage.database import Database

from .loop import AgentLoop


class PsychoAgent:
    """
    Top-level agent class.

    Owns all subsystems:
        - LLM provider (Anthropic or Ollama)
        - Memory manager (short-term + long-term)
        - Database connection
        - Agent loop (the perceive→think→act→learn cycle)

    Usage:
        agent = PsychoAgent()
        await agent.start()
        response = await agent.chat("Hello!")
        await agent.stop()
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._settings.ensure_data_dirs()

        self._db = Database(self._settings.db_path)
        self._llm: LLMProvider | None = None
        self._memory: MemoryManager | None = None
        self._loop: AgentLoop | None = None

        self._session_id = str(uuid.uuid4())[:8]  # Short readable ID
        self._started = False

    async def start(self) -> None:
        """Initialize all subsystems. Call before first chat()."""
        if self._started:
            return

        # Database
        await self._db.connect()

        # LLM provider
        self._llm = create_provider()

        # Memory
        self._memory = MemoryManager(self._db)

        # Session tracking
        await self._memory.long_term.create_session(self._session_id)

        # Agent loop
        self._loop = AgentLoop(
            session_id=self._session_id,
            llm=self._llm,
            memory=self._memory,
        )

        self._started = True
        logger.info(
            f"{AGENT_NAME} started | session={self._session_id} | "
            f"provider={self._llm.provider_name} | model={self._llm.model_name}"
        )

    async def chat(self, user_message: str) -> str:
        """Process a user message and return the agent's response."""
        if not self._started:
            await self.start()
        return await self._loop.process(user_message)

    async def stop(self) -> None:
        """Graceful shutdown: persist state and close connections."""
        if not self._started:
            return

        if self._memory:
            await self._memory.long_term.end_session(self._session_id)

        if self._db:
            await self._db.close()

        self._started = False
        logger.info(f"{AGENT_NAME} stopped | session={self._session_id}")

    # ── Accessors for the CLI layer ───────────────────────────────

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def llm(self) -> LLMProvider:
        return self._llm

    @property
    def memory(self) -> MemoryManager:
        return self._memory

    @property
    def settings(self):
        return self._settings

    async def get_stats(self) -> dict:
        if not self._memory:
            return {}
        stats = await self._memory.get_stats()
        stats["session_id"] = self._session_id
        stats["model"] = self._llm.model_name if self._llm else "unknown"
        stats["provider"] = self._llm.provider_name if self._llm else "unknown"
        return stats
