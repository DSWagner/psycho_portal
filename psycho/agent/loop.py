"""AgentLoop — the perceive → think → act → learn cycle."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from loguru import logger

from psycho.config.constants import (
    AGENT_NAME,
    DEFAULT_DOMAIN,
    MAX_CONTEXT_MEMORIES,
    SYSTEM_PROMPT_BASE,
)
from psycho.llm.base import LLMProvider, Message

from .context import AgentContext

if TYPE_CHECKING:
    from psycho.memory import MemoryManager


class AgentLoop:
    """
    The core interaction cycle.

    Phase 1: perceive → retrieve keyword-matched memories → build prompt → LLM → save
    Phase 3: adds knowledge graph context retrieval
    Phase 4: adds mistake warnings + confidence injection
    """

    def __init__(
        self,
        session_id: str,
        llm: LLMProvider,
        memory: "MemoryManager",
    ) -> None:
        self._session_id = session_id
        self._llm = llm
        self._memory = memory

    async def process(self, user_message: str) -> str:
        """Full pipeline for a single user message. Returns the agent's response."""
        ctx = AgentContext(
            session_id=self._session_id,
            interaction_id=str(uuid.uuid4()),
            user_message=user_message,
        )

        # 1. Retrieve relevant context from memory
        await self._retrieve_context(ctx)

        # 2. Build the full prompt
        system_prompt = self._build_system_prompt(ctx)
        messages = self._build_messages(ctx)

        # 3. LLM call
        try:
            response = await self._llm.complete(
                messages=messages,
                system=system_prompt,
                max_tokens=4096,
            )
            ctx.agent_response = response.content
            ctx.input_tokens = response.input_tokens
            ctx.output_tokens = response.output_tokens
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            ctx.agent_response = (
                f"I encountered an error processing your message: {e}\n"
                "Please check your API key and try again."
            )

        ctx.mark_complete()

        # 4. Save to memory
        await self._memory.add_interaction(
            session_id=self._session_id,
            user_message=user_message,
            agent_response=ctx.agent_response,
            domain=ctx.domain,
            tokens_used=ctx.total_tokens,
        )

        logger.debug(
            f"Interaction complete: {ctx.total_tokens} tokens, {ctx.latency_ms:.0f}ms"
        )
        return ctx.agent_response

    # ── Private helpers ───────────────────────────────────────────

    async def _retrieve_context(self, ctx: AgentContext) -> None:
        """Pull relevant memories to inject into the prompt."""
        memories = await self._memory.retrieve_context(
            query=ctx.user_message, domain=ctx.domain
        )
        ctx.retrieved_memories = memories
        logger.debug(f"Retrieved {len(memories)} relevant memories")

    def _build_system_prompt(self, ctx: AgentContext) -> str:
        """Assemble the system prompt with injected context."""
        parts = [SYSTEM_PROMPT_BASE.format(name=AGENT_NAME)]

        if ctx.retrieved_memories:
            parts.append("\n─── RELEVANT MEMORIES FROM PAST SESSIONS ───")
            for mem in ctx.retrieved_memories[:MAX_CONTEXT_MEMORIES]:
                parts.append(
                    f"[Past] User: {mem['user_message'][:200]}\n"
                    f"       You:  {mem['agent_response'][:300]}"
                )
            parts.append("─────────────────────────────────────────────")
            parts.append(
                "Reference the above past context naturally when relevant. "
                "Don't mechanically list it — weave it into your responses."
            )

        return "\n".join(parts)

    def _build_messages(self, ctx: AgentContext) -> list[Message]:
        """Build the message list: short-term history + current user message."""
        messages = self._memory.short_term.get_messages()
        messages.append(Message(role="user", content=ctx.user_message))
        return messages
