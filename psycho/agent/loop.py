"""AgentLoop — the perceive → think → act → learn cycle with knowledge graph."""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from loguru import logger

from psycho.config.constants import (
    AGENT_NAME,
    MAX_CONTEXT_MEMORIES,
    SYSTEM_PROMPT_BASE,
)
from psycho.llm.base import LLMProvider, Message

from .context import AgentContext

if TYPE_CHECKING:
    from psycho.knowledge.evolution import GraphEvolver
    from psycho.knowledge.extractor import KnowledgeExtractor
    from psycho.knowledge.graph import KnowledgeGraph
    from psycho.knowledge.reasoner import GraphReasoner
    from psycho.memory import MemoryManager


class AgentLoop:
    """
    Core interaction cycle — fully wired with the knowledge graph.

    Per-interaction pipeline:
      1. Semantic memory retrieval (past conversations)
      2. Knowledge graph context retrieval (structured knowledge)
      3. System prompt assembly (base + memories + graph context)
      4. LLM call
      5. Response delivery
      6. Background: 4-tier memory write + async knowledge extraction
    """

    def __init__(
        self,
        session_id: str,
        llm: LLMProvider,
        memory: "MemoryManager",
        graph: "KnowledgeGraph",
        evolver: "GraphEvolver",
        extractor: "KnowledgeExtractor",
        reasoner: "GraphReasoner",
    ) -> None:
        self._session_id = session_id
        self._llm = llm
        self._memory = memory
        self._graph = graph
        self._evolver = evolver
        self._extractor = extractor
        self._reasoner = reasoner

    async def process(self, user_message: str) -> str:
        """Full pipeline for a single user message. Returns agent's response."""
        ctx = AgentContext(
            session_id=self._session_id,
            interaction_id=str(uuid.uuid4()),
            user_message=user_message,
        )

        # 1. Parallel retrieval: semantic memories + graph context
        semantic_task = asyncio.create_task(
            self._memory.retrieve_context(query=user_message)
        )
        # Graph context is synchronous (in-process NetworkX) — run directly
        graph_context_str = self._reasoner.get_context_for_prompt(user_message)

        ctx.retrieved_memories = await semantic_task
        ctx.graph_context = [graph_context_str] if graph_context_str else []

        # 2. Build system prompt
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
                f"I encountered an error: {e}\n"
                "Please check your API key / Ollama server and try again."
            )

        ctx.mark_complete()

        # 4. Memory write (awaited — must complete before next interaction)
        await self._memory.add_interaction(
            session_id=self._session_id,
            user_message=user_message,
            agent_response=ctx.agent_response,
            domain=ctx.domain,
            tokens_used=ctx.total_tokens,
        )

        # 5. Background knowledge extraction (non-blocking — user doesn't wait)
        asyncio.create_task(
            self._extract_and_evolve(
                user_message=user_message,
                agent_response=ctx.agent_response,
                domain=ctx.domain,
            )
        )

        logger.debug(
            f"Interaction: {ctx.total_tokens} tokens, {ctx.latency_ms:.0f}ms, "
            f"graph={self._graph.get_stats()['active_nodes']} nodes"
        )
        return ctx.agent_response

    # ── System prompt assembly ────────────────────────────────────

    def _build_system_prompt(self, ctx: AgentContext) -> str:
        """Assemble the full system prompt with all context layers."""
        parts = [SYSTEM_PROMPT_BASE.format(name=AGENT_NAME)]

        # Temporal awareness
        now = datetime.now().strftime("%A, %B %d %Y at %H:%M")
        parts.append(f"\nCurrent date and time: {now}")

        # Knowledge graph context (structured knowledge)
        if ctx.graph_context:
            parts.extend(ctx.graph_context)

        # Semantic memory (relevant past conversations)
        if ctx.retrieved_memories:
            parts.append("\n─── RELEVANT PAST INTERACTIONS ───")
            for mem in ctx.retrieved_memories[:MAX_CONTEXT_MEMORIES]:
                relevance = mem.get("relevance", 0)
                rel_tag = "HIGH" if relevance > 0.7 else "MEDIUM" if relevance > 0.5 else "LOW"
                parts.append(
                    f"[{rel_tag}] User: {mem['user_message'][:200]}\n"
                    f"         You:  {mem['agent_response'][:300]}"
                )
            parts.append("──────────────────────────────────")
            parts.append(
                "Weave the above knowledge and memories naturally into your responses. "
                "Reference them as genuine memory, not as a list."
            )

        return "\n".join(parts)

    def _build_messages(self, ctx: AgentContext) -> list[Message]:
        """Build message list: short-term history + current message."""
        messages = self._memory.short_term.get_messages()
        messages.append(Message(role="user", content=ctx.user_message))
        return messages

    # ── Background extraction ─────────────────────────────────────

    async def _extract_and_evolve(
        self, user_message: str, agent_response: str, domain: str
    ) -> None:
        """
        Background task: extract knowledge from the interaction and evolve the graph.

        This runs AFTER the response is returned to the user.
        Any errors here are logged but don't affect the user experience.
        """
        try:
            extraction = await self._extractor.extract_from_interaction(
                user_message=user_message,
                agent_response=agent_response,
                session_id=self._session_id,
                domain=domain,
            )
            if not extraction.is_empty():
                stats = await self._evolver.integrate(extraction)
                if stats["nodes_added"] > 0 or stats["facts_added"] > 0:
                    self._graph.save()
                    logger.debug(
                        f"Graph evolved: +{stats['nodes_added']} nodes, "
                        f"+{stats['facts_added']} facts, "
                        f"+{stats['corrections_applied']} corrections"
                    )
        except Exception as e:
            logger.warning(f"Background extraction failed (non-critical): {e}")
