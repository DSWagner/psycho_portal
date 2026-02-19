"""AgentLoop — perceive → think → act → learn, with full self-evolution wiring."""

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
from psycho.learning.signal_detector import (
    SignalType,
    detect_signal,
    extract_correction_target,
)
from psycho.llm.base import LLMProvider, Message

from .context import AgentContext

if TYPE_CHECKING:
    from psycho.domains.base import DomainHandler, DomainResult
    from psycho.domains.router import DomainRouter
    from psycho.knowledge.evolution import GraphEvolver
    from psycho.knowledge.extractor import KnowledgeExtractor
    from psycho.knowledge.graph import KnowledgeGraph
    from psycho.knowledge.reasoner import GraphReasoner
    from psycho.learning.mistake_tracker import MistakeTracker
    from psycho.memory import MemoryManager


class AgentLoop:
    """
    Full interaction cycle with all self-evolution features active.

    Per-interaction pipeline:
      1.  Signal detection (is user correcting or confirming?)
      2.  Real-time confidence update on detected signal
      3.  Parallel retrieval: semantic memory + mistake warnings
      4.  Knowledge graph context (synchronous, in-process)
      5.  System prompt assembly
      6.  LLM call
      7.  Response delivery
      8.  Memory write (4 tiers, awaited)
      9.  Background: knowledge extraction + graph evolution
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
        mistake_tracker: "MistakeTracker",
        domain_router: "DomainRouter | None" = None,
        domain_handlers: "dict[str, DomainHandler] | None" = None,
    ) -> None:
        self._session_id = session_id
        self._llm = llm
        self._memory = memory
        self._graph = graph
        self._evolver = evolver
        self._extractor = extractor
        self._reasoner = reasoner
        self._mistake_tracker = mistake_tracker
        self._domain_router = domain_router
        self._domain_handlers = domain_handlers or {}
        self._last_domain_result = None

    async def process(self, user_message: str) -> str:
        """Full pipeline for a single user message."""
        ctx = AgentContext(
            session_id=self._session_id,
            interaction_id=str(uuid.uuid4()),
            user_message=user_message,
        )

        # 0. Domain classification (fast — keyword-first, LLM fallback)
        if self._domain_router:
            ctx.domain = await self._domain_router.classify(user_message)

        # 1. Detect correction/confirmation signal
        signal = detect_signal(user_message)
        if signal.type == SignalType.CORRECTION:
            ctx.is_correction = True
            await self._handle_correction_signal(user_message, signal)
        elif signal.type == SignalType.CONFIRMATION:
            ctx.is_confirmation = True
            await self._handle_confirmation_signal()

        # 2. Parallel: semantic memory + mistake warnings
        memory_task = asyncio.create_task(
            self._memory.retrieve_context(query=user_message)
        )
        warnings_task = asyncio.create_task(
            self._mistake_tracker.get_warnings_for_prompt(user_message)
        )
        ctx.retrieved_memories, mistake_warnings = await asyncio.gather(
            memory_task, warnings_task
        )

        # 3. Graph context (synchronous)
        graph_context_str = self._reasoner.get_context_for_prompt(user_message)
        ctx.graph_context = [graph_context_str] if graph_context_str else []
        ctx.mistake_warnings = mistake_warnings

        # 4. Build system prompt (with domain addendum + pending tasks/health context)
        domain_addendum = self._get_domain_addendum(ctx)
        domain_context = await self._get_domain_context(ctx)
        system_prompt = self._build_system_prompt(ctx, domain_addendum, domain_context)
        messages = self._build_messages(ctx)

        # 5. LLM call
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

        # 6. Domain post-processing (extract structured data, run code, log metrics)
        domain_result = await self._run_domain_handler(ctx)
        self._last_domain_result = domain_result

        # 7. Memory write (awaited)
        await self._memory.add_interaction(
            session_id=self._session_id,
            user_message=user_message,
            agent_response=ctx.agent_response,
            domain=ctx.domain,
            tokens_used=ctx.total_tokens,
        )

        # 7. Background extraction + graph evolution
        asyncio.create_task(
            self._extract_and_evolve(
                user_message=user_message,
                agent_response=ctx.agent_response,
                domain=ctx.domain,
                is_correction=ctx.is_correction,
            )
        )

        logger.debug(
            f"Interaction: {ctx.total_tokens} tokens, {ctx.latency_ms:.0f}ms | "
            f"domain={ctx.domain} | signal={signal.type.value} | "
            f"graph={self._graph.get_stats()['active_nodes']} nodes"
        )

        # Attach domain result to response for CLI rendering
        ctx.domain_result = domain_result
        return ctx.agent_response

    async def stream_process(self, user_message: str):
        """
        Streaming version of process() — yields tokens as they arrive.
        Post-processing (memory, extraction) runs after all tokens emitted.
        """
        ctx = AgentContext(
            session_id=self._session_id,
            interaction_id=str(uuid.uuid4()),
            user_message=user_message,
        )

        if self._domain_router:
            ctx.domain = await self._domain_router.classify(user_message)

        signal = detect_signal(user_message)
        if signal.type == SignalType.CORRECTION:
            ctx.is_correction = True

        memory_task = asyncio.create_task(self._memory.retrieve_context(query=user_message))
        warnings_task = asyncio.create_task(
            self._mistake_tracker.get_warnings_for_prompt(user_message)
        )
        ctx.retrieved_memories, mistake_warnings = await asyncio.gather(memory_task, warnings_task)
        ctx.graph_context = [self._reasoner.get_context_for_prompt(user_message)] if self._reasoner else []
        ctx.mistake_warnings = mistake_warnings

        domain_addendum = self._get_domain_addendum(ctx)
        domain_context = await self._get_domain_context(ctx)
        system_prompt = self._build_system_prompt(ctx, domain_addendum, domain_context)
        messages = self._build_messages(ctx)

        # Stream tokens
        full_response_parts = []
        try:
            async for token in self._llm.stream(messages=messages, system=system_prompt):
                full_response_parts.append(token)
                yield token
        except Exception as e:
            error_msg = f"Streaming error: {e}"
            yield error_msg
            full_response_parts = [error_msg]

        ctx.agent_response = "".join(full_response_parts)
        ctx.mark_complete()
        self._last_domain_result = await self._run_domain_handler(ctx)

        await self._memory.add_interaction(
            session_id=self._session_id,
            user_message=user_message,
            agent_response=ctx.agent_response,
            domain=ctx.domain,
            tokens_used=0,
        )
        asyncio.create_task(
            self._extract_and_evolve(
                user_message=user_message,
                agent_response=ctx.agent_response,
                domain=ctx.domain,
                is_correction=ctx.is_correction,
            )
        )

    # ── System prompt assembly ─────────────────────────────────────

    def _get_domain_addendum(self, ctx: AgentContext) -> str:
        handler = self._domain_handlers.get(ctx.domain)
        return handler.system_addendum(ctx) if handler else ""

    async def _get_domain_context(self, ctx: AgentContext) -> str:
        """Get domain-specific context (pending tasks, recent health metrics)."""
        handler = self._domain_handlers.get(ctx.domain)
        if handler and hasattr(handler, "get_context_for_prompt"):
            try:
                return await handler.get_context_for_prompt(ctx.session_id)
            except Exception:
                pass
        return ""

    async def _run_domain_handler(self, ctx: AgentContext):
        """Post-process response with domain handler."""
        handler = self._domain_handlers.get(ctx.domain)
        if handler:
            try:
                return await handler.post_process(ctx, ctx.agent_response)
            except Exception as e:
                logger.warning(f"Domain handler post_process failed: {e}")
        return None

    def _build_system_prompt(self, ctx: AgentContext, domain_addendum: str = "", domain_context: str = "") -> str:
        parts = [SYSTEM_PROMPT_BASE.format(name=AGENT_NAME)]

        # Datetime
        now = datetime.now().strftime("%A, %B %d %Y at %H:%M")
        parts.append(f"\nCurrent date and time: {now}")
        parts.append(f"Active domain: {ctx.domain}")

        # Domain-specific instructions
        if domain_addendum:
            parts.append(domain_addendum)

        # Domain-specific context (tasks, health metrics)
        if domain_context:
            parts.append(domain_context)

        # Knowledge graph
        if ctx.graph_context:
            parts.extend(ctx.graph_context)

        # Mistake warnings (highest priority — show before other context)
        if ctx.mistake_warnings:
            parts.append(
                self._mistake_tracker.build_warning_block(ctx.mistake_warnings)
            )

        # Semantic memory
        if ctx.retrieved_memories:
            parts.append("\n─── RELEVANT PAST INTERACTIONS ───")
            for mem in ctx.retrieved_memories[:MAX_CONTEXT_MEMORIES]:
                relevance = mem.get("relevance", 0)
                tag = "HIGH" if relevance > 0.7 else "MEDIUM" if relevance > 0.5 else "LOW"
                parts.append(
                    f"[{tag}] User: {mem['user_message'][:200]}\n"
                    f"         You:  {mem['agent_response'][:300]}"
                )
            parts.append("──────────────────────────────────")
            parts.append(
                "Weave the above knowledge and memories naturally. "
                "Reference them as genuine memory."
            )

        # Correction acknowledgment
        if ctx.is_correction:
            parts.append(
                "\nNOTE: The user appears to be correcting something. "
                "Acknowledge the correction explicitly, thank them, "
                "and provide the correct information."
            )

        return "\n".join(parts)

    def _build_messages(self, ctx: AgentContext) -> list[Message]:
        messages = self._memory.short_term.get_messages()
        messages.append(Message(role="user", content=ctx.user_message))
        return messages

    # ── Signal handlers ────────────────────────────────────────────

    async def _handle_correction_signal(
        self, user_message: str, signal
    ) -> None:
        """Immediately drop confidence on recently discussed topics."""
        # Get the most recent assistant turn to identify what was corrected
        recent_turns = self._memory.short_term.get_turns()
        if not recent_turns:
            return

        last_agent_response = recent_turns[-1].assistant if recent_turns else ""
        correction_hint = extract_correction_target(user_message, last_agent_response)

        if correction_hint:
            # Try to find the node being corrected
            node = self._graph.find_node_by_label(correction_hint.lower()[:50])
            if node:
                self._evolver.correct_node(node.id, f"User correction: {correction_hint[:100]}")
                logger.info(f"Real-time correction: '{node.display_label}' confidence reduced")

    async def _handle_confirmation_signal(self) -> None:
        """Boost confidence on the most recently discussed graph topics."""
        recent_turns = self._memory.short_term.get_turns()
        if not recent_turns:
            return

        last_user = recent_turns[-1].user if recent_turns else ""
        # Get nodes that were relevant to the last exchange
        context_items = self._graph.get_context_for_query(last_user, top_k=3)
        node_ids = [node.id for node, _ in context_items]
        if node_ids:
            self._evolver.confirm_nodes(node_ids)
            logger.debug(f"Real-time confirmation: boosted {len(node_ids)} nodes")

    # ── Background extraction ──────────────────────────────────────

    async def _extract_and_evolve(
        self,
        user_message: str,
        agent_response: str,
        domain: str,
        is_correction: bool = False,
    ) -> None:
        """Background: extract knowledge and evolve the graph."""
        try:
            extraction = await self._extractor.extract_from_interaction(
                user_message=user_message,
                agent_response=agent_response,
                session_id=self._session_id,
                domain=domain,
            )
            if not extraction.is_empty():
                stats = await self._evolver.integrate(extraction)

                # If it was a correction and the extraction found it, record the mistake
                if is_correction and extraction.corrections:
                    for corr in extraction.corrections:
                        await self._mistake_tracker.record_mistake(
                            session_id=self._session_id,
                            user_input=user_message[:400],
                            agent_response=corr.get("wrong", "")[:400],
                            correction=corr.get("correct", "")[:300],
                            domain=domain,
                        )

                if stats["nodes_added"] > 0 or stats["facts_added"] > 0:
                    self._graph.save()

        except Exception as e:
            logger.warning(f"Background extraction failed (non-critical): {e}")
