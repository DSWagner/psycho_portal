"""AgentLoop — perceive → think → act → learn, with full self-evolution wiring."""

from __future__ import annotations

import asyncio
import re
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, AsyncIterator

from loguru import logger

from psycho.config.constants import (
    AGENT_NAME,
    MAX_CONTEXT_MEMORIES,
    SYSTEM_PROMPT_BASE,
    USER_PROFILE_TEMPLATE,
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
    Core interaction cycle — every message goes through:
      0.  Domain classification
      1.  Signal detection (correction / confirmation)
      2.  Real-time confidence update on signal
      3.  Parallel: semantic memory + mistake warnings
      4.  Knowledge graph context retrieval
      5.  User profile injection
      6.  System prompt assembly
      7.  LLM call (or stream)
      8.  Domain post-processing (code run / metric log / task create)
      9.  Memory write (4 tiers, awaited)
      10. Background: knowledge extraction + graph evolution
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
        self._last_domain: str = "general"   # FIX: track domain for API layer
        self._last_search_query: str = ""    # forwarded to WebSocket done event

    # ── Main pipeline ──────────────────────────────────────────────

    async def process(self, user_message: str) -> str:
        """Full pipeline for a single user message. Returns agent's response."""
        ctx = AgentContext(
            session_id=self._session_id,
            interaction_id=str(uuid.uuid4()),
            user_message=user_message,
        )

        await self._prepare_context(ctx)

        system_prompt = self._build_system_prompt(ctx)
        messages = self._build_messages(ctx)

        try:
            response = await self._llm.complete(
                messages=messages, system=system_prompt, max_tokens=4096
            )
            ctx.agent_response = response.content
            ctx.input_tokens = response.input_tokens
            ctx.output_tokens = response.output_tokens
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            ctx.agent_response = (
                f"I ran into an error: {e}\n"
                "Check your API key / Ollama connection and try again."
            )

        ctx.mark_complete()
        self._last_domain = ctx.domain
        self._last_domain_result = await self._run_domain_handler(ctx)

        await self._memory.add_interaction(
            session_id=self._session_id,
            user_message=user_message,
            agent_response=ctx.agent_response,
            domain=ctx.domain,
            tokens_used=ctx.total_tokens,
        )

        asyncio.create_task(
            self._extract_and_evolve(
                user_message=user_message,
                agent_response=ctx.agent_response,
                domain=ctx.domain,
                is_correction=ctx.is_correction,
            )
        )

        logger.debug(
            f"Interaction: {ctx.total_tokens}t {ctx.latency_ms:.0f}ms "
            f"domain={ctx.domain} signal={ctx.signal_type}"
        )
        ctx.domain_result = self._last_domain_result
        return ctx.agent_response

    async def stream_process(self, user_message: str) -> AsyncIterator[str]:
        """Streaming version — yields tokens as they arrive."""
        ctx = AgentContext(
            session_id=self._session_id,
            interaction_id=str(uuid.uuid4()),
            user_message=user_message,
        )

        await self._prepare_context(ctx)
        system_prompt = self._build_system_prompt(ctx)
        messages = self._build_messages(ctx)

        full_parts = []
        try:
            async for token in self._llm.stream(
                messages=messages, system=system_prompt, max_tokens=4096
            ):
                full_parts.append(token)
                yield token
        except Exception as e:
            err = f"\n\nStream error: {e}"
            full_parts.append(err)
            yield err

        ctx.agent_response = "".join(full_parts)
        ctx.mark_complete()
        self._last_domain = ctx.domain
        self._last_search_query = ctx.search_query
        self._last_domain_result = await self._run_domain_handler(ctx)

        await self._memory.add_interaction(
            session_id=self._session_id,
            user_message=user_message,
            agent_response=ctx.agent_response,
            domain=ctx.domain,
            tokens_used=0,  # Streaming doesn't expose token counts
        )

        asyncio.create_task(
            self._extract_and_evolve(
                user_message=user_message,
                agent_response=ctx.agent_response,
                domain=ctx.domain,
                is_correction=ctx.is_correction,
            )
        )

    async def stream_process_with_image(
        self, user_message: str, image_data: bytes, media_type: str
    ) -> AsyncIterator[str]:
        """Streaming vision pipeline — image + optional text prompt."""
        ctx = AgentContext(
            session_id=self._session_id,
            interaction_id=str(uuid.uuid4()),
            user_message=user_message or "Describe and analyse this image in detail.",
            image_data=image_data,
            image_media_type=media_type,
        )

        await self._prepare_context(ctx)
        system_prompt = self._build_system_prompt(ctx)
        prior_messages = self._memory.short_term.get_messages()

        full_parts: list[str] = []
        try:
            async for token in self._llm.stream_with_image(
                prior_messages=prior_messages,
                image_data=image_data,
                media_type=media_type,
                user_text=ctx.user_message,
                system=system_prompt,
            ):
                full_parts.append(token)
                yield token
        except Exception as e:
            err = f"\n\nStream error: {e}"
            full_parts.append(err)
            yield err

        ctx.agent_response = "".join(full_parts)
        ctx.mark_complete()
        self._last_domain = ctx.domain
        self._last_domain_result = await self._run_domain_handler(ctx)

        await self._memory.add_interaction(
            session_id=self._session_id,
            user_message=ctx.user_message,
            agent_response=ctx.agent_response,
            domain=ctx.domain,
            tokens_used=0,
        )

        asyncio.create_task(
            self._extract_and_evolve(
                user_message=ctx.user_message,
                agent_response=ctx.agent_response,
                domain=ctx.domain,
                is_correction=ctx.is_correction,
            )
        )

    # ── Agent name resolution ──────────────────────────────────────

    def _get_agent_name(self) -> str:
        """
        Return the agent's current name.
        Checks graph for a stored 'agent_name' preference first; falls back to AGENT_NAME.
        """
        try:
            from psycho.knowledge.schema import NodeType
            prefs = self._graph.find_nodes_by_type(NodeType.PREFERENCE)
            for p in prefs:
                if p.label.startswith("agent_name:") and not p.deprecated:
                    return p.properties.get("value", AGENT_NAME)
        except Exception:
            pass
        return AGENT_NAME

    def _detect_agent_name_assignment(self, user_message: str) -> str | None:
        """
        Detect patterns like 'your name is Raz', 'call you Raz', 'from now on you are Raz'.
        Returns the new name if detected, else None.
        """
        patterns = [
            r"your name is\s+([A-Za-z][A-Za-z0-9_-]{0,30})",
            r"call you\s+([A-Za-z][A-Za-z0-9_-]{0,30})",
            r"you(?:'re| are) (?:now |called )?([A-Za-z][A-Za-z0-9_-]{0,30})",
            r"from now on[,.]?\s+(?:you(?:'re| are)|your name is)\s+([A-Za-z][A-Za-z0-9_-]{0,30})",
            r"(?:name|call) (?:you|yourself)\s+([A-Za-z][A-Za-z0-9_-]{0,30})",
        ]
        msg_lower = user_message.lower()
        for pattern in patterns:
            m = re.search(pattern, msg_lower)
            if m:
                name = m.group(1).strip().capitalize()
                # Ignore common false positives
                if name.lower() not in {"a", "an", "the", "my", "your", "their", "its", "our"}:
                    return name
        return None

    def _store_agent_name(self, name: str) -> None:
        """Persist the agent name as a high-confidence preference node in the graph."""
        try:
            from psycho.knowledge.schema import KnowledgeNode, NodeType
            node = KnowledgeNode(
                label=f"agent_name:{name.lower()}",
                type=NodeType.PREFERENCE,
                domain="general",
                confidence=0.95,
                properties={"value": name},
                source="user_assignment",
            )
            self._graph.upsert_node(node)
            self._graph.save()
            logger.info(f"Agent name stored: {name}")
        except Exception as e:
            logger.warning(f"Could not store agent name: {e}")

    # ── Context preparation (shared by process + stream_process) ──

    async def _prepare_context(self, ctx: AgentContext) -> None:
        """Build full context before LLM call: classify, signals, retrieval."""
        # -1. Detect agent name assignment before anything else
        new_name = self._detect_agent_name_assignment(ctx.user_message)
        if new_name:
            self._store_agent_name(new_name)

        # 0. Domain classification
        if self._domain_router:
            ctx.domain = await self._domain_router.classify(ctx.user_message)

        # 1. Signal detection
        signal = detect_signal(ctx.user_message)
        ctx.signal_type = signal.type.value
        if signal.type == SignalType.CORRECTION:
            ctx.is_correction = True
            await self._handle_correction_signal(ctx.user_message, signal)
        elif signal.type == SignalType.CONFIRMATION:
            ctx.is_confirmation = True
            await self._handle_confirmation_signal(ctx.user_message)

        # 2. Parallel retrieval: semantic memory + mistake warnings
        memory_task = asyncio.create_task(
            self._memory.retrieve_context(query=ctx.user_message)
        )
        warnings_task = asyncio.create_task(
            self._mistake_tracker.get_warnings_for_prompt(ctx.user_message)
        )
        ctx.retrieved_memories, ctx.mistake_warnings = await asyncio.gather(
            memory_task, warnings_task
        )

        # 3. Knowledge graph context (synchronous, in-process)
        graph_ctx = self._reasoner.get_context_for_prompt(ctx.user_message)
        ctx.graph_context = [graph_ctx] if graph_ctx else []

        # 4. Domain-specific context (tasks list, health stats)
        ctx.domain_context = await self._get_domain_context(ctx)

        # 5. Web search — inject live results when query needs current data
        try:
            from psycho.config import get_settings
            from psycho.tools.web_search import (
                extract_query, format_search_results, should_search, web_search,
            )
            s = get_settings()
            if s.web_search_enabled and should_search(ctx.user_message):
                query = extract_query(ctx.user_message)
                results = await web_search(query, brave_api_key=s.brave_api_key)
                if results:
                    ctx.search_query = query
                    ctx.search_results = format_search_results(results, query)
                    logger.debug(f"Web search injected: {len(results)} results for {query!r}")
        except Exception as e:
            logger.debug(f"Web search skipped: {e}")

    # ── System prompt assembly ─────────────────────────────────────

    def _build_system_prompt(self, ctx: AgentContext) -> str:
        # Build user profile from graph preferences + known facts
        user_profile = self._build_user_profile()

        agent_name = self._get_agent_name()
        base = SYSTEM_PROMPT_BASE.format(name=agent_name, user_profile=user_profile)
        parts = [base]

        # Datetime
        now = datetime.now().strftime("%A, %B %d %Y at %H:%M")
        parts.append(f"Current date and time: {now}")

        # Domain instructions
        domain_addendum = self._get_domain_addendum(ctx)
        if domain_addendum:
            parts.append(domain_addendum)

        # Domain context (pending tasks, health stats)
        if ctx.domain_context:
            parts.append(ctx.domain_context)

        # Live web search results
        if ctx.search_results:
            parts.append(ctx.search_results)

        # Knowledge graph
        if ctx.graph_context:
            parts.extend(ctx.graph_context)

        # Mistake warnings (highest priority — show before memory)
        if ctx.mistake_warnings:
            parts.append(
                self._mistake_tracker.build_warning_block(ctx.mistake_warnings)
            )

        # Relevant past interactions
        if ctx.retrieved_memories:
            parts.append("\n─── RELEVANT PAST CONTEXT ───")
            for mem in ctx.retrieved_memories[:MAX_CONTEXT_MEMORIES]:
                relevance = mem.get("relevance", 0)
                tag = "★★★" if relevance > 0.75 else "★★" if relevance > 0.55 else "★"
                parts.append(
                    f"[{tag}] You: {mem['user_message'][:180]}\n"
                    f"     Me: {mem['agent_response'][:280]}"
                )
            parts.append("─────────────────────────────")

        # Correction acknowledgment
        if ctx.is_correction:
            parts.append(
                "\nIMPORTANT: The user is correcting something. "
                "Acknowledge it directly and briefly — don't be defensive. "
                "Thank them, confirm the correction, and move on."
            )

        return "\n\n".join(parts)

    def _build_user_profile(self) -> str:
        """
        Build a rich user profile from the knowledge graph for system prompt injection.
        Includes: name, occupation, current projects, preferences, skills, technologies.
        """
        try:
            from psycho.knowledge.schema import NodeType
            lines = []

            # ── User identity ──────────────────────────────────────
            person_nodes = self._graph.find_nodes_by_type(NodeType.PERSON)
            user_node = next(
                (n for n in person_nodes if n.label == "user"),
                None
            )
            if user_node:
                name = user_node.properties.get("name") or user_node.display_label
                if name and name.lower() != "user":
                    lines.append(f"Name: {name}")
                for k in ("occupation", "location"):
                    if v := user_node.properties.get(k):
                        lines.append(f"{k.title()}: {v}")

            # ── Current projects and goals ─────────────────────────
            prefs = self._graph.find_nodes_by_type(NodeType.PREFERENCE)
            active = [p for p in prefs if not p.deprecated]

            project_prefs = [
                p for p in active
                if any(k in p.label for k in ("current_project:", "goal:", "working on", "building"))
            ]
            if project_prefs:
                lines.append("Current projects / goals:")
                for p in sorted(project_prefs, key=lambda x: x.confidence, reverse=True)[:4]:
                    lines.append(f"  • {p.display_label}")

            # ── Technologies and tools ─────────────────────────────
            tech_nodes = self._graph.find_nodes_by_type(NodeType.TECHNOLOGY)
            top_tech = sorted(
                [t for t in tech_nodes if not t.deprecated],
                key=lambda x: x.confidence, reverse=True
            )[:6]
            if top_tech:
                tech_labels = ", ".join(t.display_label for t in top_tech)
                lines.append(f"Known technologies: {tech_labels}")

            # ── Strong preferences ─────────────────────────────────
            strong_prefs = sorted(
                [p for p in active
                 if p.confidence > 0.65
                 and not any(k in p.label for k in ("current_project:", "goal:"))],
                key=lambda x: x.confidence, reverse=True
            )[:5]
            if strong_prefs:
                lines.append("Established preferences:")
                for p in strong_prefs:
                    lines.append(f"  • {p.display_label}")

            # ── Skills ────────────────────────────────────────────
            skills = self._graph.find_nodes_by_type(NodeType.SKILL)
            top_skills = sorted(
                [s for s in skills if not s.deprecated],
                key=lambda x: x.confidence, reverse=True
            )[:4]
            if top_skills:
                skill_labels = ", ".join(s.display_label for s in top_skills)
                lines.append(f"Skills: {skill_labels}")

            if not lines:
                return ""

            return USER_PROFILE_TEMPLATE.format(profile_lines="\n".join(lines))
        except Exception:
            return ""

    def _build_messages(self, ctx: AgentContext) -> list[Message]:
        messages = self._memory.short_term.get_messages()
        messages.append(Message(role="user", content=ctx.user_message))
        return messages

    # ── Domain helpers ─────────────────────────────────────────────

    def _get_domain_addendum(self, ctx: AgentContext) -> str:
        handler = self._domain_handlers.get(ctx.domain)
        return handler.system_addendum(ctx) if handler else ""

    async def _get_domain_context(self, ctx: AgentContext) -> str:
        handler = self._domain_handlers.get(ctx.domain)
        if handler and hasattr(handler, "get_context_for_prompt"):
            try:
                return await handler.get_context_for_prompt(ctx.session_id)
            except Exception as e:
                logger.debug(f"Domain context failed for {ctx.domain}: {e}")
        return ""

    async def _run_domain_handler(self, ctx: AgentContext):
        handler = self._domain_handlers.get(ctx.domain)
        if handler:
            try:
                return await handler.post_process(ctx, ctx.agent_response)
            except Exception as e:
                logger.warning(f"Domain handler failed ({ctx.domain}): {e}")
        return None

    # ── Signal handlers ────────────────────────────────────────────

    async def _handle_correction_signal(self, user_message: str, signal) -> None:
        """Immediately drop confidence on the corrected topic."""
        recent_turns = self._memory.short_term.get_turns()
        if not recent_turns:
            return

        last_response = recent_turns[-1].assistant
        correction_hint = extract_correction_target(user_message, last_response)

        if correction_hint:
            node = self._graph.find_node_by_label(correction_hint.lower()[:80])
            if node:
                self._evolver.correct_node(node.id, f"User correction: {correction_hint[:100]}")
                logger.info(f"Real-time correction applied: '{node.display_label}'")
            else:
                # Node doesn't exist yet — store as a raw fact to be extracted later
                # The background extraction will handle it properly
                logger.debug(
                    f"Correction target '{correction_hint[:40]}' not in graph yet; "
                    "will be captured by background extraction"
                )

    async def _handle_confirmation_signal(self, user_message: str) -> None:
        """Boost confidence on the topic the user just confirmed."""
        recent_turns = self._memory.short_term.get_turns()
        if not recent_turns:
            return

        # Use the PREVIOUS assistant turn (what was confirmed) as query
        last_agent = recent_turns[-1].assistant
        context_items = self._graph.get_context_for_query(last_agent[:200], top_k=3)
        node_ids = [node.id for node, _ in context_items]
        if node_ids:
            self._evolver.confirm_nodes(node_ids)
            logger.debug(f"Confirmation: boosted {len(node_ids)} nodes")

    # ── Background knowledge extraction ───────────────────────────

    async def _extract_and_evolve(
        self,
        user_message: str,
        agent_response: str,
        domain: str,
        is_correction: bool = False,
    ) -> None:
        """
        Background task: extract knowledge and evolve the graph.
        Errors are non-critical — logged but don't affect user experience.
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

                # Record corrections as indexed mistakes for future avoidance
                if is_correction and extraction.corrections:
                    for corr in extraction.corrections:
                        wrong = corr.get("wrong", "")
                        correct = corr.get("correct", "")
                        if wrong and correct:
                            await self._mistake_tracker.record_mistake(
                                session_id=self._session_id,
                                user_input=user_message[:400],
                                agent_response=wrong[:400],
                                correction=correct[:300],
                                domain=domain,
                            )

                if stats.get("nodes_added", 0) > 0 or stats.get("facts_added", 0) > 0:
                    self._graph.save()
                    logger.debug(
                        f"Graph evolved: +{stats.get('nodes_added',0)} nodes, "
                        f"+{stats.get('facts_added',0)} facts"
                    )

        except Exception as e:
            logger.warning(f"Background extraction skipped: {e}")
