"""AgentLoop — perceive → think → act → learn, with full self-evolution + personality wiring."""

from __future__ import annotations

import asyncio
import re
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, AsyncIterator, Optional

from loguru import logger

from psycho.config.constants import (
    AGENT_NAME,
    CORRECTION_INSTRUCTION,
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
    from psycho.personality.adapter import PersonalityAdapter
    from psycho.proactive.reminders import ReminderManager
    from psycho.proactive.calendar_manager import CalendarManager
    from psycho.proactive.checkin import CheckinEngine


class AgentLoop:
    """
    Core interaction cycle — every message goes through:
      0.  Personality trait command detection (TARS-style: "set humor to 90%")
      1.  Domain classification
      2.  Signal detection (correction / confirmation)
      3.  Real-time confidence update on signal
      4.  Parallel: semantic memory + mistake warnings + reminders
      5.  Knowledge graph context retrieval
      6.  Personality + user adaptation sections
      7.  System prompt assembly
      8.  LLM call (or stream)
      9.  Domain post-processing (code run / metric log / task create)
      10. Reminder creation if detected in message
      11. Calendar event creation if detected in message
      12. Memory write (4 tiers, awaited)
      13. Background: knowledge extraction + graph evolution
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
        personality: "PersonalityAdapter | None" = None,
        reminder_manager: "ReminderManager | None" = None,
        calendar_manager: "CalendarManager | None" = None,
        checkin_engine: "CheckinEngine | None" = None,
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
        self._personality = personality
        self._reminders = reminder_manager
        self._calendar = calendar_manager
        self._checkin = checkin_engine
        self._last_domain_result = None
        self._last_domain: str = "general"
        self._last_search_query: str = ""
        self._personality_changes: list[str] = []  # Last applied trait changes

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

        # Handle reminder/calendar extraction from message
        await self._handle_proactive_extraction(user_message, ctx)

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

        if self._personality:
            self._personality.increment_interaction()

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

        await self._handle_proactive_extraction(user_message, ctx)

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

        if self._personality:
            self._personality.increment_interaction()

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

        if self._personality:
            self._personality.increment_interaction()

    # ── Agent name resolution ──────────────────────────────────────

    def _get_agent_name(self) -> str:
        try:
            from psycho.knowledge.schema import NodeType
            prefs = self._graph.find_nodes_by_type(NodeType.PREFERENCE)
            for p in prefs:
                if p.label.startswith("agent_name:") and not p.deprecated:
                    return p.properties.get("value", AGENT_NAME)
        except Exception:
            pass
        return AGENT_NAME

    def _detect_agent_name_assignment(self, user_message: str) -> Optional[str]:
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
                if name.lower() not in {"a", "an", "the", "my", "your", "their", "its", "our"}:
                    return name
        return None

    def _store_agent_name(self, name: str) -> None:
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

    # ── Context preparation ────────────────────────────────────────

    async def _prepare_context(self, ctx: AgentContext) -> None:
        """Build full context before LLM call."""
        # -1. Detect agent name assignment
        new_name = self._detect_agent_name_assignment(ctx.user_message)
        if new_name:
            self._store_agent_name(new_name)

        # 0. Detect personality trait commands (TARS-style adjustments)
        if self._personality and self._personality.is_trait_command(ctx.user_message):
            changes = self._personality.process_trait_command(ctx.user_message)
            self._personality_changes = changes
            if changes:
                logger.info(f"Personality adjusted: {changes}")
        else:
            self._personality_changes = []

        # 1. Domain classification
        if self._domain_router:
            ctx.domain = await self._domain_router.classify(ctx.user_message)

        # 2. Signal detection
        signal = detect_signal(ctx.user_message)
        ctx.signal_type = signal.type.value
        if signal.type == SignalType.CORRECTION:
            ctx.is_correction = True
            await self._handle_correction_signal(ctx.user_message, signal)
            if self._checkin:
                self._checkin.record_stress()
        elif signal.type == SignalType.CONFIRMATION:
            ctx.is_confirmation = True
            await self._handle_confirmation_signal(ctx.user_message)

        # 3. Parallel retrieval: semantic memory + mistake warnings
        memory_task = asyncio.create_task(
            self._memory.retrieve_context(query=ctx.user_message)
        )
        warnings_task = asyncio.create_task(
            self._mistake_tracker.get_warnings_for_prompt(ctx.user_message)
        )
        ctx.retrieved_memories, ctx.mistake_warnings = await asyncio.gather(
            memory_task, warnings_task
        )

        # 4. Knowledge graph context
        graph_ctx = self._reasoner.get_context_for_prompt(ctx.user_message)
        ctx.graph_context = [graph_ctx] if graph_ctx else []

        # 5. Domain-specific context
        ctx.domain_context = await self._get_domain_context(ctx)

        # 6. Web search
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
        agent_name = self._get_agent_name()

        # Build personality and user adaptation sections
        personality_section = ""
        user_adaptation = ""
        if self._personality:
            personality_section, user_adaptation = self._personality.build_prompt_sections(
                user_message=ctx.user_message,
                conversation_length=len(self._memory.short_term.get_messages()),
            )

        # Build user profile from graph
        user_profile = self._build_user_profile()

        base = SYSTEM_PROMPT_BASE.format(
            name=agent_name,
            personality_section=personality_section,
            user_adaptation=user_adaptation,
            user_profile=user_profile,
        )
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

        # Mistake warnings
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

        # Pending reminders context
        if self._reminders:
            try:
                # Inject upcoming reminders into context
                import asyncio as _asyncio
                loop = _asyncio.get_event_loop()
                # Use the already-running loop's current context
                pending_reminders = []
                # We can't await here so skip for now — handled in proactive context
            except Exception:
                pass

        # Correction acknowledgment
        if ctx.is_correction:
            parts.append(CORRECTION_INSTRUCTION)

        # Personality change acknowledgment
        if self._personality_changes:
            changes_str = " | ".join(self._personality_changes)
            parts.append(
                f"\nPERSONALITY UPDATE: Just applied trait adjustments: {changes_str}. "
                "Acknowledge this naturally in your response — keep it brief and in-character."
            )

        # Check-in context
        if self._checkin:
            checkin_type = self._checkin.should_checkin()
            if checkin_type:
                user_name = self._get_user_name()
                checkin_ctx = self._checkin.generate_checkin_context(
                    checkin_type=checkin_type,
                    user_name=user_name,
                )
                if checkin_ctx:
                    parts.append(checkin_ctx)
                    self._checkin.record_checkin_sent(checkin_type, "")

        return "\n\n".join(parts)

    def _get_user_name(self) -> str:
        """Try to get the user's name from the knowledge graph."""
        try:
            from psycho.knowledge.schema import NodeType
            persons = self._graph.find_nodes_by_type(NodeType.PERSON)
            for p in persons:
                if p.label == "user":
                    name = p.properties.get("name", "")
                    if name and name.lower() != "user":
                        return name
        except Exception:
            pass
        return ""

    def _build_user_profile(self) -> str:
        """Build factual user profile from the knowledge graph for system prompt injection."""
        try:
            from psycho.knowledge.schema import NodeType
            lines = []

            # User identity
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

            # Current projects and goals
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

            # Technologies and tools
            tech_nodes = self._graph.find_nodes_by_type(NodeType.TECHNOLOGY)
            top_tech = sorted(
                [t for t in tech_nodes if not t.deprecated],
                key=lambda x: x.confidence, reverse=True
            )[:6]
            if top_tech:
                tech_labels = ", ".join(t.display_label for t in top_tech)
                lines.append(f"Known technologies: {tech_labels}")

            # Strong preferences (non-personality ones)
            strong_prefs = sorted(
                [p for p in active
                 if p.confidence > 0.65
                 and not any(k in p.label for k in (
                     "current_project:", "goal:", "humor_style:", "comm_style:",
                     "thinking_style:", "interest:", "hobby:", "pet_peeve:",
                     "agent_name:",
                 ))],
                key=lambda x: x.confidence, reverse=True
            )[:5]
            if strong_prefs:
                lines.append("Established preferences:")
                for p in strong_prefs:
                    lines.append(f"  • {p.display_label}")

            # Skills
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

    # ── Proactive extraction ────────────────────────────────────────

    async def _handle_proactive_extraction(self, user_message: str, ctx: AgentContext) -> None:
        """
        Check if the user's message contains a reminder or calendar event creation intent.
        Creates entries automatically without needing explicit domain routing.
        """
        if not self._reminders:
            return

        try:
            from psycho.proactive.reminders import extract_reminder_from_message
            reminder_data = extract_reminder_from_message(user_message)
            if reminder_data:
                reminder = await self._reminders.create(
                    title=reminder_data["title"],
                    due_timestamp=reminder_data["due_timestamp"],
                    notes=reminder_data.get("notes", ""),
                    priority=reminder_data.get("priority", "normal"),
                    session_id=self._session_id,
                )
                logger.info(f"Auto-created reminder: '{reminder.title}'")
        except Exception as e:
            logger.debug(f"Reminder extraction skipped: {e}")

    # ── Signal handlers ─────────────────────────────────────────────

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
                logger.debug(
                    f"Correction target '{correction_hint[:40]}' not in graph yet; "
                    "will be captured by background extraction"
                )

    async def _handle_confirmation_signal(self, user_message: str) -> None:
        """Boost confidence on the topic the user just confirmed."""
        recent_turns = self._memory.short_term.get_turns()
        if not recent_turns:
            return

        last_agent = recent_turns[-1].assistant
        context_items = self._graph.get_context_for_query(last_agent[:200], top_k=3)
        node_ids = [node.id for node, _ in context_items]
        if node_ids:
            self._evolver.confirm_nodes(node_ids)
            logger.debug(f"Confirmation: boosted {len(node_ids)} nodes")

    # ── Background knowledge extraction ────────────────────────────

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

                # Update personality adapter's user profile periodically
                if self._personality and (stats.get("preferences_added", 0) or 0) > 0:
                    self._personality.set_graph(self._graph)

        except Exception as e:
            logger.warning(f"Background extraction skipped: {e}")

    # ── Public accessors ───────────────────────────────────────────

    @property
    def last_domain(self) -> str:
        return self._last_domain

    @property
    def last_search_query(self) -> str:
        return self._last_search_query

    @property
    def last_domain_result(self):
        return self._last_domain_result

    @property
    def last_personality_changes(self) -> list[str]:
        return self._personality_changes
