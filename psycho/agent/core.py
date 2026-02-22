"""PsychoAgent — initializes and wires all subsystems."""

from __future__ import annotations

import time
import uuid
from pathlib import Path

from loguru import logger

from psycho.config import get_settings
from psycho.config.constants import AGENT_NAME
from psycho.knowledge.evolution import GraphEvolver
from psycho.knowledge.extractor import KnowledgeExtractor
from psycho.knowledge.graph import KnowledgeGraph
from psycho.knowledge.ingestion import IngestionPipeline
from psycho.knowledge.reasoner import GraphReasoner
from psycho.domains import (
    CodingHandler, DomainRouter, GeneralHandler, HealthHandler, TaskHandler
)
from psycho.learning.insight_generator import InsightGenerator
from psycho.learning.mistake_tracker import MistakeTracker
from psycho.learning.session_journal import SessionJournal
from psycho.llm import create_provider
from psycho.llm.base import LLMProvider
from psycho.memory import MemoryManager
from psycho.personality.adapter import PersonalityAdapter
from psycho.proactive.reminders import ReminderManager
from psycho.proactive.calendar_manager import CalendarManager
from psycho.proactive.checkin import CheckinEngine
from psycho.proactive.scheduler import ProactiveScheduler
from psycho.storage.database import Database
from psycho.storage.graph_store import GraphStore
from psycho.storage.vector_store import VectorStore

from .loop import AgentLoop
from .reflection import ReflectionEngine


class PsychoAgent:
    """
    Top-level agent class — the full self-evolving system with personality.

    Subsystems:
        LLM provider         — Anthropic Claude or Ollama (local)
        Memory manager       — 4-tier: short-term, long-term, semantic, episodic
        Knowledge graph      — NetworkX + ChromaDB, self-evolving
        Graph evolver        — confidence updates, dedup, maintenance
        Knowledge extractor  — LLM-powered entity/relation/personality extraction
        Graph reasoner       — context retrieval for prompt injection
        Ingestion pipeline   — file/text ingestion into the graph
        Mistake tracker      — learns from errors, warns before repeating
        Signal detector      — detects corrections/confirmations in messages
        Insight generator    — derives insights from graph + session patterns
        Session journal      — writes learning record after each session
        Reflection engine    — post-session synthesis (the self-evolution core)
        Personality adapter  — TARS/Jarvis-style adjustable personality engine
        Reminder manager     — smart reminders with natural language parsing
        Calendar manager     — local calendar with optional Google Calendar sync
        Checkin engine       — proactive check-ins based on patterns
        Proactive scheduler  — background async loop for reminders/calendar
        Agent loop           — perceive → think → act → learn cycle
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._settings.ensure_data_dirs()

        # Storage backends
        self._db = Database(self._settings.db_path)
        self._vector_store = VectorStore(self._settings.vector_path)
        self._graph_store = GraphStore(self._settings.graph_path)

        # Subsystems (initialized in start())
        self._llm: LLMProvider | None = None
        self._memory: MemoryManager | None = None
        self._graph: KnowledgeGraph | None = None
        self._evolver: GraphEvolver | None = None
        self._extractor: KnowledgeExtractor | None = None
        self._reasoner: GraphReasoner | None = None
        self._ingestion: IngestionPipeline | None = None
        self._mistake_tracker: MistakeTracker | None = None
        self._insight_generator: InsightGenerator | None = None
        self._journal: SessionJournal | None = None
        self._reflection: ReflectionEngine | None = None
        self._domain_router: DomainRouter | None = None
        self._domain_handlers: dict = {}
        self._loop: AgentLoop | None = None

        # Personality + proactive systems
        self._personality: PersonalityAdapter | None = None
        self._reminders: ReminderManager | None = None
        self._calendar: CalendarManager | None = None
        self._checkin: CheckinEngine | None = None
        self._scheduler: ProactiveScheduler | None = None

        self._session_id = str(uuid.uuid4())[:8]
        self._session_started_at = time.time()
        self._started = False
        self._session_count = 0  # Track across restarts via DB

    async def start(self) -> None:
        """Initialize all subsystems."""
        if self._started:
            return

        # Storage
        await self._db.connect()

        # LLM
        self._llm = create_provider()

        # Memory (4 tiers)
        self._memory = MemoryManager(self._db, self._vector_store)
        await self._memory.initialize()

        # Knowledge graph
        self._graph = KnowledgeGraph(self._graph_store, self._vector_store)
        self._graph.load()
        self._evolver = GraphEvolver(self._graph)
        self._extractor = KnowledgeExtractor(self._llm)
        self._reasoner = GraphReasoner(self._graph)
        self._ingestion = IngestionPipeline(
            graph=self._graph,
            evolver=self._evolver,
            extractor=self._extractor,
            memory=self._memory,
        )

        # Learning subsystems
        self._mistake_tracker = MistakeTracker(self._db, self._vector_store)
        self._insight_generator = InsightGenerator(self._llm, self._graph)
        self._journal = SessionJournal(self._settings.journal_path)

        # Reflection engine
        self._reflection = ReflectionEngine(
            llm=self._llm,
            memory=self._memory,
            graph=self._graph,
            evolver=self._evolver,
            mistake_tracker=self._mistake_tracker,
            insight_generator=self._insight_generator,
            journal=self._journal,
            reasoner=self._reasoner,
        )

        # Domain intelligence
        self._domain_router = DomainRouter(self._llm)
        self._domain_handlers = {
            "coding":  CodingHandler(self._db, self._llm),
            "health":  HealthHandler(self._db, self._llm),
            "tasks":   TaskHandler(self._db, self._llm),
            "general": GeneralHandler(self._db, self._llm),
        }

        # ── Personality engine ─────────────────────────────────────
        personality_path = self._settings.get_personality_path()
        if not personality_path.exists():
            # Create initial personality from settings
            from psycho.personality.traits import AgentPersonality
            initial = AgentPersonality.from_dict(
                self._settings.get_initial_personality_dict()
            )
            initial.save(personality_path)

        self._personality = PersonalityAdapter.create(
            personality_path=personality_path,
            graph=self._graph,
        )

        # ── Proactive subsystems ───────────────────────────────────
        self._reminders = ReminderManager(self._db)
        self._calendar = CalendarManager(
            self._db,
            google_credentials_path=self._settings.google_calendar_credentials or None,
        )
        self._checkin = CheckinEngine()

        # Initialize Google Calendar if credentials are configured
        if self._settings.google_calendar_credentials:
            try:
                await self._calendar.try_init_google(
                    self._settings.google_calendar_credentials
                )
            except Exception as e:
                logger.debug(f"Google Calendar init skipped: {e}")

        # Proactive scheduler (background task — started separately for web mode)
        self._scheduler = ProactiveScheduler(
            reminder_manager=self._reminders,
            calendar_manager=self._calendar,
            checkin_engine=self._checkin,
        )

        # Session tracking
        await self._memory.long_term.create_session(self._session_id)

        # Count sessions from DB for relationship depth
        try:
            row = await self._db.fetch_one("SELECT COUNT(*) FROM sessions")
            self._session_count = row[0] if row else 1
        except Exception:
            self._session_count = 1

        if self._personality:
            self._personality.increment_session()

        # Agent loop (fully wired with all subsystems)
        self._loop = AgentLoop(
            session_id=self._session_id,
            llm=self._llm,
            memory=self._memory,
            graph=self._graph,
            evolver=self._evolver,
            extractor=self._extractor,
            reasoner=self._reasoner,
            mistake_tracker=self._mistake_tracker,
            domain_router=self._domain_router,
            domain_handlers=self._domain_handlers,
            personality=self._personality,
            reminder_manager=self._reminders,
            calendar_manager=self._calendar,
            checkin_engine=self._checkin,
        )

        self._started = True
        g_stats = self._graph.get_stats()
        logger.info(
            f"{AGENT_NAME} started | session={self._session_id} | "
            f"provider={self._llm.provider_name} | model={self._llm.model_name} | "
            f"graph={g_stats['active_nodes']} nodes, {g_stats['total_edges']} edges | "
            f"personality={self._personality.get_trait_status() if self._personality else 'default'}"
        )

    async def start_scheduler(self) -> None:
        """Start the proactive background scheduler (call from FastAPI startup)."""
        if self._scheduler and self._settings.proactive_enabled:
            await self._scheduler.start()
            logger.info("Proactive scheduler started")

    async def chat(self, user_message: str) -> str:
        if not self._started:
            await self.start()
        return await self._loop.process(user_message)

    async def stream_chat(self, user_message: str):
        """Yield tokens as they arrive from the LLM."""
        if not self._started:
            await self.start()
        async for token in self._loop.stream_process(user_message):
            yield token

    async def stream_chat_with_image(
        self, user_message: str, image_data: bytes, media_type: str
    ):
        """Yield tokens for a vision chat message (image + optional text)."""
        if not self._started:
            await self.start()
        if not hasattr(self._llm, "stream_with_image"):
            yield "[Image chat is not supported by the current LLM provider]"
            return
        async for token in self._loop.stream_process_with_image(
            user_message, image_data, media_type
        ):
            yield token

    async def reflect(self):
        """Run post-session reflection. Returns ReflectionResult or None."""
        if not self._started:
            return None
        return await self._reflection.run(
            session_id=self._session_id,
            session_started_at=self._session_started_at,
        )

    async def ingest_file(self, path: str) -> dict:
        if not self._started:
            await self.start()
        p = Path(path)
        if p.is_dir():
            results = await self._ingestion.ingest_folder(p)
            return {
                "files_processed": len(results),
                "nodes_added": sum(r.nodes_added for r in results),
                "facts_added": sum(r.facts_added for r in results),
                "errors": [e for r in results for e in r.errors],
            }
        result = await self._ingestion.ingest_file(p)
        return {
            "nodes_added": result.nodes_added,
            "facts_added": result.facts_added,
            "edges_added": result.edges_added,
            "chunks": result.chunks_processed,
            "errors": result.errors,
        }

    async def ingest_text(
        self, text: str, source_name: str = "manual", domain: str = "general"
    ) -> dict:
        if not self._started:
            await self.start()
        result = await self._ingestion.ingest_text(text, source_name, domain)
        return {
            "nodes_added": result.nodes_added,
            "facts_added": result.facts_added,
            "chunks": result.chunks_processed,
        }

    async def stop(self, run_reflection: bool = False) -> dict | None:
        """Graceful shutdown. Optionally run post-session reflection."""
        if not self._started:
            return None

        # Stop background scheduler
        if self._scheduler and self._scheduler.is_running:
            await self._scheduler.stop()

        reflection_result = None
        if run_reflection and self._reflection:
            reflection_result = await self._reflection.run(
                session_id=self._session_id,
                session_started_at=self._session_started_at,
            )
        else:
            if self._graph:
                self._graph.save()

        if self._memory:
            await self._memory.long_term.end_session(self._session_id)

        if self._db:
            await self._db.close()

        # Save personality state
        if self._personality and self._settings.get_personality_path():
            try:
                self._personality.traits.save(self._settings.get_personality_path())
            except Exception as e:
                logger.debug(f"Personality save skipped: {e}")

        self._started = False
        logger.info(f"{AGENT_NAME} stopped | session={self._session_id}")
        return reflection_result

    # ── Accessors ─────────────────────────────────────────────────

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def session_started_at(self) -> float:
        return self._session_started_at

    @property
    def llm(self) -> LLMProvider:
        return self._llm

    @property
    def memory(self) -> MemoryManager:
        return self._memory

    @property
    def graph(self) -> KnowledgeGraph:
        return self._graph

    @property
    def reasoner(self) -> GraphReasoner:
        return self._reasoner

    @property
    def mistake_tracker(self) -> MistakeTracker:
        return self._mistake_tracker

    @property
    def personality(self) -> PersonalityAdapter | None:
        return self._personality

    @property
    def scheduler(self) -> ProactiveScheduler | None:
        return self._scheduler

    @property
    def reminder_manager(self) -> ReminderManager | None:
        return self._reminders

    @property
    def calendar_manager(self) -> CalendarManager | None:
        return self._calendar

    @property
    def settings(self):
        return self._settings

    @property
    def task_manager(self):
        h = self._domain_handlers.get("tasks")
        return h.manager if h else None

    @property
    def health_tracker(self):
        h = self._domain_handlers.get("health")
        return h._tracker if h else None

    async def get_stats(self) -> dict:
        if not self._memory:
            return {}
        stats = await self._memory.get_stats()
        stats["session_id"] = self._session_id
        stats["model"] = self._llm.model_name if self._llm else "unknown"
        stats["provider"] = self._llm.provider_name if self._llm else "unknown"
        if self._graph:
            g = self._graph.get_stats()
            stats["graph_nodes"] = g["active_nodes"]
            stats["graph_edges"] = g["total_edges"]
            stats["graph_avg_confidence"] = g["average_confidence"]
        if self._mistake_tracker:
            m = await self._mistake_tracker.get_stats()
            stats["total_mistakes"] = m["total_mistakes"]
        if self.task_manager:
            t = await self.task_manager.get_stats()
            stats["pending_tasks"] = t["pending_tasks"]
        if self.health_tracker:
            h = await self.health_tracker.get_stats()
            stats["health_entries"] = h["total_entries"]
        if self._reminders:
            r = await self._reminders.get_stats()
            stats["pending_reminders"] = r["pending"]
        if self._scheduler:
            stats["unread_notifications"] = self._scheduler.unread_count
        if self._personality:
            stats["personality"] = self._personality.traits.to_dict()
        return stats

    # ── Personality control API ────────────────────────────────────

    def set_personality_trait(self, trait: str, value: float) -> bool:
        """Directly set a personality trait. Returns True if successful."""
        if not self._personality:
            return False
        success = self._personality.traits.set_trait(trait, value)
        if success:
            path = self._settings.get_personality_path()
            self._personality.traits.save(path)
        return success

    def get_personality(self) -> dict:
        """Return current personality trait values."""
        if not self._personality:
            return {}
        return self._personality.traits.to_dict()
