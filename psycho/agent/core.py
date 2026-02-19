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
from psycho.storage.database import Database
from psycho.storage.graph_store import GraphStore
from psycho.storage.vector_store import VectorStore

from .loop import AgentLoop
from .reflection import ReflectionEngine


class PsychoAgent:
    """
    Top-level agent class — the full self-evolving system.

    Subsystems:
        LLM provider         — Anthropic Claude or Ollama (local)
        Memory manager       — 4-tier: short-term, long-term, semantic, episodic
        Knowledge graph      — NetworkX + ChromaDB, self-evolving
        Graph evolver        — confidence updates, dedup, maintenance
        Knowledge extractor  — LLM-powered entity/relation extraction
        Graph reasoner       — context retrieval for prompt injection
        Ingestion pipeline   — file/text ingestion into the graph
        Mistake tracker      — learns from errors, warns before repeating
        Signal detector      — detects corrections/confirmations in messages
        Insight generator    — derives insights from graph + session patterns
        Session journal      — writes learning record after each session
        Reflection engine    — post-session synthesis (the self-evolution core)
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

        self._session_id = str(uuid.uuid4())[:8]
        self._session_started_at = time.time()
        self._started = False

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

        # Session tracking
        await self._memory.long_term.create_session(self._session_id)

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
        )

        self._started = True
        g_stats = self._graph.get_stats()
        logger.info(
            f"{AGENT_NAME} started | session={self._session_id} | "
            f"provider={self._llm.provider_name} | model={self._llm.model_name} | "
            f"graph={g_stats['active_nodes']} nodes, {g_stats['total_edges']} edges"
        )

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

        reflection_result = None
        if run_reflection and self._reflection:
            reflection_result = await self._reflection.run(
                session_id=self._session_id,
                session_started_at=self._session_started_at,
            )
        else:
            # Always save graph on exit even without reflection
            if self._graph:
                self._graph.save()

        if self._memory:
            await self._memory.long_term.end_session(self._session_id)

        if self._db:
            await self._db.close()

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
        return stats
