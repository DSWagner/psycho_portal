"""PsychoAgent — initializes and wires all subsystems."""

from __future__ import annotations

import uuid

from loguru import logger

from psycho.config import get_settings
from psycho.config.constants import AGENT_NAME
from psycho.knowledge.evolution import GraphEvolver
from psycho.knowledge.extractor import KnowledgeExtractor
from psycho.knowledge.graph import KnowledgeGraph
from psycho.knowledge.ingestion import IngestionPipeline
from psycho.knowledge.reasoner import GraphReasoner
from psycho.llm import create_provider
from psycho.llm.base import LLMProvider
from psycho.memory import MemoryManager
from psycho.storage.database import Database
from psycho.storage.graph_store import GraphStore
from psycho.storage.vector_store import VectorStore

from .loop import AgentLoop


class PsychoAgent:
    """
    Top-level agent class — owns and coordinates all subsystems.

    Subsystems:
        LLM provider         — Anthropic Claude or Ollama (local)
        Memory manager       — 4-tier: short-term, long-term, semantic, episodic
        Knowledge graph      — NetworkX + ChromaDB, self-evolving
        Graph evolver        — confidence updates, dedup, maintenance
        Graph extractor      — LLM-powered entity/relation extraction
        Graph reasoner       — context retrieval for prompt injection
        Ingestion pipeline   — file/text ingestion into the graph
        Agent loop           — perceive → think → act → learn cycle

    Usage:
        agent = PsychoAgent()
        await agent.start()
        response = await agent.chat("Hello!")
        await agent.ingest_file("notes.pdf")
        await agent.stop()
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
        self._loop: AgentLoop | None = None

        self._session_id = str(uuid.uuid4())[:8]
        self._started = False

    async def start(self) -> None:
        """Initialize all subsystems. Call before first chat()."""
        if self._started:
            return

        # 1. Database
        await self._db.connect()

        # 2. LLM provider
        self._llm = create_provider()

        # 3. Memory (4 tiers)
        self._memory = MemoryManager(self._db, self._vector_store)
        await self._memory.initialize()

        # 4. Knowledge graph subsystem
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

        # 5. Session tracking
        await self._memory.long_term.create_session(self._session_id)

        # 6. Agent loop
        self._loop = AgentLoop(
            session_id=self._session_id,
            llm=self._llm,
            memory=self._memory,
            graph=self._graph,
            evolver=self._evolver,
            extractor=self._extractor,
            reasoner=self._reasoner,
        )

        self._started = True
        graph_stats = self._graph.get_stats()
        logger.info(
            f"{AGENT_NAME} started | session={self._session_id} | "
            f"provider={self._llm.provider_name} | model={self._llm.model_name} | "
            f"graph={graph_stats['active_nodes']} nodes"
        )

    async def chat(self, user_message: str) -> str:
        """Process a user message and return the agent's response."""
        if not self._started:
            await self.start()
        return await self._loop.process(user_message)

    async def ingest_file(self, path: str) -> dict:
        """Ingest a file or folder into the knowledge graph."""
        if not self._started:
            await self.start()
        from pathlib import Path
        p = Path(path)
        if p.is_dir():
            results = await self._ingestion.ingest_folder(p)
            total_nodes = sum(r.nodes_added for r in results)
            total_facts = sum(r.facts_added for r in results)
            return {
                "files_processed": len(results),
                "nodes_added": total_nodes,
                "facts_added": total_facts,
                "errors": [e for r in results for e in r.errors],
            }
        else:
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
        """Ingest raw text into the knowledge graph."""
        if not self._started:
            await self.start()
        result = await self._ingestion.ingest_text(text, source_name, domain)
        return {
            "nodes_added": result.nodes_added,
            "facts_added": result.facts_added,
            "chunks": result.chunks_processed,
        }

    async def stop(self) -> None:
        """Graceful shutdown: save graph, close connections."""
        if not self._started:
            return

        if self._memory:
            await self._memory.long_term.end_session(self._session_id)

        if self._graph:
            self._graph.save()

        if self._db:
            await self._db.close()

        self._started = False
        logger.info(f"{AGENT_NAME} stopped | session={self._session_id}")

    # ── Accessors ─────────────────────────────────────────────────

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
    def graph(self) -> KnowledgeGraph:
        return self._graph

    @property
    def reasoner(self) -> GraphReasoner:
        return self._reasoner

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
        if self._graph:
            graph_stats = self._graph.get_stats()
            stats["graph_nodes"] = graph_stats["active_nodes"]
            stats["graph_edges"] = graph_stats["total_edges"]
            stats["graph_avg_confidence"] = graph_stats["average_confidence"]
        return stats
