"""File ingestion pipeline — ingest any file into the knowledge graph."""

from __future__ import annotations

import ast
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, AsyncIterator

from loguru import logger

from psycho.knowledge.schema import EdgeType, KnowledgeEdge, KnowledgeNode, NodeType

if TYPE_CHECKING:
    from psycho.knowledge.evolution import GraphEvolver
    from psycho.knowledge.extractor import KnowledgeExtractor
    from psycho.knowledge.graph import KnowledgeGraph
    from psycho.memory.manager import MemoryManager

# ── Supported file types ──────────────────────────────────────────────────────

SUPPORTED_EXTENSIONS = {
    # Text / documents
    ".txt", ".md", ".rst", ".org", ".tex",
    # Code
    ".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".rs", ".java",
    ".cpp", ".c", ".h", ".cs", ".rb", ".php", ".swift", ".kt",
    ".sh", ".bash", ".zsh", ".fish",
    # Data
    ".json", ".yaml", ".yml", ".toml", ".csv", ".xml",
    # PDF
    ".pdf",
    # Images (vision-extracted)
    ".png", ".jpg", ".jpeg", ".gif", ".webp",
}

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
IMAGE_MEDIA_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}

# Chunk sizes
CHUNK_SIZE = 1500        # characters per chunk
CHUNK_OVERLAP = 200      # overlap between chunks
MIN_CHUNK_SIZE = 100     # skip chunks smaller than this


@dataclass
class IngestionResult:
    """Result of ingesting a single file or text block."""

    source_path: str
    source_type: str
    chunks_processed: int = 0
    nodes_added: int = 0
    edges_added: int = 0
    facts_added: int = 0
    errors: list[str] = field(default_factory=list)
    file_node_id: str = ""
    elapsed_seconds: float = 0.0

    def summary(self) -> str:
        return (
            f"{self.source_path}: {self.chunks_processed} chunks → "
            f"{self.nodes_added} nodes, {self.edges_added} edges, "
            f"{self.facts_added} facts in {self.elapsed_seconds:.1f}s"
        )


class IngestionPipeline:
    """
    Ingests files and raw text into the knowledge graph.

    Supported inputs:
        Files:  .txt .md .py .js .ts .json .yaml .csv .pdf (and more)
        Folders: recursively processes all supported files
        Raw text: pass directly as a string

    Pipeline:
        1. Extract raw text (format-aware)
        2. Chunk into overlapping segments
        3. Run LLM extraction on each chunk
        4. Integrate into graph via GraphEvolver
        5. Link all knowledge to a FILE node
        6. Store chunks in semantic memory (ChromaDB) for retrieval
    """

    def __init__(
        self,
        graph: "KnowledgeGraph",
        evolver: "GraphEvolver",
        extractor: "KnowledgeExtractor",
        memory: "MemoryManager",
    ) -> None:
        self._graph = graph
        self._evolver = evolver
        self._extractor = extractor
        self._memory = memory

    # ── Public API ────────────────────────────────────────────────

    async def ingest_file(self, file_path: str | Path) -> IngestionResult:
        """Ingest a single file into the knowledge graph."""
        path = Path(file_path)
        start = time.time()
        result = IngestionResult(
            source_path=str(path),
            source_type=path.suffix.lower(),
        )

        if not path.exists():
            result.errors.append(f"File not found: {path}")
            return result

        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            result.errors.append(f"Unsupported file type: {path.suffix}")
            return result

        logger.info(f"Ingesting file: {path}")

        try:
            # 1. Extract text from file
            text, metadata = await self._extract_text(path)
            if not text:
                result.errors.append("No text extracted")
                return result

            # 2. Create FILE node
            file_node = KnowledgeNode.create(
                type=NodeType.FILE,
                label=path.name.lower(),
                domain=self._guess_domain(path),
                properties={
                    "path": str(path),
                    "extension": path.suffix,
                    "size_bytes": path.stat().st_size,
                    **metadata,
                },
                confidence=0.9,
                sources=[str(path)],
            )
            file_node_id = self._graph.upsert_node(file_node)
            result.file_node_id = file_node_id

            # 3. Process chunks
            chunks = list(self._chunk_text(text))
            result.chunks_processed = len(chunks)
            logger.info(f"  {len(chunks)} chunks to process from {path.name}")

            for i, chunk in enumerate(chunks):
                chunk_stats = await self._process_chunk(
                    chunk=chunk,
                    source_name=f"{path.name}:chunk{i+1}",
                    domain=self._guess_domain(path),
                    file_node_id=file_node_id,
                )
                result.nodes_added += chunk_stats.get("nodes_added", 0)
                result.edges_added += chunk_stats.get("edges_added", 0)
                result.facts_added += chunk_stats.get("facts_added", 0)

            # 4. Store full text in semantic memory for retrieval
            await self._memory.semantic.store_interaction(
                session_id="file_ingestion",
                user_message=f"File: {path.name}",
                agent_response=text[:1000],  # summary/preview
                domain=self._guess_domain(path),
                interaction_id=f"file_{file_node_id[:8]}",
            )

            # 5. Persist graph
            self._graph.save()

        except Exception as e:
            result.errors.append(str(e))
            logger.error(f"Ingestion failed for {path}: {e}")

        result.elapsed_seconds = time.time() - start
        logger.info(f"Ingestion complete: {result.summary()}")
        return result

    async def ingest_folder(self, folder_path: str | Path) -> list[IngestionResult]:
        """Recursively ingest all supported files in a folder."""
        folder = Path(folder_path)
        if not folder.is_dir():
            logger.error(f"Not a directory: {folder}")
            return []

        results = []
        files = [
            p for p in folder.rglob("*")
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        logger.info(f"Found {len(files)} files in {folder}")

        for file_path in files:
            result = await self.ingest_file(file_path)
            results.append(result)

        return results

    async def ingest_text(
        self,
        text: str,
        source_name: str = "manual_input",
        domain: str = "general",
    ) -> IngestionResult:
        """Ingest raw text directly (e.g., pasted content, API response)."""
        start = time.time()
        result = IngestionResult(source_path=source_name, source_type="text")

        # Create a TEXT node
        text_node = KnowledgeNode.create(
            type=NodeType.FILE,
            label=source_name.lower(),
            domain=domain,
            properties={"type": "raw_text", "length": len(text)},
            confidence=0.8,
            sources=[source_name],
        )
        file_node_id = self._graph.upsert_node(text_node)
        result.file_node_id = file_node_id

        chunks = list(self._chunk_text(text))
        result.chunks_processed = len(chunks)

        for i, chunk in enumerate(chunks):
            chunk_stats = await self._process_chunk(
                chunk=chunk,
                source_name=f"{source_name}:chunk{i+1}",
                domain=domain,
                file_node_id=file_node_id,
            )
            result.nodes_added += chunk_stats.get("nodes_added", 0)
            result.edges_added += chunk_stats.get("edges_added", 0)
            result.facts_added += chunk_stats.get("facts_added", 0)

        self._graph.save()
        result.elapsed_seconds = time.time() - start
        return result

    # ── Text extraction (format-aware) ────────────────────────────

    async def _extract_text(self, path: Path) -> tuple[str, dict]:
        """Extract raw text + metadata from a file."""
        ext = path.suffix.lower()
        metadata = {}

        if ext in IMAGE_EXTENSIONS:
            return await self._extract_image(path)

        if ext == ".pdf":
            return await self._extract_pdf(path)

        if ext in (".py",):
            return self._extract_python(path)

        if ext in (".json",):
            return self._extract_json(path)

        if ext in (".yaml", ".yml"):
            return self._extract_yaml(path)

        if ext in (".csv",):
            return self._extract_csv(path)

        if ext in (".md", ".rst", ".org"):
            text = path.read_text(encoding="utf-8", errors="replace")
            # Strip markdown headers for metadata
            title_match = re.match(r"^#+ (.+)", text)
            if title_match:
                metadata["title"] = title_match.group(1)
            return text, metadata

        # Default: read as plain text
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
            return text, metadata
        except Exception as e:
            return "", {"error": str(e)}

    async def _extract_image(self, path: Path) -> tuple[str, dict]:
        """Extract all knowledge from an image using Claude vision."""
        try:
            llm = self._extractor._llm
            if not hasattr(llm, "complete_with_image"):
                return "", {"error": "Vision not supported by current LLM provider (Anthropic only)"}

            image_data = path.read_bytes()
            media_type = IMAGE_MEDIA_TYPES.get(path.suffix.lower(), "image/jpeg")

            prompt = (
                "Analyze this image completely and extract ALL knowledge from it.\n\n"
                "Cover everything:\n"
                "1. All visible text — read every word, number, label, caption exactly as written\n"
                "2. Diagrams, charts, graphs — describe what they show and what data they contain\n"
                "3. Screenshots of code or terminals — transcribe the code/output verbatim\n"
                "4. People, objects, scenes — describe with full detail\n"
                "5. UI/interfaces — describe every element, button, and label\n"
                "6. Mathematical formulas or symbols — describe them precisely\n"
                "7. Any relationships, patterns, or conclusions that can be inferred\n\n"
                "Be exhaustive. Every piece of information matters for a knowledge graph."
            )

            description = await llm.complete_with_image(
                image_data=image_data,
                media_type=media_type,
                prompt=prompt,
            )

            return description, {
                "format": "image",
                "media_type": media_type,
                "size_bytes": len(image_data),
                "vision_extracted": True,
            }
        except Exception as e:
            logger.error(f"Image extraction failed for {path}: {e}")
            return "", {"error": str(e)}

    def _extract_python(self, path: Path) -> tuple[str, dict]:
        """Extract Python source with AST-aware structure."""
        try:
            source = path.read_text(encoding="utf-8", errors="replace")
            lines = []

            # Parse AST for structure
            try:
                tree = ast.parse(source)
                for node in ast.walk(tree):
                    if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                        docstring = ast.get_docstring(node) or ""
                        kind = "class" if isinstance(node, ast.ClassDef) else "function"
                        lines.append(f"{kind}: {node.name}")
                        if docstring:
                            lines.append(f"  doc: {docstring[:200]}")
                    elif isinstance(node, ast.Import):
                        for alias in node.names:
                            lines.append(f"import: {alias.name}")
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            lines.append(f"from {node.module} import {', '.join(a.name for a in node.names)}")
            except SyntaxError:
                pass

            structured = "\n".join(lines)
            return f"# File: {path.name}\n{structured}\n\n# Source:\n{source[:3000]}", {
                "language": "python",
                "ast_parsed": True,
            }
        except Exception as e:
            return path.read_text(encoding="utf-8", errors="replace"), {"language": "python", "error": str(e)}

    def _extract_json(self, path: Path) -> tuple[str, dict]:
        """Extract JSON with schema summary."""
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            # Convert to readable summary
            text = json.dumps(data, indent=2)[:5000]
            keys = list(data.keys()) if isinstance(data, dict) else []
            return text, {"format": "json", "top_keys": keys[:10]}
        except Exception:
            return path.read_text(encoding="utf-8", errors="replace"), {"format": "json"}

    def _extract_yaml(self, path: Path) -> tuple[str, dict]:
        """Extract YAML."""
        try:
            import yaml
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
            text = str(data)[:5000]
            return text, {"format": "yaml"}
        except Exception:
            return path.read_text(encoding="utf-8", errors="replace"), {"format": "yaml"}

    def _extract_csv(self, path: Path) -> tuple[str, dict]:
        """Extract CSV with header + sample rows."""
        import csv
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                reader = csv.reader(f)
                rows = list(reader)
            headers = rows[0] if rows else []
            sample = rows[1:6] if len(rows) > 1 else []
            text = f"CSV file: {path.name}\nColumns: {', '.join(headers)}\nSample rows:\n"
            for row in sample:
                text += "  " + " | ".join(row[:10]) + "\n"
            return text, {"format": "csv", "columns": headers, "row_count": len(rows) - 1}
        except Exception:
            return path.read_text(encoding="utf-8", errors="replace"), {"format": "csv"}

    async def _extract_pdf(self, path: Path) -> tuple[str, dict]:
        """Extract text from PDF (requires pypdf)."""
        try:
            from pypdf import PdfReader
            reader = PdfReader(str(path))
            pages = []
            for page in reader.pages[:50]:  # max 50 pages
                text = page.extract_text()
                if text:
                    pages.append(text)
            return "\n\n".join(pages), {
                "format": "pdf",
                "page_count": len(reader.pages),
            }
        except ImportError:
            return "", {"error": "pypdf not installed. Run: pip install pypdf"}
        except Exception as e:
            return "", {"error": str(e)}

    # ── Chunking ──────────────────────────────────────────────────

    def _chunk_text(self, text: str) -> list[str]:
        """
        Split text into overlapping chunks.
        Tries to split on paragraph boundaries first.
        """
        if len(text) <= CHUNK_SIZE:
            return [text] if len(text) >= MIN_CHUNK_SIZE else []

        chunks = []
        paragraphs = re.split(r"\n{2,}", text)

        current_chunk = ""
        for para in paragraphs:
            if len(current_chunk) + len(para) <= CHUNK_SIZE:
                current_chunk += para + "\n\n"
            else:
                if len(current_chunk) >= MIN_CHUNK_SIZE:
                    chunks.append(current_chunk.strip())
                # Start new chunk with overlap
                overlap = current_chunk[-CHUNK_OVERLAP:] if len(current_chunk) > CHUNK_OVERLAP else current_chunk
                current_chunk = overlap + para + "\n\n"

        if current_chunk.strip() and len(current_chunk) >= MIN_CHUNK_SIZE:
            chunks.append(current_chunk.strip())

        return chunks

    # ── Chunk processing ──────────────────────────────────────────

    async def _process_chunk(
        self,
        chunk: str,
        source_name: str,
        domain: str,
        file_node_id: str,
    ) -> dict:
        """Extract knowledge from a single chunk and integrate into graph."""
        extraction = await self._extractor.extract_from_text(
            text=chunk,
            source_name=source_name,
            domain=domain,
        )

        stats = await self._evolver.integrate(extraction)

        # Link all new nodes to the FILE node via EXTRACTED_FROM
        for node in extraction.entities + extraction.facts + extraction.preferences:
            existing = self._graph.find_node_by_label(node.label)
            if existing:
                link_edge = KnowledgeEdge(
                    source_id=existing.id,
                    target_id=file_node_id,
                    type=EdgeType.EXTRACTED_FROM,
                    confidence=0.8,
                )
                self._graph.upsert_edge(link_edge)

        # Store chunk in semantic memory for later retrieval
        if len(chunk) >= MIN_CHUNK_SIZE:
            await self._memory.semantic.store_interaction(
                session_id="file_ingestion",
                user_message=f"From {source_name}:",
                agent_response=chunk[:500],
                domain=domain,
            )

        return stats

    @staticmethod
    def _guess_domain(path: Path) -> str:
        """Guess the knowledge domain from file path/extension."""
        name = path.name.lower()
        ext = path.suffix.lower()

        if ext in (".py", ".js", ".ts", ".go", ".rs", ".java", ".cpp", ".c"):
            return "coding"
        if any(k in name for k in ("health", "diet", "nutrition", "exercise", "medical")):
            return "health"
        if any(k in name for k in ("math", "calculus", "algebra", "statistics")):
            return "math"
        if any(k in name for k in ("science", "physics", "chemistry", "biology")):
            return "science"
        return "general"
