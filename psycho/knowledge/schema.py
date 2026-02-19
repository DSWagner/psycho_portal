"""Knowledge graph schema — node types, edge types, and data models."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class NodeType(str, Enum):
    CONCEPT    = "concept"     # Abstract ideas: "recursion", "ketosis", "async"
    ENTITY     = "entity"      # Named things: "Python", "asyncio", "OpenAI"
    PERSON     = "person"      # People: "Guido van Rossum", "user"
    FACT       = "fact"        # Specific true statements
    PREFERENCE = "preference"  # User preferences: "prefers dark mode"
    SKILL      = "skill"       # Capabilities: "can write SQL", "knows Python"
    MISTAKE    = "mistake"     # Recorded errors made by agent
    QUESTION   = "question"    # Open questions not yet answered
    DOMAIN     = "domain"      # Top-level areas: "coding", "health"
    TOPIC      = "topic"       # Sub-topics: "async programming", "nutrition"
    FILE       = "file"        # Ingested file references
    EVENT      = "event"       # Time-bound occurrences
    TECHNOLOGY = "technology"  # Tools, frameworks, languages


class EdgeType(str, Enum):
    # Semantic relationships
    IS_A          = "is_a"           # Python IS_A programming language
    HAS_PROPERTY  = "has_property"   # Python HAS_PROPERTY dynamic typing
    PART_OF       = "part_of"        # asyncio PART_OF Python stdlib
    RELATES_TO    = "relates_to"     # asyncio RELATES_TO concurrency
    DEPENDS_ON    = "depends_on"     # FastAPI DEPENDS_ON Starlette
    CAUSES        = "causes"         # race condition CAUSES data corruption
    USED_IN       = "used_in"        # asyncio.Lock USED_IN concurrency control
    SIMILAR_TO    = "similar_to"     # asyncio SIMILAR_TO trio (fuzzy dedup)

    # Knowledge quality relationships
    CONTRADICTS   = "contradicts"    # fact A CONTRADICTS fact B
    SUPPORTS      = "supports"       # evidence SUPPORTS fact
    CORRECTS      = "corrects"       # new knowledge CORRECTS old mistake

    # User relationships
    PREFERRED_BY  = "preferred_by"   # Python PREFERRED_BY user
    KNOWS         = "knows"          # user KNOWS recursion (at what level)
    DISLIKES      = "dislikes"       # user DISLIKES Java verbosity

    # Provenance relationships
    EXTRACTED_FROM = "extracted_from"  # knowledge EXTRACTED_FROM session/file
    INFERRED_FROM  = "inferred_from"   # knowledge INFERRED_FROM other knowledge
    MENTIONED_IN   = "mentioned_in"    # entity MENTIONED_IN file/session

    # Structural
    AUTHORED_BY    = "authored_by"    # file AUTHORED_BY person
    CONTAINS       = "contains"       # file CONTAINS concept


# Confidence level labels for display
def confidence_label(conf: float) -> str:
    if conf >= 0.8:
        return "HIGH"
    elif conf >= 0.5:
        return "MEDIUM"
    elif conf >= 0.2:
        return "LOW"
    return "UNCERTAIN"


def confidence_bar(conf: float, width: int = 10) -> str:
    filled = round(conf * width)
    return "█" * filled + "░" * (width - filled)


@dataclass
class KnowledgeNode:
    """A node in the knowledge graph."""

    id: str
    type: NodeType
    label: str                              # Primary name (normalized lowercase)
    display_label: str = ""                 # Original casing for display
    properties: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.5
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)
    access_count: int = 0
    domain: str = "general"
    sources: list[str] = field(default_factory=list)   # session IDs / file paths
    embedding_id: str = ""                  # ChromaDB doc ID for this node
    deprecated: bool = False
    deprecation_reason: str = ""

    def __post_init__(self) -> None:
        if not self.display_label:
            self.display_label = self.label

    @classmethod
    def create(
        cls,
        type: NodeType,
        label: str,
        domain: str = "general",
        properties: dict | None = None,
        confidence: float = 0.5,
        sources: list[str] | None = None,
    ) -> "KnowledgeNode":
        node_id = str(uuid.uuid4())
        return cls(
            id=node_id,
            type=type,
            label=label.lower().strip(),
            display_label=label.strip(),
            properties=properties or {},
            confidence=confidence,
            domain=domain,
            sources=sources or [],
        )

    def touch(self) -> None:
        """Mark as accessed."""
        self.last_accessed = time.time()
        self.access_count += 1

    def update_confidence(self, delta: float) -> None:
        """Apply a confidence delta, clamped to [0.05, 0.95]."""
        self.confidence = max(0.05, min(0.95, self.confidence + delta))
        self.last_updated = time.time()

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type.value,
            "label": self.label,
            "display_label": self.display_label,
            "properties": self.properties,
            "confidence": self.confidence,
            "created_at": self.created_at,
            "last_accessed": self.last_accessed,
            "last_updated": self.last_updated,
            "access_count": self.access_count,
            "domain": self.domain,
            "sources": self.sources,
            "embedding_id": self.embedding_id,
            "deprecated": self.deprecated,
            "deprecation_reason": self.deprecation_reason,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "KnowledgeNode":
        d = d.copy()
        d["type"] = NodeType(d["type"])
        return cls(**d)

    def to_text(self) -> str:
        """Compact text representation for embedding."""
        parts = [f"{self.type.value}: {self.display_label}"]
        if self.domain != "general":
            parts.append(f"domain: {self.domain}")
        for k, v in list(self.properties.items())[:3]:
            parts.append(f"{k}: {v}")
        return " | ".join(parts)

    def confidence_display(self) -> str:
        return (
            f"{confidence_bar(self.confidence)} "
            f"{self.confidence:.2f} [{confidence_label(self.confidence)}]"
        )


@dataclass
class KnowledgeEdge:
    """A directed edge between two knowledge nodes."""

    source_id: str
    target_id: str
    type: EdgeType
    confidence: float = 0.5
    weight: float = 1.0                     # Reinforcement count / strength
    properties: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    last_reinforced: float = field(default_factory=time.time)

    def reinforce(self, amount: float = 0.1) -> None:
        self.weight += amount
        self.confidence = min(0.95, self.confidence + 0.03)
        self.last_reinforced = time.time()

    def to_dict(self) -> dict:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "type": self.type.value,
            "confidence": self.confidence,
            "weight": self.weight,
            "properties": self.properties,
            "created_at": self.created_at,
            "last_reinforced": self.last_reinforced,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "KnowledgeEdge":
        d = d.copy()
        d["type"] = EdgeType(d["type"])
        return cls(**d)
