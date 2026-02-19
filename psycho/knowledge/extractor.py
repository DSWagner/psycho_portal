"""LLM-powered knowledge extractor — mines entities and relationships from text."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from loguru import logger

from psycho.knowledge.schema import EdgeType, KnowledgeEdge, KnowledgeNode, NodeType

if TYPE_CHECKING:
    from psycho.llm.base import LLMProvider


# ── Type maps ─────────────────────────────────────────────────────────────────

def _try_repair_json(raw: str) -> dict | None:
    """
    Attempt to repair truncated JSON by closing any open arrays/objects.
    Returns parsed dict or None if unrepairable.
    """
    # Walk through and try to close at reasonable cut-off points
    for end in range(len(raw) - 1, max(len(raw) - 200, 0), -1):
        candidate = raw[:end]
        # Try closing with common endings
        for suffix in ("}}", "]}}", "]}", "}"):
            try:
                result = json.loads(candidate + suffix)
                if isinstance(result, dict):
                    return result
            except json.JSONDecodeError:
                continue
    return None


_NODE_TYPE_MAP: dict[str, NodeType] = {
    "concept": NodeType.CONCEPT,
    "entity": NodeType.ENTITY,
    "person": NodeType.PERSON,
    "technology": NodeType.TECHNOLOGY,
    "tool": NodeType.TECHNOLOGY,
    "framework": NodeType.TECHNOLOGY,
    "language": NodeType.TECHNOLOGY,
    "library": NodeType.TECHNOLOGY,
    "fact": NodeType.FACT,
    "preference": NodeType.PREFERENCE,
    "skill": NodeType.SKILL,
    "question": NodeType.QUESTION,
    "event": NodeType.EVENT,
    "topic": NodeType.TOPIC,
}

_EDGE_TYPE_MAP: dict[str, EdgeType] = {
    "is_a": EdgeType.IS_A,
    "has_property": EdgeType.HAS_PROPERTY,
    "part_of": EdgeType.PART_OF,
    "relates_to": EdgeType.RELATES_TO,
    "depends_on": EdgeType.DEPENDS_ON,
    "causes": EdgeType.CAUSES,
    "used_in": EdgeType.USED_IN,
    "contradicts": EdgeType.CONTRADICTS,
    "supports": EdgeType.SUPPORTS,
    "corrects": EdgeType.CORRECTS,
    "preferred_by": EdgeType.PREFERRED_BY,
    "knows": EdgeType.KNOWS,
    "mentions": EdgeType.MENTIONED_IN,
    "mentioned_in": EdgeType.MENTIONED_IN,
    "authored_by": EdgeType.AUTHORED_BY,
    "similar_to": EdgeType.SIMILAR_TO,
}


# ── Prompt templates ──────────────────────────────────────────────────────────

CONVERSATION_EXTRACTION_PROMPT = """\
Extract structured knowledge from this conversation exchange to build a personal knowledge graph.
Return ONLY a valid JSON object. No explanation, no markdown, just JSON.

User message: {user_message}
Assistant response: {agent_response}

Focus on extracting knowledge ABOUT THE USER: their projects, preferences, skills, goals, habits, and identity.
Also extract factual knowledge discussed.

Extract:
{{
  "entities": [
    {{"label": "lowercase name", "type": "concept|entity|person|technology|fact|preference|skill|topic", "domain": "coding|health|tasks|general|science|math|finance|other", "properties": {{}}}}
  ],
  "relationships": [
    {{"from_label": "source", "to_label": "target", "type": "is_a|part_of|relates_to|has_property|depends_on|causes|used_in|supports|preferred_by|knows|works_on"}}
  ],
  "user_preferences": [
    {{"label": "specific preference (e.g. prefers dark mode, uses Python 3.12, works in VS Code)", "domain": "domain"}}
  ],
  "user_identity": [
    {{"key": "name|occupation|location|current_project|goal|habit|language|framework", "value": "extracted value"}}
  ],
  "corrections": [
    {{"wrong_label": "incorrect info", "correct_label": "correct info", "explanation": "what changed"}}
  ],
  "key_facts": [
    "specific factual statement extracted verbatim or paraphrased"
  ],
  "open_questions": [
    "unresolved question raised"
  ]
}}

Rules:
- Normalize entity labels to lowercase
- PRIORITIZE user identity/preferences/projects — these are the most valuable signals
- user_identity: extract whenever user says "I am", "I use", "I work on", "my name is", "I prefer", "I'm building"
- Only extract what was EXPLICITLY stated or clearly implied
- Skip trivial exchanges (greetings only, single-word answers)
- Properties should be specific: {{"version": "3.12", "paradigm": "OOP"}}
- Be selective: 3-8 entities per exchange is ideal
- Return empty arrays if nothing relevant"""


TEXT_EXTRACTION_PROMPT = """\
Extract structured knowledge from this text chunk.
Return ONLY a valid JSON object. No explanation, no markdown, just JSON.

Source: {source_name}
Text: {text}

Extract:
{{
  "entities": [
    {{"label": "name", "type": "concept|entity|person|technology|fact|topic", "domain": "domain", "properties": {{}}}}
  ],
  "relationships": [
    {{"from_label": "source", "to_label": "target", "type": "relationship_type"}}
  ],
  "key_facts": [
    "specific factual statement"
  ],
  "summary": "1-2 sentence summary of this chunk"
}}

Rules:
- Normalize labels to lowercase
- Focus on durable knowledge (not opinions or temporal statements)
- Extract up to 10 entities per chunk
- Return empty arrays if nothing worth extracting"""


@dataclass
class ExtractionResult:
    """Result of a single extraction pass."""

    entities: list[KnowledgeNode] = field(default_factory=list)
    edges: list[KnowledgeEdge] = field(default_factory=list)
    preferences: list[KnowledgeNode] = field(default_factory=list)
    corrections: list[dict] = field(default_factory=list)
    questions: list[KnowledgeNode] = field(default_factory=list)
    facts: list[KnowledgeNode] = field(default_factory=list)
    source: str = ""
    raw: dict = field(default_factory=dict)

    def is_empty(self) -> bool:
        return not any([
            self.entities, self.edges, self.preferences,
            self.corrections, self.questions, self.facts,
        ])


class KnowledgeExtractor:
    """
    Extracts structured knowledge from text using the LLM.

    Designed to be cheap: uses haiku with small max_tokens.
    Runs as a background task after each interaction.
    """

    def __init__(self, llm: "LLMProvider") -> None:
        self._llm = llm

    # ── Public API ────────────────────────────────────────────────

    async def extract_from_interaction(
        self,
        user_message: str,
        agent_response: str,
        session_id: str,
        domain: str = "general",
    ) -> ExtractionResult:
        """Extract knowledge from a conversation turn."""
        # Skip trivial exchanges
        if len(user_message) < 20 and len(agent_response) < 50:
            return ExtractionResult(source=session_id)

        prompt = CONVERSATION_EXTRACTION_PROMPT.format(
            user_message=user_message[:1000],
            agent_response=agent_response[:1500],
        )

        return await self._run_extraction(prompt, source=session_id, domain=domain)

    async def extract_from_text(
        self,
        text: str,
        source_name: str,
        domain: str = "general",
    ) -> ExtractionResult:
        """Extract knowledge from a text chunk (file ingestion)."""
        prompt = TEXT_EXTRACTION_PROMPT.format(
            source_name=source_name,
            text=text[:3000],
        )
        return await self._run_extraction(prompt, source=source_name, domain=domain)

    # ── Internal ──────────────────────────────────────────────────

    async def _run_extraction(
        self, prompt: str, source: str, domain: str
    ) -> ExtractionResult:
        """Call LLM and parse the extraction response."""
        from psycho.llm.base import Message

        from psycho.config.constants import EXTRACTION_MAX_TOKENS
        try:
            response = await self._llm.complete(
                messages=[Message(role="user", content=prompt)],
                system=(
                    "You are a precise knowledge extraction engine. "
                    "Output ONLY valid JSON. Never add explanations or markdown. "
                    "Keep properties short — max 30 chars per value. "
                    "Limit entities to 8 max. Output must be complete valid JSON."
                ),
                max_tokens=EXTRACTION_MAX_TOKENS,
                temperature=0.1,
            )
            raw_text = response.content.strip()
        except Exception as e:
            logger.warning(f"Extraction LLM call failed: {e}")
            return ExtractionResult(source=source)

        # Parse JSON — strip any accidental markdown fences
        raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text, flags=re.MULTILINE)
        raw_text = re.sub(r"\s*```$", "", raw_text, flags=re.MULTILINE)

        try:
            raw = json.loads(raw_text)
        except json.JSONDecodeError as e:
            # Try to salvage partial JSON — truncate at the last valid array close
            salvaged = _try_repair_json(raw_text)
            if salvaged:
                logger.debug("Extraction JSON repaired from truncated response")
                raw = salvaged
            else:
                logger.warning(f"Extraction JSON parse failed: {e} | raw: {raw_text[:150]}")
                return ExtractionResult(source=source)

        return self._parse_extraction(raw, source=source, domain=domain)

    def _parse_extraction(
        self, raw: dict, source: str, domain: str
    ) -> ExtractionResult:
        """Convert raw JSON dict into typed ExtractionResult."""
        result = ExtractionResult(source=source, raw=raw)

        # ── Entities ──────────────────────────────────────────────
        label_to_node: dict[str, KnowledgeNode] = {}

        for e in raw.get("entities", []):
            label = str(e.get("label", "")).strip().lower()
            if not label or len(label) < 2:
                continue
            node_type = _NODE_TYPE_MAP.get(
                str(e.get("type", "concept")).lower(), NodeType.CONCEPT
            )
            node = KnowledgeNode.create(
                type=node_type,
                label=label,
                domain=e.get("domain", domain),
                properties=e.get("properties", {}),
                confidence=0.5,
                sources=[source],
            )
            result.entities.append(node)
            label_to_node[label] = node

        # ── Relationships ─────────────────────────────────────────
        for r in raw.get("relationships", []):
            from_label = str(r.get("from_label", "")).strip().lower()
            to_label = str(r.get("to_label", "")).strip().lower()
            rel_type_str = str(r.get("type", "relates_to")).lower()
            rel_type = _EDGE_TYPE_MAP.get(rel_type_str, EdgeType.RELATES_TO)

            # Only build edges if both nodes exist in our extraction
            from_node = label_to_node.get(from_label)
            to_node = label_to_node.get(to_label)

            if from_node and to_node:
                edge = KnowledgeEdge(
                    source_id=from_node.id,
                    target_id=to_node.id,
                    type=rel_type,
                    confidence=0.6,
                )
                result.edges.append(edge)

        # ── User Preferences ──────────────────────────────────────
        for p in raw.get("user_preferences", []):
            label = str(p.get("label", "")).strip().lower()
            if not label:
                continue
            node = KnowledgeNode.create(
                type=NodeType.PREFERENCE,
                label=label,
                domain=p.get("domain", domain),
                confidence=0.75,   # Preferences get higher initial confidence
                sources=[source],
            )
            result.preferences.append(node)

        # ── User Identity (name, projects, occupation, goals) ──────
        for item in raw.get("user_identity", []):
            key = str(item.get("key", "")).strip().lower()
            value = str(item.get("value", "")).strip()
            if not key or not value or len(value) < 1:
                continue

            if key == "name":
                # High-confidence person node for the user
                # Preserve proper capitalization for names
                display_name = " ".join(w.capitalize() for w in value.split())
                node = KnowledgeNode.create(
                    type=NodeType.PERSON,
                    label="user",
                    domain="general",
                    confidence=0.95,
                    properties={"name": display_name, "role": "user"},
                    sources=[source],
                )
                node.display_label = display_name
                result.entities.append(node)
            elif key in ("current_project", "goal", "occupation", "location"):
                # Preference node linking user to their project/goal
                pref_label = f"{key}: {value.lower()}"
                node = KnowledgeNode.create(
                    type=NodeType.PREFERENCE,
                    label=pref_label,
                    domain=domain,
                    confidence=0.8,
                    properties={key: value},
                    sources=[source],
                )
                node.display_label = f"{key.replace('_', ' ').title()}: {value}"
                result.preferences.append(node)
            elif key in ("language", "framework", "tool"):
                # Technology node
                node = KnowledgeNode.create(
                    type=NodeType.TECHNOLOGY,
                    label=value.lower(),
                    domain="coding",
                    confidence=0.75,
                    sources=[source],
                )
                result.entities.append(node)
            else:
                # Generic preference
                node = KnowledgeNode.create(
                    type=NodeType.PREFERENCE,
                    label=f"{key}: {value.lower()}",
                    domain=domain,
                    confidence=0.7,
                    sources=[source],
                )
                result.preferences.append(node)

        # ── Corrections ───────────────────────────────────────────
        for c in raw.get("corrections", []):
            wrong = str(c.get("wrong_label", "")).strip().lower()
            correct = str(c.get("correct_label", "")).strip().lower()
            if wrong and correct:
                result.corrections.append({
                    "wrong": wrong,
                    "correct": correct,
                    "explanation": c.get("explanation", ""),
                })

        # ── Key Facts ─────────────────────────────────────────────
        for fact_text in raw.get("key_facts", []):
            if not fact_text or len(str(fact_text)) < 10:
                continue
            node = KnowledgeNode.create(
                type=NodeType.FACT,
                label=str(fact_text).lower()[:200],
                domain=domain,
                confidence=0.6,
                sources=[source],
            )
            node.display_label = str(fact_text)[:200]
            result.facts.append(node)

        # ── Open Questions ────────────────────────────────────────
        for q in raw.get("open_questions", []):
            if not q:
                continue
            node = KnowledgeNode.create(
                type=NodeType.QUESTION,
                label=str(q).lower()[:200],
                domain=domain,
                confidence=0.5,
                sources=[source],
            )
            node.display_label = str(q)[:200]
            result.questions.append(node)

        total = (
            len(result.entities) + len(result.facts)
            + len(result.preferences) + len(result.corrections)
        )
        logger.debug(
            f"Extraction from '{source[:20]}': "
            f"{len(result.entities)} entities, {len(result.edges)} edges, "
            f"{len(result.facts)} facts, {len(result.corrections)} corrections"
        )
        return result
