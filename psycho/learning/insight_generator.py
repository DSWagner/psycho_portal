"""
Insight generator — derives deeper understanding by combining
multiple knowledge graph nodes and session patterns.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from loguru import logger

from psycho.knowledge.schema import KnowledgeNode, NodeType

if TYPE_CHECKING:
    from psycho.knowledge.graph import KnowledgeGraph
    from psycho.llm.base import LLMProvider

INSIGHT_PROMPT = """\
You are analyzing an agent's knowledge graph to derive insights.

The agent knows these things (with confidence scores):
{knowledge_summary}

Recent session summary:
{session_summary}

Based on patterns and connections in this knowledge, generate insights.
An insight is something that can be INFERRED from combining multiple facts —
not just restating what's already known.

Examples of good insights:
- "User consistently uses Python for async work + prefers minimal deps → likely values pragmatism over perfection"
- "User asked about both asyncio and threading → probably debugging a concurrency issue"
- "Multiple health questions about sleep + performance → optimizing for productivity"

Output JSON only:
{
  "insights": [
    {
      "insight": "derived understanding",
      "basis": "which facts it's derived from",
      "confidence": 0.0-1.0,
      "domain": "domain",
      "actionable": "how the agent should behave differently based on this"
    }
  ]
}

Rules:
- Generate 2-5 insights maximum
- Only include insights with confidence >= 0.4
- Insights must be SPECIFIC to this user, not generic
- Return empty array if no meaningful insights can be derived
- Output ONLY the JSON object"""


class InsightGenerator:
    """
    Mines the knowledge graph and session history to derive insights
    about the user's goals, patterns, and preferences.

    Insights are added to the graph as CONCEPT nodes with type=TOPIC
    and linked to their source nodes via INFERRED_FROM edges.
    """

    def __init__(self, llm: "LLMProvider", graph: "KnowledgeGraph") -> None:
        self._llm = llm
        self._graph = graph

    async def generate_insights(
        self, session_summary: str, max_nodes: int = 30
    ) -> list[KnowledgeNode]:
        """
        Derive insights from the current knowledge graph state.

        Returns list of new insight nodes added to the graph.
        """
        # Gather top nodes as context
        top_nodes = self._graph.get_top_nodes(max_nodes)
        if len(top_nodes) < 5:
            logger.debug("Too few graph nodes for meaningful insight generation")
            return []

        # Build knowledge summary
        knowledge_summary = self._build_knowledge_summary(top_nodes)

        # LLM call
        from psycho.llm.base import Message

        prompt = INSIGHT_PROMPT.format(
            knowledge_summary=knowledge_summary,
            session_summary=session_summary or "No summary available.",
        )

        try:
            response = await self._llm.complete(
                messages=[Message(role="user", content=prompt)],
                system=(
                    "You are a precise knowledge analysis engine. "
                    "Output ONLY valid JSON. Never add explanations."
                ),
                max_tokens=1024,
                temperature=0.3,
            )
            raw = response.content.strip()
        except Exception as e:
            logger.warning(f"Insight generation LLM call failed: {e}")
            return []

        # Parse JSON
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
        raw = re.sub(r"\s*```$", "", raw, flags=re.MULTILINE)

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            logger.warning(f"Insight JSON parse failed: {e}")
            return []

        # Create insight nodes
        added = []
        for item in data.get("insights", []):
            if not item.get("insight") or item.get("confidence", 0) < 0.4:
                continue

            node = KnowledgeNode.create(
                type=NodeType.CONCEPT,
                label=item["insight"].lower()[:200],
                domain=item.get("domain", "general"),
                confidence=min(float(item.get("confidence", 0.5)), 0.75),
                properties={
                    "basis": item.get("basis", ""),
                    "actionable": item.get("actionable", ""),
                    "generated_by": "insight_generator",
                },
                sources=["reflection"],
            )
            node.display_label = item["insight"][:200]

            canonical_id = self._graph.upsert_node(node)
            if canonical_id:
                added.append(node)
                logger.debug(f"Insight added: '{item['insight'][:60]}'")

        logger.info(f"Generated {len(added)} insights from {len(top_nodes)} graph nodes")
        return added

    @staticmethod
    def _build_knowledge_summary(nodes: list[KnowledgeNode]) -> str:
        """Build a compact summary of graph nodes for the prompt."""
        lines = []
        # Group by type
        by_type: dict[str, list[KnowledgeNode]] = {}
        for n in nodes:
            by_type.setdefault(n.type.value, []).append(n)

        for node_type, type_nodes in by_type.items():
            lines.append(f"\n[{node_type.upper()}]")
            for n in type_nodes[:8]:  # max 8 per type
                props = ""
                if n.properties:
                    first = list(n.properties.items())[0]
                    props = f" ({first[0]}: {str(first[1])[:30]})"
                lines.append(
                    f"  - {n.display_label}{props} "
                    f"[conf={n.confidence:.2f}, domain={n.domain}]"
                )

        return "\n".join(lines)
