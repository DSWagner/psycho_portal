"""
ReflectionEngine — post-session synthesis that makes the agent genuinely smarter.

Runs at session end (triggered by `exit` or inactivity).
Coordinates: interaction review → LLM synthesis → graph evolution →
             insight generation → mistake recording → journal writing.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from psycho.knowledge.evolution import GraphEvolver
    from psycho.knowledge.graph import KnowledgeGraph
    from psycho.knowledge.reasoner import GraphReasoner
    from psycho.learning.insight_generator import InsightGenerator
    from psycho.learning.mistake_tracker import MistakeTracker
    from psycho.learning.session_journal import SessionJournal
    from psycho.llm.base import LLMProvider
    from psycho.memory.manager import MemoryManager


# ── Reflection prompt ──────────────────────────────────────────────────────────

REFLECTION_PROMPT = """\
You are reviewing your performance in a conversation session to learn and improve.

Session interactions (most recent first):
{interactions}

Your current knowledge about the topics discussed:
{graph_context}

Provide a thorough learning review as JSON. Be honest and specific.

{{
  "session_summary": "1-2 paragraph summary of what was discussed and accomplished",
  "quality_score": 0.0_to_1.0,
  "key_learnings": [
    {{"fact": "specific new fact learned", "domain": "domain", "confidence": 0.5}}
  ],
  "corrections_detected": [
    {{
      "wrong": "what the agent incorrectly stated",
      "correct": "what the correct information is",
      "context": "brief context of when this happened",
      "user_input_that_triggered": "the user message that exposed the error"
    }}
  ],
  "patterns_observed": [
    {{"pattern": "observed user behavior/preference", "implication": "what this means for future interactions"}}
  ],
  "knowledge_gaps": [
    {{"topic": "topic name", "why_insufficient": "why knowledge was lacking"}}
  ],
  "insights": [
    {{"insight": "deeper inference combining multiple facts", "basis": "which facts", "confidence": 0.6}}
  ],
  "nodes_to_boost": ["label of node to increase confidence"],
  "nodes_to_drop": ["label of node to decrease confidence (agent was wrong about this)"]
}}

Rules:
- Be specific: name exact topics, not vague generalities
- corrections_detected: only include EXPLICIT corrections by the user
- quality_score: 1.0 = perfect responses; 0.0 = consistently wrong
- Return ONLY the JSON object, no other text"""


@dataclass
class ReflectionResult:
    """Output of a reflection run."""

    session_id: str
    session_summary: str = ""
    quality_score: float = 0.0
    key_learnings: list[dict] = field(default_factory=list)
    corrections_detected: list[dict] = field(default_factory=list)
    patterns_observed: list[dict] = field(default_factory=list)
    knowledge_gaps: list[dict] = field(default_factory=list)
    insights: list[dict] = field(default_factory=list)
    nodes_to_boost: list[str] = field(default_factory=list)
    nodes_to_drop: list[str] = field(default_factory=list)
    graph_changes: dict = field(default_factory=dict)
    elapsed_seconds: float = 0.0
    error: str = ""

    def is_meaningful(self) -> bool:
        return bool(
            self.key_learnings
            or self.corrections_detected
            or self.insights
            or self.patterns_observed
        )

    def display_summary(self) -> str:
        """Short summary for CLI display."""
        parts = [
            f"Quality: {self.quality_score:.2f}",
            f"Learnings: {len(self.key_learnings)}",
            f"Corrections: {len(self.corrections_detected)}",
            f"Insights: {len(self.insights)}",
            f"Graph nodes added: {self.graph_changes.get('nodes_added', 0)}",
        ]
        return " · ".join(parts)


class ReflectionEngine:
    """
    Coordinates the full post-session reflection pipeline.

    Pipeline:
        1. Retrieve session interactions
        2. Get relevant graph context
        3. Run LLM reflection synthesis
        4. Apply graph confidence updates (boost/drop)
        5. Record detected mistakes in MistakeTracker
        6. Generate deeper insights via InsightGenerator
        7. Run graph maintenance (prune/dedup/decay)
        8. Write session journal
        9. Save graph
    """

    def __init__(
        self,
        llm: "LLMProvider",
        memory: "MemoryManager",
        graph: "KnowledgeGraph",
        evolver: "GraphEvolver",
        mistake_tracker: "MistakeTracker",
        insight_generator: "InsightGenerator",
        journal: "SessionJournal",
        reasoner: "GraphReasoner | None" = None,
    ) -> None:
        self._llm = llm
        self._memory = memory
        self._graph = graph
        self._evolver = evolver
        self._mistake_tracker = mistake_tracker
        self._insight_generator = insight_generator
        self._journal = journal
        self._reasoner = reasoner

    async def run(
        self, session_id: str, session_started_at: float
    ) -> ReflectionResult:
        """
        Run the full post-session reflection pipeline.

        This is called when the user exits the chat.
        Returns a ReflectionResult with all learning outcomes.
        """
        start = time.time()
        result = ReflectionResult(session_id=session_id)
        logger.info(f"Starting post-session reflection for session {session_id}")

        try:
            # 1. Get session interactions
            interactions = await self._memory.long_term.get_recent_interactions(
                limit=25
            )
            if not interactions:
                logger.info("No interactions to reflect on")
                return result

            # 2. Get graph context for topics discussed
            queries = [i["user_message"][:100] for i in interactions[:3]]
            combined_query = " ".join(queries)
            graph_context = (
                self._reasoner.get_context_for_prompt(combined_query, max_nodes=8)
                if self._reasoner else ""
            )

            # 3. LLM synthesis
            reflection_data = await self._synthesize(interactions, graph_context)
            if not reflection_data:
                logger.warning("LLM synthesis returned no data")
                return result

            # Populate result
            result.session_summary = reflection_data.get("session_summary", "")
            result.quality_score = float(reflection_data.get("quality_score", 0.5))
            result.key_learnings = reflection_data.get("key_learnings", [])
            result.corrections_detected = reflection_data.get("corrections_detected", [])
            result.patterns_observed = reflection_data.get("patterns_observed", [])
            result.knowledge_gaps = reflection_data.get("knowledge_gaps", [])
            result.insights = reflection_data.get("insights", [])
            result.nodes_to_boost = reflection_data.get("nodes_to_boost", [])
            result.nodes_to_drop = reflection_data.get("nodes_to_drop", [])

            # 4. Apply graph confidence updates
            graph_changes = await self._apply_graph_updates(result)
            result.graph_changes = graph_changes

            # 5. Record detected mistakes
            await self._record_mistakes(result, interactions, session_id)

            # 6. Generate additional insights — FIX: store results in result
            if result.session_summary:
                new_insights = await self._insight_generator.generate_insights(
                    session_summary=result.session_summary,
                    max_nodes=25,
                )
                for node in new_insights:
                    result.insights.append({
                        "insight": node.display_label,
                        "basis": node.properties.get("basis", ""),
                        "confidence": node.confidence,
                        "actionable": node.properties.get("actionable", ""),
                    })

            # 7. Graph maintenance (prune deprecated, merge near-duplicates)
            maintenance = self._evolver.run_full_maintenance()
            result.graph_changes.update(maintenance)

            # 8. Save session journal
            message_count = len(interactions)
            self._journal.write(
                session_id=session_id,
                started_at=session_started_at,
                reflection_data=reflection_data,
                graph_changes=result.graph_changes,
                message_count=message_count,
            )

            # 9. Graph already saved by run_full_maintenance

        except Exception as e:
            result.error = str(e)
            logger.error(f"Reflection failed: {e}", exc_info=True)
            # Save graph even if reflection fails
            try:
                self._graph.save()
            except Exception:
                pass

        result.elapsed_seconds = time.time() - start
        logger.info(
            f"Reflection complete in {result.elapsed_seconds:.1f}s: "
            f"{result.display_summary()}"
        )
        return result

    # ── Private helpers ───────────────────────────────────────────

    async def _synthesize(
        self, interactions: list[dict], graph_context: str
    ) -> dict:
        """Call LLM for reflection synthesis. Returns parsed dict or {}."""
        from psycho.llm.base import Message

        # Format recent interactions (most recent first, limited to 20)
        interaction_lines = []
        for i in interactions[:20]:
            interaction_lines.append(
                f"User: {i['user_message'][:200]}\n"
                f"Agent: {i['agent_response'][:300]}"
            )
        interactions_text = "\n\n---\n\n".join(interaction_lines)

        graph_text = graph_context or "(knowledge graph is empty)"

        prompt = REFLECTION_PROMPT.format(
            interactions=interactions_text[:4000],
            graph_context=graph_text[:1500],
        )

        try:
            response = await self._llm.complete(
                messages=[Message(role="user", content=prompt)],
                system=(
                    "You are a precise self-evaluation engine for an AI assistant. "
                    "Output ONLY valid JSON. Be honest about mistakes and gaps."
                ),
                max_tokens=2048,
                temperature=0.2,
            )
            raw = response.content.strip()
        except Exception as e:
            logger.error(f"Reflection LLM call failed: {e}")
            return {}

        # Strip markdown fences
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
        raw = re.sub(r"\s*```$", "", raw, flags=re.MULTILINE)

        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            logger.warning(f"Reflection JSON parse failed: {e} | raw: {raw[:300]}")
            return {}

    async def _apply_graph_updates(self, result: ReflectionResult) -> dict:
        """Apply confidence updates to the graph based on reflection output."""
        changes = {
            "nodes_boosted": 0,
            "nodes_dropped": 0,
            "corrections_applied": 0,
            "facts_added": 0,
        }

        # Boost nodes the agent was correct about
        for label in result.nodes_to_boost:
            node = self._graph.find_node_by_label(label.lower())
            if node:
                self._evolver.confirm_nodes([node.id])
                changes["nodes_boosted"] += 1

        # Drop nodes the agent was wrong about
        for label in result.nodes_to_drop:
            node = self._graph.find_node_by_label(label.lower())
            if node:
                self._evolver.correct_node(node.id, "Detected as incorrect in reflection")
                changes["nodes_dropped"] += 1

        # Add key learnings as fact nodes
        from psycho.knowledge.extractor import ExtractionResult
        from psycho.knowledge.schema import KnowledgeNode, NodeType

        for learning in result.key_learnings:
            if isinstance(learning, dict):
                fact_text = learning.get("fact", "")
                domain = learning.get("domain", "general")
                confidence = float(learning.get("confidence", 0.5))
            else:
                fact_text = str(learning)
                domain = "general"
                confidence = 0.5

            if fact_text and len(fact_text) > 10:
                node = KnowledgeNode.create(
                    type=NodeType.FACT,
                    label=fact_text.lower()[:200],
                    domain=domain,
                    confidence=confidence,
                    sources=[f"reflection_{result.session_id}"],
                )
                node.display_label = fact_text[:200]
                self._graph.upsert_node(node)
                changes["facts_added"] += 1

        # Handle explicit corrections
        for correction in result.corrections_detected:
            if isinstance(correction, dict):
                wrong = correction.get("wrong", "").lower()
                correct = correction.get("correct", "")
                if wrong:
                    node = self._graph.find_node_by_label(wrong)
                    if node:
                        self._evolver.correct_node(
                            node.id,
                            f"Corrected in reflection: {correct[:100]}"
                        )
                        changes["corrections_applied"] += 1

        return changes

    async def _record_mistakes(
        self,
        result: ReflectionResult,
        interactions: list[dict],
        session_id: str,
    ) -> None:
        """Record detected corrections as mistakes for future avoidance."""
        for correction in result.corrections_detected:
            if not isinstance(correction, dict):
                continue

            wrong = correction.get("wrong", "")
            correct = correction.get("correct", "")
            if not wrong or not correct:
                continue

            # Find the interaction where this correction happened
            trigger = correction.get("user_input_that_triggered", "")
            context = correction.get("context", "")

            # Find matching interaction
            user_input = trigger or wrong
            agent_resp = wrong  # the wrong statement

            await self._mistake_tracker.record_mistake(
                session_id=session_id,
                user_input=user_input[:400],
                agent_response=agent_resp[:400],
                correction=correct[:300],
                domain=interactions[0].get("domain", "general") if interactions else "general",
                error_pattern=context[:200],
            )
