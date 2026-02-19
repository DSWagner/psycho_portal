"""GraphEvolver — applies extracted knowledge to the graph and handles self-evolution."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from loguru import logger

from psycho.config.constants import (
    CONFIDENCE_TIME_DECAY,
    CONFIDENCE_USED_IN_RESPONSE,
    CONFIDENCE_USER_CONFIRM,
    CONFIDENCE_USER_CORRECT,
    CONFIDENCE_CONSISTENT,
    MIN_CONFIDENCE_THRESHOLD,
    CONFIDENCE_INFERRED,
)
from psycho.knowledge.schema import (
    EdgeType,
    KnowledgeEdge,
    KnowledgeNode,
    NodeType,
)

if TYPE_CHECKING:
    from psycho.knowledge.extractor import ExtractionResult
    from psycho.knowledge.graph import KnowledgeGraph


class GraphEvolver:
    """
    Applies extraction results to the knowledge graph and maintains its health.

    Responsibilities:
        - Integrate new entities, edges, and facts from extractions
        - Apply confidence updates (confirmations, corrections, decay)
        - Deduplicate similar nodes
        - Prune deprecated / very low confidence nodes
        - Derive inferred edges during reflection
    """

    def __init__(self, graph: "KnowledgeGraph") -> None:
        self._graph = graph

    # ── Primary: integrate extraction results ─────────────────────

    async def integrate(self, result: "ExtractionResult") -> dict:
        """
        Integrate a full ExtractionResult into the knowledge graph.

        Returns a summary dict of what was added/updated.
        """
        g = self._graph
        stats = {
            "nodes_added": 0,
            "nodes_updated": 0,
            "edges_added": 0,
            "facts_added": 0,
            "preferences_added": 0,
            "corrections_applied": 0,
        }

        if result.is_empty():
            return stats

        # Build a label→id map for cross-linking edges correctly
        label_to_id: dict[str, str] = {}

        # ── 1. Add/update entities ─────────────────────────────────
        for node in result.entities:
            # Special handling: PERSON "user" node — merge name into existing
            if node.type.value == "person" and node.label == "user":
                existing = g.find_node_by_label("user", node.type)
                if existing:
                    # Merge name and properties into existing user node
                    existing.update_confidence(0.05)
                    for k, v in node.properties.items():
                        existing.properties[k] = v
                    if node.display_label and node.display_label != "user":
                        existing.display_label = node.display_label
                    label_to_id[node.label] = existing.id
                    stats["nodes_updated"] += 1
                else:
                    canonical_id = g.upsert_node(node)
                    label_to_id[node.label] = canonical_id
                    stats["nodes_added"] += 1
                continue

            existing = g.find_node_by_label(node.label, node.type)
            if existing:
                existing.update_confidence(CONFIDENCE_CONSISTENT)
                for src in node.sources:
                    if src not in existing.sources:
                        existing.sources.append(src)
                # Merge any new properties
                for k, v in node.properties.items():
                    if k not in existing.properties:
                        existing.properties[k] = v
                label_to_id[node.label] = existing.id
                stats["nodes_updated"] += 1
            else:
                canonical_id = g.upsert_node(node)
                label_to_id[node.label] = canonical_id
                stats["nodes_added"] += 1

        # ── 2. Add edges (using resolved IDs) ─────────────────────
        for edge in result.edges:
            # The edge IDs from the extractor are provisional (based on new nodes).
            # Look up canonical IDs by label where possible.
            src_node = g.get_node(edge.source_id)
            tgt_node = g.get_node(edge.target_id)
            if src_node and tgt_node:
                g.upsert_edge(edge)
                stats["edges_added"] += 1

        # ── 3. Add facts ───────────────────────────────────────────
        for fact_node in result.facts:
            existing = g.find_node_by_label(fact_node.label)
            if not existing:
                g.upsert_node(fact_node)
                stats["facts_added"] += 1
            else:
                existing.update_confidence(CONFIDENCE_CONSISTENT)

        # ── 4. Add preferences ─────────────────────────────────────
        for pref_node in result.preferences:
            existing = g.find_node_by_label(pref_node.label, NodeType.PREFERENCE)
            if not existing:
                g.upsert_node(pref_node)
                stats["preferences_added"] += 1
            else:
                # Reinforce existing preference
                existing.update_confidence(0.05)

        # ── 5. Add questions ───────────────────────────────────────
        for q_node in result.questions:
            if not g.find_node_by_label(q_node.label, NodeType.QUESTION):
                g.upsert_node(q_node)

        # ── 6. Apply corrections ───────────────────────────────────
        for correction in result.corrections:
            wrong_node = g.find_node_by_label(correction["wrong"])
            correct_node = g.find_node_by_label(correction["correct"])

            if wrong_node:
                # Drop confidence on the wrong belief
                wrong_node.update_confidence(CONFIDENCE_USER_CORRECT)
                wrong_node.properties["correction_note"] = correction.get(
                    "explanation", "User corrected this"
                )
                logger.info(
                    f"Correction applied: '{wrong_node.display_label}' "
                    f"confidence → {wrong_node.confidence:.2f}"
                )
                stats["corrections_applied"] += 1

            if correct_node and wrong_node:
                # Add CORRECTS edge
                corrects_edge = KnowledgeEdge(
                    source_id=correct_node.id,
                    target_id=wrong_node.id,
                    type=EdgeType.CORRECTS,
                    confidence=0.9,
                    properties={"explanation": correction.get("explanation", "")},
                )
                g.upsert_edge(corrects_edge)
            elif correct_node:
                # Boost the correct one
                correct_node.update_confidence(CONFIDENCE_USER_CONFIRM)

        # Recompute PageRank after significant changes
        if stats["nodes_added"] > 3:
            g.compute_pagerank()

        if any(v > 0 for v in stats.values()):
            g._dirty = True
            logger.debug(f"Graph evolution: {stats}")

        return stats

    # ── Confidence management ──────────────────────────────────────

    def confirm_nodes(self, node_ids: list[str]) -> None:
        """Boost confidence for nodes explicitly confirmed by the user."""
        for nid in node_ids:
            node = self._graph.get_node(nid)
            if node:
                node.update_confidence(CONFIDENCE_USER_CONFIRM)

    def correct_node(self, node_id: str, correction_note: str = "") -> None:
        """Drop confidence for a node the user corrected."""
        node = self._graph.get_node(node_id)
        if node:
            node.update_confidence(CONFIDENCE_USER_CORRECT)
            if correction_note:
                node.properties["correction_note"] = correction_note
            logger.info(f"Node corrected: '{node.display_label}' conf={node.confidence:.2f}")

    def boost_used_nodes(self, node_ids: list[str]) -> None:
        """Slightly boost nodes that were actually used in a response."""
        for nid in node_ids:
            node = self._graph.get_node(nid)
            if node:
                node.update_confidence(CONFIDENCE_USED_IN_RESPONSE)

    # ── Graph maintenance ──────────────────────────────────────────

    def apply_time_decay(self) -> int:
        """
        Apply time-based confidence decay to all nodes.
        Nodes not accessed for a long time slowly lose confidence.
        Returns number of nodes decayed.
        """
        now = time.time()
        decayed = 0
        for _, attrs in self._graph._g.nodes(data=True):
            node: KnowledgeNode = attrs.get("data")
            if not node or node.deprecated:
                continue
            days_idle = (now - node.last_accessed) / 86400
            if days_idle > 1:
                decay = CONFIDENCE_TIME_DECAY * days_idle
                node.update_confidence(-decay)
                decayed += 1

        if decayed:
            self._graph._dirty = True
            logger.debug(f"Time decay applied to {decayed} nodes")
        return decayed

    def prune_low_confidence(self) -> int:
        """
        Mark nodes below MIN_CONFIDENCE_THRESHOLD as deprecated.
        Returns number of nodes deprecated.
        """
        pruned = 0
        for _, attrs in self._graph._g.nodes(data=True):
            node: KnowledgeNode = attrs.get("data")
            if (
                node
                and not node.deprecated
                and node.confidence < MIN_CONFIDENCE_THRESHOLD
            ):
                self._graph.deprecate_node(
                    node.id, reason=f"confidence below threshold ({node.confidence:.3f})"
                )
                pruned += 1

        if pruned:
            logger.info(f"Pruned {pruned} low-confidence nodes")
        return pruned

    def find_and_merge_duplicates(self, similarity_threshold: float = 0.92) -> int:
        """
        Find nodes with very similar labels and merge them.
        Uses simple normalized string similarity for now.
        Returns number of merges performed.
        """
        from difflib import SequenceMatcher

        nodes = [
            attrs["data"]
            for _, attrs in self._graph._g.nodes(data=True)
            if attrs.get("data") and not attrs["data"].deprecated
        ]

        merged = 0
        merged_ids: set[str] = set()

        for i, node_a in enumerate(nodes):
            if node_a.id in merged_ids:
                continue
            for node_b in nodes[i + 1:]:
                if node_b.id in merged_ids or node_a.type != node_b.type:
                    continue
                sim = SequenceMatcher(None, node_a.label, node_b.label).ratio()
                if sim >= similarity_threshold:
                    # Keep the higher-confidence node
                    keep_id = (
                        node_a.id if node_a.confidence >= node_b.confidence else node_b.id
                    )
                    drop_id = node_b.id if keep_id == node_a.id else node_a.id
                    self._graph.merge_nodes(keep_id, drop_id)
                    merged_ids.add(drop_id)
                    merged += 1
                    logger.info(f"Merged duplicate: '{node_a.label}' ~ '{node_b.label}'")

        return merged

    def add_inferred_edges(self, node_ids: list[str]) -> int:
        """
        Attempt to infer new relationships from existing graph patterns.
        E.g., if A relates_to B and B relates_to C → infer A relates_to C.
        Returns number of inferred edges added.
        """
        inferred = 0
        g = self._graph

        for nid in node_ids:
            if not g._g.has_node(nid):
                continue
            # Transitive relates_to
            for _, neighbor1, _ in g._g.edges(nid, data=True):
                for _, neighbor2, e2_attrs in g._g.edges(neighbor1, data=True):
                    e2: KnowledgeEdge = e2_attrs.get("data")
                    if (
                        e2
                        and e2.type == EdgeType.RELATES_TO
                        and neighbor2 != nid
                        and not g._g.has_edge(nid, neighbor2)
                    ):
                        inferred_edge = KnowledgeEdge(
                            source_id=nid,
                            target_id=neighbor2,
                            type=EdgeType.RELATES_TO,
                            confidence=CONFIDENCE_INFERRED,
                            properties={"inferred": True},
                        )
                        g.upsert_edge(inferred_edge)
                        inferred += 1

        if inferred:
            logger.debug(f"Inferred {inferred} new edges")
        return inferred

    def run_full_maintenance(self) -> dict:
        """Run all maintenance operations. Called during post-session reflection."""
        pruned = self.prune_low_confidence()
        merged = self.find_and_merge_duplicates()
        decayed = self.apply_time_decay()
        self._graph.compute_pagerank()
        self._graph.save()

        result = {"pruned": pruned, "merged": merged, "decayed": decayed}
        self._graph._store.record_evolution_event(
            {"type": "maintenance", "result": result}
        )
        return result
