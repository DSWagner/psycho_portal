"""GraphReasoner — assembles knowledge graph context for the agent's system prompt."""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from psycho.knowledge.schema import (
    KnowledgeEdge,
    KnowledgeNode,
    confidence_bar,
    confidence_label,
)

if TYPE_CHECKING:
    from psycho.knowledge.graph import KnowledgeGraph

# Max characters injected into the system prompt
MAX_CONTEXT_CHARS = 2400
MAX_NODES_IN_CONTEXT = 12


class GraphReasoner:
    """
    Retrieves relevant knowledge from the graph and formats it for prompt injection.

    The formatted context looks like:
    ─── KNOWLEDGE GRAPH (8 nodes) ───
    • [TECHNOLOGY] python — programming language (HIGH ████████░░ 0.91)
      └─ has_property: dynamic typing | part_of: scripting languages
    • [PREFERENCE] prefers dark mode in editors (HIGH ███████░░░ 0.87)
    • [FACT] asyncio.lock() prevents race conditions (MEDIUM █████░░░░░ 0.65)
    ─────────────────────────────────
    """

    def __init__(self, graph: "KnowledgeGraph") -> None:
        self._graph = graph

    def get_context_for_prompt(self, query: str, max_nodes: int = MAX_NODES_IN_CONTEXT) -> str:
        """
        Build a formatted context block for injection into the system prompt.

        Returns empty string if graph has no relevant nodes.
        """
        if self._graph.get_stats()["active_nodes"] == 0:
            return ""

        context_items = self._graph.get_context_for_query(query, top_k=max_nodes)
        if not context_items:
            return ""

        lines = [f"\n─── KNOWLEDGE GRAPH ({len(context_items)} relevant nodes) ───"]
        total_chars = len(lines[0])

        for node, edges in context_items:
            # Build node line
            node_line = self._format_node(node)
            edge_line = self._format_edges(edges)

            chunk = node_line
            if edge_line:
                chunk += "\n" + edge_line

            # Budget check
            if total_chars + len(chunk) > MAX_CONTEXT_CHARS:
                lines.append(f"  ... and {len(context_items) - lines.count('•')} more")
                break

            lines.append(chunk)
            total_chars += len(chunk)

        lines.append("─" * 35)
        lines.append(
            "Use this knowledge naturally. Hedge on MEDIUM/LOW confidence items."
        )

        result = "\n".join(lines)
        logger.debug(f"Graph context: {len(context_items)} nodes, {len(result)} chars")
        return result

    def get_relevant_nodes(self, query: str, top_k: int = 10) -> list[KnowledgeNode]:
        """Return just the nodes without formatting (for programmatic use)."""
        return [node for node, _ in self._graph.get_context_for_query(query, top_k)]

    def format_full_graph_summary(self, max_nodes: int = 30) -> str:
        """Format a complete graph summary for `/graph` command display."""
        stats = self._graph.get_stats()
        top_nodes = self._graph.get_top_nodes(max_nodes)

        lines = [
            f"Knowledge Graph Summary",
            f"  Active nodes:  {stats['active_nodes']}",
            f"  Total edges:   {stats['total_edges']}",
            f"  Avg confidence: {stats['average_confidence']:.2f}",
            f"  Contradictions: {stats['contradictions']}",
            "",
            "Top Nodes by Importance:",
        ]

        for node in top_nodes:
            domain = f" [{node.domain}]" if node.domain != "general" else ""
            lines.append(
                f"  {confidence_bar(node.confidence, 8)} "
                f"[{node.type.value}] {node.display_label}{domain}"
            )

        return "\n".join(lines)

    def format_node_detail(self, node: KnowledgeNode) -> str:
        """Format a single node with all its context for inspection."""
        lines = [
            f"[{node.type.value.upper()}] {node.display_label}",
            f"  Confidence: {node.confidence_display()}",
            f"  Domain:     {node.domain}",
            f"  Created:    {_format_time(node.created_at)}",
            f"  Accessed:   {node.access_count}x",
        ]
        if node.properties:
            lines.append("  Properties:")
            for k, v in node.properties.items():
                lines.append(f"    {k}: {v}")
        if node.sources:
            lines.append(f"  Sources: {', '.join(s[:20] for s in node.sources[:3])}")
        if node.deprecated:
            lines.append(f"  DEPRECATED: {node.deprecation_reason}")

        # Edges
        out_edges = self._graph.get_edges_from(node.id)
        in_edges = self._graph.get_edges_to(node.id)
        if out_edges:
            lines.append("  Outgoing edges:")
            for target, edge in out_edges[:5]:
                lines.append(f"    → [{edge.type.value}] {target.display_label} ({edge.confidence:.2f})")
        if in_edges:
            lines.append("  Incoming edges:")
            for source, edge in in_edges[:5]:
                lines.append(f"    ← [{edge.type.value}] {source.display_label} ({edge.confidence:.2f})")

        return "\n".join(lines)

    # ── Private formatting ────────────────────────────────────────

    @staticmethod
    def _format_node(node: KnowledgeNode) -> str:
        conf_label = confidence_label(node.confidence)
        conf = f"{conf_label} {confidence_bar(node.confidence, 8)} {node.confidence:.2f}"
        domain = f" [{node.domain}]" if node.domain != "general" else ""
        desc = node.properties.get("description", "")
        desc_str = f" — {desc}" if desc else ""
        return f"• [{node.type.value.upper()}] {node.display_label}{desc_str} ({conf}){domain}"

    @staticmethod
    def _format_edges(edges: list[tuple[KnowledgeNode, KnowledgeEdge]]) -> str:
        if not edges:
            return ""
        parts = []
        for target, edge in edges[:4]:  # max 4 edges per node
            parts.append(f"{edge.type.value}: {target.display_label}")
        return "  └─ " + " | ".join(parts)


def _format_time(timestamp: float) -> str:
    import datetime
    return datetime.datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d")
