"""KnowledgeGraph — NetworkX-backed graph with ChromaDB semantic indexing."""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

import networkx as nx
from loguru import logger

from psycho.knowledge.schema import (
    EdgeType,
    KnowledgeEdge,
    KnowledgeNode,
    NodeType,
    confidence_label,
)
from psycho.storage.graph_store import GraphStore
from psycho.storage.vector_store import VectorStore

if TYPE_CHECKING:
    pass

# Weights for node ranking: confidence × pagerank × recency
CONFIDENCE_WEIGHT = 0.5
PAGERANK_WEIGHT = 0.3
RECENCY_WEIGHT = 0.2
RECENCY_HALF_LIFE_DAYS = 30.0   # Nodes not accessed for 30d lose half their recency score


class KnowledgeGraph:
    """
    The agent's persistent, self-evolving knowledge base.

    Architecture:
        - NetworkX DiGraph for in-memory traversal and analysis
        - ChromaDB collection for semantic node lookup
        - JSON persistence via GraphStore

    Key operations:
        upsert_node()           — add or merge a node
        upsert_edge()           — add or reinforce an edge
        get_context_for_query() — hybrid semantic+structural retrieval
        find_node_by_label()    — exact/fuzzy node lookup
        merge_nodes()           — consolidate duplicate nodes
        compute_pagerank()      — importance scoring
    """

    GRAPH_NODES_COLLECTION = "graph_nodes"

    def __init__(self, store: GraphStore, vector_store: VectorStore) -> None:
        self._g: nx.DiGraph = nx.DiGraph()
        self._store = store
        self._vs = vector_store
        self._pagerank: dict[str, float] = {}
        self._dirty = False

    # ── Lifecycle ─────────────────────────────────────────────────

    def load(self) -> None:
        """Load graph from disk into memory."""
        data = self._store.load()
        if not data:
            logger.info("Knowledge graph: no saved graph found, starting fresh")
            return

        loaded_nodes = 0
        for nid, nd in data.get("nodes", {}).items():
            try:
                node = KnowledgeNode.from_dict(nd)
                self._g.add_node(node.id, data=node)
                loaded_nodes += 1
            except Exception as e:
                logger.warning(f"Skipping malformed node {nid}: {e}")

        loaded_edges = 0
        for ed in data.get("edges", []):
            try:
                edge = KnowledgeEdge.from_dict(ed)
                if self._g.has_node(edge.source_id) and self._g.has_node(edge.target_id):
                    self._g.add_edge(edge.source_id, edge.target_id, data=edge)
                    loaded_edges += 1
            except Exception as e:
                logger.warning(f"Skipping malformed edge: {e}")

        logger.info(f"Knowledge graph loaded: {loaded_nodes} nodes, {loaded_edges} edges")
        if loaded_nodes > 0:
            self._recompute_pagerank()

    def save(self) -> None:
        """Persist graph to disk."""
        if not self._dirty and self._g.number_of_nodes() > 0:
            # Still save if graph has nodes (in case of first run)
            pass

        nodes = {}
        for nid, attrs in self._g.nodes(data=True):
            node: KnowledgeNode = attrs.get("data")
            if node:
                nodes[nid] = node.to_dict()

        edges = []
        for src, tgt, attrs in self._g.edges(data=True):
            edge: KnowledgeEdge = attrs.get("data")
            if edge:
                edges.append(edge.to_dict())

        node_type_counts = {}
        for node in nodes.values():
            t = node.get("type", "unknown")
            node_type_counts[t] = node_type_counts.get(t, 0) + 1

        self._store.save(
            nodes,
            edges,
            extra_meta={
                "node_type_counts": node_type_counts,
                "average_confidence": (
                    sum(n.get("confidence", 0) for n in nodes.values()) / len(nodes)
                    if nodes
                    else 0.0
                ),
            },
        )
        self._dirty = False

    # ── Node Operations ───────────────────────────────────────────

    def upsert_node(self, node: KnowledgeNode) -> str:
        """
        Add a new node or merge with an existing one by label+type.

        Returns the canonical node ID.
        """
        existing = self.find_node_by_label(node.label, node.type)

        if existing:
            # Reinforce existing node
            existing.update_confidence(0.03)
            existing.last_updated = time.time()
            # Merge sources
            for src in node.sources:
                if src not in existing.sources:
                    existing.sources.append(src)
            # Merge properties
            for k, v in node.properties.items():
                if k not in existing.properties:
                    existing.properties[k] = v
            self._dirty = True
            logger.debug(f"Node reinforced: '{existing.display_label}' (conf={existing.confidence:.2f})")
            return existing.id
        else:
            # New node
            self._g.add_node(node.id, data=node)
            self._dirty = True

            # Index in ChromaDB for semantic lookup
            self._index_node(node)

            logger.debug(f"Node added: [{node.type.value}] '{node.display_label}' (id={node.id[:8]})")
            return node.id

    def get_node(self, node_id: str) -> KnowledgeNode | None:
        attrs = self._g.nodes.get(node_id)
        if attrs:
            node: KnowledgeNode = attrs.get("data")
            if node:
                node.touch()
                return node
        return None

    def find_node_by_label(
        self, label: str, node_type: NodeType | None = None
    ) -> KnowledgeNode | None:
        """Find node by normalized label (case-insensitive)."""
        normalized = label.lower().strip()
        for _, attrs in self._g.nodes(data=True):
            node: KnowledgeNode = attrs.get("data")
            if node and node.label == normalized:
                if node_type is None or node.type == node_type:
                    return node
        return None

    def find_nodes_by_type(self, node_type: NodeType) -> list[KnowledgeNode]:
        return [
            attrs["data"]
            for _, attrs in self._g.nodes(data=True)
            if attrs.get("data") and attrs["data"].type == node_type
        ]

    def search_nodes_by_label_prefix(self, prefix: str) -> list[KnowledgeNode]:
        """Find nodes whose label starts with the given prefix."""
        prefix = prefix.lower()
        return [
            attrs["data"]
            for _, attrs in self._g.nodes(data=True)
            if attrs.get("data") and attrs["data"].label.startswith(prefix)
            and not attrs["data"].deprecated
        ]

    def deprecate_node(self, node_id: str, reason: str = "") -> None:
        """Soft-delete: mark as deprecated, keep for history."""
        node = self.get_node(node_id)
        if node:
            node.deprecated = True
            node.deprecation_reason = reason
            node.update_confidence(-0.4)
            self._dirty = True
            logger.info(f"Node deprecated: '{node.display_label}' — {reason}")

    # ── Edge Operations ───────────────────────────────────────────

    def upsert_edge(self, edge: KnowledgeEdge) -> None:
        """Add or reinforce an edge."""
        if not self._g.has_node(edge.source_id) or not self._g.has_node(edge.target_id):
            return  # Don't add orphan edges

        if self._g.has_edge(edge.source_id, edge.target_id):
            existing: KnowledgeEdge = self._g[edge.source_id][edge.target_id].get("data")
            if existing and existing.type == edge.type:
                existing.reinforce()
                self._dirty = True
                return

        self._g.add_edge(edge.source_id, edge.target_id, data=edge)
        self._dirty = True

    def get_edges_from(self, node_id: str) -> list[tuple[KnowledgeNode, KnowledgeEdge]]:
        """Get all outgoing edges from a node."""
        result = []
        for _, tgt, attrs in self._g.edges(node_id, data=True):
            edge: KnowledgeEdge = attrs.get("data")
            target_node = self.get_node(tgt)
            if edge and target_node and not target_node.deprecated:
                result.append((target_node, edge))
        return result

    def get_edges_to(self, node_id: str) -> list[tuple[KnowledgeNode, KnowledgeEdge]]:
        """Get all incoming edges to a node."""
        result = []
        for src, _, attrs in self._g.in_edges(node_id, data=True):
            edge: KnowledgeEdge = attrs.get("data")
            source_node = self.get_node(src)
            if edge and source_node and not source_node.deprecated:
                result.append((source_node, edge))
        return result

    # ── Context Retrieval ─────────────────────────────────────────

    def get_context_for_query(
        self, query: str, top_k: int = 12
    ) -> list[tuple[KnowledgeNode, list[tuple[KnowledgeNode, KnowledgeEdge]]]]:
        """
        Hybrid semantic + structural context retrieval.

        Algorithm:
            1. Semantic search on graph_nodes ChromaDB collection
            2. Expand each hit via 1-hop ego_graph (neighbors)
            3. Score: confidence × pagerank × recency
            4. Return top_k nodes with their edges

        Returns list of (node, [(neighbor, edge), ...]) tuples.
        """
        if self._g.number_of_nodes() == 0:
            return []

        # 1. Semantic search for seed nodes
        seed_ids = self._semantic_search_nodes(query, k=8)

        # 2. Expand seeds via graph neighborhood
        candidate_ids: set[str] = set(seed_ids)
        for seed_id in seed_ids:
            if self._g.has_node(seed_id):
                # Add 1-hop successors and predecessors
                candidate_ids.update(self._g.successors(seed_id))
                candidate_ids.update(self._g.predecessors(seed_id))

        # 3. Score and rank candidates
        now = time.time()
        scored = []
        for nid in candidate_ids:
            node = self.get_node(nid)
            if not node or node.deprecated:
                continue

            # Recency score: exponential decay
            days_idle = (now - node.last_accessed) / 86400
            recency = 2 ** (-days_idle / RECENCY_HALF_LIFE_DAYS)

            # PageRank score (computed lazily)
            pr = self._pagerank.get(nid, 0.01)

            score = (
                CONFIDENCE_WEIGHT * node.confidence
                + PAGERANK_WEIGHT * min(pr * 100, 1.0)  # normalize pagerank
                + RECENCY_WEIGHT * recency
            )
            scored.append((score, node))

        scored.sort(key=lambda x: x[0], reverse=True)
        top_nodes = [node for _, node in scored[:top_k]]

        # 4. Build result with edges
        result = []
        for node in top_nodes:
            edges = self.get_edges_from(node.id)
            result.append((node, edges))

        return result

    # ── Graph Analysis ────────────────────────────────────────────

    def compute_pagerank(self) -> dict[str, float]:
        """Compute PageRank for all nodes. Important nodes = many connections."""
        if self._g.number_of_nodes() < 2:
            return {}
        try:
            pr = nx.pagerank(self._g, alpha=0.85, max_iter=100)
            self._pagerank = pr
            logger.debug(f"PageRank computed for {len(pr)} nodes")
            return pr
        except Exception as e:
            logger.warning(f"PageRank computation failed: {e}")
            return {}

    def find_contradictions(self) -> list[tuple[KnowledgeNode, KnowledgeNode]]:
        """Find node pairs connected by CONTRADICTS edges."""
        contradictions = []
        for src, tgt, attrs in self._g.edges(data=True):
            edge: KnowledgeEdge = attrs.get("data")
            if edge and edge.type == EdgeType.CONTRADICTS:
                src_node = self.get_node(src)
                tgt_node = self.get_node(tgt)
                if src_node and tgt_node:
                    contradictions.append((src_node, tgt_node))
        return contradictions

    def merge_nodes(self, keep_id: str, merge_id: str) -> str:
        """
        Merge node `merge_id` into `keep_id`.

        - All edges pointing to/from merge_id are redirected to keep_id
        - Sources and properties are merged
        - Confidence is averaged
        - merge_id is deprecated
        """
        keep = self.get_node(keep_id)
        merge = self.get_node(merge_id)
        if not keep or not merge:
            return keep_id

        # Merge metadata
        keep.confidence = (keep.confidence + merge.confidence) / 2
        for src in merge.sources:
            if src not in keep.sources:
                keep.sources.append(src)
        for k, v in merge.properties.items():
            if k not in keep.properties:
                keep.properties[k] = v
        keep.last_updated = time.time()

        # Redirect edges
        for src, tgt, attrs in list(self._g.in_edges(merge_id, data=True)):
            edge: KnowledgeEdge = attrs.get("data")
            if edge and src != keep_id:
                new_edge = KnowledgeEdge(
                    source_id=src,
                    target_id=keep_id,
                    type=edge.type,
                    confidence=edge.confidence,
                    weight=edge.weight,
                    properties=edge.properties,
                )
                self.upsert_edge(new_edge)

        for src, tgt, attrs in list(self._g.out_edges(merge_id, data=True)):
            edge: KnowledgeEdge = attrs.get("data")
            if edge and tgt != keep_id:
                new_edge = KnowledgeEdge(
                    source_id=keep_id,
                    target_id=tgt,
                    type=edge.type,
                    confidence=edge.confidence,
                    weight=edge.weight,
                    properties=edge.properties,
                )
                self.upsert_edge(new_edge)

        # Add SIMILAR_TO edge as historical record
        similar_edge = KnowledgeEdge(
            source_id=keep_id,
            target_id=merge_id,
            type=EdgeType.SIMILAR_TO,
            confidence=0.9,
            properties={"reason": "merged"},
        )
        self.upsert_edge(similar_edge)

        # Deprecate the merged node
        self.deprecate_node(merge_id, reason=f"merged into {keep.display_label}")

        self._dirty = True
        logger.info(f"Merged '{merge.display_label}' → '{keep.display_label}'")
        return keep_id

    # ── Statistics ────────────────────────────────────────────────

    def get_stats(self) -> dict:
        nodes = [attrs["data"] for _, attrs in self._g.nodes(data=True) if attrs.get("data")]
        active = [n for n in nodes if not n.deprecated]
        deprecated = [n for n in nodes if n.deprecated]

        type_counts: dict[str, int] = {}
        for n in active:
            type_counts[n.type.value] = type_counts.get(n.type.value, 0) + 1

        avg_conf = sum(n.confidence for n in active) / len(active) if active else 0.0

        return {
            "total_nodes": len(nodes),
            "active_nodes": len(active),
            "deprecated_nodes": len(deprecated),
            "total_edges": self._g.number_of_edges(),
            "node_types": type_counts,
            "average_confidence": round(avg_conf, 3),
            "contradictions": len(self.find_contradictions()),
        }

    def get_top_nodes(self, n: int = 20) -> list[KnowledgeNode]:
        """Return top N most important active nodes by confidence × PageRank."""
        if not self._pagerank:
            self._recompute_pagerank()
        nodes = [
            attrs["data"]
            for _, attrs in self._g.nodes(data=True)
            if attrs.get("data") and not attrs["data"].deprecated
        ]
        nodes.sort(
            key=lambda x: x.confidence * self._pagerank.get(x.id, 0.01),
            reverse=True,
        )
        return nodes[:n]

    # ── Internal helpers ──────────────────────────────────────────

    def _index_node(self, node: KnowledgeNode) -> None:
        """Store node text embedding in ChromaDB for semantic lookup."""
        try:
            doc_id = f"gnode_{node.id}"
            self._vs.add(
                collection=self.GRAPH_NODES_COLLECTION,
                doc_id=doc_id,
                text=node.to_text(),
                metadata={
                    "node_id": node.id,
                    "type": node.type.value,
                    "label": node.label,
                    "domain": node.domain,
                    "confidence": node.confidence,
                },
            )
            node.embedding_id = doc_id
        except Exception as e:
            logger.warning(f"Failed to index node '{node.label}': {e}")

    def _semantic_search_nodes(self, query: str, k: int = 8) -> list[str]:
        """Search ChromaDB graph_nodes collection, return node IDs."""
        try:
            hits = self._vs.search(
                collection=self.GRAPH_NODES_COLLECTION,
                query=query,
                top_k=k,
            )
            return [hit["metadata"]["node_id"] for hit in hits if "node_id" in hit["metadata"]]
        except Exception as e:
            logger.warning(f"Semantic node search failed: {e}")
            return []

    def _recompute_pagerank(self) -> None:
        if self._g.number_of_nodes() >= 2:
            try:
                self._pagerank = nx.pagerank(self._g, alpha=0.85, max_iter=100)
            except Exception:
                pass
