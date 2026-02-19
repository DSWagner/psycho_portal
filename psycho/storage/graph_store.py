"""NetworkX graph serialization — save/load the knowledge graph as JSON."""

from __future__ import annotations

import json
import time
from pathlib import Path

from loguru import logger

GRAPH_FILE = "knowledge_graph.json"
METADATA_FILE = "graph_metadata.json"
SCHEMA_VERSION = 2


class GraphStore:
    """
    Persists a NetworkX knowledge graph to disk as JSON.

    Format:
        knowledge_graph.json  — all nodes and edges
        graph_metadata.json   — stats, version, evolution history
    """

    def __init__(self, graph_dir: Path) -> None:
        self._dir = graph_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._graph_file = self._dir / GRAPH_FILE
        self._meta_file = self._dir / METADATA_FILE

    def save(self, nodes: dict, edges: list, extra_meta: dict | None = None) -> None:
        """Serialize graph to JSON."""
        data = {
            "schema_version": SCHEMA_VERSION,
            "saved_at": time.time(),
            "nodes": nodes,      # {node_id: node_dict}
            "edges": edges,      # [edge_dict, ...]
        }
        with open(self._graph_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        # Update metadata
        meta = self._load_meta()
        meta.update(
            {
                "schema_version": SCHEMA_VERSION,
                "last_saved": time.time(),
                "total_nodes": len(nodes),
                "total_edges": len(edges),
                "active_nodes": sum(
                    1 for n in nodes.values() if not n.get("deprecated", False)
                ),
            }
        )
        if extra_meta:
            meta.update(extra_meta)
        meta.setdefault("created_at", time.time())
        meta.setdefault("evolution_history", [])

        with open(self._meta_file, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        logger.debug(
            f"Graph saved: {len(nodes)} nodes, {len(edges)} edges → {self._graph_file}"
        )

    def load(self) -> dict | None:
        """Load graph from JSON. Returns None if no saved graph exists."""
        if not self._graph_file.exists():
            return None
        try:
            with open(self._graph_file, encoding="utf-8") as f:
                data = json.load(f)
            logger.info(
                f"Graph loaded: {len(data.get('nodes', {}))} nodes, "
                f"{len(data.get('edges', []))} edges"
            )
            return data
        except (json.JSONDecodeError, OSError) as e:
            logger.error(f"Failed to load graph: {e}")
            return None

    def load_metadata(self) -> dict:
        return self._load_meta()

    def record_evolution_event(self, event: dict) -> None:
        """Append an evolution event to the graph history."""
        meta = self._load_meta()
        history = meta.setdefault("evolution_history", [])
        history.append({"timestamp": time.time(), **event})
        # Keep last 200 events
        meta["evolution_history"] = history[-200:]
        with open(self._meta_file, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    def export_cypher(self, nodes: dict, edges: list) -> str:
        """Export graph as Neo4j Cypher CREATE statements."""
        lines = ["// PsychoPortal Knowledge Graph — Cypher Export"]
        for nid, n in nodes.items():
            props = json.dumps(n.get("properties", {}))
            lines.append(
                f"CREATE (n{nid[:8]}:{n['type'].upper()} "
                f"{{id: '{nid}', label: '{n['label']}', "
                f"confidence: {n['confidence']:.3f}}})"
            )
        for e in edges:
            src = e["source_id"][:8]
            tgt = e["target_id"][:8]
            rel = e["type"].upper().replace("-", "_")
            lines.append(f"CREATE (n{src})-[:{rel} {{confidence: {e['confidence']:.3f}}}]->(n{tgt})")
        return "\n".join(lines)

    def export_d3(self, nodes: dict, edges: list) -> dict:
        """Export as D3.js-compatible JSON for visualization."""
        return {
            "nodes": [
                {
                    "id": nid,
                    "label": n["display_label"],
                    "type": n["type"],
                    "confidence": n["confidence"],
                    "domain": n["domain"],
                    "deprecated": n.get("deprecated", False),
                }
                for nid, n in nodes.items()
            ],
            "links": [
                {
                    "source": e["source_id"],
                    "target": e["target_id"],
                    "type": e["type"],
                    "confidence": e["confidence"],
                }
                for e in edges
            ],
        }

    def _load_meta(self) -> dict:
        if self._meta_file.exists():
            try:
                with open(self._meta_file, encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {}
