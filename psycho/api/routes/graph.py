"""Graph routes — GET /api/graph, GET/DELETE /api/graph/node/{id}."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from psycho.api.schemas import GraphResponse, GraphNode, GraphEdge

router = APIRouter(prefix="/api", tags=["graph"])


@router.get("/graph", response_model=GraphResponse)
async def get_graph(request: Request, limit: int = 200):
    """Get the knowledge graph in D3.js-compatible format."""
    agent = request.app.state.agent
    g = agent.graph

    nodes = []
    links = []

    for nid, attrs in list(g._g.nodes(data=True))[:limit]:
        node = attrs.get("data")
        if node and not node.deprecated:
            nodes.append(GraphNode(
                id=node.id,
                label=node.display_label[:40],
                type=node.type.value,
                confidence=node.confidence,
                domain=node.domain,
            ))

    for src, tgt, attrs in g._g.edges(data=True):
        edge = attrs.get("data")
        if edge:
            links.append(GraphEdge(
                source=src,
                target=tgt,
                type=edge.type.value,
                confidence=edge.confidence,
            ))

    stats = g.get_stats()
    return GraphResponse(nodes=nodes, links=links, stats=stats)


@router.get("/graph/node/{node_id}")
async def get_node(node_id: str, request: Request):
    """Get full details for a single graph node including its edges."""
    agent = request.app.state.agent
    g = agent.graph

    node = g.get_node(node_id)
    if not node:
        return JSONResponse(status_code=404, content={"error": "Node not found"})

    out_edges = []
    for target, edge in g.get_edges_from(node_id):
        out_edges.append({
            "direction": "out",
            "node_id": target.id,
            "node_label": target.display_label,
            "edge_type": edge.type.value,
            "confidence": edge.confidence,
        })

    in_edges = []
    for source, edge in g.get_edges_to(node_id):
        in_edges.append({
            "direction": "in",
            "node_id": source.id,
            "node_label": source.display_label,
            "edge_type": edge.type.value,
            "confidence": edge.confidence,
        })

    return {
        "id": node.id,
        "label": node.display_label,
        "type": node.type.value,
        "domain": node.domain,
        "confidence": round(node.confidence, 3),
        "deprecated": node.deprecated,
        "sources": node.sources,
        "properties": node.properties,
        "edges": out_edges + in_edges,
        "edge_count": len(out_edges) + len(in_edges),
    }


@router.delete("/graph/node/{node_id}")
async def delete_node(node_id: str, request: Request):
    """Soft-delete (deprecate) a graph node."""
    agent = request.app.state.agent
    g = agent.graph

    node = g.get_node(node_id)
    if not node:
        return JSONResponse(status_code=404, content={"error": "Node not found"})

    label = node.display_label
    g.deprecate_node(node_id, reason="Deleted by user via graph explorer")
    g.save()
    return {"success": True, "deleted": label, "node_id": node_id}
