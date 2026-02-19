"""Graph routes — GET /api/graph."""

from __future__ import annotations

from fastapi import APIRouter, Request

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
