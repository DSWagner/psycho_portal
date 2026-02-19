"""Pydantic schemas for the FastAPI API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=8000)


class ChatResponse(BaseModel):
    response: str
    domain: str = "general"
    session_id: str = ""
    tokens_used: int = 0
    actions_taken: list[str] = []


class IngestRequest(BaseModel):
    text: str = Field(..., min_length=1)
    source_name: str = "api_input"
    domain: str = "general"


class IngestResponse(BaseModel):
    nodes_added: int = 0
    facts_added: int = 0
    chunks_processed: int = 0


class TaskCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=300)
    priority: str = "normal"
    due_date: str | None = None
    description: str = ""


class TaskResponse(BaseModel):
    id: str
    title: str
    priority: str
    status: str
    due_date: str | None
    created_at: float


class HealthMetricCreate(BaseModel):
    metric_type: str
    value: float
    unit: str
    notes: str = ""


class GraphNode(BaseModel):
    id: str
    label: str
    type: str
    confidence: float
    domain: str


class GraphEdge(BaseModel):
    source: str
    target: str
    type: str
    confidence: float


class GraphResponse(BaseModel):
    nodes: list[GraphNode]
    links: list[GraphEdge]
    stats: dict[str, Any]


class StatsResponse(BaseModel):
    sessions: int = 0
    interactions: int = 0
    facts: int = 0
    graph_nodes: int = 0
    graph_edges: int = 0
    graph_avg_confidence: float = 0.0
    pending_tasks: int = 0
    health_entries: int = 0
    total_mistakes: int = 0
    session_id: str = ""
    model: str = ""
    provider: str = ""
