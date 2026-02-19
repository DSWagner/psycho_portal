"""Health metrics routes."""

from __future__ import annotations

from fastapi import APIRouter, Request

from psycho.api.schemas import HealthMetricCreate

router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health-metrics")
async def get_health_metrics(request: Request, days: int = 30):
    ht = request.app.state.agent.health_tracker
    summary = await ht.get_summary(days=days)
    return {"summary": summary, "days": days}


@router.post("/health-metrics")
async def log_health_metric(req: HealthMetricCreate, request: Request):
    ht = request.app.state.agent.health_tracker
    agent = request.app.state.agent
    mid = await ht.log_metric(
        metric_type=req.metric_type,
        value=req.value,
        unit=req.unit,
        notes=req.notes,
        session_id=agent.session_id,
    )
    return {"id": mid, "logged": True}
