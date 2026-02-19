"""Tasks routes."""

from __future__ import annotations

from fastapi import APIRouter, Request, HTTPException

from psycho.api.schemas import TaskCreate, TaskResponse

router = APIRouter(prefix="/api", tags=["tasks"])


@router.get("/tasks")
async def list_tasks(request: Request, status: str = "pending"):
    tasks = await request.app.state.agent.task_manager.get_all(status=status)
    return {"tasks": tasks, "count": len(tasks)}


@router.post("/tasks", response_model=TaskResponse)
async def create_task(req: TaskCreate, request: Request):
    tm = request.app.state.agent.task_manager
    import time
    task_id = await tm.create_task(
        title=req.title,
        description=req.description,
        priority=req.priority,
        due_date=req.due_date,
    )
    return TaskResponse(
        id=task_id,
        title=req.title,
        priority=req.priority,
        status="pending",
        due_date=req.due_date,
        created_at=time.time(),
    )


@router.patch("/tasks/{task_id}/complete")
async def complete_task(task_id: str, request: Request):
    tm = request.app.state.agent.task_manager
    await tm.complete_task(task_id)
    return {"status": "completed", "id": task_id}
