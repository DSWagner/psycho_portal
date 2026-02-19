"""Chat routes — POST /api/chat, GET /api/history, WebSocket /ws/chat."""

from __future__ import annotations

import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import JSONResponse

from psycho.api.schemas import ChatRequest, ChatResponse, IngestRequest, IngestResponse

router = APIRouter(prefix="/api", tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request):
    """Send a message and get a response (non-streaming)."""
    agent = request.app.state.agent
    response = await agent.chat(req.message)

    loop = getattr(agent, '_loop', None)
    domain_result = getattr(loop, '_last_domain_result', None)
    domain = getattr(loop, '_last_domain', "general")   # FIX: use _last_domain not _session_id
    actions = domain_result.actions_taken if domain_result else []

    return ChatResponse(
        response=response,
        domain=domain,
        session_id=agent.session_id,
        actions_taken=actions,
    )


@router.get("/history")
async def history(request: Request, limit: int = 20):
    """Get recent chat history."""
    agent = request.app.state.agent
    items = await agent.memory.get_recent_history(limit=limit)
    return {"history": items, "count": len(items)}


@router.post("/ingest", response_model=IngestResponse)
async def ingest_text(req: IngestRequest, request: Request):
    """Ingest raw text into the knowledge graph."""
    agent = request.app.state.agent
    result = await agent.ingest_text(
        text=req.text,
        source_name=req.source_name,
        domain=req.domain,
    )
    return IngestResponse(**result)


@router.get("/stats")
async def stats(request: Request):
    """Get agent statistics."""
    agent = request.app.state.agent
    return await agent.get_stats()


# ── WebSocket streaming ───────────────────────────────────────────────────────

async def ws_chat_handler(websocket: WebSocket, agent):
    """Handle a single WebSocket chat session with streaming."""
    await websocket.accept()
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json({"type": "error", "message": "Invalid JSON"})
                continue

            msg_type = data.get("type", "chat")
            message = data.get("message", "").strip()

            if msg_type == "ping":
                await websocket.send_json({"type": "pong"})
                continue

            if not message:
                await websocket.send_json({"type": "error", "message": "Empty message"})
                continue

            full_response = []
            try:
                async for token in agent.stream_chat(message):
                    full_response.append(token)
                    await websocket.send_json({"type": "token", "token": token})
            except Exception as e:
                await websocket.send_json({"type": "error", "message": str(e)})
                continue

            loop = getattr(agent, '_loop', None)
            domain_result = getattr(loop, '_last_domain_result', None)
            actions = domain_result.actions_taken if domain_result else []
            domain = getattr(loop, '_last_domain', "general")   # FIX: _last_domain not _session_id

            await websocket.send_json({
                "type": "done",
                "response": "".join(full_response),
                "domain": domain,
                "actions": actions,
                "session_id": agent.session_id,
            })

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
