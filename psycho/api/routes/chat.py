"""Chat routes — POST /api/chat, GET /api/history, WebSocket /ws/chat."""

from __future__ import annotations

import asyncio
import base64
import json
import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Request, UploadFile, File
from fastapi.responses import JSONResponse
from loguru import logger

from psycho.api.schemas import ChatRequest, ChatResponse, IngestRequest, IngestResponse

router = APIRouter(prefix="/api", tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request):
    """Send a message and get a response (non-streaming)."""
    agent = request.app.state.agent
    response = await agent.chat(req.message)

    loop = getattr(agent, '_loop', None)
    domain_result = getattr(loop, '_last_domain_result', None)
    domain = getattr(loop, '_last_domain', "general")
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


@router.get("/sessions")
async def get_sessions(request: Request, limit: int = 30):
    """List all past sessions."""
    agent = request.app.state.agent
    sessions = await agent.memory.long_term.get_sessions(limit=limit)
    return {"sessions": sessions}


@router.get("/sessions/{session_id}/messages")
async def get_session_messages(session_id: str, request: Request):
    """Get all messages for a specific session."""
    agent = request.app.state.agent
    messages = await agent.memory.long_term.get_interactions_for_session(session_id)
    return {"messages": messages, "session_id": session_id}


# ── Background ingest helper ──────────────────────────────────────────────────

async def _ingest_file_bg(agent, tmp_path: str) -> None:
    """Run file ingest in the background, then clean up the temp file."""
    try:
        await agent.ingest_file(tmp_path)
    except Exception as exc:
        logger.warning(f"Background file ingest failed ({tmp_path}): {exc}")
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@router.post("/upload")
async def upload_file(request: Request, file: UploadFile = File(...)):
    """Upload a file — returns immediately while ingest runs in the background."""
    from psycho.knowledge.ingestion import SUPPORTED_EXTENSIONS

    agent = request.app.state.agent
    suffix = Path(file.filename).suffix.lower()

    if suffix not in SUPPORTED_EXTENSIONS:
        return JSONResponse(
            status_code=400,
            content={"error": f"Unsupported file type: {suffix}. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"},
        )

    # Write to temp file, then hand off to a background task
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    asyncio.create_task(_ingest_file_bg(agent, tmp_path))

    return {
        "filename": file.filename,
        "nodes_added": 0,
        "facts_added": 0,
        "edges_added": 0,
        "chunks": 0,
        "errors": [],
        "status": "processing",
    }


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

            # ── File chat (text file content + user prompt) ────────
            if msg_type == "file_chat":
                file_content = data.get("file_content", "")
                filename = data.get("filename", "unknown")
                user_prompt = message or "Please analyse and summarise this file."

                # Cap to avoid exceeding LLM context window
                content_preview = file_content[:12000]
                if len(file_content) > 12000:
                    content_preview += "\n\n[…file truncated for context…]"

                enriched = (
                    f"The user has shared a file named '{filename}'.\n\n"
                    f"File contents:\n```\n{content_preview}\n```\n\n"
                    f"User: {user_prompt}"
                )

                # Kick off background graph ingestion (don't wait)
                asyncio.create_task(
                    agent.ingest_text(file_content[:20000], source_name=filename)
                )

                full_response = []
                try:
                    async for token in agent.stream_chat(enriched):
                        full_response.append(token)
                        await websocket.send_json({"type": "token", "token": token})
                except Exception as e:
                    await websocket.send_json({"type": "error", "message": str(e)})
                    continue

                loop = getattr(agent, '_loop', None)
                domain_result = getattr(loop, '_last_domain_result', None)
                actions = domain_result.actions_taken if domain_result else []
                domain = getattr(loop, '_last_domain', "general")
                await websocket.send_json({
                    "type": "done",
                    "response": "".join(full_response),
                    "domain": domain,
                    "actions": actions,
                    "session_id": agent.session_id,
                })
                continue

            # ── Image chat ─────────────────────────────────────────
            if msg_type == "image_chat":
                image_b64 = data.get("image", "")
                media_type = data.get("media_type", "image/jpeg")
                if not image_b64:
                    await websocket.send_json({"type": "error", "message": "No image data"})
                    continue
                try:
                    image_bytes = base64.b64decode(image_b64)
                except Exception:
                    await websocket.send_json({"type": "error", "message": "Invalid base64 image"})
                    continue

                full_response = []
                try:
                    async for token in agent.stream_chat_with_image(
                        message, image_bytes, media_type
                    ):
                        full_response.append(token)
                        await websocket.send_json({"type": "token", "token": token})
                except Exception as e:
                    await websocket.send_json({"type": "error", "message": str(e)})
                    continue

                loop = getattr(agent, '_loop', None)
                domain_result = getattr(loop, '_last_domain_result', None)
                actions = domain_result.actions_taken if domain_result else []
                domain = getattr(loop, '_last_domain', "general")
                await websocket.send_json({
                    "type": "done",
                    "response": "".join(full_response),
                    "domain": domain,
                    "actions": actions,
                    "session_id": agent.session_id,
                })
                continue

            # ── Regular text chat ──────────────────────────────────
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
            domain = getattr(loop, '_last_domain', "general")
            search_query = ""
            if hasattr(loop, '_last_search_query'):
                search_query = loop._last_search_query or ""

            await websocket.send_json({
                "type": "done",
                "response": "".join(full_response),
                "domain": domain,
                "actions": actions,
                "session_id": agent.session_id,
                "search_query": search_query,
            })

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
