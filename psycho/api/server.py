"""FastAPI application — full REST API + WebSocket streaming + web UI."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

from psycho.api.routes.chat import router as chat_router, ws_chat_handler
from psycho.api.routes.graph import router as graph_router
from psycho.api.routes.health_metrics import router as health_router
from psycho.api.routes.tasks import router as tasks_router

STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize agent on startup, shut down cleanly on exit."""
    from psycho.agent.core import PsychoAgent

    logger.info("PsychoPortal API starting…")
    agent = PsychoAgent()
    await agent.start()
    app.state.agent = agent
    logger.info(
        f"Agent ready | session={agent.session_id} | "
        f"model={agent.llm.model_name}"
    )
    yield
    # Shutdown
    logger.info("PsychoPortal API shutting down…")
    await agent.stop(run_reflection=True)


def create_app() -> FastAPI:
    app = FastAPI(
        title="PsychoPortal API",
        description="Self-evolving AI personal assistant",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS — allow all origins in dev
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # REST routes
    app.include_router(chat_router)
    app.include_router(graph_router)
    app.include_router(health_router)
    app.include_router(tasks_router)

    # WebSocket streaming
    @app.websocket("/ws/chat")
    async def websocket_chat(websocket: WebSocket):
        await ws_chat_handler(websocket, app.state.agent)

    # Serve static web UI
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def root():
        index = STATIC_DIR / "index.html"
        if index.exists():
            return FileResponse(str(index))
        return HTMLResponse(
            "<h1>PsychoPortal API</h1>"
            "<p>Visit <a href='/docs'>/docs</a> for the API documentation.</p>"
            "<p>The web UI file (psycho/api/static/index.html) was not found.</p>"
        )

    @app.get("/api/ping")
    async def ping():
        return {"status": "ok", "service": "PsychoPortal"}

    return app
