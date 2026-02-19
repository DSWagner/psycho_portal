"""FastAPI application factory — Phase 6 placeholder."""

from __future__ import annotations


def create_app():
    """Create and configure the FastAPI application."""
    try:
        from fastapi import FastAPI

        app = FastAPI(
            title="PsychoPortal API",
            description="Self-evolving AI personal assistant",
            version="0.1.0",
        )

        @app.get("/health")
        async def health():
            return {"status": "ok", "version": "0.1.0"}

        @app.get("/")
        async def root():
            return {
                "name": "PsychoPortal",
                "status": "running",
                "note": "Full API routes coming in Phase 6",
            }

        return app
    except ImportError:
        raise ImportError("Install fastapi and uvicorn: pip install fastapi uvicorn[standard]")
