"""Click CLI commands — psycho chat | psycho stats | psycho serve."""

from __future__ import annotations

import asyncio

import click
from loguru import logger
from rich.console import Console

console = Console()


@click.group()
@click.version_option(version="0.1.0", prog_name="psycho")
def cli() -> None:
    """PsychoPortal — self-evolving AI personal assistant."""
    pass


@cli.command()
def chat() -> None:
    """Start an interactive chat session with the dashboard UI."""
    from psycho.agent import PsychoAgent
    from psycho.cli.chat_view import ChatView

    agent = PsychoAgent()
    view = ChatView(agent)

    try:
        asyncio.run(view.run())
    except KeyboardInterrupt:
        pass


@cli.command()
def stats() -> None:
    """Display memory statistics and session history."""
    from psycho.agent import PsychoAgent
    from psycho.cli import ui

    async def _run():
        agent = PsychoAgent()
        await agent.start()
        stats = await agent.get_stats()

        # Recent interactions
        recent = await agent.memory.get_recent_history(limit=10)

        await agent.stop()
        return stats, recent

    s, recent = asyncio.run(_run())

    ui.render_welcome()
    ui.render_exit_summary(s)

    if recent:
        from rich.table import Table
        from rich import box

        table = Table(
            title="Recent Interactions",
            box=box.SIMPLE_HEAVY,
            border_style="grey35",
            show_header=True,
        )
        table.add_column("Domain", style="dim", width=10)
        table.add_column("User Message", width=50)
        table.add_column("Response Preview", width=60)

        for item in reversed(recent):
            table.add_row(
                item.get("domain", "general"),
                item["user_message"][:50],
                item["agent_response"][:60] + "...",
            )

        console.print(table)


@cli.command()
@click.option("--host", default="0.0.0.0", help="Bind host")
@click.option("--port", default=8000, help="Bind port")
def serve(host: str, port: int) -> None:
    """Start the FastAPI HTTP server (for web/mobile integrations)."""
    try:
        import uvicorn
        from psycho.api.server import create_app

        app = create_app()
        console.print(f"[green]Starting PsychoPortal API on http://{host}:{port}[/green]")
        uvicorn.run(app, host=host, port=port)
    except ImportError as e:
        console.print(f"[red]FastAPI/uvicorn not installed: {e}[/red]")
        raise SystemExit(1)
