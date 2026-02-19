"""Rich UI primitives and rendering helpers."""

from __future__ import annotations

import time
from datetime import datetime

from rich import box
from rich.columns import Columns
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from psycho.config.constants import (
    AGENT_NAME,
    AGENT_VERSION,
    COLOR_ACCENT,
    COLOR_AGENT,
    COLOR_DIM,
    COLOR_ERROR,
    COLOR_SYSTEM,
    COLOR_USER,
)

console = Console()


def render_header(session_id: str, model: str, provider: str) -> Panel:
    """Top-of-screen header panel."""
    header = Text()
    header.append(f" {AGENT_NAME} ", style=f"bold {COLOR_ACCENT}")
    header.append(f"v{AGENT_VERSION}", style=COLOR_DIM)
    header.append("  │  ", style=COLOR_DIM)
    header.append("session: ", style=COLOR_DIM)
    header.append(session_id, style="bold white")
    header.append("  │  ", style=COLOR_DIM)
    header.append("model: ", style=COLOR_DIM)
    header.append(model, style=f"bold {COLOR_SYSTEM}")
    header.append(f" ({provider})", style=COLOR_DIM)
    header.append("  │  ", style=COLOR_DIM)
    header.append(datetime.now().strftime("%H:%M"), style=COLOR_DIM)

    return Panel(header, box=box.HORIZONTALS, style="grey23", padding=(0, 1))


def render_stats_panel(stats: dict) -> Panel:
    """Right-side stats panel."""
    table = Table.grid(padding=(0, 1))
    table.add_column(style=COLOR_DIM, justify="right")
    table.add_column(style="bold white")

    table.add_row("sessions", str(stats.get("sessions", 0)))
    table.add_row("messages", str(stats.get("interactions", 0)))
    table.add_row("facts", str(stats.get("facts", 0)))
    table.add_row("memory", str(stats.get("short_term_turns", 0)) + " turns")

    return Panel(
        table,
        title=f"[{COLOR_ACCENT}]STATS[/{COLOR_ACCENT}]",
        border_style="grey35",
        padding=(0, 1),
    )


def render_graph_panel(node_count: int = 0, edge_count: int = 0) -> Panel:
    """Knowledge graph stats panel (Phase 3 will populate this)."""
    table = Table.grid(padding=(0, 1))
    table.add_column(style=COLOR_DIM, justify="right")
    table.add_column(style="bold white")
    table.add_row("nodes", str(node_count))
    table.add_row("edges", str(edge_count))

    return Panel(
        table,
        title=f"[{COLOR_ACCENT}]GRAPH[/{COLOR_ACCENT}]",
        border_style="grey35",
        padding=(0, 1),
    )


def render_user_message(message: str) -> None:
    """Print a user message turn."""
    console.print()
    label = Text(" YOU ", style=f"bold reverse {COLOR_USER}")
    console.print(label, end=" ")
    console.print(Text(message, style="white"))


def render_agent_message(message: str, latency_ms: float = 0) -> None:
    """Print the agent's response with Markdown rendering."""
    console.print()
    label = Text(f" {AGENT_NAME.upper()} ", style=f"bold reverse {COLOR_AGENT}")
    if latency_ms > 0:
        latency_text = Text(f" {latency_ms:.0f}ms", style=COLOR_DIM)
        console.print(label, latency_text)
    else:
        console.print(label)

    # Render as Markdown inside a subtle panel
    md = Markdown(message)
    console.print(
        Panel(md, border_style="grey23", padding=(0, 2)),
        markup=False,
    )


def render_streaming_start() -> None:
    """Print the agent label before streaming output."""
    console.print()
    label = Text(f" {AGENT_NAME.upper()} ", style=f"bold reverse {COLOR_AGENT}")
    console.print(label)
    console.print()


def render_separator() -> None:
    console.rule(style="grey23")


def render_welcome() -> None:
    """Full welcome screen on first launch."""
    console.clear()
    welcome = f"""
[bold {COLOR_ACCENT}]██████╗ ███████╗██╗   ██╗ ██████╗██╗  ██╗ ██████╗[/bold {COLOR_ACCENT}]
[bold {COLOR_ACCENT}]██╔══██╗██╔════╝╚██╗ ██╔╝██╔════╝██║  ██║██╔═══██╗[/bold {COLOR_ACCENT}]
[bold {COLOR_ACCENT}]██████╔╝███████╗ ╚████╔╝ ██║     ███████║██║   ██║[/bold {COLOR_ACCENT}]
[bold {COLOR_ACCENT}]██╔═══╝ ╚════██║  ╚██╔╝  ██║     ██╔══██║██║   ██║[/bold {COLOR_ACCENT}]
[bold {COLOR_ACCENT}]██║     ███████║   ██║   ╚██████╗██║  ██║╚██████╔╝[/bold {COLOR_ACCENT}]
[bold {COLOR_ACCENT}]╚═╝     ╚══════╝   ╚═╝    ╚═════╝╚═╝  ╚═╝ ╚═════╝ [/bold {COLOR_ACCENT}]  [dim]PORTAL[/dim]
"""
    console.print(welcome)
    console.print(
        Panel(
            f"[{COLOR_DIM}]Self-evolving AI assistant · Persistent knowledge graph · Local-first[/{COLOR_DIM}]",
            box=box.SIMPLE,
            padding=(0, 4),
        )
    )


def render_system_message(message: str, style: str = COLOR_SYSTEM) -> None:
    console.print(f"[{style}]  {message}[/{style}]")


def render_error(message: str) -> None:
    console.print(f"[{COLOR_ERROR}] ERROR  {message}[/{COLOR_ERROR}]")


def render_exit_summary(stats: dict) -> None:
    """Display session summary on exit."""
    console.print()
    table = Table(
        title="Session Summary",
        box=box.ROUNDED,
        border_style="grey35",
        show_header=False,
    )
    table.add_column(style=COLOR_DIM, justify="right")
    table.add_column(style="bold white")

    table.add_row("Session ID", stats.get("session_id", "—"))
    table.add_row("Messages this session", str(stats.get("short_term_turns", 0)))
    table.add_row("Total sessions", str(stats.get("sessions", 0)))
    table.add_row("Total messages", str(stats.get("interactions", 0)))
    table.add_row("Facts stored", str(stats.get("facts", 0)))
    table.add_row("Model", stats.get("model", "—"))

    console.print(table)
    console.print(
        f"\n[{COLOR_DIM}]Memory persisted. See you next time.[/{COLOR_DIM}]\n"
    )
