"""Dashboard chat view — Rich panels + prompt_toolkit input."""

from __future__ import annotations

import asyncio
import sys
import time
from typing import TYPE_CHECKING

from loguru import logger
from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style
from rich import box
from rich.columns import Columns
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.text import Text

from psycho.config.constants import (
    AGENT_NAME,
    COLOR_ACCENT,
    COLOR_DIM,
    MAX_CHAT_DISPLAY_MESSAGES,
)

from . import ui

if TYPE_CHECKING:
    from psycho.agent import PsychoAgent


# Prompt toolkit style matching the Rich color scheme
PT_STYLE = Style.from_dict(
    {
        "prompt": "ansicyan bold",
        "": "ansiwhite",
    }
)

PROMPT_TEXT = HTML('<prompt> › </prompt>')

# Commands the user can type
COMMANDS = {
    "/help":          "Show available commands",
    "/stats":         "Show memory and knowledge graph statistics",
    "/graph":         "Inspect the knowledge graph",
    "/facts":         "List stored facts",
    "/ingest <path>": "Ingest a file or folder into the knowledge graph",
    "/clear":         "Clear the screen",
    "/exit":          "Exit the chat (also: quit, exit, bye)",
}

EXIT_PHRASES = {"/exit", "exit", "quit", "bye", "/quit"}


class ChatView:
    """
    Interactive dashboard chat interface.

    Layout:
        ┌── header ──────────────────────────────────────────┐
        │  PSYCHO PORTAL  session: xxx  model: haiku         │
        └────────────────────────────────────────────────────┘
        ┌── chat (messages) ──────────────────────────────────┐
        │  [You] ...                                          │
        │  [PSYCHOPORTAL] ...                                 │
        └────────────────────────────────────────────────────┘
        ┌── input ────────────────────────────────────────────┐
        │  › type here                                        │
        └────────────────────────────────────────────────────┘

    Stats are rendered in a sidebar when the terminal is wide enough (>120 cols).
    """

    def __init__(self, agent: "PsychoAgent") -> None:
        self._agent = agent
        self._console = ui.console
        self._session = PromptSession(
            history=FileHistory(".psycho_history"),
            auto_suggest=AutoSuggestFromHistory(),
            style=PT_STYLE,
            mouse_support=False,
        )
        self._turn_count = 0

    async def run(self) -> None:
        """Main chat loop."""
        # Welcome screen
        ui.render_welcome()

        # Startup
        await self._agent.start()
        stats = await self._agent.get_stats()

        # Header
        self._console.print(
            ui.render_header(
                session_id=self._agent.session_id,
                model=self._agent.llm.model_name,
                provider=self._agent.llm.provider_name,
            )
        )

        # Greeting
        ui.render_system_message(
            f"Memory loaded · {stats.get('interactions', 0)} past interactions · "
            f"Type /help for commands"
        )
        ui.render_separator()

        # Main loop
        while True:
            try:
                user_input = await self._session.prompt_async(PROMPT_TEXT)
            except (EOFError, KeyboardInterrupt):
                await self._handle_exit()
                break

            user_input = user_input.strip()
            if not user_input:
                continue

            # Built-in commands
            if user_input.lower() in EXIT_PHRASES:
                await self._handle_exit()
                break

            if user_input.startswith("/"):
                handled = await self._handle_command(user_input)
                if handled:
                    continue

            # Regular chat
            await self._handle_chat(user_input)

    async def _handle_chat(self, user_input: str) -> None:
        """Process a chat message and render the response."""
        ui.render_user_message(user_input)

        start = time.time()

        # Show a spinner while waiting
        with self._console.status(
            f"[{COLOR_DIM}]Thinking...[/{COLOR_DIM}]", spinner="dots"
        ):
            response = await self._agent.chat(user_input)

        latency_ms = (time.time() - start) * 1000
        self._turn_count += 1

        ui.render_agent_message(response, latency_ms=latency_ms)

    async def _handle_command(self, cmd: str) -> bool:
        """Handle slash commands. Returns True if command was recognized."""
        cmd_lower = cmd.lower().split()[0]

        if cmd_lower == "/help":
            self._show_help()
            return True

        if cmd_lower == "/stats":
            await self._show_stats()
            return True

        if cmd_lower == "/clear":
            self._console.clear()
            self._console.print(
                ui.render_header(
                    session_id=self._agent.session_id,
                    model=self._agent.llm.model_name,
                    provider=self._agent.llm.provider_name,
                )
            )
            return True

        if cmd_lower == "/facts":
            await self._show_facts()
            return True

        if cmd_lower == "/graph":
            await self._show_graph()
            return True

        if cmd_lower == "/ingest":
            parts = cmd.split(maxsplit=1)
            if len(parts) < 2:
                ui.render_system_message("Usage: /ingest <file_path_or_folder>", style="yellow")
            else:
                await self._handle_ingest(parts[1].strip())
            return True

        ui.render_system_message(
            f"Unknown command: {cmd}. Type /help for available commands.",
            style="yellow",
        )
        return True

    def _show_help(self) -> None:
        from rich.table import Table

        table = Table(
            title="Available Commands",
            box=box.SIMPLE_HEAVY,
            border_style="grey35",
            show_header=True,
            header_style=f"bold {COLOR_ACCENT}",
        )
        table.add_column("Command", style="bold white", no_wrap=True)
        table.add_column("Description", style=COLOR_DIM)

        for cmd, desc in COMMANDS.items():
            table.add_row(cmd, desc)

        self._console.print(table)

    async def _show_stats(self) -> None:
        stats = await self._agent.get_stats()
        ui.render_exit_summary(stats)

    async def _show_facts(self) -> None:
        facts = await self._agent.memory.long_term.get_facts(limit=20)
        if not facts:
            ui.render_system_message("No facts stored yet.")
            return

        from rich.table import Table

        table = Table(
            title="Stored Facts",
            box=box.SIMPLE_HEAVY,
            border_style="grey35",
            show_header=True,
        )
        table.add_column("Confidence", style="bold", justify="center", width=12)
        table.add_column("Domain", style=COLOR_DIM, width=10)
        table.add_column("Content")

        for fact in facts:
            conf = fact["confidence"]
            conf_style = "green" if conf > 0.7 else "yellow" if conf > 0.4 else "red"
            table.add_row(
                f"[{conf_style}]{conf:.2f}[/{conf_style}]",
                fact["domain"],
                fact["content"][:120],
            )

        self._console.print(table)

    async def _show_graph(self) -> None:
        from psycho.knowledge.schema import confidence_bar, confidence_label

        g = self._agent.graph
        stats = g.get_stats()
        top_nodes = g.get_top_nodes(20)

        from rich.table import Table
        from rich import box

        self._console.print(
            f"\n[magenta]Knowledge Graph[/magenta] — "
            f"{stats['active_nodes']} nodes · {stats['total_edges']} edges · "
            f"avg confidence {stats['average_confidence']:.2f}\n"
        )
        if not top_nodes:
            ui.render_system_message("Graph is empty. Chat more to build it, or use /ingest.")
            return

        table = Table(box=box.SIMPLE_HEAVY, border_style="grey35", show_header=True)
        table.add_column("Type", style="dim", width=12)
        table.add_column("Node", width=35)
        table.add_column("Domain", width=12)
        table.add_column("Confidence", width=22)

        for node in top_nodes[:20]:
            conf_style = "green" if node.confidence > 0.7 else "yellow" if node.confidence > 0.4 else "red"
            table.add_row(
                node.type.value,
                node.display_label[:35],
                node.domain,
                f"[{conf_style}]{confidence_bar(node.confidence, 8)} {node.confidence:.2f}[/{conf_style}]",
            )
        self._console.print(table)

    async def _handle_ingest(self, path: str) -> None:
        import os
        if not os.path.exists(path):
            ui.render_error(f"Path not found: {path}")
            return

        ui.render_system_message(f"Ingesting: {path}")
        with self._console.status("[dim]Processing…[/dim]", spinner="dots"):
            result = await self._agent.ingest_file(path)

        if result.get("errors"):
            for err in result["errors"]:
                ui.render_error(err)
        nodes = result.get("nodes_added", 0)
        facts = result.get("facts_added", 0)
        edges = result.get("edges_added", 0)
        ui.render_system_message(
            f"Ingestion complete: {nodes} nodes, {facts} facts, {edges} edges added to graph."
        )

    async def _handle_exit(self) -> None:
        ui.render_separator()
        ui.render_system_message("Saving session and shutting down...")
        stats = await self._agent.get_stats()
        await self._agent.stop()
        ui.render_exit_summary(stats)
