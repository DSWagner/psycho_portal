"""Click CLI commands — psycho chat | psycho stats | psycho graph | psycho ingest | psycho serve."""

from __future__ import annotations

import asyncio

import click
from rich.console import Console
from rich import box
from rich.table import Table
from rich.panel import Panel

console = Console()


@click.group()
@click.version_option(version="0.1.0", prog_name="psycho")
def cli() -> None:
    """PsychoPortal — self-evolving AI personal assistant with knowledge graph."""
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
    """Display memory and knowledge graph statistics."""
    from psycho.agent import PsychoAgent
    from psycho.cli import ui

    async def _run():
        agent = PsychoAgent()
        await agent.start()
        s = await agent.get_stats()
        recent = await agent.memory.get_recent_history(limit=8)
        await agent.stop()
        return s, recent

    s, recent = asyncio.run(_run())
    ui.render_welcome()
    ui.render_exit_summary(s)

    if recent:
        table = Table(
            title="Recent Interactions",
            box=box.SIMPLE_HEAVY,
            border_style="grey35",
        )
        table.add_column("Domain", style="dim", width=10)
        table.add_column("User", width=45)
        table.add_column("Response Preview", width=55)
        for item in reversed(recent):
            table.add_row(
                item.get("domain", "general"),
                item["user_message"][:45],
                item["agent_response"][:55] + "…",
            )
        console.print(table)


@cli.command()
@click.option("--top", default=25, help="Number of top nodes to display")
@click.option("--type", "node_type", default=None, help="Filter by node type (concept/fact/preference/…)")
@click.option("--export", "export_path", default=None, help="Export graph to JSON file")
def graph(top: int, node_type: str | None, export_path: str | None) -> None:
    """Inspect the knowledge graph — top nodes, stats, optional export."""
    from psycho.agent import PsychoAgent
    from psycho.knowledge.schema import NodeType, confidence_bar, confidence_label

    async def _run():
        agent = PsychoAgent()
        await agent.start()

        g = agent.graph
        stats = g.get_stats()
        top_nodes = g.get_top_nodes(top)

        # Filter by type if requested
        if node_type:
            try:
                nt = NodeType(node_type)
                top_nodes = [n for n in top_nodes if n.type == nt]
            except ValueError:
                console.print(f"[red]Unknown node type '{node_type}'. "
                               f"Valid: {', '.join(t.value for t in NodeType)}[/red]")

        if export_path:
            import json
            from pathlib import Path
            d3_data = agent.graph._store.export_d3(
                {nid: attrs["data"].to_dict() for nid, attrs in g._g.nodes(data=True) if attrs.get("data")},
                [attrs["data"].to_dict() for _, __, attrs in g._g.edges(data=True) if attrs.get("data")],
            )
            Path(export_path).write_text(json.dumps(d3_data, indent=2))
            console.print(f"[green]Graph exported to {export_path}[/green]")

        await agent.stop()
        return stats, top_nodes

    stats, nodes = asyncio.run(_run())

    # Header stats
    console.print(Panel(
        f"[bold magenta]Knowledge Graph[/bold magenta]\n"
        f"  Active nodes:      [bold]{stats['active_nodes']}[/bold]\n"
        f"  Total edges:       [bold]{stats['total_edges']}[/bold]\n"
        f"  Avg confidence:    [bold]{stats['average_confidence']:.2f}[/bold]\n"
        f"  Contradictions:    [bold]{stats['contradictions']}[/bold]",
        box=box.ROUNDED, border_style="grey35",
    ))

    if not nodes:
        console.print("[dim]No nodes found.[/dim]")
        return

    # Node table
    table = Table(
        title=f"Top {len(nodes)} Knowledge Nodes",
        box=box.SIMPLE_HEAVY,
        border_style="grey35",
        show_header=True,
    )
    table.add_column("Type", style="dim", width=12)
    table.add_column("Label", width=35)
    table.add_column("Domain", width=12)
    table.add_column("Confidence", width=20)
    table.add_column("Props", style="dim", width=30)

    for node in nodes:
        conf_label = confidence_label(node.confidence)
        conf_style = "green" if node.confidence > 0.7 else "yellow" if node.confidence > 0.4 else "red"
        props_str = ", ".join(f"{k}={str(v)[:15]}" for k, v in list(node.properties.items())[:2])
        table.add_row(
            node.type.value,
            node.display_label[:35],
            node.domain,
            f"[{conf_style}]{confidence_bar(node.confidence, 8)} {node.confidence:.2f}[/{conf_style}]",
            props_str,
        )

    console.print(table)

    # Node type breakdown
    if stats.get("node_types"):
        type_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
        type_table.add_column("Type", style="dim")
        type_table.add_column("Count", style="bold")
        for t, count in sorted(stats["node_types"].items(), key=lambda x: -x[1]):
            type_table.add_row(t, str(count))
        console.print(Panel(type_table, title="By Type", border_style="grey23", expand=False))


@cli.command()
@click.argument("path")
@click.option("--domain", default=None, help="Override domain (coding/health/general/…)")
@click.option("--text", is_flag=True, help="Treat PATH as raw text to ingest")
def ingest(path: str, domain: str | None, text: bool) -> None:
    """
    Ingest a file, folder, or raw text into the knowledge graph.

    \b
    Examples:
        psycho ingest notes.md
        psycho ingest ./docs/
        psycho ingest "Python uses indentation" --text
        psycho ingest report.pdf --domain health
    """
    from psycho.agent import PsychoAgent

    async def _run():
        agent = PsychoAgent()
        await agent.start()

        if text:
            result = await agent.ingest_text(
                text=path,
                source_name="cli_text",
                domain=domain or "general",
            )
            console.print(f"[green]Text ingested:[/green] {result['nodes_added']} nodes, {result['facts_added']} facts")
        else:
            from pathlib import Path as P
            p = P(path)
            if p.is_dir():
                console.print(f"[dim]Scanning folder: {path}[/dim]")
            else:
                console.print(f"[dim]Ingesting: {path}[/dim]")

            with console.status("[dim]Processing…[/dim]", spinner="dots"):
                result = await agent.ingest_file(path)

            if result.get("files_processed"):
                console.print(
                    f"[green]Folder ingested:[/green] "
                    f"{result['files_processed']} files → "
                    f"{result['nodes_added']} nodes, {result['facts_added']} facts"
                )
            else:
                console.print(
                    f"[green]File ingested:[/green] "
                    f"{result['nodes_added']} nodes, "
                    f"{result['facts_added']} facts, "
                    f"{result['edges_added']} edges "
                    f"({result['chunks']} chunks)"
                )
            if result.get("errors"):
                for err in result["errors"]:
                    console.print(f"[red]  Error: {err}[/red]")

        graph_stats = agent.graph.get_stats()
        console.print(
            f"[dim]Graph total: {graph_stats['active_nodes']} nodes, "
            f"{graph_stats['total_edges']} edges[/dim]"
        )
        await agent.stop()

    asyncio.run(_run())


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
