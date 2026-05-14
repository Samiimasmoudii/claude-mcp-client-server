"""
Rich-based terminal UI utilities.
All output in the app goes through this module — never print() directly.
"""
import json
from rich.console import Console
from rich.markdown import Markdown
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.style import Style

console = Console()


class StreamRenderer:
    """Live markdown renderer for streaming LLM responses."""

    def __init__(self):
        self._buf = ""
        self._live: Live | None = None

    def __enter__(self):
        self._live = Live(console=console, refresh_per_second=20, vertical_overflow="visible")
        self._live.__enter__()
        return self

    def __exit__(self, *args):
        if self._live:
            # Render the finished buffer one last time as plain markdown
            self._live.__exit__(*args)
            self._live = None

    def write(self, token: str):
        self._buf += token
        if self._live:
            self._live.update(Markdown(self._buf))

    @property
    def text(self) -> str:
        return self._buf


def print_response(text: str):
    """Render a completed response as Markdown."""
    if text.strip():
        console.print(Markdown(text))


def print_tool_call(tool_name: str, tool_input: dict):
    snippet = json.dumps(tool_input, indent=2)
    if len(snippet) > 400:
        snippet = snippet[:400] + "\n…"
    console.print(Panel(
        f"[dim]{snippet}[/]",
        title=f"[bold cyan]⚙  {tool_name}[/]",
        border_style="cyan",
        padding=(0, 1),
    ))


def print_tool_result(tool_name: str, result: str):
    snippet = result[:300] + ("…" if len(result) > 300 else "")
    console.print(f"[dim]  ↳ {tool_name}: {snippet}[/]")


def print_token_usage(prompt: int, completion: int):
    total = prompt + completion
    console.print(
        f"[dim]  tokens: {prompt} in · {completion} out · {total} total[/]",
        highlight=False,
    )


def print_info(msg: str):
    console.print(f"[bold green]✓[/] {msg}")


def print_warning(msg: str):
    console.print(f"[bold yellow]⚠[/]  {msg}")


def print_error(msg: str):
    console.print(f"[bold red]✗[/] {msg}")


def print_sessions_table(sessions: list[dict]):
    if not sessions:
        console.print("[dim]No saved sessions.[/]")
        return
    t = Table(show_header=True, header_style="bold cyan")
    t.add_column("Name")
    t.add_column("Last updated")
    for s in sessions:
        t.add_row(s["name"], s["updated_at"])
    console.print(t)


def print_history_results(results: list[dict]):
    if not results:
        console.print("[dim]No matches.[/]")
        return
    for r in results:
        console.print(
            f"[dim]{r['timestamp']}[/] [bold]{r['session']}[/] [{r['role']}] {r['content'][:120]}"
        )


def print_mcp_status(clients: dict):
    t = Table(show_header=True, header_style="bold cyan")
    t.add_column("Server")
    t.add_column("Status")
    for name in clients:
        t.add_row(name, "[green]connected[/]")
    console.print(t)


async def print_mcp_tools(clients: dict):
    t = Table(show_header=True, header_style="bold cyan")
    t.add_column("Server")
    t.add_column("Tool")
    t.add_column("Description")
    for name, client in clients.items():
        try:
            tools = await client.list_tools()
            for tool in tools:
                t.add_row(name, tool.name, (tool.description or "")[:60])
        except Exception as e:
            t.add_row(name, "[red]error[/]", str(e))
    console.print(t)
