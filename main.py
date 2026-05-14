import argparse
import asyncio
import os
import sys
from dotenv import load_dotenv
from contextlib import AsyncExitStack

from mcp_client import MCPClient
from core.providers.factory import create_provider
from core.session import SessionManager
from core.permissions import PermissionManager
from core.cli_chat import CliChat
from core.cli import CliApp
from core import ui

load_dotenv()


def _parse_args():
    parser = argparse.ArgumentParser(
        description="MCP Chat — multi-provider LLM client",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                          # interactive chat
  python main.py -p openai               # use OpenAI
  python main.py -p gemini -y            # Gemini, auto-approve tools
  cat logs.txt | python main.py "summarize errors"
  git diff | python main.py "review this PR"
        """,
    )
    parser.add_argument(
        "--provider", "-p",
        metavar="NAME",
        help="LLM provider: anthropic | openai | gemini | ollama | openai-compatible | openrouter",
    )
    parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Auto-approve all tool executions (skip permission prompts)",
    )
    parser.add_argument(
        "--server", "-s",
        action="append",
        dest="servers",
        default=[],
        metavar="SCRIPT",
        help="Extra MCP server script to connect (repeatable)",
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        help="One-shot prompt — pipe stdin + this text and exit",
    )
    return parser.parse_args()


async def main():
    args = _parse_args()

    llm_provider = create_provider(provider_name=args.provider)
    session_manager = SessionManager()
    permission_manager = PermissionManager(auto_approve=args.yes)

    command, run_args = (
        ("uv", ["run", "mcp_server.py"])
        if os.getenv("USE_UV", "0") == "1"
        else ("python", ["mcp_server.py"])
    )

    async with AsyncExitStack() as stack:
        doc_client = await stack.enter_async_context(
            MCPClient(command=command, args=run_args)
        )
        clients = {"doc_client": doc_client}

        for i, script in enumerate(args.servers):
            client = await stack.enter_async_context(
                MCPClient(command="uv", args=["run", script])
            )
            clients[f"server_{i}_{script}"] = client

        chat = CliChat(
            doc_client=doc_client,
            clients=clients,
            llm_provider=llm_provider,
            session_manager=session_manager,
            permission_manager=permission_manager,
        )

        # ── piped / one-shot mode ─────────────────────────────────────────────
        pipe_data = sys.stdin.read().strip() if not sys.stdin.isatty() else ""
        if pipe_data or args.prompt:
            query = args.prompt or ""
            if pipe_data:
                query = f"{query}\n\n{pipe_data}".strip()
            response = await chat.run(query)
            # In pipe mode, write plain text to stdout for scripting
            if not sys.stdout.isatty():
                sys.stdout.write(response + "\n")
            return

        # ── interactive mode ──────────────────────────────────────────────────
        cli = CliApp(chat)
        await cli.initialize()
        ui.console.print(
            f"[bold green]MCP Chat[/] · [dim]{llm_provider.__class__.__name__}[/] · "
            f"[dim]type /help for commands, Ctrl+C to exit[/]"
        )
        await cli.run()


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    asyncio.run(main())
