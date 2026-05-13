import argparse
import asyncio
import os
from dotenv import load_dotenv
from contextlib import AsyncExitStack

from mcp_client import MCPClient
from core.providers.factory import create_provider

from core.cli_chat import CliChat
from core.cli import CliApp

load_dotenv()


def _parse_args():
    parser = argparse.ArgumentParser(description="MCP Chat — multi-provider LLM client")
    parser.add_argument(
        "--provider", "-p",
        metavar="NAME",
        help="LLM provider: anthropic | openai | gemini | ollama | openai-compatible",
    )
    parser.add_argument(
        "servers",
        nargs="*",
        metavar="SERVER_SCRIPT",
        help="Extra MCP server scripts to connect to",
    )
    return parser.parse_args()


async def main():
    args = _parse_args()
    llm_provider = create_provider(provider_name=args.provider)

    server_scripts = args.servers
    clients = {}

    command, args = (
        ("uv", ["run", "mcp_server.py"])
        if os.getenv("USE_UV", "0") == "1"
        else ("python", ["mcp_server.py"])
    )

    async with AsyncExitStack() as stack:
        doc_client = await stack.enter_async_context(
            MCPClient(command=command, args=args)
        )
        clients["doc_client"] = doc_client

        for i, server_script in enumerate(server_scripts):
            client_id = f"client_{i}_{server_script}"
            client = await stack.enter_async_context(
                MCPClient(command="uv", args=["run", server_script])
            )
            clients[client_id] = client

        chat = CliChat(
            doc_client=doc_client,
            clients=clients,
            llm_provider=llm_provider,
        )

        cli = CliApp(chat)
        await cli.initialize()
        await cli.run()


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    asyncio.run(main())
