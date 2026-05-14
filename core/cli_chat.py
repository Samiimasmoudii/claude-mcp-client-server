from typing import List, Tuple
from mcp.types import Prompt, PromptMessage

from core.chat import Chat
from core.providers.base import LLMProvider
from core.session import SessionManager
from core.permissions import PermissionManager
from core import ui
from mcp_client import MCPClient


class CliChat(Chat):
    def __init__(
        self,
        doc_client: MCPClient,
        clients: dict[str, MCPClient],
        llm_provider: LLMProvider,
        session_manager: SessionManager | None = None,
        permission_manager: PermissionManager | None = None,
    ):
        super().__init__(
            clients=clients,
            llm_provider=llm_provider,
            session_manager=session_manager,
            permission_manager=permission_manager,
        )
        self.doc_client: MCPClient = doc_client

    # ── MCP helpers ───────────────────────────────────────────────────────────

    async def list_prompts(self) -> list[Prompt]:
        return await self.doc_client.list_prompts()

    async def list_docs_ids(self) -> list[str]:
        return await self.doc_client.read_resource("docs://documents")

    async def get_doc_content(self, doc_id: str) -> str:
        return await self.doc_client.read_resource(f"docs://documents/{doc_id}")

    async def get_prompt(self, command: str, doc_id: str) -> list[PromptMessage]:
        return await self.doc_client.get_prompt(command, {"doc_id": doc_id})

    # ── @ resource injection ──────────────────────────────────────────────────

    async def _extract_resources(self, query: str) -> str:
        mentions = [word[1:] for word in query.split() if word.startswith("@")]
        doc_ids = await self.list_docs_ids()
        mentioned: list[Tuple[str, str]] = []
        for doc_id in doc_ids:
            if doc_id in mentions:
                content = await self.get_doc_content(doc_id)
                mentioned.append((doc_id, content))
        return "".join(
            f'\n<document id="{doc_id}">\n{content}\n</document>\n'
            for doc_id, content in mentioned
        )

    # ── built-in command dispatch ─────────────────────────────────────────────

    async def _process_command(self, query: str) -> bool:
        if not query.startswith("/"):
            return False

        parts = query.split()
        cmd = parts[0].lstrip("/").lower()

        # /session save|load|list|delete
        if cmd == "session":
            await self._handle_session(parts[1:])
            return True

        # /history search <query>
        if cmd == "history":
            await self._handle_history(parts[1:])
            return True

        # /mcp list|tools
        if cmd == "mcp":
            await self._handle_mcp(parts[1:])
            return True

        # /clear
        if cmd == "clear":
            self.messages.clear()
            ui.print_info("Conversation cleared.")
            return True

        # /help
        if cmd == "help":
            self._print_help()
            return True

        # MCP-defined prompt commands: /summarize doc.md etc.
        if len(parts) >= 2:
            prompt_messages = await self.doc_client.get_prompt(cmd, {"doc_id": parts[1]})
            self.messages += convert_prompt_messages_to_message_params(prompt_messages)
            return True

        return False

    async def _handle_session(self, args: list[str]):
        if not self.session_manager:
            ui.print_warning("Session persistence is disabled (no session manager).")
            return
        sub = args[0].lower() if args else "list"

        if sub == "save":
            name = args[1] if len(args) > 1 else self.session_name
            self.session_manager.save(name, self.messages)
            self.session_name = name
            ui.print_info(f"Session saved as '{name}'.")

        elif sub == "load":
            if len(args) < 2:
                ui.print_error("Usage: /session load <name>")
                return
            name = args[1]
            loaded = self.session_manager.load(name)
            if loaded is None:
                ui.print_error(f"Session '{name}' not found.")
            else:
                self.messages = loaded
                self.session_name = name
                ui.print_info(f"Session '{name}' loaded ({len(loaded)} messages).")

        elif sub == "delete":
            if len(args) < 2:
                ui.print_error("Usage: /session delete <name>")
                return
            self.session_manager.delete(args[1])
            ui.print_info(f"Session '{args[1]}' deleted.")

        else:  # list
            ui.print_sessions_table(self.session_manager.list_sessions())

    async def _handle_history(self, args: list[str]):
        if not self.session_manager:
            ui.print_warning("Session persistence is disabled.")
            return
        sub = args[0].lower() if args else ""
        if sub == "search" and len(args) > 1:
            results = self.session_manager.search(" ".join(args[1:]))
            ui.print_history_results(results)
        else:
            ui.print_error("Usage: /history search <query>")

    async def _handle_mcp(self, args: list[str]):
        sub = args[0].lower() if args else "list"
        if sub == "tools":
            await ui.print_mcp_tools(self.clients)
        else:  # list / status
            ui.print_mcp_status(self.clients)

    def _print_help(self):
        from rich.table import Table
        t = ui.console.print
        table = Table(show_header=True, header_style="bold cyan", title="Available commands")
        table.add_column("Command", style="bold")
        table.add_column("Description")
        rows = [
            ("/help", "Show this help"),
            ("/clear", "Clear the current conversation"),
            ("/session list", "List saved sessions"),
            ("/session save [name]", "Save current session"),
            ("/session load <name>", "Load a saved session"),
            ("/session delete <name>", "Delete a saved session"),
            ("/history search <q>", "Search message history"),
            ("/mcp list", "Show connected MCP servers"),
            ("/mcp tools", "Show all available tools"),
            ("@<doc_id>", "Inject a document into the query"),
            ("/<prompt> <doc>", "Run an MCP-defined prompt"),
        ]
        for cmd, desc in rows:
            table.add_row(cmd, desc)
        ui.console.print(table)

    # ── query processing ──────────────────────────────────────────────────────

    async def _process_query(self, query: str):
        if await self._process_command(query):
            return

        added_resources = await self._extract_resources(query)

        prompt = f"""The user has a question:
<query>
{query}
</query>

The following context may be useful in answering their question:
<context>
{added_resources}
</context>

Note: references like "@report.docx" are document mentions — the actual name is "report.docx".
If the document content is included above, you don't need to read it with a tool.
Answer directly and concisely. Don't refer to the context block itself."""

        self.messages.append({"role": "user", "content": prompt})


# ── helpers ───────────────────────────────────────────────────────────────────

def convert_prompt_message_to_message_param(prompt_message: "PromptMessage") -> dict:
    role = "user" if prompt_message.role == "user" else "assistant"
    content = prompt_message.content
    if isinstance(content, dict) or hasattr(content, "__dict__"):
        content_type = content.get("type") if isinstance(content, dict) else getattr(content, "type", None)
        if content_type == "text":
            text = content.get("text", "") if isinstance(content, dict) else getattr(content, "text", "")
            return {"role": role, "content": text}
    if isinstance(content, list):
        text_blocks = []
        for item in content:
            item_type = item.get("type") if isinstance(item, dict) else getattr(item, "type", None)
            if item_type == "text":
                text = item.get("text", "") if isinstance(item, dict) else getattr(item, "text", "")
                text_blocks.append({"type": "text", "text": text})
        if text_blocks:
            return {"role": role, "content": text_blocks}
    return {"role": role, "content": ""}


def convert_prompt_messages_to_message_params(prompt_messages: List[PromptMessage]) -> List[dict]:
    return [convert_prompt_message_to_message_param(m) for m in prompt_messages]
