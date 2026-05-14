from typing import Optional
from core.providers.base import LLMProvider
from core.models import TokenEvent, FinalEvent
from core.tools import ToolManager
from core.session import SessionManager
from core.permissions import PermissionManager
from core import ui
from mcp_client import MCPClient


class Chat:
    def __init__(
        self,
        llm_provider: LLMProvider,
        clients: dict[str, MCPClient],
        session_manager: Optional[SessionManager] = None,
        permission_manager: Optional[PermissionManager] = None,
    ):
        self.llm_provider = llm_provider
        self.clients = clients
        self.session_manager = session_manager
        self.permission_manager = permission_manager
        self.messages: list = []
        self.session_name: str = "default"

    async def _process_query(self, query: str):
        self.messages.append({"role": "user", "content": query})

    async def run(self, query: str) -> str:
        await self._process_query(query)

        if self.session_manager:
            self.session_manager.log(self.session_name, "user", query)

        tools = await ToolManager.get_all_tools(self.clients)
        final_text = ""

        while True:
            response = None

            if self.llm_provider.capabilities.streaming:
                with ui.StreamRenderer() as renderer:
                    async for event in self.llm_provider.stream(
                        messages=self.messages, tools=tools
                    ):
                        if isinstance(event, TokenEvent):
                            renderer.write(event.token)
                        elif isinstance(event, FinalEvent):
                            response = event.response
            else:
                response = await self.llm_provider.chat(
                    messages=self.messages, tools=tools
                )
                text = self.llm_provider.text_from_message(response)
                if text:
                    ui.print_response(text)

            if response is None:
                break

            # Show token usage
            if response.usage.total > 0:
                ui.print_token_usage(response.usage.prompt_tokens, response.usage.completion_tokens)

            self.llm_provider.add_assistant_message(self.messages, response)

            if response.stop_reason == "tool_use":
                # Show any reasoning text before tool calls
                reasoning = self.llm_provider.text_from_message(response)
                if reasoning and not self.llm_provider.capabilities.streaming:
                    ui.print_response(reasoning)

                tool_result_parts = await ToolManager.execute_tool_requests(
                    self.clients, response, self.permission_manager
                )
                self.llm_provider.add_user_message(self.messages, tool_result_parts)
            else:
                final_text = self.llm_provider.text_from_message(response)
                break

        if self.session_manager and final_text:
            self.session_manager.log(self.session_name, "assistant", final_text)

        return final_text
