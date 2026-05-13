from core.providers.base import LLMProvider
from mcp_client import MCPClient
from core.tools import ToolManager


class Chat:
    def __init__(self, llm_provider: LLMProvider, clients: dict[str, MCPClient]):
        self.llm_provider: LLMProvider = llm_provider
        self.clients: dict[str, MCPClient] = clients
        self.messages: list = []

    async def _process_query(self, query: str):
        self.messages.append({"role": "user", "content": query})

    async def run(self, query: str) -> str:
        final_text_response = ""

        await self._process_query(query)

        while True:
            response = self.llm_provider.chat(
                messages=self.messages,
                tools=await ToolManager.get_all_tools(self.clients),
            )

            self.llm_provider.add_assistant_message(self.messages, response)

            if response.stop_reason == "tool_use":
                print(self.llm_provider.text_from_message(response))
                tool_result_parts = await ToolManager.execute_tool_requests(
                    self.clients, response
                )
                self.llm_provider.add_user_message(self.messages, tool_result_parts)
            else:
                final_text_response = self.llm_provider.text_from_message(response)
                break

        return final_text_response
