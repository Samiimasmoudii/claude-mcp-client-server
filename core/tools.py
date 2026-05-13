import json
from typing import Optional
from mcp.types import CallToolResult, TextContent
from mcp_client import MCPClient
from core.providers.base import LLMResponse, ToolUseBlock


class ToolManager:
    @classmethod
    async def get_all_tools(cls, clients: dict[str, MCPClient]) -> list:
        tools = []
        for client in clients.values():
            tool_models = await client.list_tools()
            tools += [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.inputSchema,
                }
                for t in tool_models
            ]
        return tools

    @classmethod
    async def _find_client_with_tool(
        cls, clients: list[MCPClient], tool_name: str
    ) -> Optional[MCPClient]:
        for client in clients:
            tools = await client.list_tools()
            if any(t.name == tool_name for t in tools):
                return client
        return None

    @classmethod
    def _build_tool_result(cls, tool_use_id: str, content: str, is_error: bool) -> dict:
        return {
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": content,
            "is_error": is_error,
        }

    @classmethod
    async def execute_tool_requests(
        cls, clients: dict[str, MCPClient], response: LLMResponse
    ) -> list:
        tool_requests = [b for b in response.content if isinstance(b, ToolUseBlock)]
        tool_result_blocks = []

        for tool_request in tool_requests:
            tool_use_id = tool_request.id
            tool_name = tool_request.name
            tool_input = tool_request.input

            client = await cls._find_client_with_tool(list(clients.values()), tool_name)

            if not client:
                tool_result_blocks.append(
                    cls._build_tool_result(tool_use_id, "Could not find that tool", True)
                )
                continue

            tool_output: Optional[CallToolResult] = None
            try:
                tool_output = await client.call_tool(tool_name, tool_input)
                items = tool_output.content if tool_output else []
                content_json = json.dumps(
                    [item.text for item in items if isinstance(item, TextContent)]
                )
                tool_result_blocks.append(
                    cls._build_tool_result(
                        tool_use_id,
                        content_json,
                        bool(tool_output and tool_output.isError),
                    )
                )
            except Exception as e:
                error_message = f"Error executing tool '{tool_name}': {e}"
                print(error_message)
                tool_result_blocks.append(
                    cls._build_tool_result(
                        tool_use_id, json.dumps({"error": error_message}), True
                    )
                )

        return tool_result_blocks
