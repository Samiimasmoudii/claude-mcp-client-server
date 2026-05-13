import json
from typing import Optional
from openai import OpenAI
from core.providers.base import LLMProvider, LLMResponse, TextBlock, ToolUseBlock


class OpenAIProvider(LLMProvider):
    def __init__(self, model: str, api_key: str, base_url: Optional[str] = None):
        self.model = model
        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self.client = OpenAI(**kwargs)

    def _convert_messages(self, messages: list) -> list:
        """Convert normalized message format to OpenAI wire format."""
        result = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == "user" and isinstance(content, list):
                tool_results = [b for b in content if isinstance(b, dict) and b.get("type") == "tool_result"]
                if tool_results:
                    for tr in tool_results:
                        c = tr["content"]
                        result.append({
                            "role": "tool",
                            "tool_call_id": tr["tool_use_id"],
                            "content": c if isinstance(c, str) else json.dumps(c),
                        })
                    continue

            if role == "assistant" and isinstance(content, list):
                text_parts = [b for b in content if isinstance(b, dict) and b.get("type") == "text"]
                tool_use_parts = [b for b in content if isinstance(b, dict) and b.get("type") == "tool_use"]

                text = "\n".join(b["text"] for b in text_parts) or None
                tool_calls = [
                    {
                        "id": b["id"],
                        "type": "function",
                        "function": {"name": b["name"], "arguments": json.dumps(b["input"])},
                    }
                    for b in tool_use_parts
                ] or None

                oai_msg = {"role": "assistant", "content": text}
                if tool_calls:
                    oai_msg["tool_calls"] = tool_calls
                result.append(oai_msg)
                continue

            result.append({
                "role": role,
                "content": content if isinstance(content, str) else json.dumps(content),
            })

        return result

    def _convert_tools(self, tools: list) -> list:
        """Convert MCP tool definitions to OpenAI function-calling format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t["input_schema"],
                },
            }
            for t in tools
        ]

    def chat(
        self,
        messages: list,
        system: Optional[str] = None,
        temperature: float = 1.0,
        stop_sequences: list = [],
        tools: Optional[list] = None,
    ) -> LLMResponse:
        oai_messages = []
        if system:
            oai_messages.append({"role": "system", "content": system})
        oai_messages += self._convert_messages(messages)

        params = {
            "model": self.model,
            "messages": oai_messages,
            "temperature": temperature,
            "max_tokens": 8000,
        }
        if stop_sequences:
            params["stop"] = stop_sequences
        if tools:
            params["tools"] = self._convert_tools(tools)

        raw = self.client.chat.completions.create(**params)
        choice = raw.choices[0]
        msg = choice.message

        content = []
        if msg.content:
            content.append(TextBlock(text=msg.content))
        if msg.tool_calls:
            for tc in msg.tool_calls:
                content.append(ToolUseBlock(
                    id=tc.id,
                    name=tc.function.name,
                    input=json.loads(tc.function.arguments),
                ))

        stop_reason = "tool_use" if choice.finish_reason == "tool_calls" else "end_turn"
        return LLMResponse(stop_reason=stop_reason, content=content)
