from typing import Optional
from anthropic import Anthropic
from core.providers.base import LLMProvider, LLMResponse, TextBlock, ToolUseBlock


class AnthropicProvider(LLMProvider):
    def __init__(self, model: str):
        self.client = Anthropic()
        self.model = model

    def chat(
        self,
        messages: list,
        system: Optional[str] = None,
        temperature: float = 1.0,
        stop_sequences: list = [],
        tools: Optional[list] = None,
    ) -> LLMResponse:
        params = {
            "model": self.model,
            "max_tokens": 8000,
            "messages": messages,
            "temperature": temperature,
            "stop_sequences": stop_sequences,
        }
        if tools:
            params["tools"] = tools
        if system:
            params["system"] = system

        raw = self.client.messages.create(**params)

        content = []
        for block in raw.content:
            if block.type == "text":
                content.append(TextBlock(text=block.text))
            elif block.type == "tool_use":
                content.append(ToolUseBlock(id=block.id, name=block.name, input=block.input))

        stop_reason = "tool_use" if raw.stop_reason == "tool_use" else "end_turn"
        return LLMResponse(stop_reason=stop_reason, content=content)
