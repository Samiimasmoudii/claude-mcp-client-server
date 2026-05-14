from typing import Optional
from anthropic import AsyncAnthropic
from core.models import ProviderCapabilities, TokenUsage, TokenEvent, FinalEvent
from core.providers.base import LLMProvider, LLMResponse, TextBlock, ToolUseBlock


class AnthropicProvider(LLMProvider):
    def __init__(self, model: str, api_key: Optional[str] = None):
        self.client = AsyncAnthropic(api_key=api_key) if api_key else AsyncAnthropic()
        self.model = model

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(streaming=True, tools=True, system_prompt=True)

    def _build_params(self, messages, system, temperature, stop_sequences, tools) -> dict:
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
        return params

    def _convert_raw(self, raw) -> LLMResponse:
        content = []
        for block in raw.content:
            if block.type == "text":
                content.append(TextBlock(text=block.text))
            elif block.type == "tool_use":
                content.append(ToolUseBlock(id=block.id, name=block.name, input=block.input))
        usage = TokenUsage(
            prompt_tokens=raw.usage.input_tokens,
            completion_tokens=raw.usage.output_tokens,
        )
        stop_reason = "tool_use" if raw.stop_reason == "tool_use" else "end_turn"
        return LLMResponse(stop_reason=stop_reason, content=content, usage=usage)

    async def chat(
        self,
        messages: list,
        system: Optional[str] = None,
        temperature: float = 1.0,
        stop_sequences: list = [],
        tools: Optional[list] = None,
    ) -> LLMResponse:
        raw = await self.client.messages.create(**self._build_params(messages, system, temperature, stop_sequences, tools))
        return self._convert_raw(raw)

    async def stream(self, messages, system=None, temperature=1.0, stop_sequences=[], tools=None):
        async with self.client.messages.stream(**self._build_params(messages, system, temperature, stop_sequences, tools)) as s:
            async for text in s.text_stream:
                yield TokenEvent(token=text)
            yield FinalEvent(response=self._convert_raw(await s.get_final_message()))
