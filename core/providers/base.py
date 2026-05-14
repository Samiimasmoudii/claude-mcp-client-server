from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, AsyncGenerator
from core.models import ProviderCapabilities, TokenUsage, TokenEvent, FinalEvent


@dataclass
class TextBlock:
    type: str = "text"
    text: str = ""


@dataclass
class ToolUseBlock:
    type: str = "tool_use"
    id: str = ""
    name: str = ""
    input: dict = field(default_factory=dict)


@dataclass
class LLMResponse:
    stop_reason: str  # "tool_use" | "end_turn"
    content: list  # list[TextBlock | ToolUseBlock]
    usage: TokenUsage = field(default_factory=TokenUsage)


class LLMProvider(ABC):
    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities()

    @abstractmethod
    async def chat(
        self,
        messages: list,
        system: Optional[str] = None,
        temperature: float = 1.0,
        stop_sequences: list = [],
        tools: Optional[list] = None,
    ) -> LLMResponse:
        pass

    async def stream(
        self,
        messages: list,
        system: Optional[str] = None,
        temperature: float = 1.0,
        stop_sequences: list = [],
        tools: Optional[list] = None,
    ) -> AsyncGenerator[TokenEvent | FinalEvent, None]:
        """Default: non-streaming fallback. Override in streaming providers."""
        response = await self.chat(messages, system, temperature, stop_sequences, tools)
        yield TokenEvent(token=self.text_from_message(response))
        yield FinalEvent(response=response)

    def add_user_message(self, messages: list, message):
        messages.append({"role": "user", "content": message})

    def add_assistant_message(self, messages: list, message):
        if isinstance(message, LLMResponse):
            content = []
            for block in message.content:
                if isinstance(block, TextBlock):
                    content.append({"type": "text", "text": block.text})
                elif isinstance(block, ToolUseBlock):
                    content.append({
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    })
            messages.append({"role": "assistant", "content": content})
        else:
            messages.append({"role": "assistant", "content": message})

    def text_from_message(self, response: LLMResponse) -> str:
        return "\n".join(
            block.text for block in response.content if isinstance(block, TextBlock)
        )
