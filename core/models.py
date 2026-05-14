from dataclasses import dataclass, field
from typing import Optional, Any


@dataclass
class ProviderCapabilities:
    streaming: bool = False
    tools: bool = True
    vision: bool = False
    json_mode: bool = False
    system_prompt: bool = True


@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class ToolResult:
    content: str
    artifacts: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    error: Optional[str] = None


# Stream events — yielded by LLMProvider.stream()
@dataclass
class TokenEvent:
    token: str


@dataclass
class FinalEvent:
    response: Any  # LLMResponse; typed as Any to avoid circular import
