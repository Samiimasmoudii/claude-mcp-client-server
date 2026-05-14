import json
from typing import Optional
from openai import AsyncOpenAI
from core.models import ProviderCapabilities, TokenUsage, TokenEvent, FinalEvent
from core.providers.base import LLMProvider, LLMResponse, TextBlock, ToolUseBlock


class OpenAIProvider(LLMProvider):
    def __init__(self, model: str, api_key: str, base_url: Optional[str] = None):
        self.model = model
        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self.client = AsyncOpenAI(**kwargs)

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(streaming=True, tools=True, system_prompt=True)

    def _convert_messages(self, messages: list) -> list:
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

    def _build_params(self, messages, system, temperature, stop_sequences, tools) -> dict:
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
        return params

    async def chat(
        self,
        messages: list,
        system: Optional[str] = None,
        temperature: float = 1.0,
        stop_sequences: list = [],
        tools: Optional[list] = None,
    ) -> LLMResponse:
        raw = await self.client.chat.completions.create(**self._build_params(messages, system, temperature, stop_sequences, tools))
        choice = raw.choices[0]
        msg = choice.message
        content = []
        if msg.content:
            content.append(TextBlock(text=msg.content))
        if msg.tool_calls:
            for tc in msg.tool_calls:
                content.append(ToolUseBlock(id=tc.id, name=tc.function.name, input=json.loads(tc.function.arguments)))
        usage = TokenUsage(
            prompt_tokens=raw.usage.prompt_tokens if raw.usage else 0,
            completion_tokens=raw.usage.completion_tokens if raw.usage else 0,
        )
        stop_reason = "tool_use" if choice.finish_reason == "tool_calls" else "end_turn"
        return LLMResponse(stop_reason=stop_reason, content=content, usage=usage)

    async def stream(self, messages, system=None, temperature=1.0, stop_sequences=[], tools=None):
        params = self._build_params(messages, system, temperature, stop_sequences, tools)
        params["stream"] = True
        params["stream_options"] = {"include_usage": True}

        text = ""
        tc_by_idx: dict[int, dict] = {}
        finish_reason = None
        usage = TokenUsage()

        response = await self.client.chat.completions.create(**params)
        async for chunk in response:
            if chunk.usage:
                usage = TokenUsage(
                    prompt_tokens=chunk.usage.prompt_tokens,
                    completion_tokens=chunk.usage.completion_tokens,
                )
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            if choice.finish_reason:
                finish_reason = choice.finish_reason
            delta = choice.delta
            if delta.content:
                text += delta.content
                yield TokenEvent(token=delta.content)
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in tc_by_idx:
                        tc_by_idx[idx] = {"id": "", "name": "", "arguments": ""}
                    if tc.id:
                        tc_by_idx[idx]["id"] = tc.id
                    if tc.function and tc.function.name:
                        tc_by_idx[idx]["name"] = tc.function.name
                    if tc.function and tc.function.arguments:
                        tc_by_idx[idx]["arguments"] += tc.function.arguments

        content = []
        if text:
            content.append(TextBlock(text=text))
        for idx in sorted(tc_by_idx):
            tc = tc_by_idx[idx]
            try:
                tool_input = json.loads(tc["arguments"] or "{}")
            except json.JSONDecodeError:
                tool_input = {}
            content.append(ToolUseBlock(id=tc["id"], name=tc["name"], input=tool_input))

        stop_reason = "tool_use" if finish_reason == "tool_calls" else "end_turn"
        yield FinalEvent(response=LLMResponse(stop_reason=stop_reason, content=content, usage=usage))
