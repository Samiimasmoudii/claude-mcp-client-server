import os
from core.providers.base import LLMProvider
from core.providers.anthropic_provider import AnthropicProvider
from core.providers.openai_provider import OpenAIProvider

_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
_OLLAMA_BASE_URL = "http://localhost:11434/v1"


def create_provider() -> LLMProvider:
    """
    Create an LLM provider from environment variables.

    LLM_PROVIDER controls which backend is used (default: anthropic):
      anthropic       — ANTHROPIC_API_KEY + ANTHROPIC_MODEL (or CLAUDE_MODEL)
      openai          — OPENAI_API_KEY + OPENAI_MODEL (default: gpt-4o)
      gemini          — GEMINI_API_KEY + GEMINI_MODEL (default: gemini-2.0-flash)
      ollama          — no key needed + OLLAMA_MODEL (default: llama3.2)
      openai-compatible — OPENAI_API_KEY + OPENAI_MODEL + OPENAI_BASE_URL
    """
    provider = os.getenv("LLM_PROVIDER", "anthropic").lower()

    if provider == "anthropic":
        model = os.getenv("ANTHROPIC_MODEL") or os.getenv("CLAUDE_MODEL", "")
        assert model, "Set ANTHROPIC_MODEL (or CLAUDE_MODEL) in your .env"
        return AnthropicProvider(model=model)

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY", "")
        model = os.getenv("OPENAI_MODEL", "gpt-4o")
        assert api_key, "Set OPENAI_API_KEY in your .env"
        return OpenAIProvider(model=model, api_key=api_key)

    if provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY", "")
        model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        assert api_key, "Set GEMINI_API_KEY in your .env"
        return OpenAIProvider(model=model, api_key=api_key, base_url=_GEMINI_BASE_URL)

    if provider == "ollama":
        model = os.getenv("OLLAMA_MODEL", "llama3.2")
        return OpenAIProvider(model=model, api_key="ollama", base_url=_OLLAMA_BASE_URL)

    if provider == "openai-compatible":
        api_key = os.getenv("OPENAI_API_KEY", "")
        model = os.getenv("OPENAI_MODEL", "")
        base_url = os.getenv("OPENAI_BASE_URL", "")
        assert api_key, "Set OPENAI_API_KEY in your .env"
        assert model, "Set OPENAI_MODEL in your .env"
        assert base_url, "Set OPENAI_BASE_URL in your .env"
        return OpenAIProvider(model=model, api_key=api_key, base_url=base_url)

    raise ValueError(
        f"Unknown LLM_PROVIDER '{provider}'. "
        "Supported values: anthropic, openai, gemini, ollama, openai-compatible"
    )
