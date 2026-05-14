import os
import getpass
from core.providers.base import LLMProvider
from core.providers.anthropic_provider import AnthropicProvider
from core.providers.openai_provider import OpenAIProvider

_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
_OLLAMA_BASE_URL = "http://localhost:11434/v1"

_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

_PROVIDERS = ["anthropic", "openai", "gemini", "ollama", "openrouter", "openai-compatible"]
_DEFAULTS = {
    "anthropic": "claude-sonnet-4-6",
    "openai": "gpt-4o",
    "gemini": "gemini-2.0-flash",
    "ollama": "llama3.2",
    "openrouter": "openai/gpt-4o",
}


def _ask(prompt: str, default: str = "") -> str:
    hint = f" [{default}]" if default else ""
    value = input(f"{prompt}{hint}: ").strip()
    return value or default


def _ask_secret(prompt: str) -> str:
    return getpass.getpass(f"{prompt}: ")


def _select_provider() -> str:
    print("\nSelect a provider:")
    for i, name in enumerate(_PROVIDERS, 1):
        print(f"  {i}. {name}")
    while True:
        raw = input("Enter number or name [1]: ").strip() or "1"
        if raw.lower() in _PROVIDERS:
            return raw.lower()
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(_PROVIDERS):
                return _PROVIDERS[idx]
        except ValueError:
            pass
        print("  Invalid — enter a number or a provider name.")


def create_provider(provider_name: str = None) -> LLMProvider:
    """
    Create an LLM provider. Resolution order for the provider name:
      1. provider_name argument (from --provider CLI flag)
      2. LLM_PROVIDER env var / .env
      3. Interactive prompt

    Missing API keys and models are also prompted interactively.
    """
    provider = (provider_name or os.getenv("LLM_PROVIDER", "")).lower() or _select_provider()

    if provider == "anthropic":
        model = (
            os.getenv("ANTHROPIC_MODEL")
            or os.getenv("CLAUDE_MODEL")
            or _ask("Model", _DEFAULTS["anthropic"])
        )
        api_key = os.getenv("ANTHROPIC_API_KEY") or _ask_secret("Anthropic API key")
        return AnthropicProvider(model=model, api_key=api_key)

    if provider == "openai":
        model = os.getenv("OPENAI_MODEL") or _ask("Model", _DEFAULTS["openai"])
        api_key = os.getenv("OPENAI_API_KEY") or _ask_secret("OpenAI API key")
        return OpenAIProvider(model=model, api_key=api_key)

    if provider == "gemini":
        model = os.getenv("GEMINI_MODEL") or _ask("Model", _DEFAULTS["gemini"])
        api_key = os.getenv("GEMINI_API_KEY") or _ask_secret("Gemini API key")
        return OpenAIProvider(model=model, api_key=api_key, base_url=_GEMINI_BASE_URL)

    if provider == "ollama":
        model = os.getenv("OLLAMA_MODEL") or _ask("Model", _DEFAULTS["ollama"])
        return OpenAIProvider(model=model, api_key="ollama", base_url=_OLLAMA_BASE_URL)

    if provider == "openrouter":
        model = os.getenv("OPENROUTER_MODEL") or _ask("Model", _DEFAULTS["openrouter"])
        api_key = os.getenv("OPENROUTER_API_KEY") or _ask_secret("OpenRouter API key")
        return OpenAIProvider(model=model, api_key=api_key, base_url=_OPENROUTER_BASE_URL)

    if provider == "openai-compatible":
        base_url = os.getenv("OPENAI_BASE_URL") or _ask("Base URL")
        model = os.getenv("OPENAI_MODEL") or _ask("Model")
        api_key = os.getenv("OPENAI_API_KEY") or _ask_secret("API key")
        return OpenAIProvider(model=model, api_key=api_key, base_url=base_url)

    raise ValueError(
        f"Unknown provider '{provider}'. "
        f"Supported: {', '.join(_PROVIDERS)}"
    )
