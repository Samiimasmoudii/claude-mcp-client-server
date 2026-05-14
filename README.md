# MCP Chat

A command-line chat client that connects to any LLM — Claude, GPT-4o, Gemini, Ollama, OpenRouter, or any OpenAI-compatible endpoint — via the [MCP (Model Context Protocol)](https://github.com/modelcontextprotocol) architecture. Supports streaming responses, document retrieval, persistent sessions, slash-command prompts, and pluggable tool servers.

## Prerequisites

- Python 3.10+
- An API key for your chosen provider (**not needed for Ollama**)

## Setup

### 1. Install dependencies

**With uv (recommended)**

```bash
pip install uv
uv venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
uv pip install -e .
```

For OpenAI / Gemini / Ollama / OpenRouter, also install the OpenAI SDK:

```bash
uv pip install openai
```

**With pip**

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
pip install openai   # if using OpenAI / Gemini / Ollama / OpenRouter
```

### 2. Configure your provider

Copy `.env.example` to `.env` and fill in the section for your chosen provider:

```bash
cp .env.example .env   # Windows: copy .env.example .env
```

| Provider | `.env` variables | Default model |
|---|---|---|
| **Anthropic** (Claude) | `ANTHROPIC_API_KEY`, `ANTHROPIC_MODEL` | `claude-sonnet-4-6` |
| **OpenAI** (GPT) | `OPENAI_API_KEY`, `OPENAI_MODEL` | `gpt-4o` |
| **Google Gemini** | `GEMINI_API_KEY`, `GEMINI_MODEL` | `gemini-2.0-flash` |
| **Ollama** (local, free) | `OLLAMA_MODEL` — **no key** | `llama3.2` |
| **OpenRouter** | `OPENROUTER_API_KEY`, `OPENROUTER_MODEL` | `openai/gpt-4o` |
| **OpenAI-compatible** | `OPENAI_API_KEY`, `OPENAI_MODEL`, `OPENAI_BASE_URL` | — |

Set `LLM_PROVIDER` in `.env` to the matching value.

> **Skip the .env entirely:** if `LLM_PROVIDER` is not set the app will ask you to pick a provider and enter your API key interactively at startup.

---

### Provider quick-start guides

#### Anthropic (Claude)

1. Get an API key at <https://console.anthropic.com>
2. Set in `.env`:
   ```
   LLM_PROVIDER=anthropic
   ANTHROPIC_API_KEY=sk-ant-...
   ANTHROPIC_MODEL=claude-sonnet-4-6
   ```

#### OpenAI (GPT)

1. Get an API key at <https://platform.openai.com>
2. Set in `.env`:
   ```
   LLM_PROVIDER=openai
   OPENAI_API_KEY=sk-...
   OPENAI_MODEL=gpt-4o
   ```

#### Google Gemini

1. Get an API key at <https://aistudio.google.com/apikey>
2. Set in `.env`:
   ```
   LLM_PROVIDER=gemini
   GEMINI_API_KEY=...
   GEMINI_MODEL=gemini-2.0-flash
   ```

#### Ollama (local, no cost, no internet)

Ollama runs models entirely on your machine. You must have it **installed and running** before starting MCP Chat.

1. **Install Ollama** — download from <https://ollama.com/download> (Windows/Mac/Linux)

2. **Start the Ollama server** (it must keep running in a separate terminal):
   ```bash
   ollama serve
   ```
   You should see: `Ollama is running on 127.0.0.1:11434`

3. **Pull the model** you want to use (one-time download):
   ```bash
   ollama pull llama3.2        # ~2 GB, good all-rounder
   ollama pull mistral         # ~4 GB, stronger reasoning
   ollama pull codellama       # ~4 GB, code-focused
   ollama list                 # see what you have
   ```

4. **Set in `.env`**:
   ```
   LLM_PROVIDER=ollama
   OLLAMA_MODEL=llama3.2
   ```

> **Common error:** `Connection refused` / `WinError 10061` means Ollama isn't running.
> Fix: open a terminal and run `ollama serve`, then retry.

#### OpenRouter (100+ models, one API key)

1. Get an API key at <https://openrouter.ai/keys>
2. Set in `.env`:
   ```
   LLM_PROVIDER=openrouter
   OPENROUTER_API_KEY=sk-or-...
   OPENROUTER_MODEL=openai/gpt-4o    # or anthropic/claude-3-5-sonnet, meta-llama/llama-3.1-8b-instruct, etc.
   ```
   Browse available models at <https://openrouter.ai/models>

---

## Running

```bash
# Use the provider configured in .env
uv run main.py

# Override the provider on the command line (prompts for key if not in .env)
uv run main.py --provider ollama
uv run main.py -p openai

# Auto-approve all tool executions (useful for scripting)
uv run main.py -p anthropic --yes

# Connect extra MCP tool servers
uv run main.py --server my_tools.py
```

Without uv, replace `uv run` with `python`.

## Shell piping

MCP Chat reads from stdin when it's piped, making it easy to use in scripts:

```bash
cat logs.txt | uv run main.py "summarize the errors"
git diff | uv run main.py "review this PR"
cat report.md | uv run main.py "extract action items as JSON"
```

The response is written to stdout as plain text, so you can chain it further:

```bash
git diff | uv run main.py "write a commit message" | pbcopy
```

## Usage

### Chat

Type any message and press Enter:

```
> What is the MCP protocol?
```

Responses are streamed live and rendered as Markdown (on providers that support streaming).

### Document retrieval

Use `@` to inject a document from the MCP server into your query:

```
> Summarise @report.md
> Compare @spec.md and @impl.md
```

### Slash commands

| Command | Description |
|---|---|
| `/help` | Show all commands |
| `/clear` | Clear the current conversation |
| `/session list` | List saved sessions |
| `/session save <name>` | Save the current conversation |
| `/session load <name>` | Resume a saved conversation |
| `/session delete <name>` | Delete a saved session |
| `/history search <query>` | Full-text search across all sessions |
| `/mcp list` | Show connected MCP servers |
| `/mcp tools` | Show all available tools |
| `/<prompt> <doc>` | Run an MCP-defined prompt template |

Tab-completion works for `@` document IDs and `/` commands. Press `Ctrl+C` to exit.

### Tool permissions

Before any MCP tool runs, you are prompted:

```
  Allow? [y] once  [a] always  [n] deny:
```

- **y** — run this once
- **a** — always allow this tool for the session
- **n** — deny and tell the model it was blocked

Use `--yes` / `-y` to skip all prompts (auto-approve everything).

## Development

### Adding documents

Edit the `docs` dictionary in `mcp_server.py`.

### Adding a new LLM provider

1. Subclass `LLMProvider` from `core/providers/base.py`
2. Implement `async chat()` and optionally override `async stream()`
3. Register the new name in `core/providers/factory.py`

### Connecting additional tool servers

```bash
uv run main.py --server my_tools.py --server another_server.py
```

Sessions are stored at `~/.mcp-chat/sessions.db` (SQLite).
