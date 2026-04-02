# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

A standalone, bidirectional protocol converter between the OpenAI Chat Completion API and the Anthropic Messages API. Two directions:
1. **OpenAI → Anthropic → OpenAI**: Accept OpenAI-format requests, forward to Anthropic backend, convert responses back
2. **Anthropic → OpenAI → Anthropic**: Accept Anthropic-format requests, forward to OpenAI-compatible backend, convert responses back

The core conversion library is zero-dependency (only `typing_extensions`). The optional proxy servers add FastAPI/httpx/uvicorn.

## Development Commands

```bash
# Install
pip install -e .              # Core library only
pip install -e ".[server]"    # With proxy servers
pip install -e ".[dev]"       # With dev tools (pytest, ruff, mypy)

# Test
python -m pytest tests/ -v                          # All tests
python -m pytest tests/test_servers.py -v            # Server tests only
python -m pytest tests/test_openai_to_anthropic.py::TestClassName::test_name -v  # Single test

# Lint (matches CI)
ruff check .                  # Lint
ruff format --check .         # Format check
ruff format .                 # Apply formatting
mypy --ignore-missing-imports --no-strict-optional openai_anthropic_converter/

# Run proxy servers
./start_openai_server.sh      # Port 8001, forwards to Anthropic backend
./start_anthropic_server.sh   # Port 8002, forwards to OpenAI backend
```

## Architecture

### Core Library (`openai_anthropic_converter/`)

**Stateless, dict-in/dict-out converters.** Two facade classes with all-static methods:
- `OpenAIToAnthropicConverter` — delegates to `openai_to_anthropic/{request,response,stream}.py`
- `AnthropicToOpenAIConverter` — delegates to `anthropic_to_openai/{request,response,stream}.py`

**Streaming uses explicit state machines**, not stateless transforms:
- `AnthropicSSEToOpenAIStream` — processes Anthropic event sequence (message_start → content_block_start → content_block_delta → content_block_stop → message_delta → message_stop) and emits OpenAI chunk dicts
- `OpenAIToAnthropicSSEStream` — the harder direction; must detect content type transitions (text→tool_use, thinking→text) and synthesize Anthropic block boundary events that OpenAI doesn't emit

### Proxy Servers (`servers/`)

Two FastAPI apps (`openai_server.py`, `anthropic_server.py`) that handle HTTP forwarding and SSE parsing. Each exposes `/docs`, `/redoc`, `/openapi.json`, `/debug` (interactive playground), and `/health`.

### Types (`types/`)

TypedDicts for both wire formats — documentation/IDE support only, converters operate on raw dicts.

## Key Design Patterns

**Tool name truncation/restoration**: Anthropic→OpenAI direction truncates tool names >64 chars with SHA-256 hash suffix. `convert_request()` returns `(request, tool_name_mapping)` — the mapping must be threaded through to `convert_response()`/`convert_stream()` to restore original names.

**Defensive parameter handling**: Unknown/unsupported parameters are silently dropped via `request.pop()`, not rejected. This is intentional for forward compatibility.

**Dual auth headers**: The OpenAI server sends both `x-api-key` and `Authorization: Bearer` to support both standard Anthropic and DashScope/Bailian backends.

**Bailian/DashScope compatibility**: Shims for Alibaba Cloud's platform are marked with `# [Bailian compat]` comments throughout the codebase (`enable_thinking`, `thinking_budget`, `enable_search`, `reasoning_content` in streaming deltas).

**Schema filtering for Anthropic**: When using native `output_format` (Claude 4.5+), JSON schemas are recursively filtered to remove unsupported fields (minItems, maxItems, etc.) with constraint info moved to descriptions. `$ref`/`$defs` resolution handles circular references.

## Testing

Tests use `httpx.ASGITransport` for in-process FastAPI testing with mocked backends via `unittest.mock.patch`. The mock client is created before patching to avoid interference with the test client.

CI runs lint + test across Python 3.9, 3.11, 3.12.

## Code Style

Ruff with line-length 100, rules `E,F,W,I`, `E501` ignored. All public APIs have type hints.
