# OpenAI ↔ Anthropic Protocol Converter

English | [中文](README_zh.md)

OpenAI and Anthropic define the two dominant LLM API protocols today. Many model providers only support one; many clients and frameworks only speak the other. This project bridges that gap with **complete, bidirectional** protocol conversion — any OpenAI client can call an Anthropic backend, and vice versa.

**Design philosophy:** The core library is a pure conversion layer — **stateless, zero-dependency, dict-in/dict-out**. No HTTP clients, no framework coupling, no global state. Conversion functions take a plain dict and return a plain dict. This makes it trivially embeddable in any Python application, testable without mocking, and composable with any transport layer. The optional proxy servers are a thin convenience layer on top, not the other way around.

---

## Two Ways to Use

### 1. Library API — Embed in Your Application

Import the converter as a library and perform protocol conversion in your own code. Best when you need **full control over request/response handling**.

```bash
pip install -e .
```

**OpenAI format → Anthropic format (calling an Anthropic backend)**

```python
from openai_anthropic_converter import OpenAIToAnthropicConverter

# Convert request
anthropic_req = OpenAIToAnthropicConverter.convert_request(openai_request)
# ... send to Anthropic API ...
openai_resp = OpenAIToAnthropicConverter.convert_response(anthropic_response)

# Streaming
async for chunk in OpenAIToAnthropicConverter.aconvert_stream(anthropic_sse_events):
    print(chunk)  # OpenAI chat.completion.chunk format
```

**Anthropic format → OpenAI format (calling an OpenAI backend)**

```python
import json
from openai_anthropic_converter import AnthropicToOpenAIConverter

# Convert request (returns a tuple with tool name mapping)
openai_req, tool_name_mapping = AnthropicToOpenAIConverter.convert_request(anthropic_request)
# ... send to OpenAI API ...
anthropic_resp = AnthropicToOpenAIConverter.convert_response(
    openai_response, tool_name_mapping=tool_name_mapping
)

# Async streaming
async for event in AnthropicToOpenAIConverter.aconvert_stream(
    openai_chunks, tool_name_mapping=tool_name_mapping
):
    print(f"event: {event['type']}\ndata: {json.dumps(event)}\n")
```

> **Note:** `AnthropicToOpenAIConverter.convert_request()` returns `(request, tool_name_mapping)`. OpenAI enforces a 64-character tool name limit — the converter auto-truncates long names with a SHA-256 hash suffix and builds a reverse mapping. Pass this mapping to `convert_response()` or `aconvert_stream()` to restore the original names.

---

### 2. HTTP Proxy Server — Drop-in Replacement

Launch a proxy server that exposes one protocol's interface and auto-forwards to a backend speaking the other. Best when you want to **use existing clients without any code changes**.

```bash
pip install -e ".[server]"
```

**OpenAI-compatible server** (makes an Anthropic backend look like OpenAI)

```bash
# Start the server
./start_openai_server.sh
# Or with explicit args:
python -m openai_anthropic_converter.servers.openai_server \
    --backend-url https://api.anthropic.com/v1 \
    --backend-api-key $ANTHROPIC_API_KEY \
    --port 8001

# Call it with any OpenAI client
curl http://localhost:8001/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"claude-sonnet-4-20250514","messages":[{"role":"user","content":"Hello!"}],"max_tokens":1024}'
```

**Anthropic-compatible server** (makes an OpenAI backend look like Anthropic)

```bash
# Start the server
./start_anthropic_server.sh
# Or with explicit args:
python -m openai_anthropic_converter.servers.anthropic_server \
    --backend-url https://api.openai.com/v1 \
    --backend-api-key $OPENAI_API_KEY \
    --port 8002

# Call it with any Anthropic client
curl http://localhost:8002/v1/messages \
    -H "Content-Type: application/json" \
    -H "x-api-key: anything" \
    -H "anthropic-version: 2023-06-01" \
    -d '{"model":"gpt-4o","messages":[{"role":"user","content":"Hello!"}],"max_tokens":1024}'
```

Both servers provide: Swagger UI (`/docs`), ReDoc (`/redoc`), OpenAPI JSON (`/openapi.json`), interactive debug playground (`/debug`), and health check (`/health`).

Configuration is auto-loaded from `.env` and can be overridden with environment variables or CLI args.

---

## Feature Coverage

| Feature | Supported |
|---------|-----------|
| Text messages | ✅ |
| System messages | ✅ (inline ↔ separate param) |
| Tool/Function calling | ✅ (definitions + invocations + results) |
| Tool name >64 char truncation & restoration | ✅ |
| Images (base64 + URL) | ✅ |
| Thinking/Reasoning | ✅ (thinking ↔ reasoning_effort) |
| Streaming | ✅ (sync + async) |
| Usage / cache token stats | ✅ |
| response_format ↔ output_format | ✅ |
| stop ↔ stop_sequences | ✅ |
| context_management | ✅ |
| Message alternation merging | ✅ (Anthropic requires strict user/assistant alternation) |

## Design Highlights

- **Zero external dependencies** — core library is pure dict in/out; TypedDicts are for type annotations only
- **Streaming via state machines** — `AnthropicSSEToOpenAIStream` / `OpenAIToAnthropicSSEStream` track content block type and index internally
- **Forward-compatible** — unknown parameters are silently dropped, not rejected

## Installation & Development

```bash
pip install -e .              # Core library only
pip install -e ".[server]"    # With proxy servers
pip install -e ".[dev]"       # With dev tools (pytest, ruff, mypy)

python -m pytest tests/ -v    # Run tests
ruff check . && ruff format --check .    # Lint
mypy --ignore-missing-imports openai_anthropic_converter/
```

## Project Structure

```
openai_anthropic_converter/
├── __init__.py                    # Exports both Converter classes
├── constants.py                   # Constants (stop reason maps, model lists)
├── utils.py                       # Utilities (name truncation, JSON schema filtering)
├── types/                         # TypedDict definitions
├── openai_to_anthropic/           # Direction 1: OpenAI → Anthropic → OpenAI
│   ├── converter.py               # OpenAIToAnthropicConverter facade
│   ├── request.py                 # Request conversion
│   ├── response.py                # Response conversion
│   └── stream.py                  # Streaming conversion (state machine)
├── anthropic_to_openai/           # Direction 2: Anthropic → OpenAI → Anthropic
│   ├── converter.py               # AnthropicToOpenAIConverter facade
│   ├── request.py                 # Request conversion
│   ├── response.py                # Response conversion
│   └── stream.py                  # Streaming conversion (state machine)
└── servers/                       # HTTP proxy servers (requires [server] extras)
    ├── openai_server.py           # /v1/chat/completions endpoint
    ├── anthropic_server.py        # /v1/messages endpoint
    ├── schemas.py                 # Pydantic models (OpenAPI docs)
    └── debug_page.py              # Interactive debug playground
```
