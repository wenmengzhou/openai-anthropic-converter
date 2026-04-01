# OpenAI-Anthropic Protocol Converter

A standalone, bidirectional protocol converter between the OpenAI Chat Completion API and the Anthropic Messages API. Supports both streaming and non-streaming requests/responses.

## Installation

Zero external dependencies (only requires `typing_extensions`). Simply copy the `openai_anthropic_converter/` directory into your project.

## Core API

### Converter 1: `OpenAIToAnthropicConverter`

Converts OpenAI-format requests to Anthropic format for sending, and converts Anthropic responses back to OpenAI format.

```python
from openai_anthropic_converter import OpenAIToAnthropicConverter

# Non-streaming
anthropic_req = OpenAIToAnthropicConverter.convert_request(openai_request)
# ... send to Anthropic API ...
openai_resp = OpenAIToAnthropicConverter.convert_response(anthropic_response)

# Streaming
for chunk in OpenAIToAnthropicConverter.convert_stream(anthropic_sse_events):
    print(chunk)  # OpenAI chat.completion.chunk format

# Async streaming
async for chunk in OpenAIToAnthropicConverter.aconvert_stream(anthropic_sse_events):
    print(chunk)
```

### Converter 2: `AnthropicToOpenAIConverter`

Converts Anthropic `/v1/messages` requests to OpenAI format for forwarding, and converts OpenAI responses back to Anthropic format.

```python
from openai_anthropic_converter import AnthropicToOpenAIConverter

# Non-streaming (note: convert_request returns a tuple)
openai_req, tool_name_mapping = AnthropicToOpenAIConverter.convert_request(anthropic_request)
# ... send to OpenAI-compatible API ...
anthropic_resp = AnthropicToOpenAIConverter.convert_response(
    openai_response, tool_name_mapping=tool_name_mapping
)

# Streaming
for event in AnthropicToOpenAIConverter.convert_stream(
    openai_chunks, tool_name_mapping=tool_name_mapping
):
    # Each event is an Anthropic SSE event dict (message_start, content_block_delta, etc.)
    print(f"event: {event['type']}\ndata: {json.dumps(event)}\n")

# Async streaming
async for event in AnthropicToOpenAIConverter.aconvert_stream(
    openai_chunks, tool_name_mapping=tool_name_mapping
):
    print(event)
```

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
| Message alternation merging | ✅ |

## Design Notes

- **Zero LiteLLM dependency** — pure dict in/out, TypedDicts for type annotations only
- **Streaming via state machines** — `AnthropicSSEToOpenAIStream` and `OpenAIToAnthropicSSEStream` track content block type/index internally
- **Tool name truncation recovery** — `AnthropicToOpenAIConverter.convert_request()` returns `(request, tool_name_mapping)`, pass the mapping to `convert_response()` to restore original names exceeding OpenAI's 64-char limit

## Project Structure

```
openai_anthropic_converter/
├── __init__.py                    # Exports both Converter classes
├── constants.py                   # Constants (stop reason maps, defaults)
├── utils.py                       # Utilities (tool name truncation, schema filtering)
├── types/
│   ├── anthropic_types.py         # Anthropic protocol TypedDicts
│   ├── openai_types.py            # OpenAI protocol TypedDicts
│   └── streaming_types.py         # Streaming event types
├── openai_to_anthropic/
│   ├── converter.py               # OpenAIToAnthropicConverter main class
│   ├── request.py                 # OpenAI request → Anthropic request
│   ├── response.py                # Anthropic response → OpenAI response
│   └── stream.py                  # Anthropic SSE → OpenAI chunks
├── anthropic_to_openai/
│   ├── converter.py               # AnthropicToOpenAIConverter main class
│   ├── request.py                 # Anthropic request → OpenAI request
│   ├── response.py                # OpenAI response → Anthropic response
│   └── stream.py                  # OpenAI chunks → Anthropic SSE
└── tests/
    ├── test_openai_to_anthropic.py
    └── test_anthropic_to_openai.py
```

## Running Tests

```bash
python -m pytest openai_anthropic_converter/tests/ -v
```
