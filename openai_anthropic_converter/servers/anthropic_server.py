"""
Anthropic-compatible API server.

Exposes an Anthropic /v1/messages endpoint that forwards requests
to an OpenAI ChatCompletion API backend, converting protocols in both directions.

Features:
    - Swagger UI at /docs, ReDoc at /redoc, OpenAPI JSON at /openapi.json
    - Interactive debug playground at /debug
    - Full SSE streaming and non-streaming support
    - Token counting endpoint (stub)

Usage:
    # Forward to OpenAI API:
    python -m openai_anthropic_converter.servers.anthropic_server \
        --backend-url https://api.openai.com/v1 \
        --backend-api-key $OPENAI_API_KEY \
        --port 8002

    # Then call it as if it were an Anthropic API:
    curl http://localhost:8002/v1/messages \
        -H "Content-Type: application/json" \
        -H "x-api-key: anything" \
        -H "anthropic-version: 2023-06-01" \
        -d '{"model":"gpt-4o","messages":[{"role":"user","content":"Hello!"}],"max_tokens":1024}'
"""

import argparse
import json
import logging
import os
import sys
from typing import Any, AsyncIterator, Dict

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import httpx as _httpx
import openai
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse

from openai_anthropic_converter import AnthropicToOpenAIConverter

from .schemas import (
    AnthropicCountTokensRequest,
    AnthropicCountTokensResponse,
    AnthropicErrorResponse,
    AnthropicMessagesRequest,
    AnthropicMessagesResponse,
    HealthResponse,
)


def _custom_openapi():
    """Inject request body schemas into the OpenAPI spec."""
    if app.openapi_schema:
        return app.openapi_schema
    from fastapi.openapi.utils import get_openapi

    schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    components = schema.setdefault("components", {})
    schemas = components.setdefault("schemas", {})

    # Add request schemas
    for model_cls in (AnthropicMessagesRequest, AnthropicCountTokensRequest):
        model_schema = model_cls.model_json_schema(ref_template="#/components/schemas/{model}")
        name = model_cls.__name__
        # Move nested $defs to top-level components/schemas
        for def_name, definition in model_schema.pop("$defs", {}).items():
            schemas[def_name] = definition
        schemas[name] = model_schema

    app.openapi_schema = schema
    return schema


logger = logging.getLogger("anthropic_server")

app = FastAPI(
    title="Anthropic-compatible Proxy (OpenAI backend)",
    description=(
        "Converts Anthropic Messages API requests to OpenAI ChatCompletion format, "
        "forwards to an OpenAI-compatible backend, and converts responses back.\n\n"
        "Supports SSE streaming, tool calling (with name truncation/restoration), "
        "extended thinking, structured output, and web search."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.openapi = _custom_openapi  # type: ignore[method-assign]

# Global config — set via configure() or CLI args
_config: Dict[str, Any] = {
    "backend_base_url": "https://api.openai.com/v1",
    "backend_api_key": "",
    "timeout": 3600,
    "models": ["gpt-4o", "gpt-4o-mini"],
}

# OpenAI SDK client — initialized in configure()
_client: openai.AsyncOpenAI | None = None


def configure(**kwargs: Any) -> None:
    """Update server configuration and (re)create the OpenAI client."""
    global _client
    _config.update(kwargs)
    _client = openai.AsyncOpenAI(
        api_key=_config["backend_api_key"],
        base_url=_config["backend_base_url"],
        timeout=_config["timeout"],
        # Bypass system proxy to avoid SSE buffering
        http_client=_httpx.AsyncClient(trust_env=False),
    )


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)


# ── Health check ────────────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


# ── Debug playground ──────────────────────────────────────────────────


@app.get("/debug", response_class=HTMLResponse, include_in_schema=False)
async def debug_playground():
    """Interactive debug playground for testing API requests."""
    from .debug_page import get_debug_html

    return HTMLResponse(get_debug_html("anthropic", models=_config["models"]))


# ── Messages endpoint ──────────────────────────────────────────────────


@app.post(
    "/v1/messages",
    response_model=AnthropicMessagesResponse,
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                "application/json": {
                    "schema": {
                        "$ref": "#/components/schemas/AnthropicMessagesRequest",
                    }
                }
            },
        },
    },
    responses={
        200: {
            "description": "Successful non-streaming response",
            "content": {
                "application/json": {
                    "example": {
                        "id": "msg_abc123",
                        "type": "message",
                        "role": "assistant",
                        "model": "gpt-4o",
                        "content": [{"type": "text", "text": "Hello! How can I help you today?"}],
                        "stop_reason": "end_turn",
                        "stop_sequence": None,
                        "usage": {"input_tokens": 10, "output_tokens": 8},
                    }
                },
                "text/event-stream": {
                    "example": (
                        'event: message_start\ndata: {"type":"message_start","message":{...}}\n\n'
                        "event: content_block_start\ndata: {...}\n\n"
                        "event: content_block_delta\ndata: {...}\n\n"
                        'event: message_stop\ndata: {"type":"message_stop"}\n\n'
                    )
                },
            },
        },
        400: {
            "description": "Invalid request",
            "model": AnthropicErrorResponse,
        },
        401: {
            "description": "Authentication error",
            "model": AnthropicErrorResponse,
        },
        429: {
            "description": "Rate limit error",
            "model": AnthropicErrorResponse,
        },
        502: {
            "description": "Backend error",
            "model": AnthropicErrorResponse,
        },
        504: {
            "description": "Backend timeout",
            "model": AnthropicErrorResponse,
        },
    },
    tags=["Messages"],
    summary="Create a message",
    description=(
        "Anthropic-compatible `/v1/messages` endpoint.\n\n"
        "Accepts an Anthropic Messages request, converts it to OpenAI format, "
        "forwards to the OpenAI backend, and converts the response back.\n\n"
        "**Streaming**: Set `stream: true` to receive Server-Sent Events in Anthropic format "
        "(message_start, content_block_start, content_block_delta, content_block_stop, "
        "message_delta, message_stop).\n\n"
        "**Parameter mapping**:\n"
        "- `system` → system role message\n"
        "- `tools` → OpenAI tool definitions (names >64 chars truncated with hash)\n"
        "- `tool_choice` → OpenAI tool_choice (any→required)\n"
        "- `thinking` → reasoning_effort\n"
        "- `output_config` → response_format\n"
        "- `stop_sequences` → stop\n"
        "- `metadata.user_id` → user\n\n"
        "**Dropped params** (no OpenAI equivalent): `top_k`, `context_management`, `cache_control`"
    ),
)
async def messages(request: Request):
    try:
        anthropic_request = await request.json()
    except Exception:
        raise HTTPException(
            status_code=400,
            detail={
                "type": "error",
                "error": {"type": "invalid_request_error", "message": "Invalid JSON body"},
            },
        )

    is_stream = anthropic_request.get("stream", False)
    logger.info(
        "Incoming request: model=%s stream=%s messages=%d",
        anthropic_request.get("model", "?"),
        is_stream,
        len(anthropic_request.get("messages", [])),
    )

    # Convert Anthropic request -> OpenAI request
    try:
        openai_request, tool_name_mapping = AnthropicToOpenAIConverter.convert_request(
            anthropic_request
        )
    except Exception as e:
        logger.error("Request conversion failed: %s", e)
        return _anthropic_error(400, "invalid_request_error", f"Request conversion error: {e}")

    # Set stream flag for OpenAI
    openai_request["stream"] = is_stream
    if is_stream:
        # Request usage in streaming mode so we can include it in message_delta
        openai_request["stream_options"] = {"include_usage": True}

    logger.debug(
        "Converted request -> OpenAI:\n%s",
        json.dumps(openai_request, indent=2, ensure_ascii=False),
    )

    if is_stream:
        return StreamingResponse(
            _stream_response(openai_request, tool_name_mapping),
            media_type="text/event-stream; charset=utf-8",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
                "Transfer-Encoding": "chunked",
            },
        )
    else:
        return await _non_stream_response(openai_request, tool_name_mapping)


async def _non_stream_response(
    openai_request: Dict[str, Any],
    tool_name_mapping: Dict[str, str],
) -> JSONResponse:
    """Send non-streaming request to OpenAI and convert response."""
    assert _client is not None, "Client not initialized. Call configure() first."
    params = dict(openai_request)
    model = params.pop("model")
    messages = params.pop("messages")
    params.pop("stream", None)
    params.pop("stream_options", None)

    try:
        response = await _client.chat.completions.create(
            model=model,
            messages=messages,
            extra_body=params,
        )
    except openai.APITimeoutError:
        return _anthropic_error(504, "api_error", "Backend request timed out")
    except openai.APIConnectionError as e:
        return _anthropic_error(502, "api_error", f"Backend connection error: {e}")
    except openai.APIStatusError as e:
        logger.error(
            "Backend (OpenAI) error %d:\n  URL: %s\n  Response: %s",
            e.status_code,
            _config["backend_base_url"],
            str(e.body)[:2000],
        )
        error_type = _map_status_to_error_type(e.status_code)
        return _anthropic_error(e.status_code, error_type, str(e.body)[:500])

    openai_response = response.model_dump()
    logger.debug(
        "Backend (OpenAI) response:\n%s",
        json.dumps(openai_response, indent=2, ensure_ascii=False)[:3000],
    )

    # Convert OpenAI response -> Anthropic response
    try:
        anthropic_response = AnthropicToOpenAIConverter.convert_response(
            openai_response,
            tool_name_mapping=tool_name_mapping,
        )
    except Exception as e:
        logger.error(
            "Response conversion failed: %s\n  Original OpenAI response: %s",
            e,
            json.dumps(openai_response, ensure_ascii=False)[:1000],
        )
        return _anthropic_error(502, "api_error", f"Response conversion error: {e}")

    return JSONResponse(content=anthropic_response)


def _log_stream_summary(
    direction: str,
    backend_chunks: list[Dict[str, Any]],
    output_events: list[Dict[str, Any]],
) -> None:
    """Log a structured summary of a completed stream."""
    model = ""
    content = ""
    thinking = ""
    tool_calls: list[str] = []
    usage: Dict[str, Any] = {}
    finish_reason = ""

    for chunk in backend_chunks:
        if not model:
            model = chunk.get("model", "")
        choices = chunk.get("choices", [])
        if choices:
            choice = choices[0]
            delta = choice.get("delta", {})
            if delta.get("content"):
                content += delta["content"]
            if delta.get("reasoning_content"):
                thinking += delta["reasoning_content"]
            if delta.get("tool_calls"):
                for tc in delta["tool_calls"]:
                    fn = tc.get("function", {})
                    if fn.get("name"):
                        tool_calls.append(f"{fn['name']}:")
                    if fn.get("arguments") and tool_calls:
                        tool_calls[-1] += fn["arguments"]
            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]
        if chunk.get("usage"):
            usage.update(chunk["usage"])

    lines = [f"[{direction}] Stream complete"]
    if model:
        lines.append(f"  model: {model}")
    if finish_reason:
        lines.append(f"  finish_reason: {finish_reason}")
    if usage:
        lines.append(f"  usage: {json.dumps(usage)}")
    lines.append(f"  events: {len(backend_chunks)} backend → {len(output_events)} output")
    if thinking:
        lines.append(f"  thinking: ({len(thinking)} chars) {thinking[:200]}...")
    if content:
        lines.append(f"  content: ({len(content)} chars) {content[:300]}...")
    if tool_calls:
        lines.append(f"  tool_calls: {', '.join(tc[:100] for tc in tool_calls)}")

    logger.debug("\n".join(lines))


async def _stream_response(
    openai_request: Dict[str, Any],
    tool_name_mapping: Dict[str, str],
) -> AsyncIterator[str]:
    """Stream request to OpenAI using the SDK, convert chunks to Anthropic SSE events."""
    assert _client is not None, "Client not initialized. Call configure() first."
    params = dict(openai_request)
    model_name = params.pop("model")
    messages = params.pop("messages")
    params.pop("stream", None)

    # Extract stream_options before passing to extra_body
    stream_options = params.pop("stream_options", None)

    try:
        stream = await _client.chat.completions.create(
            model=model_name,
            messages=messages,
            stream=True,
            stream_options=stream_options,
            extra_body=params,
        )

        all_openai_chunks: list[Dict[str, Any]] = []
        all_anthropic_events: list[Dict[str, Any]] = []

        async def _chunk_iter() -> AsyncIterator[Dict[str, Any]]:
            """Iterate SDK stream, collecting chunks and yielding dicts."""
            async for chunk in stream:
                chunk_dict = chunk.model_dump()
                all_openai_chunks.append(chunk_dict)
                yield chunk_dict

        async for event in AnthropicToOpenAIConverter.aconvert_stream(
            _chunk_iter(),
            model=model_name,
            tool_name_mapping=tool_name_mapping,
        ):
            all_anthropic_events.append(event)
            event_type = event.get("type", "unknown")
            yield f"event: {event_type}\ndata: {json.dumps(event)}\n\n"

        _log_stream_summary(
            "OpenAI→Anthropic",
            all_openai_chunks,
            all_anthropic_events,
        )

    except openai.APITimeoutError:
        error_event = {
            "type": "error",
            "error": {"type": "api_error", "message": "Backend request timed out"},
        }
        yield f"event: error\ndata: {json.dumps(error_event)}\n\n"
    except openai.APIConnectionError as e:
        error_event = {
            "type": "error",
            "error": {"type": "api_error", "message": f"Backend connection error: {e}"},
        }
        yield f"event: error\ndata: {json.dumps(error_event)}\n\n"
    except openai.APIStatusError as e:
        logger.error(
            "Backend (OpenAI) stream error %d:\n  URL: %s\n  Response: %s",
            e.status_code,
            _config["backend_base_url"],
            str(e.body)[:2000],
        )
        error_event = {
            "type": "error",
            "error": {
                "type": _map_status_to_error_type(e.status_code),
                "message": str(e.body)[:500],
            },
        }
        yield f"event: error\ndata: {json.dumps(error_event)}\n\n"


# ── Count tokens endpoint (stub) ───────────────────────────────────────


@app.post(
    "/v1/messages/count_tokens",
    response_model=AnthropicCountTokensResponse,
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                "application/json": {
                    "schema": {
                        "$ref": "#/components/schemas/AnthropicCountTokensRequest",
                    }
                }
            },
        },
    },
    tags=["Messages"],
    summary="Count tokens (stub)",
    description=(
        "Rough token count estimate (~4 chars per token). "
        "For accurate counts, use the actual Anthropic API."
    ),
)
async def count_tokens(request: Request):
    """Stub token counting endpoint. Returns a rough estimate."""
    try:
        body = await request.json()
    except Exception:
        return _anthropic_error(400, "invalid_request_error", "Invalid JSON body")

    # Rough estimate: ~4 chars per token
    total_chars = 0
    for msg in body.get("messages", []):
        content = msg.get("content", "")
        if isinstance(content, str):
            total_chars += len(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    total_chars += len(block.get("text", ""))
                elif isinstance(block, str):
                    total_chars += len(block)

    system = body.get("system", "")
    if isinstance(system, str):
        total_chars += len(system)
    elif isinstance(system, list):
        for block in system:
            if isinstance(block, dict):
                total_chars += len(block.get("text", ""))

    return JSONResponse(content={"input_tokens": max(1, total_chars // 4)})


# ── Error helpers ───────────────────────────────────────────────────────


def _anthropic_error(status_code: int, error_type: str, message: str) -> JSONResponse:
    """Create an Anthropic-format error response."""
    return JSONResponse(
        status_code=status_code,
        content={
            "type": "error",
            "error": {"type": error_type, "message": message},
        },
    )


def _map_status_to_error_type(status_code: int) -> str:
    """Map HTTP status code to Anthropic error type."""
    mapping = {
        400: "invalid_request_error",
        401: "authentication_error",
        403: "permission_error",
        404: "not_found_error",
        413: "request_too_large",
        429: "rate_limit_error",
        500: "api_error",
        529: "overloaded_error",
    }
    return mapping.get(status_code, "api_error")


# ── CLI entry point ─────────────────────────────────────────────────────


def _normalize_base_url(url: str) -> str:
    """
    Normalize a URL to a base URL suitable for the OpenAI SDK.

    The SDK appends /chat/completions automatically, so strip that if present.

    Accepts:
      - https://api.openai.com/v1                          -> keep as-is
      - https://api.openai.com/v1/chat/completions          -> strip /chat/completions
      - https://dashscope.aliyuncs.com/compatible-mode/v1   -> keep as-is
      - https://dashscope.aliyuncs.com/compatible-mode/v1/  -> strip trailing /
    """
    url = url.rstrip("/")
    if url.endswith("/chat/completions"):
        url = url[: -len("/chat/completions")]
    return url


def main():
    parser = argparse.ArgumentParser(
        description="Anthropic-compatible API server backed by OpenAI ChatCompletion API"
    )
    parser.add_argument(
        "--backend-url",
        default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        help="OpenAI base URL or chat completions URL. "
        "Automatically normalized for the SDK. "
        "(default: $OPENAI_BASE_URL or https://api.openai.com/v1)",
    )
    parser.add_argument(
        "--backend-api-key",
        default=os.environ.get("OPENAI_API_KEY", ""),
        help="OpenAI API key (default: $OPENAI_API_KEY)",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8002, help="Bind port (default: 8002)")
    parser.add_argument("--timeout", type=int, default=3600, help="Backend timeout in seconds")
    parser.add_argument(
        "--models",
        default=os.environ.get("ANTHROPIC_SERVER_MODELS", os.environ.get("MODELS", "")),
        help="Comma-separated list of available model names for debug page. "
        "(default: $ANTHROPIC_SERVER_MODELS or $MODELS or gpt-4o,gpt-4o-mini)",
    )
    parser.add_argument(
        "--log-level", default="info", choices=["debug", "info", "warning", "error"]
    )
    parser.add_argument(
        "--log-dir",
        default=os.environ.get("LOG_DIR", "logs"),
        help="Directory for log files (default: $LOG_DIR or logs/)",
    )

    args = parser.parse_args()

    from .logging_config import setup_logging

    setup_logging("anthropic_server", level=args.log_level, log_dir=args.log_dir)

    if not args.backend_api_key:
        logger.error("No API key provided. Set --backend-api-key or $OPENAI_API_KEY")
        sys.exit(1)

    backend_base_url = _normalize_base_url(args.backend_url)

    models = (
        [m.strip() for m in args.models.split(",") if m.strip()]
        if args.models
        else ["gpt-4o", "gpt-4o-mini"]
    )

    configure(
        backend_base_url=backend_base_url,
        backend_api_key=args.backend_api_key,
        timeout=args.timeout,
        models=models,
    )

    logger.info("Starting Anthropic-compatible server on %s:%d", args.host, args.port)
    logger.info("Backend: %s", backend_base_url)
    logger.info("Swagger UI: http://%s:%d/docs", args.host, args.port)
    logger.info("Debug playground: http://%s:%d/debug", args.host, args.port)

    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
