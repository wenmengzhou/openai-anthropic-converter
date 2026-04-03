"""
OpenAI-compatible API server.

Exposes an OpenAI /v1/chat/completions endpoint that forwards requests
to an Anthropic Messages API backend, converting protocols in both directions.

Features:
    - Swagger UI at /docs, ReDoc at /redoc, OpenAPI JSON at /openapi.json
    - Interactive debug playground at /debug
    - Full streaming and non-streaming support

Usage:
    # Forward to Anthropic API:
    python -m openai_anthropic_converter.servers.openai_server \
        --backend-url https://api.anthropic.com/v1/messages \
        --backend-api-key $ANTHROPIC_API_KEY \
        --port 8001

    # Then call it as if it were an OpenAI API:
    curl http://localhost:8001/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{"model":"claude-sonnet-4-20250514","messages":[{"role":"user","content":"Hello!"}],"max_tokens":1024}'
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

import anthropic
import httpx as _httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse

from openai_anthropic_converter import OpenAIToAnthropicConverter
from openai_anthropic_converter.openai_to_anthropic.stream import AnthropicSSEToOpenAIStream

from .schemas import (
    HealthResponse,
    OpenAIChatCompletionRequest,
    OpenAIChatCompletionResponse,
    OpenAIModelsResponse,
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
    # Add request schemas to components
    components = schema.setdefault("components", {})
    schemas = components.setdefault("schemas", {})
    schemas["OpenAIChatCompletionRequest"] = OpenAIChatCompletionRequest.model_json_schema(
        ref_template="#/components/schemas/{model}"
    )
    # Inline nested $defs into components/schemas
    for name, definition in schemas.get("OpenAIChatCompletionRequest", {}).pop("$defs", {}).items():
        schemas[name] = definition

    app.openapi_schema = schema
    return schema


logger = logging.getLogger("openai_server")

app = FastAPI(
    title="OpenAI-compatible Proxy (Anthropic backend)",
    description=(
        "Converts OpenAI Chat Completion requests to Anthropic Messages API format, "
        "forwards to an Anthropic backend, and converts responses back.\n\n"
        "Supports streaming, tool calling, extended thinking, JSON schema output, "
        "and Bailian/DashScope compatibility."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.openapi = _custom_openapi  # type: ignore[method-assign]

# Global config — set via configure() or CLI args
_config: Dict[str, Any] = {
    "backend_base_url": "https://api.anthropic.com",
    "backend_api_key": "",
    "anthropic_version": "2023-06-01",
    "timeout": 3600,
    "default_max_tokens": 4096,
    "models": ["claude-sonnet-4-20250514", "claude-opus-4-20250514"],
}

# Anthropic SDK client — initialized in configure()
_client: anthropic.AsyncAnthropic | None = None


def configure(**kwargs: Any) -> None:
    """Update server configuration and (re)create the Anthropic client."""
    global _client
    _config.update(kwargs)
    _client = anthropic.AsyncAnthropic(
        api_key=_config["backend_api_key"],
        base_url=_config["backend_base_url"],
        # DashScope/Bailian compatibility: send both x-api-key (SDK default) and Bearer
        default_headers={"Authorization": f"Bearer {_config['backend_api_key']}"},
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


@app.get("/v1/models", response_model=OpenAIModelsResponse, tags=["Models"])
async def list_models():
    """List available models. Returns the configured model list."""
    return {
        "object": "list",
        "data": [{"id": m, "object": "model", "owned_by": "system"} for m in _config["models"]],
    }


# ── Debug playground ──────────────────────────────────────────────────


@app.get("/debug", response_class=HTMLResponse, include_in_schema=False)
async def debug_playground():
    """Interactive debug playground for testing API requests."""
    from .debug_page import get_debug_html

    return HTMLResponse(get_debug_html("openai", models=_config["models"]))


# ── Chat Completions ───────────────────────────────────────────────────


@app.post(
    "/v1/chat/completions",
    response_model=OpenAIChatCompletionResponse,
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                "application/json": {
                    "schema": {
                        "$ref": "#/components/schemas/OpenAIChatCompletionRequest",
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
                        "id": "chatcmpl-abc123",
                        "object": "chat.completion",
                        "created": 1700000000,
                        "model": "claude-sonnet-4-20250514",
                        "choices": [
                            {
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": "Hello! How can I help you today?",
                                },
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 10,
                            "completion_tokens": 8,
                            "total_tokens": 18,
                        },
                    }
                },
                "text/event-stream": {
                    "example": 'data: {"id":"chatcmpl-abc123","choices":[{"delta":{"content":"Hello"},"index":0}]}\n\ndata: [DONE]\n\n'
                },
            },
        },
        400: {"description": "Invalid request (bad JSON or conversion error)"},
        502: {"description": "Backend error"},
        504: {"description": "Backend timeout"},
    },
    tags=["Chat"],
    summary="Create chat completion",
    description=(
        "OpenAI-compatible `/v1/chat/completions` endpoint.\n\n"
        "Accepts an OpenAI ChatCompletion request, converts it to Anthropic format, "
        "forwards to the Anthropic backend, and converts the response back.\n\n"
        "**Streaming**: Set `stream: true` to receive Server-Sent Events.\n\n"
        "**Parameter mapping**:\n"
        "- `tools` → Anthropic tool definitions\n"
        "- `tool_choice` → Anthropic tool_choice (required→any)\n"
        "- `response_format` → output_config (Claude 4.5+) or tool-based JSON mode\n"
        "- `reasoning_effort` → thinking.budget_tokens\n"
        "- `stop` → stop_sequences\n"
        "- `user` → metadata.user_id\n\n"
        "**[Bailian/DashScope compat]**:\n"
        "- `enable_thinking` + `thinking_budget` → thinking param\n"
        "- `enable_search` → web_search tool"
    ),
)
async def chat_completions(request: Request):
    """
    Accepts an OpenAI ChatCompletion request, converts to Anthropic,
    forwards to backend, and converts response back.
    """
    try:
        openai_request = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    is_stream = openai_request.get("stream", False)
    logger.info(
        "Incoming request: model=%s stream=%s messages=%d",
        openai_request.get("model", "?"),
        is_stream,
        len(openai_request.get("messages", [])),
    )

    # Convert OpenAI request -> Anthropic request
    try:
        anthropic_request = OpenAIToAnthropicConverter.convert_request(
            openai_request,
            default_max_tokens=_config["default_max_tokens"],
        )
    except Exception as e:
        logger.error("Request conversion failed: %s", e)
        raise HTTPException(status_code=400, detail=f"Request conversion error: {e}")

    logger.debug(
        "Converted request -> Anthropic:\n%s",
        json.dumps(anthropic_request, indent=2, ensure_ascii=False),
    )

    if is_stream:
        return StreamingResponse(
            _stream_response(anthropic_request),
            media_type="text/event-stream; charset=utf-8",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
                "Transfer-Encoding": "chunked",
            },
        )
    else:
        return await _non_stream_response(anthropic_request)


# SDK named parameters (beyond model/messages/max_tokens/stream) that should be
# extracted from extra_body and passed as keyword args to messages.create/stream.
_SDK_NAMED_PARAMS = {
    "system",
    "temperature",
    "top_p",
    "top_k",
    "stop_sequences",
    "thinking",
    "tools",
    "tool_choice",
    "metadata",
    "output_config",
    "service_tier",
    "cache_control",
}


def _extract_sdk_params(params: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Split params into SDK named kwargs and extra_body remainder."""
    sdk_kwargs: Dict[str, Any] = {}
    extra: Dict[str, Any] = {}
    for k, v in params.items():
        if k in _SDK_NAMED_PARAMS:
            sdk_kwargs[k] = v
        else:
            extra[k] = v
    return sdk_kwargs, extra


async def _non_stream_response(
    anthropic_request: Dict[str, Any],
) -> JSONResponse:
    """Send non-streaming request to Anthropic and convert response."""
    assert _client is not None, "Client not initialized. Call configure() first."
    params = dict(anthropic_request)
    model = params.pop("model")
    messages = params.pop("messages")
    max_tokens = params.pop("max_tokens")
    params.pop("stream", None)
    sdk_kwargs, extra = _extract_sdk_params(params)

    try:
        response = await _client.messages.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            **sdk_kwargs,
            extra_body=extra if extra else None,
        )
    except anthropic.APITimeoutError:
        raise HTTPException(status_code=504, detail="Backend request timed out")
    except anthropic.APIConnectionError as e:
        raise HTTPException(status_code=502, detail=f"Backend connection error: {e}")
    except anthropic.APIStatusError as e:
        logger.error(
            "Backend (Anthropic) error %d:\n  URL: %s\n  Response: %s",
            e.status_code,
            _config["backend_base_url"],
            str(e.body)[:2000],
        )
        raise HTTPException(
            status_code=e.status_code,
            detail=f"Backend error: {str(e.body)[:500]}",
        )

    anthropic_response = response.model_dump()
    logger.debug(
        "Backend (Anthropic) response:\n%s",
        json.dumps(anthropic_response, indent=2, ensure_ascii=False)[:3000],
    )

    # Convert Anthropic response -> OpenAI response
    try:
        openai_response = OpenAIToAnthropicConverter.convert_response(anthropic_response)
    except Exception as e:
        logger.error(
            "Response conversion failed: %s\n  Original Anthropic response: %s",
            e,
            json.dumps(anthropic_response, ensure_ascii=False)[:1000],
        )
        raise HTTPException(status_code=502, detail=f"Response conversion error: {e}")

    return JSONResponse(content=openai_response)


def _log_stream_summary(
    direction: str,
    backend_events: list[Dict[str, Any]],
    output_chunks: list[Dict[str, Any]],
) -> None:
    """Log a structured summary of a completed stream."""
    model = ""
    content = ""
    thinking = ""
    tool_calls: list[str] = []
    usage: Dict[str, Any] = {}
    stop_reason = ""

    for ev in backend_events:
        ev_type = ev.get("type", "")
        if ev_type == "message_start":
            msg = ev.get("message", {})
            model = msg.get("model", "")
            usage.update(msg.get("usage", {}))
        elif ev_type == "content_block_delta":
            delta = ev.get("delta", {})
            if delta.get("type") == "text_delta":
                content += delta.get("text", "")
            elif delta.get("type") == "thinking_delta":
                thinking += delta.get("thinking", "")
            elif delta.get("type") == "input_json_delta":
                if tool_calls:
                    tool_calls[-1] += delta.get("partial_json", "")
        elif ev_type == "content_block_start":
            block = ev.get("content_block", {})
            if block.get("type") in ("tool_use", "server_tool_use", "mcp_tool_use"):
                tool_calls.append(f"{block.get('name', 'tool')}:")
        elif ev_type == "message_delta":
            delta = ev.get("delta", {})
            stop_reason = delta.get("stop_reason", "")
            usage.update(ev.get("usage", {}))

    lines = [f"[{direction}] Stream complete"]
    if model:
        lines.append(f"  model: {model}")
    if stop_reason:
        lines.append(f"  stop_reason: {stop_reason}")
    if usage:
        lines.append(f"  usage: {json.dumps(usage)}")
    lines.append(f"  events: {len(backend_events)} backend → {len(output_chunks)} output")
    if thinking:
        lines.append(f"  thinking: ({len(thinking)} chars) {thinking[:200]}...")
    if content:
        lines.append(f"  content: ({len(content)} chars) {content[:300]}...")
    if tool_calls:
        lines.append(f"  tool_calls: {', '.join(tc[:100] for tc in tool_calls)}")

    logger.debug("\n".join(lines))


async def _stream_response(
    anthropic_request: Dict[str, Any],
) -> AsyncIterator[str]:
    """Stream request to Anthropic using the SDK, convert events to OpenAI chunks."""
    assert _client is not None, "Client not initialized. Call configure() first."
    params = dict(anthropic_request)
    model = params.pop("model")
    messages = params.pop("messages")
    max_tokens = params.pop("max_tokens")
    params.pop("stream", None)
    sdk_kwargs, extra = _extract_sdk_params(params)

    try:
        async with _client.messages.stream(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            **sdk_kwargs,
            extra_body=extra if extra else None,
        ) as stream:
            converter = AnthropicSSEToOpenAIStream()
            all_anthropic_events: list[Dict[str, Any]] = []
            all_openai_chunks: list[Dict[str, Any]] = []

            async for event in stream:
                event_dict = event.model_dump() if hasattr(event, "model_dump") else {}
                # SDK events have a 'type' attr directly
                if not event_dict.get("type") and hasattr(event, "type"):
                    event_dict["type"] = event.type
                all_anthropic_events.append(event_dict)

                openai_chunk = converter.process_event(event_dict)
                if openai_chunk is not None:
                    all_openai_chunks.append(openai_chunk)
                    yield f"data: {json.dumps(openai_chunk)}\n\n"

            yield "data: [DONE]\n\n"

            _log_stream_summary("Anthropic→OpenAI", all_anthropic_events, all_openai_chunks)

    except anthropic.APITimeoutError:
        error_chunk = {"error": {"message": "Backend request timed out", "type": "timeout"}}
        yield f"data: {json.dumps(error_chunk)}\n\n"
    except anthropic.APIConnectionError as e:
        error_chunk = {
            "error": {"message": f"Backend connection error: {e}", "type": "connection_error"}
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"
    except anthropic.APIStatusError as e:
        logger.error(
            "Backend (Anthropic) stream error %d:\n  URL: %s\n  Response: %s",
            e.status_code,
            _config["backend_base_url"],
            str(e.body)[:2000],
        )
        error_chunk = {
            "error": {
                "message": f"Backend error {e.status_code}: {str(e.body)[:500]}",
                "type": "backend_error",
            }
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"


# ── CLI entry point ─────────────────────────────────────────────────────


def _normalize_base_url(url: str) -> str:
    """
    Normalize a URL to a base URL suitable for the Anthropic SDK.

    The SDK appends /v1/messages automatically, so strip that if present.

    Accepts:
      - https://api.anthropic.com                       -> keep as-is
      - https://api.anthropic.com/v1                    -> strip /v1
      - https://api.anthropic.com/v1/messages           -> strip /v1/messages
      - https://xxx.example.com/anthropic-native/v1     -> strip /v1
      - https://xxx.example.com/anthropic-native        -> keep as-is
    """
    url = url.rstrip("/")
    if url.endswith("/v1/messages"):
        url = url[: -len("/v1/messages")]
    elif url.endswith("/v1"):
        url = url[: -len("/v1")]
    return url


def main():
    parser = argparse.ArgumentParser(
        description="OpenAI-compatible API server backed by Anthropic Messages API"
    )
    parser.add_argument(
        "--backend-url",
        default=os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1"),
        help="Anthropic base URL or messages endpoint URL. "
        "Automatically normalized for the SDK. "
        "(default: $ANTHROPIC_BASE_URL or https://api.anthropic.com/v1)",
    )
    parser.add_argument(
        "--backend-api-key",
        default=os.environ.get("ANTHROPIC_API_KEY", ""),
        help="Anthropic API key (default: $ANTHROPIC_API_KEY)",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8001, help="Bind port (default: 8001)")
    parser.add_argument("--timeout", type=int, default=3600, help="Backend timeout in seconds")
    parser.add_argument("--default-max-tokens", type=int, default=4096)
    parser.add_argument(
        "--models",
        default=os.environ.get("OPENAI_SERVER_MODELS", os.environ.get("MODELS", "")),
        help="Comma-separated list of available model names for /v1/models and debug page. "
        "(default: $OPENAI_SERVER_MODELS or $MODELS or claude-sonnet-4-20250514,claude-opus-4-20250514)",
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

    setup_logging("openai_server", level=args.log_level, log_dir=args.log_dir)

    if not args.backend_api_key:
        logger.error("No API key provided. Set --backend-api-key or $ANTHROPIC_API_KEY")
        sys.exit(1)

    backend_base_url = _normalize_base_url(args.backend_url)

    models = (
        [m.strip() for m in args.models.split(",") if m.strip()]
        if args.models
        else ["claude-sonnet-4-20250514", "claude-opus-4-20250514"]
    )

    configure(
        backend_base_url=backend_base_url,
        backend_api_key=args.backend_api_key,
        timeout=args.timeout,
        default_max_tokens=args.default_max_tokens,
        models=models,
    )

    logger.info("Starting OpenAI-compatible server on %s:%d", args.host, args.port)
    logger.info("Backend: %s", backend_base_url)
    logger.info("Swagger UI: http://%s:%d/docs", args.host, args.port)
    logger.info("Debug playground: http://%s:%d/debug", args.host, args.port)

    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
