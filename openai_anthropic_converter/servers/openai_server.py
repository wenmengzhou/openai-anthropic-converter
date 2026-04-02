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
from typing import Any, AsyncIterator, Dict, Union

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from openai_anthropic_converter import OpenAIToAnthropicConverter

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
    "backend_url": "https://api.anthropic.com/v1/messages",
    "backend_api_key": "",
    "anthropic_version": "2023-06-01",
    "timeout": 300,
    "default_max_tokens": 4096,
}


def configure(**kwargs: Any) -> None:
    """Update server configuration. Call before starting the server."""
    _config.update(kwargs)


# ── Health check ────────────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/v1/models", response_model=OpenAIModelsResponse, tags=["Models"])
async def list_models():
    """List available models. Returns a static list of Anthropic models."""
    return {
        "object": "list",
        "data": [
            {"id": "claude-sonnet-4-20250514", "object": "model", "owned_by": "anthropic"},
            {"id": "claude-opus-4-20250514", "object": "model", "owned_by": "anthropic"},
        ],
    }


# ── Debug playground ──────────────────────────────────────────────────


@app.get("/debug", response_class=HTMLResponse, include_in_schema=False)
async def debug_playground():
    """Interactive debug playground for testing API requests."""
    from .debug_page import get_debug_html

    return HTMLResponse(get_debug_html("openai"))


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
                    "example": "data: {\"id\":\"chatcmpl-abc123\",\"choices\":[{\"delta\":{\"content\":\"Hello\"},\"index\":0}]}\n\ndata: [DONE]\n\n"
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
        "- `response_format` → output_format (Claude 4.5+) or tool-based JSON mode\n"
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

    # Convert OpenAI request -> Anthropic request
    try:
        anthropic_request = OpenAIToAnthropicConverter.convert_request(
            openai_request,
            default_max_tokens=_config["default_max_tokens"],
        )
    except Exception as e:
        logger.error("Request conversion failed: %s", e)
        raise HTTPException(status_code=400, detail=f"Request conversion error: {e}")

    # Set stream flag for Anthropic
    anthropic_request["stream"] = is_stream

    # Build headers for Anthropic API
    headers = {
        "Content-Type": "application/json",
        "x-api-key": _config["backend_api_key"],
        "anthropic-version": _config["anthropic_version"],
    }

    if is_stream:
        return StreamingResponse(
            _stream_response(anthropic_request, headers),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    else:
        return await _non_stream_response(anthropic_request, headers)


async def _non_stream_response(
    anthropic_request: Dict[str, Any],
    headers: Dict[str, str],
) -> JSONResponse:
    """Send non-streaming request to Anthropic and convert response."""
    async with httpx.AsyncClient(timeout=_config["timeout"]) as client:
        try:
            resp = await client.post(
                _config["backend_url"],
                json=anthropic_request,
                headers=headers,
            )
        except httpx.TimeoutException:
            raise HTTPException(status_code=504, detail="Backend request timed out")
        except httpx.ConnectError as e:
            raise HTTPException(status_code=502, detail=f"Backend connection error: {e}")

    if resp.status_code != 200:
        logger.error("Backend error %d: %s", resp.status_code, resp.text[:500])
        raise HTTPException(
            status_code=resp.status_code,
            detail=f"Backend error: {resp.text[:500]}",
        )

    try:
        anthropic_response = resp.json()
    except Exception:
        raise HTTPException(status_code=502, detail="Invalid JSON from backend")

    # Convert Anthropic response -> OpenAI response
    try:
        openai_response = OpenAIToAnthropicConverter.convert_response(anthropic_response)
    except Exception as e:
        logger.error("Response conversion failed: %s", e)
        raise HTTPException(status_code=502, detail=f"Response conversion error: {e}")

    return JSONResponse(content=openai_response)


async def _stream_response(
    anthropic_request: Dict[str, Any],
    headers: Dict[str, str],
) -> AsyncIterator[str]:
    """Stream request to Anthropic, convert SSE events to OpenAI chunks."""
    async with httpx.AsyncClient(timeout=_config["timeout"]) as client:
        try:
            async with client.stream(
                "POST",
                _config["backend_url"],
                json=anthropic_request,
                headers=headers,
            ) as resp:
                if resp.status_code != 200:
                    body = await resp.aread()
                    error_chunk = {
                        "error": {
                            "message": f"Backend error {resp.status_code}: {body.decode()[:500]}",
                            "type": "backend_error",
                        }
                    }
                    yield f"data: {json.dumps(error_chunk)}\n\n"
                    return

                # Parse Anthropic SSE and convert to OpenAI chunks
                async for openai_chunk in _parse_and_convert_sse(resp):
                    yield f"data: {json.dumps(openai_chunk)}\n\n"

                yield "data: [DONE]\n\n"

        except httpx.TimeoutException:
            error_chunk = {"error": {"message": "Backend request timed out", "type": "timeout"}}
            yield f"data: {json.dumps(error_chunk)}\n\n"
        except httpx.ConnectError as e:
            error_chunk = {
                "error": {"message": f"Backend connection error: {e}", "type": "connection_error"}
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"


async def _parse_and_convert_sse(
    resp: httpx.Response,
) -> AsyncIterator[Dict[str, Any]]:
    """Parse Anthropic SSE stream and yield converted OpenAI chunks."""

    async def _anthropic_events() -> AsyncIterator[Dict[str, Any]]:
        """Parse raw SSE lines into Anthropic event dicts."""
        async for line in resp.aiter_lines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("event:"):
                # Event type line — we get the type from the data payload
                continue
            if line.startswith("data:"):
                data_str = line[5:].strip()
                if not data_str or data_str == "[DONE]":
                    continue
                try:
                    yield json.loads(data_str)
                except json.JSONDecodeError:
                    logger.warning("Failed to parse SSE data: %s", data_str[:200])

    async for chunk in OpenAIToAnthropicConverter.aconvert_stream(_anthropic_events()):
        yield chunk


# ── CLI entry point ─────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="OpenAI-compatible API server backed by Anthropic Messages API"
    )
    parser.add_argument(
        "--backend-url",
        default=os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1/messages"),
        help="Anthropic Messages API URL (default: https://api.anthropic.com/v1/messages)",
    )
    parser.add_argument(
        "--backend-api-key",
        default=os.environ.get("ANTHROPIC_API_KEY", ""),
        help="Anthropic API key (default: $ANTHROPIC_API_KEY)",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8001, help="Bind port (default: 8001)")
    parser.add_argument("--timeout", type=int, default=300, help="Backend timeout in seconds")
    parser.add_argument("--default-max-tokens", type=int, default=4096)
    parser.add_argument(
        "--log-level", default="info", choices=["debug", "info", "warning", "error"]
    )

    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper()))

    if not args.backend_api_key:
        logger.error("No API key provided. Set --backend-api-key or $ANTHROPIC_API_KEY")
        sys.exit(1)

    configure(
        backend_url=args.backend_url,
        backend_api_key=args.backend_api_key,
        timeout=args.timeout,
        default_max_tokens=args.default_max_tokens,
    )

    logger.info("Starting OpenAI-compatible server on %s:%d", args.host, args.port)
    logger.info("Backend: %s", args.backend_url)
    logger.info("Swagger UI: http://%s:%d/docs", args.host, args.port)
    logger.info("Debug playground: http://%s:%d/debug", args.host, args.port)

    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
