"""
Anthropic-compatible API server.

Exposes an Anthropic /v1/messages endpoint that forwards requests
to an OpenAI ChatCompletion API backend, converting protocols in both directions.

Usage:
    # Forward to OpenAI API:
    python -m openai_anthropic_converter.servers.anthropic_server \
        --backend-url https://api.openai.com/v1/chat/completions \
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

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

# Add parent to path so the package can be imported when running as __main__
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from openai_anthropic_converter import AnthropicToOpenAIConverter

logger = logging.getLogger("anthropic_server")

app = FastAPI(title="Anthropic-compatible Proxy (OpenAI backend)")

# Global config — set via configure() or CLI args
_config: Dict[str, Any] = {
    "backend_url": "https://api.openai.com/v1/chat/completions",
    "backend_api_key": "",
    "timeout": 300,
}


def configure(**kwargs: Any) -> None:
    """Update server configuration. Call before starting the server."""
    _config.update(kwargs)


# ── Health check ────────────────────────────────────────────────────────


@app.get("/health")
async def health():
    return {"status": "ok"}


# ── Messages endpoint ──────────────────────────────────────────────────


@app.post("/v1/messages")
async def messages(request: Request):
    """
    Anthropic-compatible /v1/messages endpoint.

    Accepts an Anthropic Messages request, converts it to OpenAI format,
    forwards to the OpenAI backend, and converts the response back.
    """
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

    # Build headers for OpenAI API
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {_config['backend_api_key']}",
    }

    if is_stream:
        return StreamingResponse(
            _stream_response(openai_request, headers, tool_name_mapping),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    else:
        return await _non_stream_response(openai_request, headers, tool_name_mapping)


async def _non_stream_response(
    openai_request: Dict[str, Any],
    headers: Dict[str, str],
    tool_name_mapping: Dict[str, str],
) -> JSONResponse:
    """Send non-streaming request to OpenAI and convert response."""
    async with httpx.AsyncClient(timeout=_config["timeout"]) as client:
        try:
            resp = await client.post(
                _config["backend_url"],
                json=openai_request,
                headers=headers,
            )
        except httpx.TimeoutException:
            return _anthropic_error(504, "api_error", "Backend request timed out")
        except httpx.ConnectError as e:
            return _anthropic_error(502, "api_error", f"Backend connection error: {e}")

    if resp.status_code != 200:
        logger.error("Backend error %d: %s", resp.status_code, resp.text[:500])
        error_type = _map_status_to_error_type(resp.status_code)
        return _anthropic_error(resp.status_code, error_type, resp.text[:500])

    try:
        openai_response = resp.json()
    except Exception:
        return _anthropic_error(502, "api_error", "Invalid JSON from backend")

    # Convert OpenAI response -> Anthropic response
    try:
        anthropic_response = AnthropicToOpenAIConverter.convert_response(
            openai_response,
            tool_name_mapping=tool_name_mapping,
        )
    except Exception as e:
        logger.error("Response conversion failed: %s", e)
        return _anthropic_error(502, "api_error", f"Response conversion error: {e}")

    return JSONResponse(content=anthropic_response)


async def _stream_response(
    openai_request: Dict[str, Any],
    headers: Dict[str, str],
    tool_name_mapping: Dict[str, str],
) -> AsyncIterator[str]:
    """Stream request to OpenAI, convert chunks to Anthropic SSE events."""
    async with httpx.AsyncClient(timeout=_config["timeout"]) as client:
        try:
            async with client.stream(
                "POST",
                _config["backend_url"],
                json=openai_request,
                headers=headers,
            ) as resp:
                if resp.status_code != 200:
                    body = await resp.aread()
                    error_event = {
                        "type": "error",
                        "error": {
                            "type": _map_status_to_error_type(resp.status_code),
                            "message": body.decode()[:500],
                        },
                    }
                    yield f"event: error\ndata: {json.dumps(error_event)}\n\n"
                    return

                # Parse OpenAI SSE and convert to Anthropic events
                model = openai_request.get("model", "unknown")
                async for event in _parse_and_convert_sse(resp, model, tool_name_mapping):
                    event_type = event.get("type", "unknown")
                    yield f"event: {event_type}\ndata: {json.dumps(event)}\n\n"

        except httpx.TimeoutException:
            error_event = {
                "type": "error",
                "error": {"type": "api_error", "message": "Backend request timed out"},
            }
            yield f"event: error\ndata: {json.dumps(error_event)}\n\n"
        except httpx.ConnectError as e:
            error_event = {
                "type": "error",
                "error": {"type": "api_error", "message": f"Backend connection error: {e}"},
            }
            yield f"event: error\ndata: {json.dumps(error_event)}\n\n"


async def _parse_and_convert_sse(
    resp: httpx.Response,
    model: str,
    tool_name_mapping: Dict[str, str],
) -> AsyncIterator[Dict[str, Any]]:
    """Parse OpenAI SSE stream and yield converted Anthropic events."""

    async def _openai_chunks() -> AsyncIterator[Dict[str, Any]]:
        """Parse raw SSE lines into OpenAI chunk dicts."""
        async for line in resp.aiter_lines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("data:"):
                data_str = line[5:].strip()
                if not data_str or data_str == "[DONE]":
                    continue
                try:
                    yield json.loads(data_str)
                except json.JSONDecodeError:
                    logger.warning("Failed to parse SSE data: %s", data_str[:200])

    async for event in AnthropicToOpenAIConverter.aconvert_stream(
        _openai_chunks(),
        model=model,
        tool_name_mapping=tool_name_mapping,
    ):
        yield event


# ── Count tokens endpoint (stub) ───────────────────────────────────────


@app.post("/v1/messages/count_tokens")
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


def main():
    parser = argparse.ArgumentParser(
        description="Anthropic-compatible API server backed by OpenAI ChatCompletion API"
    )
    parser.add_argument(
        "--backend-url",
        default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1/chat/completions"),
        help="OpenAI ChatCompletion API URL (default: https://api.openai.com/v1/chat/completions)",
    )
    parser.add_argument(
        "--backend-api-key",
        default=os.environ.get("OPENAI_API_KEY", ""),
        help="OpenAI API key (default: $OPENAI_API_KEY)",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8002, help="Bind port (default: 8002)")
    parser.add_argument("--timeout", type=int, default=300, help="Backend timeout in seconds")
    parser.add_argument(
        "--log-level", default="info", choices=["debug", "info", "warning", "error"]
    )

    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper()))

    if not args.backend_api_key:
        logger.error("No API key provided. Set --backend-api-key or $OPENAI_API_KEY")
        sys.exit(1)

    configure(
        backend_url=args.backend_url,
        backend_api_key=args.backend_api_key,
        timeout=args.timeout,
    )

    logger.info("Starting Anthropic-compatible server on %s:%d", args.host, args.port)
    logger.info("Backend: %s", args.backend_url)

    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
