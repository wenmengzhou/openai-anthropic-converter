"""
Tests for OpenAI and Anthropic proxy servers.

Uses httpx.AsyncClient with ASGITransport to test FastAPI apps directly,
and unittest.mock to mock backend HTTP calls.
"""

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from openai_anthropic_converter.servers.anthropic_server import app as anthropic_app
from openai_anthropic_converter.servers.anthropic_server import configure as anthropic_configure
from openai_anthropic_converter.servers.openai_server import app as openai_app
from openai_anthropic_converter.servers.openai_server import configure as openai_configure

# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def setup_configs():
    """Configure both servers with test values before each test."""
    openai_configure(
        backend_url="https://mock-anthropic.example.com/v1/messages",
        backend_api_key="test-anthropic-key",
        timeout=30,
        default_max_tokens=1024,
    )
    anthropic_configure(
        backend_url="https://mock-openai.example.com/v1/chat/completions",
        backend_api_key="test-openai-key",
        timeout=30,
    )


def _mock_httpx_response(status_code: int, json_body: dict) -> httpx.Response:
    """Create a mock httpx.Response with the given status and JSON body."""
    return httpx.Response(
        status_code=status_code,
        json=json_body,
        request=httpx.Request("POST", "https://mock.example.com"),
    )


def _make_mock_client(post_return=None, post_side_effect=None):
    """Create a properly configured mock httpx.AsyncClient."""
    mock_client = AsyncMock()
    if post_side_effect:
        mock_client.post = AsyncMock(side_effect=post_side_effect)
    else:
        mock_client.post = AsyncMock(return_value=post_return)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    return mock_client


async def _call_openai_server(method, path, **kwargs):
    """Call the OpenAI server app directly via ASGI transport."""
    transport = httpx.ASGITransport(app=openai_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await getattr(client, method)(path, **kwargs)


async def _call_anthropic_server(method, path, **kwargs):
    """Call the Anthropic server app directly via ASGI transport."""
    transport = httpx.ASGITransport(app=anthropic_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await getattr(client, method)(path, **kwargs)


async def _patched_openai_call(mock_client, path, json_body):
    """Make a request to OpenAI server with mocked backend.

    Creates the test ASGI client BEFORE patching httpx.AsyncClient,
    so the patch only affects the server's backend calls.
    """
    transport = httpx.ASGITransport(app=openai_app)
    # Create test client before patch to avoid interference
    client = httpx.AsyncClient(transport=transport, base_url="http://test")
    try:
        with patch(
            "openai_anthropic_converter.servers.openai_server.httpx.AsyncClient",
            return_value=mock_client,
        ):
            return await client.post(path, json=json_body)
    finally:
        await client.aclose()


async def _patched_anthropic_call(mock_client, path, json_body):
    """Make a request to Anthropic server with mocked backend."""
    transport = httpx.ASGITransport(app=anthropic_app)
    client = httpx.AsyncClient(transport=transport, base_url="http://test")
    try:
        with patch(
            "openai_anthropic_converter.servers.anthropic_server.httpx.AsyncClient",
            return_value=mock_client,
        ):
            return await client.post(path, json=json_body)
    finally:
        await client.aclose()


# ── OpenAI Server Tests ──────────────────────────────────────────────────


class TestOpenAIServerHealth:
    """Test OpenAI server health and models endpoints."""

    @pytest.mark.anyio
    async def test_health(self):
        resp = await _call_openai_server("get", "/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    @pytest.mark.anyio
    async def test_list_models(self):
        resp = await _call_openai_server("get", "/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 2
        model_ids = [m["id"] for m in data["data"]]
        assert "claude-sonnet-4-20250514" in model_ids


class TestOpenAIServerNonStreaming:
    """Test OpenAI server /v1/chat/completions non-streaming."""

    @pytest.mark.anyio
    async def test_basic_completion(self):
        """Test a basic non-streaming chat completion request."""
        anthropic_response = {
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-20250514",
            "content": [{"type": "text", "text": "Hello! How can I help?"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 8},
        }

        mock_client = _make_mock_client(post_return=_mock_httpx_response(200, anthropic_response))
        resp = await _patched_openai_call(
            mock_client,
            "/v1/chat/completions",
            {
                "model": "claude-sonnet-4-20250514",
                "messages": [{"role": "user", "content": "Hello!"}],
                "max_tokens": 1024,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["choices"][0]["message"]["content"] == "Hello! How can I help?"
        assert data["choices"][0]["finish_reason"] == "stop"
        assert data["usage"]["prompt_tokens"] == 10
        assert data["usage"]["completion_tokens"] == 8

    @pytest.mark.anyio
    async def test_tool_use_completion(self):
        """Test a completion with tool use response."""
        anthropic_response = {
            "id": "msg_456",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-20250514",
            "content": [
                {"type": "text", "text": "Let me search for that."},
                {
                    "type": "tool_use",
                    "id": "tu_1",
                    "name": "search",
                    "input": {"query": "weather"},
                },
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 20, "output_tokens": 30},
        }

        mock_client = _make_mock_client(post_return=_mock_httpx_response(200, anthropic_response))
        resp = await _patched_openai_call(
            mock_client,
            "/v1/chat/completions",
            {
                "model": "claude-sonnet-4-20250514",
                "messages": [{"role": "user", "content": "What's the weather?"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "search",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ],
                "max_tokens": 1024,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["choices"][0]["finish_reason"] == "tool_calls"
        tool_calls = data["choices"][0]["message"]["tool_calls"]
        assert len(tool_calls) == 1
        assert tool_calls[0]["function"]["name"] == "search"

    @pytest.mark.anyio
    async def test_backend_error(self):
        """Test handling of backend error response."""
        mock_client = _make_mock_client(
            post_return=_mock_httpx_response(500, {"error": "Internal server error"})
        )
        resp = await _patched_openai_call(
            mock_client,
            "/v1/chat/completions",
            {
                "model": "claude-sonnet-4-20250514",
                "messages": [{"role": "user", "content": "Hello!"}],
                "max_tokens": 1024,
            },
        )
        assert resp.status_code == 500

    @pytest.mark.anyio
    async def test_backend_timeout(self):
        """Test handling of backend timeout."""
        mock_client = _make_mock_client(post_side_effect=httpx.TimeoutException("timed out"))
        resp = await _patched_openai_call(
            mock_client,
            "/v1/chat/completions",
            {
                "model": "claude-sonnet-4-20250514",
                "messages": [{"role": "user", "content": "Hello!"}],
                "max_tokens": 1024,
            },
        )
        assert resp.status_code == 504

    @pytest.mark.anyio
    async def test_backend_connect_error(self):
        """Test handling of backend connection error."""
        mock_client = _make_mock_client(post_side_effect=httpx.ConnectError("Connection refused"))
        resp = await _patched_openai_call(
            mock_client,
            "/v1/chat/completions",
            {
                "model": "claude-sonnet-4-20250514",
                "messages": [{"role": "user", "content": "Hello!"}],
                "max_tokens": 1024,
            },
        )
        assert resp.status_code == 502

    @pytest.mark.anyio
    async def test_invalid_json_body(self):
        """Test handling of invalid JSON request body."""
        transport = httpx.ASGITransport(app=openai_app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/v1/chat/completions",
                content=b"not json",
                headers={"Content-Type": "application/json"},
            )
            assert resp.status_code == 400


# ── Anthropic Server Tests ───────────────────────────────────────────────


class TestAnthropicServerHealth:
    """Test Anthropic server health endpoint."""

    @pytest.mark.anyio
    async def test_health(self):
        resp = await _call_anthropic_server("get", "/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


class TestAnthropicServerNonStreaming:
    """Test Anthropic server /v1/messages non-streaming."""

    @pytest.mark.anyio
    async def test_basic_message(self):
        """Test a basic non-streaming messages request."""
        openai_response = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Hello! How can I help?",
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

        mock_client = _make_mock_client(post_return=_mock_httpx_response(200, openai_response))
        resp = await _patched_anthropic_call(
            mock_client,
            "/v1/messages",
            {
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "Hello!"}],
                "max_tokens": 1024,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["type"] == "message"
        assert data["role"] == "assistant"
        text_blocks = [b for b in data["content"] if b["type"] == "text"]
        assert len(text_blocks) == 1
        assert text_blocks[0]["text"] == "Hello! How can I help?"
        assert data["stop_reason"] == "end_turn"
        assert data["usage"]["input_tokens"] == 10
        assert data["usage"]["output_tokens"] == 8

    @pytest.mark.anyio
    async def test_tool_use_message(self):
        """Test a messages request with tool calls in the response."""
        openai_response = {
            "id": "chatcmpl-456",
            "object": "chat.completion",
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"city": "Tokyo"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {
                "prompt_tokens": 20,
                "completion_tokens": 15,
                "total_tokens": 35,
            },
        }

        mock_client = _make_mock_client(post_return=_mock_httpx_response(200, openai_response))
        resp = await _patched_anthropic_call(
            mock_client,
            "/v1/messages",
            {
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "What's the weather in Tokyo?"}],
                "max_tokens": 1024,
                "tools": [
                    {
                        "name": "get_weather",
                        "description": "Get weather",
                        "input_schema": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                        },
                    }
                ],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["stop_reason"] == "tool_use"
        tool_blocks = [b for b in data["content"] if b["type"] == "tool_use"]
        assert len(tool_blocks) == 1
        assert tool_blocks[0]["name"] == "get_weather"
        assert tool_blocks[0]["input"] == {"city": "Tokyo"}

    @pytest.mark.anyio
    async def test_backend_error_429(self):
        """Test handling of 429 backend error response."""
        mock_client = _make_mock_client(
            post_return=_mock_httpx_response(429, {"error": "Rate limited"})
        )
        resp = await _patched_anthropic_call(
            mock_client,
            "/v1/messages",
            {
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "Hello!"}],
                "max_tokens": 1024,
            },
        )
        assert resp.status_code == 429
        data = resp.json()
        assert data["type"] == "error"
        assert data["error"]["type"] == "rate_limit_error"

    @pytest.mark.anyio
    async def test_backend_error_401(self):
        """Test handling of 401 backend error response."""
        mock_client = _make_mock_client(
            post_return=_mock_httpx_response(401, {"error": "Unauthorized"})
        )
        resp = await _patched_anthropic_call(
            mock_client,
            "/v1/messages",
            {
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "Hello!"}],
                "max_tokens": 1024,
            },
        )
        assert resp.status_code == 401
        data = resp.json()
        assert data["type"] == "error"
        assert data["error"]["type"] == "authentication_error"

    @pytest.mark.anyio
    async def test_backend_error_403(self):
        """Test handling of 403 backend error response."""
        mock_client = _make_mock_client(
            post_return=_mock_httpx_response(403, {"error": "Forbidden"})
        )
        resp = await _patched_anthropic_call(
            mock_client,
            "/v1/messages",
            {
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "Hello!"}],
                "max_tokens": 1024,
            },
        )
        assert resp.status_code == 403
        data = resp.json()
        assert data["type"] == "error"
        assert data["error"]["type"] == "permission_error"

    @pytest.mark.anyio
    async def test_backend_timeout(self):
        """Test handling of backend timeout."""
        mock_client = _make_mock_client(post_side_effect=httpx.TimeoutException("timed out"))
        resp = await _patched_anthropic_call(
            mock_client,
            "/v1/messages",
            {
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "Hello!"}],
                "max_tokens": 1024,
            },
        )
        assert resp.status_code == 504
        data = resp.json()
        assert data["type"] == "error"
        assert data["error"]["type"] == "api_error"

    @pytest.mark.anyio
    async def test_backend_connect_error(self):
        """Test handling of backend connection error."""
        mock_client = _make_mock_client(post_side_effect=httpx.ConnectError("Connection refused"))
        resp = await _patched_anthropic_call(
            mock_client,
            "/v1/messages",
            {
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "Hello!"}],
                "max_tokens": 1024,
            },
        )
        assert resp.status_code == 502
        data = resp.json()
        assert data["type"] == "error"


class TestAnthropicServerCountTokens:
    """Test Anthropic server /v1/messages/count_tokens endpoint."""

    @pytest.mark.anyio
    async def test_count_tokens_string_content(self):
        resp = await _call_anthropic_server(
            "post",
            "/v1/messages/count_tokens",
            json={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "Hello world!"}],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        # "Hello world!" = 12 chars -> 12 // 4 = 3 tokens
        assert data["input_tokens"] == 3

    @pytest.mark.anyio
    async def test_count_tokens_with_system(self):
        resp = await _call_anthropic_server(
            "post",
            "/v1/messages/count_tokens",
            json={
                "model": "gpt-4o",
                "system": "You are a helpful assistant.",
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        # "You are a helpful assistant." = 29 chars, "Hi" = 2 chars -> 31 // 4 = 7
        assert data["input_tokens"] == 7

    @pytest.mark.anyio
    async def test_count_tokens_block_content(self):
        resp = await _call_anthropic_server(
            "post",
            "/v1/messages/count_tokens",
            json={
                "model": "gpt-4o",
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "Hello world!"}],
                    }
                ],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["input_tokens"] == 3

    @pytest.mark.anyio
    async def test_count_tokens_system_list(self):
        resp = await _call_anthropic_server(
            "post",
            "/v1/messages/count_tokens",
            json={
                "model": "gpt-4o",
                "system": [{"type": "text", "text": "Be helpful."}],
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        # "Be helpful." = 11 chars, "Hi" = 2 chars -> 13 // 4 = 3
        assert data["input_tokens"] == 3
