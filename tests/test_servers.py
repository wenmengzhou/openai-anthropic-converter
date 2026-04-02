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
    # Clear cached OpenAPI schema so each test gets fresh generation
    openai_app.openapi_schema = None
    anthropic_app.openapi_schema = None

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


class TestOpenAPIAndDebug:
    """Test OpenAPI schema, Swagger UI, and debug playground endpoints."""

    @pytest.mark.anyio
    async def test_openai_server_openapi_json(self):
        """OpenAI server should serve valid OpenAPI JSON at /openapi.json."""
        resp = await _call_openai_server("get", "/openapi.json")
        assert resp.status_code == 200
        schema = resp.json()
        assert schema["openapi"].startswith("3.")
        assert "paths" in schema
        assert "/v1/chat/completions" in schema["paths"]
        assert "/v1/models" in schema["paths"]
        assert "/health" in schema["paths"]
        # Should have request body schema
        completions_post = schema["paths"]["/v1/chat/completions"]["post"]
        assert "summary" in completions_post

    @pytest.mark.anyio
    async def test_anthropic_server_openapi_json(self):
        """Anthropic server should serve valid OpenAPI JSON at /openapi.json."""
        resp = await _call_anthropic_server("get", "/openapi.json")
        assert resp.status_code == 200
        schema = resp.json()
        assert schema["openapi"].startswith("3.")
        assert "/v1/messages" in schema["paths"]
        assert "/v1/messages/count_tokens" in schema["paths"]
        assert "/health" in schema["paths"]
        messages_post = schema["paths"]["/v1/messages"]["post"]
        assert "summary" in messages_post

    @pytest.mark.anyio
    async def test_openai_server_swagger_ui(self):
        """OpenAI server should serve Swagger UI at /docs."""
        resp = await _call_openai_server("get", "/docs")
        assert resp.status_code == 200
        assert "swagger" in resp.text.lower() or "openapi" in resp.text.lower()

    @pytest.mark.anyio
    async def test_anthropic_server_swagger_ui(self):
        """Anthropic server should serve Swagger UI at /docs."""
        resp = await _call_anthropic_server("get", "/docs")
        assert resp.status_code == 200
        assert "swagger" in resp.text.lower() or "openapi" in resp.text.lower()

    @pytest.mark.anyio
    async def test_openai_server_redoc(self):
        """OpenAI server should serve ReDoc at /redoc."""
        resp = await _call_openai_server("get", "/redoc")
        assert resp.status_code == 200

    @pytest.mark.anyio
    async def test_anthropic_server_redoc(self):
        """Anthropic server should serve ReDoc at /redoc."""
        resp = await _call_anthropic_server("get", "/redoc")
        assert resp.status_code == 200

    @pytest.mark.anyio
    async def test_openai_server_debug_playground(self):
        """OpenAI server should serve debug playground at /debug."""
        resp = await _call_openai_server("get", "/debug")
        assert resp.status_code == 200
        assert "Debug Playground" in resp.text
        assert "/v1/chat/completions" in resp.text
        assert "sendRequest" in resp.text

    @pytest.mark.anyio
    async def test_anthropic_server_debug_playground(self):
        """Anthropic server should serve debug playground at /debug."""
        resp = await _call_anthropic_server("get", "/debug")
        assert resp.status_code == 200
        assert "Debug Playground" in resp.text
        assert "/v1/messages" in resp.text
        assert "sendRequest" in resp.text

    @pytest.mark.anyio
    async def test_openai_openapi_schema_has_models(self):
        """OpenAPI schema should have request and response models in components."""
        resp = await _call_openai_server("get", "/openapi.json")
        schema = resp.json()
        components = schema.get("components", {}).get("schemas", {})
        # Response models (auto-generated by FastAPI)
        assert "OpenAIChatCompletionResponse" in components
        assert "OpenAIModelsResponse" in components
        assert "HealthResponse" in components
        # Request model (injected via custom openapi())
        assert "OpenAIChatCompletionRequest" in components
        req_schema = components["OpenAIChatCompletionRequest"]
        assert "model" in req_schema.get("properties", {})
        assert "messages" in req_schema.get("properties", {})

    @pytest.mark.anyio
    async def test_anthropic_openapi_schema_has_models(self):
        """OpenAPI schema should have request, response, and error models."""
        resp = await _call_anthropic_server("get", "/openapi.json")
        schema = resp.json()
        components = schema.get("components", {}).get("schemas", {})
        # Response/error models
        assert "AnthropicMessagesResponse" in components
        assert "AnthropicErrorResponse" in components
        assert "AnthropicCountTokensResponse" in components
        # Request models (injected via custom openapi())
        assert "AnthropicMessagesRequest" in components
        assert "AnthropicCountTokensRequest" in components
        req_schema = components["AnthropicMessagesRequest"]
        assert "model" in req_schema.get("properties", {})
        assert "messages" in req_schema.get("properties", {})


# ── Logging tests ────────────────────────────────────────────────────────


class TestLoggingConfig:
    """Tests for the shared logging configuration."""

    def test_setup_logging_creates_handlers(self, tmp_path):
        """setup_logging should create console and file handlers."""
        import logging

        from openai_anthropic_converter.servers.logging_config import setup_logging

        setup_logging("test_server", level="debug", log_dir=str(tmp_path))
        root = logging.getLogger()

        handler_types = [type(h).__name__ for h in root.handlers]
        assert "StreamHandler" in handler_types
        assert "RotatingFileHandler" in handler_types

        # Cleanup
        root.handlers.clear()

    def test_setup_logging_creates_log_file(self, tmp_path):
        """setup_logging should create the log file and write to it."""
        import logging

        from openai_anthropic_converter.servers.logging_config import setup_logging

        setup_logging("test_file_server", level="info", log_dir=str(tmp_path))
        logger = logging.getLogger("test_file_server")
        logger.info("test message for file output")

        log_file = tmp_path / "test_file_server.log"
        assert log_file.exists()
        content = log_file.read_text()
        assert "test message for file output" in content

        # Cleanup
        logging.getLogger().handlers.clear()

    def test_log_format_contains_expected_fields(self, tmp_path):
        """Log output should contain timestamp, level, filename, line number."""
        import logging

        from openai_anthropic_converter.servers.logging_config import setup_logging

        setup_logging("test_format_server", level="info", log_dir=str(tmp_path))
        logger = logging.getLogger("test_format_server")
        logger.warning("format check message")

        log_file = tmp_path / "test_format_server.log"
        content = log_file.read_text()
        assert "WARNING" in content
        assert "format check message" in content
        # filename:lineno pattern
        assert "test_servers.py:" in content

        # Cleanup
        logging.getLogger().handlers.clear()

    def test_setup_logging_creates_directory(self, tmp_path):
        """setup_logging should create the log directory if it doesn't exist."""
        import logging

        from openai_anthropic_converter.servers.logging_config import setup_logging

        log_dir = tmp_path / "nested" / "log" / "dir"
        setup_logging("test_dir_server", level="info", log_dir=str(log_dir))

        assert log_dir.exists()

        # Cleanup
        logging.getLogger().handlers.clear()
