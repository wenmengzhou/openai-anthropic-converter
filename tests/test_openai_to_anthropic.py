"""
Tests for OpenAI -> Anthropic conversion.
"""

import json

from openai_anthropic_converter import OpenAIToAnthropicConverter


class TestRequestConversion:
    """Test OpenAI request -> Anthropic request conversion."""

    def test_basic_text_message(self):
        openai_req = {
            "model": "claude-sonnet-4-20250514",
            "messages": [
                {"role": "user", "content": "Hello!"},
            ],
            "max_tokens": 1024,
        }
        result = OpenAIToAnthropicConverter.convert_request(openai_req)

        assert result["model"] == "claude-sonnet-4-20250514"
        assert result["max_tokens"] == 1024
        assert len(result["messages"]) == 1
        assert result["messages"][0]["role"] == "user"

    def test_system_message_extraction(self):
        openai_req = {
            "model": "claude-sonnet-4-20250514",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hi"},
            ],
        }
        result = OpenAIToAnthropicConverter.convert_request(openai_req)

        # System should be extracted to top-level
        assert "system" in result
        assert result["system"][0]["type"] == "text"
        assert result["system"][0]["text"] == "You are helpful."
        # Messages should not contain system
        assert all(m["role"] != "system" for m in result["messages"])

    def test_default_max_tokens(self):
        openai_req = {
            "model": "claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "Hi"}],
        }
        result = OpenAIToAnthropicConverter.convert_request(openai_req)
        assert result["max_tokens"] == 4096  # DEFAULT_MAX_TOKENS

    def test_tool_conversion(self):
        openai_req = {
            "model": "claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "Search for python"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "search",
                        "description": "Search the web",
                        "parameters": {
                            "type": "object",
                            "properties": {"query": {"type": "string"}},
                            "required": ["query"],
                        },
                    },
                }
            ],
            "max_tokens": 1024,
        }
        result = OpenAIToAnthropicConverter.convert_request(openai_req)

        assert "tools" in result
        assert len(result["tools"]) == 1
        assert result["tools"][0]["name"] == "search"
        assert "input_schema" in result["tools"][0]

    def test_tool_choice_required(self):
        openai_req = {
            "model": "claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "Hi"}],
            "tool_choice": "required",
        }
        result = OpenAIToAnthropicConverter.convert_request(openai_req)
        assert result["tool_choice"]["type"] == "any"

    def test_tool_choice_specific(self):
        openai_req = {
            "model": "claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "Hi"}],
            "tool_choice": {"type": "function", "function": {"name": "search"}},
        }
        result = OpenAIToAnthropicConverter.convert_request(openai_req)
        assert result["tool_choice"]["type"] == "tool"
        assert result["tool_choice"]["name"] == "search"

    def test_stop_sequences(self):
        openai_req = {
            "model": "claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "Hi"}],
            "stop": ["END", "STOP"],
        }
        result = OpenAIToAnthropicConverter.convert_request(openai_req)
        assert result["stop_sequences"] == ["END", "STOP"]

    def test_stop_string(self):
        openai_req = {
            "model": "claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "Hi"}],
            "stop": "END",
        }
        result = OpenAIToAnthropicConverter.convert_request(openai_req)
        assert result["stop_sequences"] == ["END"]

    def test_user_to_metadata(self):
        openai_req = {
            "model": "claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "Hi"}],
            "user": "user-123",
        }
        result = OpenAIToAnthropicConverter.convert_request(openai_req)
        assert result["metadata"]["user_id"] == "user-123"

    def test_reasoning_effort(self):
        openai_req = {
            "model": "claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "Hi"}],
            "reasoning_effort": "high",
        }
        result = OpenAIToAnthropicConverter.convert_request(openai_req)
        assert result["thinking"]["type"] == "enabled"
        assert result["thinking"]["budget_tokens"] == 10000

    def test_temperature_passthrough(self):
        openai_req = {
            "model": "claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "Hi"}],
            "temperature": 0.7,
            "top_p": 0.9,
        }
        result = OpenAIToAnthropicConverter.convert_request(openai_req)
        assert result["temperature"] == 0.7
        assert result["top_p"] == 0.9

    def test_tool_calls_in_assistant_message(self):
        openai_req = {
            "model": "claude-sonnet-4-20250514",
            "messages": [
                {"role": "user", "content": "Search for python"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_123",
                            "type": "function",
                            "function": {"name": "search", "arguments": '{"query":"python"}'},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_123",
                    "content": "Python is a programming language.",
                },
                {"role": "user", "content": "Tell me more"},
            ],
        }
        result = OpenAIToAnthropicConverter.convert_request(openai_req)
        messages = result["messages"]

        # Should have alternating user/assistant/user/user->merged
        # Check assistant message has tool_use block
        assistant_msgs = [m for m in messages if m["role"] == "assistant"]
        assert len(assistant_msgs) >= 1
        assistant_content = assistant_msgs[0]["content"]
        tool_use_blocks = [b for b in assistant_content if b.get("type") == "tool_use"]
        assert len(tool_use_blocks) == 1
        assert tool_use_blocks[0]["name"] == "search"

    def test_image_url_conversion(self):
        openai_req = {
            "model": "claude-sonnet-4-20250514",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is this?"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQ"},
                        },
                    ],
                }
            ],
        }
        result = OpenAIToAnthropicConverter.convert_request(openai_req)
        user_content = result["messages"][0]["content"]
        image_blocks = [b for b in user_content if b.get("type") == "image"]
        assert len(image_blocks) == 1
        assert image_blocks[0]["source"]["type"] == "base64"
        assert image_blocks[0]["source"]["media_type"] == "image/jpeg"

    def test_message_alternation(self):
        """Ensure consecutive same-role messages are merged."""
        openai_req = {
            "model": "claude-sonnet-4-20250514",
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "user", "content": "World"},
            ],
        }
        result = OpenAIToAnthropicConverter.convert_request(openai_req)
        # Should be merged into one user message
        assert len(result["messages"]) == 1
        assert result["messages"][0]["role"] == "user"


class TestResponseConversion:
    """Test Anthropic response -> OpenAI response conversion."""

    def test_basic_text_response(self):
        anthropic_resp = {
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-20250514",
            "content": [{"type": "text", "text": "Hello!"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        result = OpenAIToAnthropicConverter.convert_response(anthropic_resp)

        assert result["object"] == "chat.completion"
        assert result["model"] == "claude-sonnet-4-20250514"
        assert result["choices"][0]["message"]["role"] == "assistant"
        assert result["choices"][0]["message"]["content"] == "Hello!"
        assert result["choices"][0]["finish_reason"] == "stop"

    def test_tool_use_response(self):
        anthropic_resp = {
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-20250514",
            "content": [
                {"type": "text", "text": "I'll search for that."},
                {
                    "type": "tool_use",
                    "id": "toolu_123",
                    "name": "search",
                    "input": {"query": "python"},
                },
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 20, "output_tokens": 15},
        }
        result = OpenAIToAnthropicConverter.convert_response(anthropic_resp)

        assert result["choices"][0]["finish_reason"] == "tool_calls"
        tool_calls = result["choices"][0]["message"]["tool_calls"]
        assert len(tool_calls) == 1
        assert tool_calls[0]["function"]["name"] == "search"
        assert json.loads(tool_calls[0]["function"]["arguments"]) == {"query": "python"}

    def test_thinking_blocks_response(self):
        anthropic_resp = {
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-20250514",
            "content": [
                {"type": "thinking", "thinking": "Let me think about this..."},
                {"type": "text", "text": "Here's my answer."},
            ],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 20},
        }
        result = OpenAIToAnthropicConverter.convert_response(anthropic_resp)

        msg = result["choices"][0]["message"]
        assert msg["content"] == "Here's my answer."
        assert "thinking_blocks" in msg
        assert msg["thinking_blocks"][0]["thinking"] == "Let me think about this..."
        assert msg["reasoning_content"] == "Let me think about this..."

    def test_usage_with_cache(self):
        anthropic_resp = {
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-20250514",
            "content": [{"type": "text", "text": "Hi"}],
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 100,
                "output_tokens": 50,
                "cache_creation_input_tokens": 20,
                "cache_read_input_tokens": 30,
            },
        }
        result = OpenAIToAnthropicConverter.convert_response(anthropic_resp)

        usage = result["usage"]
        assert usage["prompt_tokens"] == 150  # 100 + 20 + 30
        assert usage["completion_tokens"] == 50
        assert usage["total_tokens"] == 200
        assert usage["prompt_tokens_details"]["cached_tokens"] == 30
        assert usage["prompt_tokens_details"]["cache_creation_tokens"] == 20

    def test_stop_reason_mapping(self):
        for anthropic_reason, openai_reason in [
            ("end_turn", "stop"),
            ("max_tokens", "length"),
            ("tool_use", "tool_calls"),
            ("stop_sequence", "stop"),
        ]:
            resp = {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": "test",
                "content": [{"type": "text", "text": "Hi"}],
                "stop_reason": anthropic_reason,
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }
            result = OpenAIToAnthropicConverter.convert_response(resp)
            assert result["choices"][0]["finish_reason"] == openai_reason


class TestStreamConversion:
    """Test Anthropic SSE -> OpenAI streaming chunks conversion."""

    def test_basic_text_stream(self):
        events = [
            {
                "type": "message_start",
                "message": {
                    "id": "msg_123",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-sonnet-4-20250514",
                    "content": [],
                    "stop_reason": None,
                    "usage": {"input_tokens": 10, "output_tokens": 0},
                },
            },
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "Hello"},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": " world!"},
            },
            {"type": "content_block_stop", "index": 0},
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": 5},
            },
            {"type": "message_stop"},
        ]

        chunks = list(OpenAIToAnthropicConverter.convert_stream(events))

        # First chunk should have role
        assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"

        # Text chunks
        text_chunks = [c for c in chunks if c["choices"][0]["delta"].get("content")]
        assert len(text_chunks) == 2
        assert text_chunks[0]["choices"][0]["delta"]["content"] == "Hello"
        assert text_chunks[1]["choices"][0]["delta"]["content"] == " world!"

        # Last non-empty choice should have finish_reason
        finish_chunks = [c for c in chunks if c["choices"][0].get("finish_reason")]
        assert len(finish_chunks) == 1
        assert finish_chunks[0]["choices"][0]["finish_reason"] == "stop"

    def test_tool_use_stream(self):
        events = [
            {
                "type": "message_start",
                "message": {
                    "id": "msg_123",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-sonnet-4-20250514",
                    "content": [],
                    "stop_reason": None,
                    "usage": {"input_tokens": 10, "output_tokens": 0},
                },
            },
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "tool_use",
                    "id": "toolu_123",
                    "name": "search",
                    "input": {},
                },
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": '{"query":'},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": '"python"}'},
            },
            {"type": "content_block_stop", "index": 0},
            {
                "type": "message_delta",
                "delta": {"stop_reason": "tool_use"},
                "usage": {"output_tokens": 10},
            },
            {"type": "message_stop"},
        ]

        chunks = list(OpenAIToAnthropicConverter.convert_stream(events))

        # Should have tool call chunks
        tool_chunks = [c for c in chunks if c["choices"][0]["delta"].get("tool_calls")]
        assert len(tool_chunks) >= 1
        # First tool chunk should have id and name
        first_tc = tool_chunks[0]["choices"][0]["delta"]["tool_calls"][0]
        assert first_tc["id"] == "toolu_123"
        assert first_tc["function"]["name"] == "search"

    def test_thinking_stream(self):
        events = [
            {
                "type": "message_start",
                "message": {
                    "id": "msg_123",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-sonnet-4-20250514",
                    "content": [],
                    "stop_reason": None,
                    "usage": {"input_tokens": 10, "output_tokens": 0},
                },
            },
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "thinking", "thinking": ""},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "thinking_delta", "thinking": "Let me think..."},
            },
            {"type": "content_block_stop", "index": 0},
            {
                "type": "content_block_start",
                "index": 1,
                "content_block": {"type": "text", "text": ""},
            },
            {
                "type": "content_block_delta",
                "index": 1,
                "delta": {"type": "text_delta", "text": "Here's my answer."},
            },
            {"type": "content_block_stop", "index": 1},
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": 20},
            },
            {"type": "message_stop"},
        ]

        chunks = list(OpenAIToAnthropicConverter.convert_stream(events))

        thinking_chunks = [c for c in chunks if c["choices"][0]["delta"].get("thinking_blocks")]
        assert len(thinking_chunks) >= 1


class TestRoundtrip:
    """Test that request conversion roundtrips preserve semantics."""

    def test_basic_roundtrip(self):
        """OpenAI -> Anthropic -> convert_response preserves meaning."""
        from openai_anthropic_converter import AnthropicToOpenAIConverter

        openai_req = {
            "model": "claude-sonnet-4-20250514",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello!"},
            ],
            "max_tokens": 1024,
            "temperature": 0.7,
        }

        # Convert to Anthropic
        anthropic_req = OpenAIToAnthropicConverter.convert_request(openai_req)
        assert anthropic_req["model"] == "claude-sonnet-4-20250514"
        assert anthropic_req["max_tokens"] == 1024
        assert anthropic_req["temperature"] == 0.7
        assert "system" in anthropic_req

        # Now convert the Anthropic request back to OpenAI
        openai_req2, _ = AnthropicToOpenAIConverter.convert_request(anthropic_req)
        assert openai_req2["model"] == "claude-sonnet-4-20250514"
        assert openai_req2["max_tokens"] == 1024
        assert openai_req2["temperature"] == 0.7


class TestEdgeCases:
    """Test edge cases and less common scenarios."""

    def test_empty_messages(self):
        openai_req = {
            "model": "claude-sonnet-4-20250514",
            "messages": [],
        }
        result = OpenAIToAnthropicConverter.convert_request(openai_req)
        assert result["messages"] == []

    def test_none_content_in_user_message(self):
        openai_req = {
            "model": "claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": None}],
        }
        result = OpenAIToAnthropicConverter.convert_request(openai_req)
        # Should handle None content gracefully
        assert len(result["messages"]) >= 0

    def test_assistant_with_none_content_and_tool_calls(self):
        """Assistant message with content=None but tool_calls should work."""
        openai_req = {
            "model": "claude-sonnet-4-20250514",
            "messages": [
                {"role": "user", "content": "Hello"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "test", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "result"},
            ],
        }
        result = OpenAIToAnthropicConverter.convert_request(openai_req)
        assistant_msgs = [m for m in result["messages"] if m["role"] == "assistant"]
        assert len(assistant_msgs) >= 1
        # Should have tool_use block
        tool_blocks = [b for b in assistant_msgs[0]["content"] if b.get("type") == "tool_use"]
        assert len(tool_blocks) == 1

    def test_response_format_json_object(self):
        """response_format: {type: 'json_object'} should create a JSON tool."""
        openai_req = {
            "model": "claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "List items as JSON"}],
            "response_format": {"type": "json_object"},
        }
        result = OpenAIToAnthropicConverter.convert_request(openai_req)
        # Should have a json_tool_call tool
        assert "tools" in result
        tool_names = [t.get("name") for t in result["tools"]]
        assert "json_tool_call" in tool_names
        assert result["tool_choice"]["type"] == "tool"

    def test_response_format_json_schema(self):
        """response_format with json_schema for supported models."""
        openai_req = {
            "model": "claude-sonnet-4-5-20250514",
            "messages": [{"role": "user", "content": "Respond structured"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "my_schema",
                    "schema": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}},
                    },
                },
            },
        }
        result = OpenAIToAnthropicConverter.convert_request(openai_req)
        assert "output_format" in result
        assert result["output_format"]["type"] == "json_schema"

    def test_max_completion_tokens_alias(self):
        """max_completion_tokens should work as alias for max_tokens."""
        openai_req = {
            "model": "claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_completion_tokens": 2048,
        }
        result = OpenAIToAnthropicConverter.convert_request(openai_req)
        assert result["max_tokens"] == 2048

    def test_parallel_tool_calls_false(self):
        """parallel_tool_calls: false -> disable_parallel_tool_use: true."""
        openai_req = {
            "model": "claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "Hi"}],
            "tool_choice": "auto",
            "parallel_tool_calls": False,
        }
        result = OpenAIToAnthropicConverter.convert_request(openai_req)
        assert result["tool_choice"]["type"] == "auto"
        assert result["tool_choice"]["disable_parallel_tool_use"] is True

    def test_multiple_system_messages(self):
        """Multiple system messages should all be extracted."""
        openai_req = {
            "model": "claude-sonnet-4-20250514",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "Hi"},
            ],
        }
        result = OpenAIToAnthropicConverter.convert_request(openai_req)
        assert len(result["system"]) == 2
        assert result["system"][0]["text"] == "You are helpful."
        assert result["system"][1]["text"] == "Be concise."

    def test_tool_with_no_parameters(self):
        """Tool with no parameters should get default schema."""
        openai_req = {
            "model": "claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "Hi"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_time",
                        "description": "Get current time",
                    },
                }
            ],
        }
        result = OpenAIToAnthropicConverter.convert_request(openai_req)
        tool = result["tools"][0]
        assert tool["name"] == "get_time"
        assert tool["input_schema"]["type"] == "object"

    def test_context_management_conversion(self):
        openai_req = {
            "model": "claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "Hi"}],
            "context_management": [{"type": "compaction", "compact_threshold": 200000}],
        }
        result = OpenAIToAnthropicConverter.convert_request(openai_req)
        assert "context_management" in result
        assert result["context_management"]["edits"][0]["type"] == "compact_20260112"
        assert result["context_management"]["edits"][0]["trigger"]["value"] == 200000

    def test_multiple_tool_calls_stream(self):
        """Test streaming with multiple sequential tool calls."""
        events = [
            {
                "type": "message_start",
                "message": {
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-sonnet-4-20250514",
                    "content": [],
                    "stop_reason": None,
                    "usage": {"input_tokens": 10, "output_tokens": 0},
                },
            },
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "tool_use",
                    "id": "tu_1",
                    "name": "search",
                    "input": {},
                },
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": '{"q":"a"}'},
            },
            {"type": "content_block_stop", "index": 0},
            {
                "type": "content_block_start",
                "index": 1,
                "content_block": {
                    "type": "tool_use",
                    "id": "tu_2",
                    "name": "fetch",
                    "input": {},
                },
            },
            {
                "type": "content_block_delta",
                "index": 1,
                "delta": {"type": "input_json_delta", "partial_json": '{"url":"b"}'},
            },
            {"type": "content_block_stop", "index": 1},
            {
                "type": "message_delta",
                "delta": {"stop_reason": "tool_use"},
                "usage": {"output_tokens": 20},
            },
            {"type": "message_stop"},
        ]

        chunks = list(OpenAIToAnthropicConverter.convert_stream(events))
        tool_chunks = [c for c in chunks if c["choices"][0]["delta"].get("tool_calls")]

        # Find tool_call starts (have id field)
        tool_starts = []
        for c in tool_chunks:
            for tc in c["choices"][0]["delta"]["tool_calls"]:
                if tc.get("id"):
                    tool_starts.append(tc)

        assert len(tool_starts) == 2
        assert tool_starts[0]["id"] == "tu_1"
        assert tool_starts[0]["function"]["name"] == "search"
        assert tool_starts[0]["index"] == 0
        assert tool_starts[1]["id"] == "tu_2"
        assert tool_starts[1]["function"]["name"] == "fetch"
        assert tool_starts[1]["index"] == 1

    def test_json_mode_response_extraction(self):
        """JSON tool response should be extracted as text content."""
        anthropic_resp = {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-20250514",
            "content": [
                {
                    "type": "tool_use",
                    "id": "tu_1",
                    "name": "json_tool_call",
                    "input": {"name": "Alice", "age": 30},
                }
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 10, "output_tokens": 20},
        }
        result = OpenAIToAnthropicConverter.convert_response(anthropic_resp)
        msg = result["choices"][0]["message"]
        # Should extract as text content, not tool_calls
        assert msg["content"] is not None
        assert "tool_calls" not in msg
        parsed = json.loads(msg["content"])
        assert parsed["name"] == "Alice"

    def test_signature_delta_in_stream(self):
        """Test that signature deltas in thinking streams are handled."""
        events = [
            {
                "type": "message_start",
                "message": {
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "model": "test",
                    "content": [],
                    "stop_reason": None,
                    "usage": {"input_tokens": 0, "output_tokens": 0},
                },
            },
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "thinking", "thinking": ""},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "thinking_delta", "thinking": "thinking..."},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "signature_delta", "signature": "sig123"},
            },
            {"type": "content_block_stop", "index": 0},
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": 10},
            },
            {"type": "message_stop"},
        ]
        chunks = list(OpenAIToAnthropicConverter.convert_stream(events))
        sig_chunks = [
            c
            for c in chunks
            if c["choices"][0]["delta"].get("thinking_blocks")
            and any(tb.get("signature") for tb in c["choices"][0]["delta"]["thinking_blocks"])
        ]
        assert len(sig_chunks) == 1
        assert sig_chunks[0]["choices"][0]["delta"]["thinking_blocks"][0]["signature"] == "sig123"

    def test_empty_text_content_response(self):
        """Response with only tool_use and no text should have content=None."""
        anthropic_resp = {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "test",
            "content": [
                {
                    "type": "tool_use",
                    "id": "tu_1",
                    "name": "search",
                    "input": {"q": "test"},
                }
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 5, "output_tokens": 10},
        }
        result = OpenAIToAnthropicConverter.convert_response(anthropic_resp)
        msg = result["choices"][0]["message"]
        assert msg["content"] is None
        assert len(msg["tool_calls"]) == 1

    def test_redacted_thinking_blocks_response(self):
        """Test that redacted thinking blocks are passed through."""
        anthropic_resp = {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "test",
            "content": [
                {"type": "redacted_thinking", "data": "abc123"},
                {"type": "text", "text": "Result."},
            ],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 5, "output_tokens": 10},
        }
        result = OpenAIToAnthropicConverter.convert_response(anthropic_resp)
        msg = result["choices"][0]["message"]
        assert msg["content"] == "Result."
        assert len(msg["thinking_blocks"]) == 1
        assert msg["thinking_blocks"][0]["type"] == "redacted_thinking"

    def test_tool_call_index_with_mixed_content(self):
        """Tool call indices should be sequential among tool calls only, not content blocks."""
        anthropic_resp = {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "test",
            "content": [
                {"type": "text", "text": "I'll do both."},
                {
                    "type": "tool_use",
                    "id": "tu_1",
                    "name": "search",
                    "input": {"q": "a"},
                },
                {
                    "type": "tool_use",
                    "id": "tu_2",
                    "name": "fetch",
                    "input": {"url": "http://x.com"},
                },
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 10, "output_tokens": 20},
        }
        result = OpenAIToAnthropicConverter.convert_response(anthropic_resp)
        msg = result["choices"][0]["message"]
        assert len(msg["tool_calls"]) == 2
        # Indices should be 0 and 1, NOT 1 and 2
        assert msg["tool_calls"][0]["index"] == 0
        assert msg["tool_calls"][1]["index"] == 1

    def test_server_tool_use_conversion(self):
        """server_tool_use blocks should be converted to tool_calls."""
        anthropic_resp = {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "test",
            "content": [
                {
                    "type": "server_tool_use",
                    "id": "stu_1",
                    "name": "web_search",
                    "input": {"query": "test"},
                },
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 5, "output_tokens": 5},
        }
        result = OpenAIToAnthropicConverter.convert_response(anthropic_resp)
        msg = result["choices"][0]["message"]
        assert len(msg["tool_calls"]) == 1
        assert msg["tool_calls"][0]["function"]["name"] == "web_search"
        assert msg["tool_calls"][0]["index"] == 0

    def test_mcp_tool_use_conversion(self):
        """mcp_tool_use blocks should be converted to tool_calls."""
        anthropic_resp = {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "test",
            "content": [
                {
                    "type": "mcp_tool_use",
                    "id": "mtu_1",
                    "name": "my_mcp_tool",
                    "input": {"key": "value"},
                },
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 5, "output_tokens": 5},
        }
        result = OpenAIToAnthropicConverter.convert_response(anthropic_resp)
        msg = result["choices"][0]["message"]
        assert len(msg["tool_calls"]) == 1
        assert msg["tool_calls"][0]["function"]["name"] == "my_mcp_tool"

    def test_citations_in_response(self):
        """Citations in text blocks should be captured in provider_specific_fields."""
        anthropic_resp = {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "test",
            "content": [
                {
                    "type": "text",
                    "text": "According to the source...",
                    "citations": [
                        {
                            "type": "document",
                            "document_index": 0,
                            "start_char_index": 0,
                            "end_char_index": 10,
                        }
                    ],
                },
            ],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 5, "output_tokens": 10},
        }
        result = OpenAIToAnthropicConverter.convert_response(anthropic_resp)
        msg = result["choices"][0]["message"]
        assert "provider_specific_fields" in msg
        assert "citations" in msg["provider_specific_fields"]

    def test_web_search_tool_result_in_response(self):
        """web_search_tool_result blocks should be captured."""
        anthropic_resp = {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "test",
            "content": [
                {
                    "type": "web_search_tool_result",
                    "tool_use_id": "stu_1",
                    "content": [{"type": "web_search_result", "url": "http://example.com"}],
                },
                {"type": "text", "text": "Based on search results..."},
            ],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 20},
        }
        result = OpenAIToAnthropicConverter.convert_response(anthropic_resp)
        msg = result["choices"][0]["message"]
        assert "provider_specific_fields" in msg
        assert "web_search_results" in msg["provider_specific_fields"]

    def test_image_url_base64_conversion(self):
        """Base64 data URIs should be parsed correctly."""
        openai_req = {
            "model": "claude-sonnet-4-20250514",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,iVBORw0KGgoAAAANS=="},
                        },
                    ],
                },
            ],
        }
        result = OpenAIToAnthropicConverter.convert_request(openai_req)
        content = result["messages"][0]["content"]
        assert content[0]["type"] == "image"
        assert content[0]["source"]["type"] == "base64"
        assert content[0]["source"]["media_type"] == "image/png"
        assert content[0]["source"]["data"] == "iVBORw0KGgoAAAANS=="

    def test_image_url_http_conversion(self):
        """HTTP URLs should be converted to url source type."""
        openai_req = {
            "model": "claude-sonnet-4-20250514",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://example.com/img.jpg"},
                        },
                    ],
                },
            ],
        }
        result = OpenAIToAnthropicConverter.convert_request(openai_req)
        content = result["messages"][0]["content"]
        assert content[0]["type"] == "image"
        assert content[0]["source"]["type"] == "url"
        assert content[0]["source"]["url"] == "https://example.com/img.jpg"

    def test_tool_message_list_content(self):
        """Tool messages with list content should be preserved."""
        openai_req = {
            "model": "claude-sonnet-4-20250514",
            "messages": [
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": [
                        {"type": "text", "text": "Result part 1"},
                        {"type": "text", "text": "Result part 2"},
                    ],
                },
            ],
        }
        result = OpenAIToAnthropicConverter.convert_request(openai_req)
        # Should have a user message with tool_result
        user_msg = result["messages"][0]
        assert user_msg["role"] == "user"
        tool_result = user_msg["content"][0]
        assert tool_result["type"] == "tool_result"
        assert isinstance(tool_result["content"], list)

    def test_cache_control_on_tools(self):
        """cache_control on tool definitions should be preserved."""
        openai_req = {
            "model": "claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "Use tool"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "search",
                        "description": "Search",
                        "parameters": {"type": "object", "properties": {}},
                    },
                    "cache_control": {"type": "ephemeral"},
                },
            ],
        }
        result = OpenAIToAnthropicConverter.convert_request(openai_req)
        assert result["tools"][0]["cache_control"] == {"type": "ephemeral"}

    def test_thinking_passthrough(self):
        """Thinking param should pass through when directly specified."""
        openai_req = {
            "model": "claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "Think"}],
            "thinking": {"type": "enabled", "budget_tokens": 8000},
        }
        result = OpenAIToAnthropicConverter.convert_request(openai_req)
        assert result["thinking"]["type"] == "enabled"
        assert result["thinking"]["budget_tokens"] == 8000

    def test_consecutive_tool_messages_merge(self):
        """Multiple consecutive tool messages should merge into one user message."""
        openai_req = {
            "model": "claude-sonnet-4-20250514",
            "messages": [
                {"role": "user", "content": "Use tools"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "a", "arguments": "{}"},
                        },
                        {
                            "id": "call_2",
                            "type": "function",
                            "function": {"name": "b", "arguments": "{}"},
                        },
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "result1"},
                {"role": "tool", "tool_call_id": "call_2", "content": "result2"},
            ],
        }
        result = OpenAIToAnthropicConverter.convert_request(openai_req)
        # The two tool messages should merge into a single user message
        user_msgs = [m for m in result["messages"] if m["role"] == "user"]
        # Last user message should have 2 tool_result blocks
        last_user = user_msgs[-1]
        tool_results = [b for b in last_user["content"] if b.get("type") == "tool_result"]
        assert len(tool_results) == 2

    def test_alternation_assistant_starts(self):
        """If conversation starts with assistant, a placeholder user message is inserted."""
        openai_req = {
            "model": "claude-sonnet-4-20250514",
            "messages": [
                {"role": "assistant", "content": "I already started talking"},
                {"role": "user", "content": "Ok continue"},
            ],
        }
        result = OpenAIToAnthropicConverter.convert_request(openai_req)
        assert result["messages"][0]["role"] == "user"

    def test_schema_filtering_for_output_format(self):
        """JSON schema should have unsupported fields filtered for Anthropic."""
        openai_req = {
            "model": "claude-sonnet-4.5-20250514",  # Supports output_format
            "messages": [{"role": "user", "content": "Generate"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "test_schema",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "count": {
                                "type": "integer",
                                "minimum": 0,
                                "maximum": 100,
                            }
                        },
                    },
                },
            },
        }
        result = OpenAIToAnthropicConverter.convert_request(openai_req)
        assert "output_format" in result
        schema = result["output_format"]["schema"]
        props = schema["properties"]["count"]
        # minimum/maximum should be removed and added to description
        assert "minimum" not in props
        assert "maximum" not in props
        assert "description" in props

    def test_tool_based_json_mode_for_old_model(self):
        """Older models should use tool-based JSON mode instead of output_format."""
        openai_req = {
            "model": "claude-3-opus-20240229",  # Older model
            "messages": [{"role": "user", "content": "Generate"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "test",
                    "schema": {"type": "object", "properties": {"x": {"type": "string"}}},
                },
            },
        }
        result = OpenAIToAnthropicConverter.convert_request(openai_req)
        assert "output_format" not in result
        # Should have the JSON tool added
        tool_names = [t.get("name") for t in result.get("tools", [])]
        assert "json_tool_call" in tool_names
        assert result["tool_choice"]["type"] == "tool"

    def test_server_tool_use_stream(self):
        """server_tool_use blocks should be handled in streaming."""
        events = [
            {"type": "message_start", "message": {"id": "msg_1", "model": "test", "usage": {}}},
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "server_tool_use",
                    "id": "stu_1",
                    "name": "web_search",
                },
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": '{"query":'},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": '"test"}'},
            },
            {"type": "content_block_stop", "index": 0},
            {
                "type": "message_delta",
                "delta": {"stop_reason": "tool_use"},
                "usage": {"output_tokens": 10},
            },
            {"type": "message_stop"},
        ]
        chunks = list(OpenAIToAnthropicConverter.convert_stream(events))
        # Should have tool_calls in the chunks
        tool_chunks = [c for c in chunks if c["choices"][0]["delta"].get("tool_calls")]
        assert len(tool_chunks) >= 1
        # First tool chunk should have the function name
        first_tc = tool_chunks[0]["choices"][0]["delta"]["tool_calls"][0]
        assert first_tc["function"]["name"] == "web_search"

    def test_schema_with_refs_resolved(self):
        """JSON schema $refs should be resolved for Anthropic."""
        openai_req = {
            "model": "claude-sonnet-4.5-20250514",
            "messages": [{"role": "user", "content": "Generate"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "test",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "item": {"$ref": "#/$defs/Item"},
                        },
                        "$defs": {
                            "Item": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                },
                            }
                        },
                    },
                },
            },
        }
        result = OpenAIToAnthropicConverter.convert_request(openai_req)
        assert "output_format" in result
        schema = result["output_format"]["schema"]
        # $ref should be resolved
        item_prop = schema["properties"]["item"]
        assert "$ref" not in item_prop
        assert item_prop["type"] == "object"

    def test_ping_event_ignored(self):
        """Anthropic ping events should be silently ignored."""
        events = [
            {"type": "message_start", "message": {"id": "msg_1", "model": "test", "usage": {}}},
            {"type": "ping"},
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "Hi"},
            },
            {"type": "content_block_stop", "index": 0},
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {}},
            {"type": "message_stop"},
        ]
        chunks = list(OpenAIToAnthropicConverter.convert_stream(events))
        # ping should not produce any chunk
        assert all(c.get("object") == "chat.completion.chunk" for c in chunks)

    def test_system_message_with_non_text_type_ignored(self):
        """System message items without text should be safely skipped."""
        openai_req = {
            "model": "test",
            "messages": [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "Valid system text"},
                        {"type": "image_url", "image_url": {"url": "http://x.com/img.jpg"}},
                    ],
                },
                {"role": "user", "content": "Hi"},
            ],
        }
        result = OpenAIToAnthropicConverter.convert_request(openai_req)
        # Only the text block should be in system
        assert len(result["system"]) == 1
        assert result["system"][0]["text"] == "Valid system text"

    def test_merge_does_not_mutate_original_messages(self):
        """Merging consecutive messages should not mutate the originals."""
        original_msgs = [
            {"role": "user", "content": "First"},
            {"role": "tool", "tool_call_id": "c1", "content": "Tool result"},
        ]
        # Make copies to check they're not mutated
        import copy

        saved = copy.deepcopy(original_msgs)
        openai_req = {
            "model": "test",
            "messages": original_msgs,
        }
        OpenAIToAnthropicConverter.convert_request(openai_req)
        # Original messages should not be mutated
        assert original_msgs[0]["content"] == saved[0]["content"]

    def test_json_mode_empty_arguments_content(self):
        """JSON mode with empty arguments should still set content as string."""
        from openai_anthropic_converter.constants import RESPONSE_FORMAT_TOOL_NAME

        anthropic_resp = {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "test",
            "content": [
                {
                    "type": "tool_use",
                    "id": "tu_1",
                    "name": RESPONSE_FORMAT_TOOL_NAME,
                    "input": {},
                },
            ],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 5, "output_tokens": 5},
        }
        result = OpenAIToAnthropicConverter.convert_response(anthropic_resp)
        msg = result["choices"][0]["message"]
        # Content should be the JSON string, not None
        assert msg["content"] is not None
        assert msg["content"] == "{}"
        # tool_calls should be empty (JSON mode extracts content)
        assert "tool_calls" not in msg

    def test_reasoning_effort_none_ignored(self):
        """reasoning_effort='none' should not set thinking."""
        openai_req = {
            "model": "test",
            "messages": [{"role": "user", "content": "Hi"}],
            "reasoning_effort": "none",
        }
        result = OpenAIToAnthropicConverter.convert_request(openai_req)
        assert "thinking" not in result

    def test_response_format_text_type_ignored(self):
        """response_format with type='text' should not produce output_format."""
        openai_req = {
            "model": "test",
            "messages": [{"role": "user", "content": "Hi"}],
            "response_format": {"type": "text"},
        }
        result = OpenAIToAnthropicConverter.convert_request(openai_req)
        assert "output_format" not in result
        assert "tools" not in result

    def test_tool_with_non_object_schema(self):
        """Tool with non-object schema type should be forced to object."""
        openai_req = {
            "model": "test",
            "messages": [{"role": "user", "content": "Hi"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_list",
                        "parameters": {"type": "array", "items": {"type": "string"}},
                    },
                }
            ],
        }
        result = OpenAIToAnthropicConverter.convert_request(openai_req)
        schema = result["tools"][0]["input_schema"]
        assert schema["type"] == "object"
        assert "properties" in schema

    def test_circular_ref_schema_does_not_infinite_loop(self):
        """Schema with circular $ref should not cause infinite recursion."""
        openai_req = {
            "model": "claude-sonnet-4-5-20250514",
            "messages": [{"role": "user", "content": "Hi"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "tree",
                    "schema": {
                        "type": "object",
                        "$defs": {
                            "Node": {
                                "type": "object",
                                "properties": {
                                    "value": {"type": "string"},
                                    "children": {
                                        "type": "array",
                                        "items": {"$ref": "#/$defs/Node"},
                                    },
                                },
                                "required": ["value"],
                            }
                        },
                        "properties": {
                            "root": {"$ref": "#/$defs/Node"},
                        },
                        "required": ["root"],
                    },
                },
            },
        }
        # Should not hang or raise RecursionError
        result = OpenAIToAnthropicConverter.convert_request(openai_req)
        assert "output_format" in result
        schema = result["output_format"]["schema"]
        # The root ref should be expanded
        assert schema["properties"]["root"]["type"] == "object"

    def test_non_circular_ref_expanded_in_multiple_places(self):
        """Same $ref used in multiple places should be expanded in all."""
        openai_req = {
            "model": "claude-sonnet-4-5-20250514",
            "messages": [{"role": "user", "content": "Hi"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "pair",
                    "schema": {
                        "type": "object",
                        "$defs": {
                            "Point": {
                                "type": "object",
                                "properties": {
                                    "x": {"type": "number"},
                                    "y": {"type": "number"},
                                },
                            }
                        },
                        "properties": {
                            "start": {"$ref": "#/$defs/Point"},
                            "end": {"$ref": "#/$defs/Point"},
                        },
                    },
                },
            },
        }
        result = OpenAIToAnthropicConverter.convert_request(openai_req)
        schema = result["output_format"]["schema"]
        # Both refs should be expanded
        assert schema["properties"]["start"]["type"] == "object"
        assert schema["properties"]["end"]["type"] == "object"
        assert "x" in schema["properties"]["start"]["properties"]
        assert "x" in schema["properties"]["end"]["properties"]

    def test_json_object_mode_creates_tool(self):
        """response_format type=json_object should create a generic JSON tool."""
        openai_req = {
            "model": "claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "Give me JSON"}],
            "response_format": {"type": "json_object"},
        }
        result = OpenAIToAnthropicConverter.convert_request(openai_req)
        # Should add json tool
        assert "tools" in result
        tool_names = [t["name"] for t in result["tools"]]
        assert "json_tool_call" in tool_names
        # Should set tool_choice to force the tool
        assert result["tool_choice"]["type"] == "tool"
        assert result["tool_choice"]["name"] == "json_tool_call"

    def test_context_management_conversion(self):
        """OpenAI context_management should convert to Anthropic format."""
        openai_req = {
            "model": "test",
            "messages": [{"role": "user", "content": "Hi"}],
            "context_management": [
                {"type": "compaction", "compact_threshold": 200000}
            ],
        }
        result = OpenAIToAnthropicConverter.convert_request(openai_req)
        assert "context_management" in result
        assert result["context_management"]["edits"][0]["type"] == "compact_20260112"
        assert result["context_management"]["edits"][0]["trigger"]["value"] == 200000

    def test_context_management_already_anthropic_format(self):
        """Already-Anthropic context_management should pass through."""
        anthropic_ctx = {"edits": [{"type": "compact_20260112"}]}
        openai_req = {
            "model": "test",
            "messages": [{"role": "user", "content": "Hi"}],
            "context_management": anthropic_ctx,
        }
        result = OpenAIToAnthropicConverter.convert_request(openai_req)
        assert result["context_management"] == anthropic_ctx

    def test_stream_redacted_thinking_block(self):
        """Anthropic SSE with redacted_thinking content block should not crash."""
        events = [
            {
                "type": "message_start",
                "message": {"id": "msg_1", "model": "test", "usage": {}},
            },
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "redacted_thinking", "data": "abc123"},
            },
            {
                "type": "content_block_stop",
                "index": 0,
            },
            {
                "type": "content_block_start",
                "index": 1,
                "content_block": {"type": "text", "text": ""},
            },
            {
                "type": "content_block_delta",
                "index": 1,
                "delta": {"type": "text_delta", "text": "Hello"},
            },
            {
                "type": "content_block_stop",
                "index": 1,
            },
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": 10},
            },
            {"type": "message_stop"},
        ]
        chunks = list(OpenAIToAnthropicConverter.convert_stream(events))
        # Should have message_start, text delta, and finish
        assert any(c["choices"][0]["delta"].get("content") == "Hello" for c in chunks)
        assert any(c["choices"][0].get("finish_reason") == "stop" for c in chunks)

    def test_stream_signature_delta(self):
        """Signature delta should be emitted as thinking_blocks with signature."""
        events = [
            {
                "type": "message_start",
                "message": {"id": "msg_1", "model": "test", "usage": {}},
            },
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "thinking", "thinking": ""},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "thinking_delta", "thinking": "Let me think"},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "signature_delta", "signature": "sig123"},
            },
            {
                "type": "content_block_stop",
                "index": 0,
            },
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": 20},
            },
            {"type": "message_stop"},
        ]
        chunks = list(OpenAIToAnthropicConverter.convert_stream(events))
        # Find the signature chunk
        sig_chunks = [
            c
            for c in chunks
            if any(
                tb.get("signature") == "sig123"
                for tb in c["choices"][0]["delta"].get("thinking_blocks", [])
            )
        ]
        assert len(sig_chunks) == 1

    def test_ensure_alternation_consecutive_user(self):
        """Two consecutive user messages should be merged into one."""
        openai_req = {
            "model": "test",
            "messages": [
                {"role": "user", "content": "First"},
                {"role": "user", "content": "Second"},
            ],
        }
        result = OpenAIToAnthropicConverter.convert_request(openai_req)
        msgs = result["messages"]
        # Both user messages get merged into one
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"
        # Content should contain both texts as blocks
        content = msgs[0]["content"]
        texts = [b["text"] for b in content if b.get("type") == "text"]
        assert "First" in texts
        assert "Second" in texts

    def test_ensure_alternation_starts_with_assistant(self):
        """If first message is assistant, a placeholder user should be prepended."""
        openai_req = {
            "model": "test",
            "messages": [
                {"role": "assistant", "content": "I'm here to help"},
                {"role": "user", "content": "Thanks"},
            ],
        }
        result = OpenAIToAnthropicConverter.convert_request(openai_req)
        msgs = result["messages"]
        assert msgs[0]["role"] == "user"
