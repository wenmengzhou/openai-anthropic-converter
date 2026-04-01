"""
Tests for Anthropic -> OpenAI conversion.
"""

import json
import pytest

from openai_anthropic_converter import AnthropicToOpenAIConverter


class TestRequestConversion:
    """Test Anthropic request -> OpenAI request conversion."""

    def test_basic_text_message(self):
        anthropic_req = {
            "model": "claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "Hello!"}],
            "max_tokens": 1024,
        }
        result, mapping = AnthropicToOpenAIConverter.convert_request(anthropic_req)

        assert result["model"] == "claude-sonnet-4-20250514"
        assert result["max_tokens"] == 1024
        assert len(result["messages"]) == 1
        assert result["messages"][0]["role"] == "user"
        assert mapping == {}

    def test_system_message(self):
        anthropic_req = {
            "model": "claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "Hi"}],
            "system": "You are helpful.",
            "max_tokens": 1024,
        }
        result, _ = AnthropicToOpenAIConverter.convert_request(anthropic_req)

        # System should be first message
        assert result["messages"][0]["role"] == "system"
        assert result["messages"][0]["content"] == "You are helpful."
        assert result["messages"][1]["role"] == "user"

    def test_system_message_list(self):
        anthropic_req = {
            "model": "claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "Hi"}],
            "system": [{"type": "text", "text": "System prompt here."}],
            "max_tokens": 1024,
        }
        result, _ = AnthropicToOpenAIConverter.convert_request(anthropic_req)

        assert result["messages"][0]["role"] == "system"

    def test_tool_conversion(self):
        anthropic_req = {
            "model": "claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "Search"}],
            "tools": [{
                "name": "search",
                "description": "Search the web",
                "input_schema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                },
            }],
            "max_tokens": 1024,
        }
        result, mapping = AnthropicToOpenAIConverter.convert_request(anthropic_req)

        assert "tools" in result
        assert result["tools"][0]["type"] == "function"
        assert result["tools"][0]["function"]["name"] == "search"
        assert "parameters" in result["tools"][0]["function"]
        assert mapping == {}

    def test_tool_name_truncation(self):
        """Tool names > 64 chars should be truncated with hash suffix."""
        long_name = "a" * 100
        anthropic_req = {
            "model": "claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "Hi"}],
            "tools": [{"name": long_name, "input_schema": {"type": "object", "properties": {}}}],
            "max_tokens": 1024,
        }
        result, mapping = AnthropicToOpenAIConverter.convert_request(anthropic_req)

        truncated = result["tools"][0]["function"]["name"]
        assert len(truncated) <= 64
        assert mapping[truncated] == long_name

    def test_tool_choice_any(self):
        anthropic_req = {
            "model": "claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "Hi"}],
            "tool_choice": {"type": "any"},
            "max_tokens": 1024,
        }
        result, _ = AnthropicToOpenAIConverter.convert_request(anthropic_req)
        assert result["tool_choice"] == "required"

    def test_tool_choice_tool(self):
        anthropic_req = {
            "model": "claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "Hi"}],
            "tool_choice": {"type": "tool", "name": "search"},
            "max_tokens": 1024,
        }
        result, _ = AnthropicToOpenAIConverter.convert_request(anthropic_req)
        assert result["tool_choice"]["type"] == "function"
        assert result["tool_choice"]["function"]["name"] == "search"

    def test_thinking_to_reasoning_effort(self):
        anthropic_req = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hi"}],
            "thinking": {"type": "enabled", "budget_tokens": 10000},
            "max_tokens": 1024,
        }
        result, _ = AnthropicToOpenAIConverter.convert_request(anthropic_req)
        assert result["reasoning_effort"] == "high"

    def test_thinking_disabled(self):
        anthropic_req = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hi"}],
            "thinking": {"type": "disabled"},
            "max_tokens": 1024,
        }
        result, _ = AnthropicToOpenAIConverter.convert_request(anthropic_req)
        assert "reasoning_effort" not in result

    def test_metadata_user_id(self):
        anthropic_req = {
            "model": "claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "Hi"}],
            "metadata": {"user_id": "user-123"},
            "max_tokens": 1024,
        }
        result, _ = AnthropicToOpenAIConverter.convert_request(anthropic_req)
        assert result["user"] == "user-123"

    def test_stop_sequences(self):
        anthropic_req = {
            "model": "claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "Hi"}],
            "stop_sequences": ["END"],
            "max_tokens": 1024,
        }
        result, _ = AnthropicToOpenAIConverter.convert_request(anthropic_req)
        assert result["stop"] == ["END"]

    def test_output_format_to_response_format(self):
        anthropic_req = {
            "model": "claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "Hi"}],
            "output_format": {
                "type": "json_schema",
                "schema": {"type": "object", "properties": {"name": {"type": "string"}}},
            },
            "max_tokens": 1024,
        }
        result, _ = AnthropicToOpenAIConverter.convert_request(anthropic_req)
        assert result["response_format"]["type"] == "json_schema"
        assert "json_schema" in result["response_format"]

    def test_image_in_user_message(self):
        anthropic_req = {
            "model": "claude-sonnet-4-20250514",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is this?"},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": "iVBOR...",
                        },
                    },
                ],
            }],
            "max_tokens": 1024,
        }
        result, _ = AnthropicToOpenAIConverter.convert_request(anthropic_req)

        user_content = result["messages"][0]["content"]
        assert len(user_content) == 2
        assert user_content[0]["type"] == "text"
        image_item = user_content[1]
        assert image_item["type"] == "image_url"
        assert "base64" in image_item["image_url"]["url"]

    def test_tool_result_conversion(self):
        anthropic_req = {
            "model": "claude-sonnet-4-20250514",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "call_123",
                            "content": "Result text here",
                        },
                    ],
                },
            ],
            "max_tokens": 1024,
        }
        result, _ = AnthropicToOpenAIConverter.convert_request(anthropic_req)

        # tool_result should become a tool role message
        tool_msgs = [m for m in result["messages"] if m.get("role") == "tool"]
        assert len(tool_msgs) == 1
        assert tool_msgs[0]["tool_call_id"] == "call_123"
        assert tool_msgs[0]["content"] == "Result text here"

    def test_assistant_with_tool_use(self):
        anthropic_req = {
            "model": "claude-sonnet-4-20250514",
            "messages": [
                {"role": "user", "content": "Search for python"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "I'll search for that."},
                        {
                            "type": "tool_use",
                            "id": "toolu_123",
                            "name": "search",
                            "input": {"query": "python"},
                        },
                    ],
                },
            ],
            "max_tokens": 1024,
        }
        result, _ = AnthropicToOpenAIConverter.convert_request(anthropic_req)

        assistant_msgs = [m for m in result["messages"] if m.get("role") == "assistant"]
        assert len(assistant_msgs) == 1
        assert assistant_msgs[0]["content"] == "I'll search for that."
        assert len(assistant_msgs[0]["tool_calls"]) == 1
        assert assistant_msgs[0]["tool_calls"][0]["function"]["name"] == "search"

    def test_web_search_tool_filtered(self):
        """Web search tools should be filtered out and converted to web_search_options."""
        anthropic_req = {
            "model": "claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "Search"}],
            "tools": [
                {"type": "web_search_20250305", "name": "web_search"},
                {"name": "custom_tool", "input_schema": {"type": "object", "properties": {}}},
            ],
            "max_tokens": 1024,
        }
        result, _ = AnthropicToOpenAIConverter.convert_request(anthropic_req)

        assert "web_search_options" in result
        # Only custom_tool should remain
        assert len(result["tools"]) == 1
        assert result["tools"][0]["function"]["name"] == "custom_tool"


class TestResponseConversion:
    """Test OpenAI response -> Anthropic response conversion."""

    def test_basic_text_response(self):
        openai_resp = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "claude-sonnet-4-20250514",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello!",
                },
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }
        result = AnthropicToOpenAIConverter.convert_response(openai_resp)

        assert result["type"] == "message"
        assert result["role"] == "assistant"
        assert result["stop_reason"] == "end_turn"
        assert len(result["content"]) == 1
        assert result["content"][0]["type"] == "text"
        assert result["content"][0]["text"] == "Hello!"

    def test_tool_calls_response(self):
        openai_resp = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "claude-sonnet-4-20250514",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Searching...",
                    "tool_calls": [{
                        "id": "call_123",
                        "type": "function",
                        "function": {
                            "name": "search",
                            "arguments": '{"query":"python"}',
                        },
                    }],
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {"prompt_tokens": 20, "completion_tokens": 15, "total_tokens": 35},
        }
        result = AnthropicToOpenAIConverter.convert_response(openai_resp)

        assert result["stop_reason"] == "tool_use"
        # Should have text + tool_use blocks
        text_blocks = [b for b in result["content"] if b["type"] == "text"]
        tool_blocks = [b for b in result["content"] if b["type"] == "tool_use"]
        assert len(text_blocks) == 1
        assert text_blocks[0]["text"] == "Searching..."
        assert len(tool_blocks) == 1
        assert tool_blocks[0]["name"] == "search"
        assert tool_blocks[0]["input"] == {"query": "python"}

    def test_tool_name_restoration(self):
        """Truncated tool names should be restored via mapping."""
        long_name = "a" * 100
        from openai_anthropic_converter.utils import truncate_tool_name

        truncated = truncate_tool_name(long_name)
        mapping = {truncated: long_name}

        openai_resp = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "test",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": truncated, "arguments": "{}"},
                    }],
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        result = AnthropicToOpenAIConverter.convert_response(
            openai_resp, tool_name_mapping=mapping
        )

        tool_blocks = [b for b in result["content"] if b["type"] == "tool_use"]
        assert tool_blocks[0]["name"] == long_name

    def test_thinking_blocks_response(self):
        openai_resp = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "test",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Answer",
                    "thinking_blocks": [
                        {"type": "thinking", "thinking": "Let me think...", "signature": "sig123"},
                    ],
                },
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        }
        result = AnthropicToOpenAIConverter.convert_response(openai_resp)

        thinking_blocks = [b for b in result["content"] if b["type"] == "thinking"]
        text_blocks = [b for b in result["content"] if b["type"] == "text"]
        assert len(thinking_blocks) == 1
        assert thinking_blocks[0]["thinking"] == "Let me think..."
        assert len(text_blocks) == 1

    def test_finish_reason_mapping(self):
        for openai_reason, anthropic_reason in [
            ("stop", "end_turn"),
            ("length", "max_tokens"),
            ("tool_calls", "tool_use"),
            ("content_filter", "end_turn"),
        ]:
            resp = {
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "created": 1700000000,
                "model": "test",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hi"},
                    "finish_reason": openai_reason,
                }],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
            result = AnthropicToOpenAIConverter.convert_response(resp)
            assert result["stop_reason"] == anthropic_reason

    def test_usage_conversion(self):
        openai_resp = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "test",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Hi"},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": 150,
                "completion_tokens": 50,
                "total_tokens": 200,
                "prompt_tokens_details": {
                    "cached_tokens": 30,
                    "cache_creation_tokens": 20,
                },
            },
        }
        result = AnthropicToOpenAIConverter.convert_response(openai_resp)

        usage = result["usage"]
        assert usage["input_tokens"] == 100  # 150 - 30 - 20
        assert usage["output_tokens"] == 50
        assert usage["cache_creation_input_tokens"] == 20
        assert usage["cache_read_input_tokens"] == 30


class TestStreamConversion:
    """Test OpenAI chunks -> Anthropic SSE conversion."""

    def test_basic_text_stream(self):
        chunks = [
            {
                "id": "chatcmpl-123",
                "object": "chat.completion.chunk",
                "created": 1700000000,
                "model": "claude-sonnet-4-20250514",
                "choices": [{
                    "index": 0,
                    "delta": {"role": "assistant", "content": ""},
                    "finish_reason": None,
                }],
            },
            {
                "id": "chatcmpl-123",
                "object": "chat.completion.chunk",
                "created": 1700000000,
                "model": "claude-sonnet-4-20250514",
                "choices": [{
                    "index": 0,
                    "delta": {"content": "Hello"},
                    "finish_reason": None,
                }],
            },
            {
                "id": "chatcmpl-123",
                "object": "chat.completion.chunk",
                "created": 1700000000,
                "model": "claude-sonnet-4-20250514",
                "choices": [{
                    "index": 0,
                    "delta": {"content": " world!"},
                    "finish_reason": None,
                }],
            },
            {
                "id": "chatcmpl-123",
                "object": "chat.completion.chunk",
                "created": 1700000000,
                "model": "claude-sonnet-4-20250514",
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }],
            },
        ]

        events = list(AnthropicToOpenAIConverter.convert_stream(chunks))

        # Should start with message_start
        assert events[0]["type"] == "message_start"

        # Should have content_block_start
        block_starts = [e for e in events if e["type"] == "content_block_start"]
        assert len(block_starts) >= 1

        # Should have text deltas
        text_deltas = [
            e for e in events
            if e["type"] == "content_block_delta" and e["delta"]["type"] == "text_delta"
        ]
        assert len(text_deltas) == 2
        assert text_deltas[0]["delta"]["text"] == "Hello"
        assert text_deltas[1]["delta"]["text"] == " world!"

        # Should end with message_delta + message_stop
        assert events[-1]["type"] == "message_stop"
        assert events[-2]["type"] == "message_delta"
        assert events[-2]["delta"]["stop_reason"] == "end_turn"

    def test_tool_call_stream(self):
        chunks = [
            {
                "id": "chatcmpl-123",
                "object": "chat.completion.chunk",
                "created": 1700000000,
                "model": "test",
                "choices": [{
                    "index": 0,
                    "delta": {"role": "assistant", "content": ""},
                    "finish_reason": None,
                }],
            },
            {
                "id": "chatcmpl-123",
                "object": "chat.completion.chunk",
                "created": 1700000000,
                "model": "test",
                "choices": [{
                    "index": 0,
                    "delta": {
                        "tool_calls": [{
                            "index": 0,
                            "id": "call_123",
                            "type": "function",
                            "function": {"name": "search", "arguments": ""},
                        }],
                    },
                    "finish_reason": None,
                }],
            },
            {
                "id": "chatcmpl-123",
                "object": "chat.completion.chunk",
                "created": 1700000000,
                "model": "test",
                "choices": [{
                    "index": 0,
                    "delta": {
                        "tool_calls": [{
                            "index": 0,
                            "function": {"arguments": '{"query":"python"}'},
                        }],
                    },
                    "finish_reason": None,
                }],
            },
            {
                "id": "chatcmpl-123",
                "object": "chat.completion.chunk",
                "created": 1700000000,
                "model": "test",
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "tool_calls",
                }],
            },
        ]

        events = list(AnthropicToOpenAIConverter.convert_stream(chunks))

        # Should have tool_use content_block_start
        tool_starts = [
            e for e in events
            if e["type"] == "content_block_start"
            and e.get("content_block", {}).get("type") == "tool_use"
        ]
        assert len(tool_starts) == 1
        assert tool_starts[0]["content_block"]["name"] == "search"

        # Should have input_json_delta
        json_deltas = [
            e for e in events
            if e["type"] == "content_block_delta"
            and e["delta"]["type"] == "input_json_delta"
        ]
        assert len(json_deltas) >= 1

        # Should end with tool_use stop_reason
        message_delta = [e for e in events if e["type"] == "message_delta"]
        assert message_delta[0]["delta"]["stop_reason"] == "tool_use"

    def test_tool_name_restoration_in_stream(self):
        """Truncated tool names should be restored in stream events."""
        long_name = "a" * 100
        from openai_anthropic_converter.utils import truncate_tool_name

        truncated = truncate_tool_name(long_name)
        mapping = {truncated: long_name}

        chunks = [
            {
                "id": "chatcmpl-123",
                "object": "chat.completion.chunk",
                "created": 1700000000,
                "model": "test",
                "choices": [{
                    "index": 0,
                    "delta": {"role": "assistant"},
                    "finish_reason": None,
                }],
            },
            {
                "id": "chatcmpl-123",
                "object": "chat.completion.chunk",
                "created": 1700000000,
                "model": "test",
                "choices": [{
                    "index": 0,
                    "delta": {
                        "tool_calls": [{
                            "index": 0,
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": truncated, "arguments": ""},
                        }],
                    },
                    "finish_reason": None,
                }],
            },
            {
                "id": "chatcmpl-123",
                "object": "chat.completion.chunk",
                "created": 1700000000,
                "model": "test",
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "tool_calls",
                }],
            },
        ]

        events = list(AnthropicToOpenAIConverter.convert_stream(
            chunks, tool_name_mapping=mapping
        ))

        tool_starts = [
            e for e in events
            if e["type"] == "content_block_start"
            and e.get("content_block", {}).get("type") == "tool_use"
        ]
        assert tool_starts[0]["content_block"]["name"] == long_name
