"""
Tests for Anthropic -> OpenAI conversion.
"""

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
            "tools": [
                {
                    "name": "search",
                    "description": "Search the web",
                    "input_schema": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                    },
                }
            ],
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
            "messages": [
                {
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
                }
            ],
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
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Hello!",
                    },
                    "finish_reason": "stop",
                }
            ],
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
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Searching...",
                        "tool_calls": [
                            {
                                "id": "call_123",
                                "type": "function",
                                "function": {
                                    "name": "search",
                                    "arguments": '{"query":"python"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
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
                                "function": {"name": truncated, "arguments": "{}"},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        result = AnthropicToOpenAIConverter.convert_response(openai_resp, tool_name_mapping=mapping)

        tool_blocks = [b for b in result["content"] if b["type"] == "tool_use"]
        assert tool_blocks[0]["name"] == long_name

    def test_thinking_blocks_response(self):
        openai_resp = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "test",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Answer",
                        "thinking_blocks": [
                            {
                                "type": "thinking",
                                "thinking": "Let me think...",
                                "signature": "sig123",
                            },
                        ],
                    },
                    "finish_reason": "stop",
                }
            ],
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
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hi"},
                        "finish_reason": openai_reason,
                    }
                ],
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
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hi"},
                    "finish_reason": "stop",
                }
            ],
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
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": ""},
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": "chatcmpl-123",
                "object": "chat.completion.chunk",
                "created": 1700000000,
                "model": "claude-sonnet-4-20250514",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": "Hello"},
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": "chatcmpl-123",
                "object": "chat.completion.chunk",
                "created": 1700000000,
                "model": "claude-sonnet-4-20250514",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": " world!"},
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": "chatcmpl-123",
                "object": "chat.completion.chunk",
                "created": 1700000000,
                "model": "claude-sonnet-4-20250514",
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop",
                    }
                ],
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
            e
            for e in events
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
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": ""},
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": "chatcmpl-123",
                "object": "chat.completion.chunk",
                "created": 1700000000,
                "model": "test",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_123",
                                    "type": "function",
                                    "function": {"name": "search", "arguments": ""},
                                }
                            ],
                        },
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": "chatcmpl-123",
                "object": "chat.completion.chunk",
                "created": 1700000000,
                "model": "test",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {"arguments": '{"query":"python"}'},
                                }
                            ],
                        },
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": "chatcmpl-123",
                "object": "chat.completion.chunk",
                "created": 1700000000,
                "model": "test",
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "tool_calls",
                    }
                ],
            },
        ]

        events = list(AnthropicToOpenAIConverter.convert_stream(chunks))

        # Should have tool_use content_block_start
        tool_starts = [
            e
            for e in events
            if e["type"] == "content_block_start"
            and e.get("content_block", {}).get("type") == "tool_use"
        ]
        assert len(tool_starts) == 1
        assert tool_starts[0]["content_block"]["name"] == "search"

        # Should have input_json_delta
        json_deltas = [
            e
            for e in events
            if e["type"] == "content_block_delta" and e["delta"]["type"] == "input_json_delta"
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
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant"},
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": "chatcmpl-123",
                "object": "chat.completion.chunk",
                "created": 1700000000,
                "model": "test",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {"name": truncated, "arguments": ""},
                                }
                            ],
                        },
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": "chatcmpl-123",
                "object": "chat.completion.chunk",
                "created": 1700000000,
                "model": "test",
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "tool_calls",
                    }
                ],
            },
        ]

        events = list(AnthropicToOpenAIConverter.convert_stream(chunks, tool_name_mapping=mapping))

        tool_starts = [
            e
            for e in events
            if e["type"] == "content_block_start"
            and e.get("content_block", {}).get("type") == "tool_use"
        ]
        assert tool_starts[0]["content_block"]["name"] == long_name


class TestEdgeCases:
    """Edge case tests for Anthropic -> OpenAI conversion."""

    # --- Request edge cases ---

    def test_system_message_with_cache_control(self):
        """System blocks with cache_control should be preserved."""
        anthropic_req = {
            "model": "test",
            "messages": [{"role": "user", "content": "Hi"}],
            "system": [
                {
                    "type": "text",
                    "text": "System prompt",
                    "cache_control": {"type": "ephemeral"},
                },
            ],
            "max_tokens": 1024,
        }
        result, _ = AnthropicToOpenAIConverter.convert_request(anthropic_req)

        sys_msg = result["messages"][0]
        assert sys_msg["role"] == "system"
        # Content should be list format (with cache_control)
        assert isinstance(sys_msg["content"], list)
        assert sys_msg["content"][0]["cache_control"] == {"type": "ephemeral"}

    def test_tool_result_with_none_content(self):
        """tool_result with content=None should produce empty string content."""
        anthropic_req = {
            "model": "test",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "call_1",
                            "content": None,
                        },
                    ],
                },
            ],
            "max_tokens": 1024,
        }
        result, _ = AnthropicToOpenAIConverter.convert_request(anthropic_req)
        tool_msgs = [m for m in result["messages"] if m["role"] == "tool"]
        assert len(tool_msgs) == 1
        assert tool_msgs[0]["content"] == ""

    def test_tool_result_with_list_content(self):
        """tool_result with list content should be converted correctly."""
        anthropic_req = {
            "model": "test",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "call_1",
                            "content": [
                                {"type": "text", "text": "Part 1"},
                                {"type": "text", "text": "Part 2"},
                            ],
                        },
                    ],
                },
            ],
            "max_tokens": 1024,
        }
        result, _ = AnthropicToOpenAIConverter.convert_request(anthropic_req)
        tool_msgs = [m for m in result["messages"] if m["role"] == "tool"]
        assert len(tool_msgs) == 1
        # Multi-part content should become list
        assert isinstance(tool_msgs[0]["content"], list)
        assert len(tool_msgs[0]["content"]) == 2

    def test_assistant_with_thinking_and_redacted_thinking(self):
        """Assistant messages with thinking + redacted_thinking blocks."""
        anthropic_req = {
            "model": "test",
            "messages": [
                {"role": "user", "content": "Think hard"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "deep thoughts", "signature": "sig1"},
                        {"type": "redacted_thinking", "data": "redacted_data"},
                        {"type": "text", "text": "Answer"},
                    ],
                },
            ],
            "max_tokens": 1024,
        }
        result, _ = AnthropicToOpenAIConverter.convert_request(anthropic_req)

        assistant_msg = [m for m in result["messages"] if m["role"] == "assistant"][0]
        assert assistant_msg["content"] == "Answer"
        assert len(assistant_msg["thinking_blocks"]) == 2
        assert assistant_msg["thinking_blocks"][0]["type"] == "thinking"
        assert assistant_msg["thinking_blocks"][1]["type"] == "redacted_thinking"

    def test_assistant_with_cache_control_on_text(self):
        """Text blocks with cache_control should use list format content."""
        anthropic_req = {
            "model": "test",
            "messages": [
                {"role": "user", "content": "Hi"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": "cached response",
                            "cache_control": {"type": "ephemeral"},
                        },
                    ],
                },
            ],
            "max_tokens": 1024,
        }
        result, _ = AnthropicToOpenAIConverter.convert_request(anthropic_req)

        assistant_msg = [m for m in result["messages"] if m["role"] == "assistant"][0]
        # Should use list format because of cache_control
        assert isinstance(assistant_msg["content"], list)
        assert assistant_msg["content"][0]["cache_control"] == {"type": "ephemeral"}

    def test_thinking_adaptive_type(self):
        """thinking.type='adaptive' should map to reasoning_effort."""
        anthropic_req = {
            "model": "test",
            "messages": [{"role": "user", "content": "Hi"}],
            "thinking": {"type": "adaptive", "budget_tokens": 5000},
            "max_tokens": 1024,
        }
        result, _ = AnthropicToOpenAIConverter.convert_request(anthropic_req)
        assert result["reasoning_effort"] == "medium"

    def test_thinking_budget_minimal(self):
        """Very low budget_tokens should map to 'minimal'."""
        anthropic_req = {
            "model": "test",
            "messages": [{"role": "user", "content": "Hi"}],
            "thinking": {"type": "enabled", "budget_tokens": 500},
            "max_tokens": 1024,
        }
        result, _ = AnthropicToOpenAIConverter.convert_request(anthropic_req)
        assert result["reasoning_effort"] == "minimal"

    def test_thinking_budget_low(self):
        """budget_tokens ~2000 should map to 'low'."""
        anthropic_req = {
            "model": "test",
            "messages": [{"role": "user", "content": "Hi"}],
            "thinking": {"type": "enabled", "budget_tokens": 2000},
            "max_tokens": 1024,
        }
        result, _ = AnthropicToOpenAIConverter.convert_request(anthropic_req)
        assert result["reasoning_effort"] == "low"

    def test_tool_choice_none(self):
        """tool_choice type 'none' should pass through."""
        anthropic_req = {
            "model": "test",
            "messages": [{"role": "user", "content": "Hi"}],
            "tool_choice": {"type": "none"},
            "max_tokens": 1024,
        }
        result, _ = AnthropicToOpenAIConverter.convert_request(anthropic_req)
        assert result["tool_choice"] == "none"

    def test_tool_choice_auto(self):
        """tool_choice type 'auto' should pass through."""
        anthropic_req = {
            "model": "test",
            "messages": [{"role": "user", "content": "Hi"}],
            "tool_choice": {"type": "auto"},
            "max_tokens": 1024,
        }
        result, _ = AnthropicToOpenAIConverter.convert_request(anthropic_req)
        assert result["tool_choice"] == "auto"

    def test_temperature_and_top_p_passthrough(self):
        """temperature, top_p, stream should pass through."""
        anthropic_req = {
            "model": "test",
            "messages": [{"role": "user", "content": "Hi"}],
            "temperature": 0.7,
            "top_p": 0.9,
            "stream": True,
            "max_tokens": 1024,
        }
        result, _ = AnthropicToOpenAIConverter.convert_request(anthropic_req)
        assert result["temperature"] == 0.7
        assert result["top_p"] == 0.9
        assert result["stream"] is True

    def test_document_block_conversion(self):
        """Document blocks should be converted to image_url format."""
        anthropic_req = {
            "model": "test",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "document",
                            "source": {
                                "type": "base64",
                                "media_type": "application/pdf",
                                "data": "JVBER...",
                            },
                        },
                    ],
                },
            ],
            "max_tokens": 1024,
        }
        result, _ = AnthropicToOpenAIConverter.convert_request(anthropic_req)
        user_msg = result["messages"][0]
        assert isinstance(user_msg["content"], list)
        assert user_msg["content"][0]["type"] == "image_url"
        assert "base64" in user_msg["content"][0]["image_url"]["url"]

    def test_empty_assistant_content_skipped(self):
        """Assistant message with empty content list produces no message."""
        anthropic_req = {
            "model": "test",
            "messages": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": []},
                {"role": "user", "content": "Hello again"},
            ],
            "max_tokens": 1024,
        }
        result, _ = AnthropicToOpenAIConverter.convert_request(anthropic_req)
        # Empty assistant should be skipped
        roles = [m["role"] for m in result["messages"]]
        assert "assistant" not in roles

    def test_mixed_tool_results_and_text_in_user_message(self):
        """User message with both tool_result and text blocks."""
        anthropic_req = {
            "model": "test",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "call_1",
                            "content": "result data",
                        },
                        {"type": "text", "text": "Now what?"},
                    ],
                },
            ],
            "max_tokens": 1024,
        }
        result, _ = AnthropicToOpenAIConverter.convert_request(anthropic_req)
        # Tool messages come first, then user text
        assert result["messages"][0]["role"] == "tool"
        assert result["messages"][1]["role"] == "user"

    def test_multiple_tool_use_in_assistant(self):
        """Assistant message with multiple tool_use blocks."""
        anthropic_req = {
            "model": "test",
            "messages": [
                {"role": "user", "content": "Do two things"},
                {
                    "role": "assistant",
                    "content": [
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
                            "input": {"url": "http://example.com"},
                        },
                    ],
                },
            ],
            "max_tokens": 1024,
        }
        result, _ = AnthropicToOpenAIConverter.convert_request(anthropic_req)
        assistant_msg = [m for m in result["messages"] if m["role"] == "assistant"][0]
        assert len(assistant_msg["tool_calls"]) == 2
        assert assistant_msg["tool_calls"][0]["function"]["name"] == "search"
        assert assistant_msg["tool_calls"][1]["function"]["name"] == "fetch"
        assert assistant_msg["content"] is None  # No text

    # --- Response edge cases ---

    def test_response_with_no_content(self):
        """Response with None content should produce empty text block."""
        resp = {
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "test",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": None},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        result = AnthropicToOpenAIConverter.convert_response(resp)
        # No text block should be created for None content
        text_blocks = [b for b in result["content"] if b["type"] == "text"]
        assert len(text_blocks) == 0

    def test_response_with_empty_string_content(self):
        """Response with empty string content should produce text block."""
        resp = {
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "test",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": ""},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        result = AnthropicToOpenAIConverter.convert_response(resp)
        text_blocks = [b for b in result["content"] if b["type"] == "text"]
        assert len(text_blocks) == 1
        assert text_blocks[0]["text"] == ""

    def test_response_with_reasoning_content_fallback(self):
        """reasoning_content should become a thinking block when no thinking_blocks."""
        resp = {
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "test",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Answer",
                        "reasoning_content": "I thought about this carefully",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
        }
        result = AnthropicToOpenAIConverter.convert_response(resp)

        thinking_blocks = [b for b in result["content"] if b["type"] == "thinking"]
        assert len(thinking_blocks) == 1
        assert thinking_blocks[0]["thinking"] == "I thought about this carefully"

    def test_response_with_redacted_thinking_blocks(self):
        """Redacted thinking blocks should be preserved."""
        resp = {
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "test",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Answer",
                        "thinking_blocks": [
                            {"type": "redacted_thinking", "data": "secret_data"},
                        ],
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        result = AnthropicToOpenAIConverter.convert_response(resp)

        redacted = [b for b in result["content"] if b["type"] == "redacted_thinking"]
        assert len(redacted) == 1
        assert redacted[0]["data"] == "secret_data"

    def test_response_with_multiple_tool_calls(self):
        """Multiple tool calls should become multiple tool_use blocks."""
        resp = {
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "test",
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
                                "function": {"name": "a", "arguments": '{"x":1}'},
                            },
                            {
                                "id": "call_2",
                                "type": "function",
                                "function": {"name": "b", "arguments": '{"y":2}'},
                            },
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
        }
        result = AnthropicToOpenAIConverter.convert_response(resp)

        tool_blocks = [b for b in result["content"] if b["type"] == "tool_use"]
        assert len(tool_blocks) == 2
        assert tool_blocks[0]["name"] == "a"
        assert tool_blocks[0]["input"] == {"x": 1}
        assert tool_blocks[1]["name"] == "b"
        assert tool_blocks[1]["input"] == {"y": 2}

    def test_response_invalid_json_arguments(self):
        """Invalid JSON in tool arguments should default to empty dict."""
        resp = {
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "test",
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
                                "function": {"name": "t", "arguments": "not json!!"},
                            },
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        result = AnthropicToOpenAIConverter.convert_response(resp)

        tool_blocks = [b for b in result["content"] if b["type"] == "tool_use"]
        assert len(tool_blocks) == 1
        assert tool_blocks[0]["input"] == {}

    def test_response_no_usage(self):
        """Response with no usage should default to zeros."""
        resp = {
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "test",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hi"},
                    "finish_reason": "stop",
                }
            ],
        }
        result = AnthropicToOpenAIConverter.convert_response(resp)
        assert result["usage"]["input_tokens"] == 0
        assert result["usage"]["output_tokens"] == 0

    def test_response_function_call_finish_reason(self):
        """Deprecated 'function_call' finish_reason should map to tool_use."""
        resp = {
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "test",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hi"},
                    "finish_reason": "function_call",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        result = AnthropicToOpenAIConverter.convert_response(resp)
        assert result["stop_reason"] == "tool_use"

    # --- Streaming edge cases ---

    def test_stream_thinking_then_text(self):
        """Stream with thinking blocks followed by text content."""
        chunks = [
            _make_chunk(delta={"role": "assistant", "content": ""}),
            _make_chunk(
                delta={"thinking_blocks": [{"type": "thinking", "thinking": "Let me think..."}]}
            ),
            _make_chunk(
                delta={
                    "thinking_blocks": [{"type": "thinking", "thinking": "", "signature": "sig"}]
                }
            ),
            _make_chunk(delta={"content": "Result"}),
            _make_chunk(delta={}, finish_reason="stop"),
        ]
        events = list(AnthropicToOpenAIConverter.convert_stream(chunks))

        # Should have thinking content_block_start
        thinking_starts = [
            e
            for e in events
            if e["type"] == "content_block_start"
            and e.get("content_block", {}).get("type") == "thinking"
        ]
        assert len(thinking_starts) >= 1

        # Should have thinking_delta
        thinking_deltas = [
            e
            for e in events
            if e["type"] == "content_block_delta" and e["delta"]["type"] == "thinking_delta"
        ]
        assert len(thinking_deltas) == 1

        # Should have signature_delta
        sig_deltas = [
            e
            for e in events
            if e["type"] == "content_block_delta" and e["delta"]["type"] == "signature_delta"
        ]
        assert len(sig_deltas) == 1

        # Should have text content
        text_deltas = [
            e
            for e in events
            if e["type"] == "content_block_delta" and e["delta"]["type"] == "text_delta"
        ]
        assert len(text_deltas) == 1
        assert text_deltas[0]["delta"]["text"] == "Result"

    def test_stream_multiple_tool_calls(self):
        """Stream with multiple sequential tool calls."""
        chunks = [
            _make_chunk(delta={"role": "assistant"}),
            _make_chunk(
                delta={
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "search", "arguments": ""},
                        }
                    ]
                }
            ),
            _make_chunk(
                delta={"tool_calls": [{"index": 0, "function": {"arguments": '{"q":"test"}'}}]}
            ),
            _make_chunk(
                delta={
                    "tool_calls": [
                        {
                            "index": 1,
                            "id": "call_2",
                            "type": "function",
                            "function": {"name": "fetch", "arguments": ""},
                        }
                    ]
                }
            ),
            _make_chunk(
                delta={
                    "tool_calls": [
                        {"index": 1, "function": {"arguments": '{"url":"http://x.com"}'}}
                    ]
                }
            ),
            _make_chunk(delta={}, finish_reason="tool_calls"),
        ]
        events = list(AnthropicToOpenAIConverter.convert_stream(chunks))

        tool_starts = [
            e
            for e in events
            if e["type"] == "content_block_start"
            and e.get("content_block", {}).get("type") == "tool_use"
        ]
        assert len(tool_starts) == 2
        assert tool_starts[0]["content_block"]["name"] == "search"
        assert tool_starts[1]["content_block"]["name"] == "fetch"

        # Should have content_block_stop between tool calls
        tool_stops = [e for e in events if e["type"] == "content_block_stop"]
        assert len(tool_stops) >= 2  # At least one for each tool block

    def test_stream_usage_in_final_chunk(self):
        """Usage info in final chunk should be propagated."""
        chunks = [
            _make_chunk(delta={"role": "assistant", "content": ""}),
            _make_chunk(delta={"content": "Hi"}),
            {
                "id": "chatcmpl-123",
                "object": "chat.completion.chunk",
                "created": 1700000000,
                "model": "test",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            },
        ]
        events = list(AnthropicToOpenAIConverter.convert_stream(chunks))

        message_delta = [e for e in events if e["type"] == "message_delta"]
        assert len(message_delta) == 1
        assert message_delta[0]["usage"]["input_tokens"] == 10
        assert message_delta[0]["usage"]["output_tokens"] == 5

    def test_stream_text_then_tool_transition(self):
        """Stream transitioning from text to tool call."""
        chunks = [
            _make_chunk(delta={"role": "assistant", "content": ""}),
            _make_chunk(delta={"content": "Let me search."}),
            _make_chunk(
                delta={
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "search", "arguments": ""},
                        }
                    ]
                }
            ),
            _make_chunk(
                delta={"tool_calls": [{"index": 0, "function": {"arguments": '{"q":"x"}'}}]}
            ),
            _make_chunk(delta={}, finish_reason="tool_calls"),
        ]
        events = list(AnthropicToOpenAIConverter.convert_stream(chunks))

        # Verify event sequence: text block -> stop -> tool block -> stop
        event_types = [e["type"] for e in events]
        # Should have content_block_stop between text and tool
        text_start_idx = next(
            i
            for i, e in enumerate(events)
            if e["type"] == "content_block_start"
            and e.get("content_block", {}).get("type") == "text"
        )
        tool_start_idx = next(
            i
            for i, e in enumerate(events)
            if e["type"] == "content_block_start"
            and e.get("content_block", {}).get("type") == "tool_use"
        )
        # There should be a content_block_stop between them
        stops_between = [
            i
            for i in range(text_start_idx, tool_start_idx)
            if event_types[i] == "content_block_stop"
        ]
        assert len(stops_between) >= 1

    def test_stream_empty_content_chunks_ignored(self):
        """Chunks with empty content should not produce text deltas."""
        chunks = [
            _make_chunk(delta={"role": "assistant", "content": ""}),
            _make_chunk(delta={"content": ""}),  # Empty content
            _make_chunk(delta={"content": "Hello"}),
            _make_chunk(delta={}, finish_reason="stop"),
        ]
        events = list(AnthropicToOpenAIConverter.convert_stream(chunks))

        text_deltas = [
            e
            for e in events
            if e["type"] == "content_block_delta" and e["delta"]["type"] == "text_delta"
        ]
        # Only "Hello" should produce a delta (empty string is filtered)
        assert len(text_deltas) == 1
        assert text_deltas[0]["delta"]["text"] == "Hello"

    def test_stream_event_sequence_integrity(self):
        """Verify the full event sequence for a simple text response."""
        chunks = [
            _make_chunk(delta={"role": "assistant", "content": ""}),
            _make_chunk(delta={"content": "Hi"}),
            _make_chunk(delta={}, finish_reason="stop"),
        ]
        events = list(AnthropicToOpenAIConverter.convert_stream(chunks))

        types = [e["type"] for e in events]
        # Must start with message_start
        assert types[0] == "message_start"
        # Must end with message_delta + message_stop
        assert types[-2] == "message_delta"
        assert types[-1] == "message_stop"
        # Must have content_block_start/stop pair
        assert "content_block_start" in types
        assert "content_block_stop" in types

    def test_web_search_only_tools(self):
        """When all tools are web_search, tools key should not be in result."""
        anthropic_req = {
            "model": "test",
            "messages": [{"role": "user", "content": "Search"}],
            "tools": [{"type": "web_search_20250305", "name": "web_search"}],
            "max_tokens": 1024,
        }
        result, _ = AnthropicToOpenAIConverter.convert_request(anthropic_req)

        assert "web_search_options" in result
        assert "tools" not in result

    def test_output_format_non_json_schema_ignored(self):
        """output_format with non-json_schema type should be ignored."""
        anthropic_req = {
            "model": "test",
            "messages": [{"role": "user", "content": "Hi"}],
            "output_format": {"type": "text"},
            "max_tokens": 1024,
        }
        result, _ = AnthropicToOpenAIConverter.convert_request(anthropic_req)
        assert "response_format" not in result


def _make_chunk(delta, finish_reason=None):
    """Helper to build an OpenAI streaming chunk."""
    return {
        "id": "chatcmpl-123",
        "object": "chat.completion.chunk",
        "created": 1700000000,
        "model": "test",
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }
