"""
Anthropic SSE -> OpenAI streaming chunks conversion.

Converts Anthropic Server-Sent Events into OpenAI chat.completion.chunk format.
"""

import time
import uuid
from typing import Any, AsyncIterable, AsyncIterator, Dict, Iterable, Iterator, Optional

from ..constants import ANTHROPIC_TO_OPENAI_FINISH_REASON


class AnthropicSSEToOpenAIStream:
    """
    Stateful converter that transforms Anthropic SSE events into
    OpenAI chat.completion.chunk dicts.

    Handles the event sequence:
        message_start -> content_block_start -> content_block_delta* ->
        content_block_stop -> ... -> message_delta -> message_stop
    """

    def __init__(self) -> None:
        self.response_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        self.model = ""
        self.created = int(time.time())
        self.tool_index = 0
        self.current_content_type: Optional[str] = None  # "text", "tool_use", "thinking"
        self.current_tool_call_id: Optional[str] = None
        self.current_tool_name: Optional[str] = None

    def process_event(self, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process a single Anthropic SSE event and return an OpenAI chunk dict,
        or None if no chunk should be emitted for this event.
        """
        event_type = event.get("type", "")

        if event_type == "message_start":
            return self._handle_message_start(event)
        elif event_type == "content_block_start":
            return self._handle_content_block_start(event)
        elif event_type == "content_block_delta":
            return self._handle_content_block_delta(event)
        elif event_type == "content_block_stop":
            if self.current_content_type == "tool_use":
                self.tool_index += 1
            self.current_content_type = None
            return None
        elif event_type == "message_delta":
            return self._handle_message_delta(event)
        elif event_type == "message_stop":
            return None

        return None

    def _handle_message_start(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Handle message_start: extract model info, emit role chunk."""
        message = event.get("message", {})
        self.model = message.get("model", self.model)
        self.response_id = message.get("id", self.response_id)

        usage_info = message.get("usage", {})

        chunk = self._make_chunk(
            delta={"role": "assistant", "content": ""},
        )
        if usage_info:
            chunk["usage"] = {
                "prompt_tokens": (usage_info.get("input_tokens", 0) or 0)
                + (usage_info.get("cache_read_input_tokens", 0) or 0)
                + (usage_info.get("cache_creation_input_tokens", 0) or 0),
                "completion_tokens": usage_info.get("output_tokens", 0) or 0,
                "total_tokens": 0,
            }
            chunk["usage"]["total_tokens"] = (
                chunk["usage"]["prompt_tokens"] + chunk["usage"]["completion_tokens"]
            )
        return chunk

    def _handle_content_block_start(self, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Handle content_block_start: track content type, emit tool call start if needed."""
        content_block = event.get("content_block", {})
        block_type = content_block.get("type", "text")
        self.current_content_type = block_type

        if block_type in ("tool_use", "server_tool_use", "mcp_tool_use"):
            self.current_content_type = "tool_use"  # Normalize to tool_use
            self.current_tool_call_id = content_block.get("id", "")
            self.current_tool_name = content_block.get("name", "")
            # Emit initial tool call chunk
            return self._make_chunk(
                delta={
                    "tool_calls": [
                        {
                            "index": self.tool_index,
                            "id": self.current_tool_call_id,
                            "type": "function",
                            "function": {
                                "name": self.current_tool_name,
                                "arguments": "",
                            },
                        }
                    ],
                }
            )
        elif block_type in ("thinking", "redacted_thinking"):
            return None  # Will emit on delta
        else:
            return None  # Text blocks emit on delta

    def _handle_content_block_delta(self, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Handle content_block_delta: emit text/tool/thinking delta."""
        delta = event.get("delta", {})
        delta_type = delta.get("type", "")

        if delta_type == "text_delta":
            text = delta.get("text", "")
            if text:
                return self._make_chunk(delta={"content": text})

        elif delta_type == "input_json_delta":
            partial_json = delta.get("partial_json", "")
            if partial_json:
                return self._make_chunk(
                    delta={
                        "tool_calls": [
                            {
                                "index": self.tool_index,
                                "function": {"arguments": partial_json},
                            }
                        ],
                    }
                )

        elif delta_type == "thinking_delta":
            thinking = delta.get("thinking", "")
            if thinking:
                return self._make_chunk(
                    delta={
                        "thinking_blocks": [
                            {
                                "type": "thinking",
                                "thinking": thinking,
                            }
                        ],
                    }
                )

        elif delta_type == "signature_delta":
            signature = delta.get("signature", "")
            if signature:
                return self._make_chunk(
                    delta={
                        "thinking_blocks": [
                            {
                                "type": "thinking",
                                "thinking": "",
                                "signature": signature,
                            }
                        ],
                    }
                )

        return None

    def _handle_message_delta(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Handle message_delta: emit finish_reason and usage."""
        delta_body = event.get("delta", {})
        usage = event.get("usage", {})

        stop_reason = delta_body.get("stop_reason")
        finish_reason = (
            ANTHROPIC_TO_OPENAI_FINISH_REASON.get(stop_reason, "stop") if stop_reason else "stop"
        )

        chunk = self._make_chunk(
            delta={},
            finish_reason=finish_reason,
        )

        if usage:
            input_tokens = usage.get("input_tokens", 0) or 0
            output_tokens = usage.get("output_tokens", 0) or 0
            cache_creation = usage.get("cache_creation_input_tokens", 0) or 0
            cache_read = usage.get("cache_read_input_tokens", 0) or 0
            prompt_tokens = input_tokens + cache_creation + cache_read
            chunk["usage"] = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": prompt_tokens + output_tokens,
            }

        return chunk

    def _make_chunk(
        self,
        delta: Dict[str, Any],
        finish_reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Build an OpenAI chat.completion.chunk dict."""
        return {
            "id": self.response_id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model,
            "choices": [
                {
                    "index": 0,
                    "delta": delta,
                    "finish_reason": finish_reason,
                }
            ],
        }


def convert_stream(
    anthropic_events: Iterable[Dict[str, Any]],
) -> Iterator[Dict[str, Any]]:
    """
    Convert a stream of Anthropic SSE events to OpenAI streaming chunks.

    Args:
        anthropic_events: Iterable of Anthropic SSE event dicts

    Yields:
        OpenAI chat.completion.chunk dicts
    """
    converter = AnthropicSSEToOpenAIStream()
    for event in anthropic_events:
        chunk = converter.process_event(event)
        if chunk is not None:
            yield chunk


async def aconvert_stream(
    anthropic_events: AsyncIterable[Dict[str, Any]],
) -> AsyncIterator[Dict[str, Any]]:
    """
    Async version of convert_stream.
    """
    converter = AnthropicSSEToOpenAIStream()
    async for event in anthropic_events:
        chunk = converter.process_event(event)
        if chunk is not None:
            yield chunk
