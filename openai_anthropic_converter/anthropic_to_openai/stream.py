"""
OpenAI streaming chunks -> Anthropic SSE events conversion.

Converts OpenAI chat.completion.chunk format to Anthropic Server-Sent Events.
"""

import uuid
from collections import deque
from typing import Any, AsyncIterable, AsyncIterator, Dict, Iterable, Iterator, List, Optional

from ..constants import OPENAI_TO_ANTHROPIC_STOP_REASON


class OpenAIToAnthropicSSEStream:
    """
    Stateful converter that transforms OpenAI streaming chunks into
    Anthropic SSE event dicts.

    Synthesizes the complete Anthropic event sequence:
        message_start -> content_block_start -> content_block_delta* ->
        content_block_stop -> ... -> message_delta -> message_stop

    The core difficulty: OpenAI doesn't emit explicit block boundaries.
    We must detect content type transitions (text -> tool_use, etc.) to
    generate content_block_stop/start pairs.
    """

    def __init__(
        self,
        model: str = "unknown",
        tool_name_mapping: Optional[Dict[str, str]] = None,
    ) -> None:
        self.model = model
        self.tool_name_mapping = tool_name_mapping or {}

        self.sent_message_start = False
        self.sent_content_block_start = False
        self.current_content_type: Optional[str] = None  # "text" | "tool_use" | "thinking"
        self.current_block_index = 0
        self.response_id = f"msg_{uuid.uuid4().hex[:12]}"

        # Queue for buffered events (some chunks produce multiple events)
        self._queue: deque = deque()

        # Accumulated usage from chunks
        self._usage: Dict[str, int] = {
            "input_tokens": 0,
            "output_tokens": 0,
        }

    def process_chunk(self, chunk: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Process a single OpenAI streaming chunk and return zero or more
        Anthropic SSE events.
        """
        events: List[Dict[str, Any]] = []

        # Extract model from first chunk
        chunk_model = chunk.get("model")
        if chunk_model:
            self.model = chunk_model

        chunk_id = chunk.get("id")
        if chunk_id:
            self.response_id = chunk_id

        choices = chunk.get("choices", [])
        if not choices:
            # Usage-only chunk (some providers send final usage separately)
            usage = chunk.get("usage")
            if usage:
                self._update_usage(usage)
            return events

        choice = choices[0]
        delta = choice.get("delta", {})
        finish_reason = choice.get("finish_reason")

        # Emit message_start on first chunk
        if not self.sent_message_start:
            events.append(self._make_message_start())
            self.sent_message_start = True

        # Determine what kind of content this chunk carries
        has_text = delta.get("content") is not None and delta.get("content") != ""
        has_tool_calls = bool(delta.get("tool_calls"))
        has_thinking = bool(delta.get("thinking_blocks"))
        # [Bailian compat] DashScope sends reasoning_content in delta
        has_reasoning = (
            not has_thinking
            and delta.get("reasoning_content") is not None
            and delta.get("reasoning_content") != ""
        )

        # Handle content type transitions
        if has_text:
            events.extend(self._ensure_content_block("text"))
            events.append(
                {
                    "type": "content_block_delta",
                    "index": self.current_block_index,
                    "delta": {"type": "text_delta", "text": delta["content"]},
                }
            )

        elif has_tool_calls:
            for tc in delta["tool_calls"]:
                func = tc.get("function", {})
                tc_id = tc.get("id")

                if tc_id:
                    # New tool call - start new block
                    events.extend(self._close_current_block())

                    # Restore original name if truncated
                    name = func.get("name", "")
                    original_name = self.tool_name_mapping.get(name, name)

                    self.current_content_type = "tool_use"
                    self.sent_content_block_start = True
                    events.append(
                        {
                            "type": "content_block_start",
                            "index": self.current_block_index,
                            "content_block": {
                                "type": "tool_use",
                                "id": tc_id,
                                "name": original_name,
                                "input": {},
                            },
                        }
                    )
                elif func.get("arguments"):
                    # Continuing tool arguments
                    if not self.sent_content_block_start:
                        events.extend(self._ensure_content_block("tool_use"))
                    events.append(
                        {
                            "type": "content_block_delta",
                            "index": self.current_block_index,
                            "delta": {
                                "type": "input_json_delta",
                                "partial_json": func["arguments"],
                            },
                        }
                    )

        elif has_thinking:
            for tb in delta["thinking_blocks"]:
                thinking_text = tb.get("thinking", "")
                signature = tb.get("signature", "")

                if thinking_text:
                    events.extend(self._ensure_content_block("thinking"))
                    events.append(
                        {
                            "type": "content_block_delta",
                            "index": self.current_block_index,
                            "delta": {"type": "thinking_delta", "thinking": thinking_text},
                        }
                    )
                if signature:
                    events.extend(self._ensure_content_block("thinking"))
                    events.append(
                        {
                            "type": "content_block_delta",
                            "index": self.current_block_index,
                            "delta": {"type": "signature_delta", "signature": signature},
                        }
                    )

        elif has_reasoning:
            # [Bailian compat] DashScope/Bailian sends reasoning_content in
            # delta instead of thinking_blocks. Convert to thinking block.
            events.extend(self._ensure_content_block("thinking"))
            events.append(
                {
                    "type": "content_block_delta",
                    "index": self.current_block_index,
                    "delta": {
                        "type": "thinking_delta",
                        "thinking": delta["reasoning_content"],
                    },
                }
            )

        # Handle finish
        if finish_reason:
            events.extend(self._close_current_block())

            stop_reason = OPENAI_TO_ANTHROPIC_STOP_REASON.get(finish_reason, "end_turn")

            # Update usage from chunk if available
            usage = chunk.get("usage")
            if usage:
                self._update_usage(usage)

            events.append(
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                    "usage": dict(self._usage),
                }
            )
            events.append({"type": "message_stop"})

        return events

    def _make_message_start(self) -> Dict[str, Any]:
        """Create message_start event."""
        return {
            "type": "message_start",
            "message": {
                "id": self.response_id,
                "type": "message",
                "role": "assistant",
                "model": self.model,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 0,
                },
            },
        }

    def _ensure_content_block(self, block_type: str) -> List[Dict[str, Any]]:
        """
        Ensure a content block of the given type is open.
        If a different type is open, close it first.
        """
        events: List[Dict[str, Any]] = []

        if self.current_content_type == block_type and self.sent_content_block_start:
            return events  # Already in the right block

        # Close previous block if different type
        if self.sent_content_block_start:
            events.extend(self._close_current_block())

        # Start new block
        self.current_content_type = block_type
        self.sent_content_block_start = True

        if block_type == "text":
            events.append(
                {
                    "type": "content_block_start",
                    "index": self.current_block_index,
                    "content_block": {"type": "text", "text": ""},
                }
            )
        elif block_type == "tool_use":
            events.append(
                {
                    "type": "content_block_start",
                    "index": self.current_block_index,
                    "content_block": {
                        "type": "tool_use",
                        "id": f"toolu_{uuid.uuid4().hex[:12]}",
                        "name": "",
                        "input": {},
                    },
                }
            )
        elif block_type == "thinking":
            events.append(
                {
                    "type": "content_block_start",
                    "index": self.current_block_index,
                    "content_block": {"type": "thinking", "thinking": ""},
                }
            )

        return events

    def _close_current_block(self) -> List[Dict[str, Any]]:
        """Close the current content block if one is open."""
        events: List[Dict[str, Any]] = []
        if self.sent_content_block_start:
            events.append(
                {
                    "type": "content_block_stop",
                    "index": self.current_block_index,
                }
            )
            self.current_block_index += 1
            self.sent_content_block_start = False
            self.current_content_type = None
        return events

    def _update_usage(self, usage: Dict[str, Any]) -> None:
        """Update accumulated usage from a chunk."""
        if "prompt_tokens" in usage:
            self._usage["input_tokens"] = usage["prompt_tokens"]
        if "completion_tokens" in usage:
            self._usage["output_tokens"] = usage["completion_tokens"]


def convert_stream(
    openai_chunks: Iterable[Dict[str, Any]],
    *,
    model: str = "unknown",
    tool_name_mapping: Optional[Dict[str, str]] = None,
) -> Iterator[Dict[str, Any]]:
    """
    Convert a stream of OpenAI chunks to Anthropic SSE events.

    Args:
        openai_chunks: Iterable of OpenAI chat.completion.chunk dicts
        model: Model name for the message_start event
        tool_name_mapping: Mapping of truncated tool names to originals

    Yields:
        Anthropic SSE event dicts
    """
    converter = OpenAIToAnthropicSSEStream(model=model, tool_name_mapping=tool_name_mapping)
    for chunk in openai_chunks:
        events = converter.process_chunk(chunk)
        for event in events:
            yield event


async def aconvert_stream(
    openai_chunks: AsyncIterable[Dict[str, Any]],
    *,
    model: str = "unknown",
    tool_name_mapping: Optional[Dict[str, str]] = None,
) -> AsyncIterator[Dict[str, Any]]:
    """Async version of convert_stream."""
    converter = OpenAIToAnthropicSSEStream(model=model, tool_name_mapping=tool_name_mapping)
    async for chunk in openai_chunks:
        events = converter.process_chunk(chunk)
        for event in events:
            yield event
