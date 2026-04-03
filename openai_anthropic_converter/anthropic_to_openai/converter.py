"""
AnthropicToOpenAIConverter: Main converter class.

Converts Anthropic Messages protocol <-> OpenAI ChatCompletion protocol.

Use case: You receive Anthropic /v1/messages requests and need to forward them
to an OpenAI-compatible backend, then convert the responses back to Anthropic format.
"""

from typing import Any, AsyncIterable, AsyncIterator, Dict, Iterable, Iterator, Optional, Tuple

from .request import convert_request as _convert_request
from .response import convert_response as _convert_response
from .stream import aconvert_stream as _aconvert_stream
from .stream import convert_stream as _convert_stream


class AnthropicToOpenAIConverter:
    """
    Bidirectional converter: Anthropic request -> OpenAI request,
    OpenAI response -> Anthropic response.

    All methods are static — the converter is stateless.

    Example usage (non-streaming):
        converter = AnthropicToOpenAIConverter()

        # Convert request (returns tool_name_mapping for name restoration)
        openai_req, tool_name_mapping = converter.convert_request(anthropic_request)

        # ... send openai_req to OpenAI-compatible API, get openai_resp ...

        # Convert response (pass tool_name_mapping to restore truncated names)
        anthropic_resp = converter.convert_response(
            openai_resp, tool_name_mapping=tool_name_mapping
        )

    Example usage (streaming):
        openai_req, tool_name_mapping = converter.convert_request(anthropic_request)

        # ... send openai_req to OpenAI API with stream=True ...

        for anthropic_event in converter.convert_stream(
            openai_chunks, tool_name_mapping=tool_name_mapping
        ):
            # Send SSE event to client: f"event: {event['type']}\\ndata: {json.dumps(event)}\\n\\n"
            pass
    """

    @staticmethod
    def convert_request(
        anthropic_request: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Dict[str, str]]:
        """
        Convert an Anthropic Messages request to an OpenAI ChatCompletion request.

        Returns a tuple of (openai_request, tool_name_mapping):
        - openai_request: The converted request dict
        - tool_name_mapping: Maps truncated tool names back to originals.
          OpenAI has a 64-char tool name limit; names > 64 chars are truncated
          with a hash suffix. Pass this mapping to convert_response() to
          restore original names in tool_use blocks.

        Handles:
        - Message conversion (content blocks -> OpenAI messages)
        - System param -> system role message
        - Tool definition conversion with name truncation
        - thinking -> reasoning_effort mapping
        - output_config -> response_format conversion
        - metadata.user_id -> user mapping

        Args:
            anthropic_request: Anthropic Messages API request dict

        Returns:
            (openai_request, tool_name_mapping)
        """
        return _convert_request(anthropic_request)

    @staticmethod
    def convert_response(
        openai_response: Dict[str, Any],
        *,
        tool_name_mapping: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Convert an OpenAI ChatCompletion response to an Anthropic Messages response.

        Handles:
        - message.content -> text content block
        - tool_calls -> tool_use content blocks (with name restoration)
        - thinking_blocks -> thinking/redacted_thinking blocks
        - finish_reason -> stop_reason mapping
        - Usage conversion

        Args:
            openai_response: OpenAI ChatCompletion response dict
            tool_name_mapping: Mapping from convert_request() for name restoration

        Returns:
            Anthropic Messages API response dict
        """
        return _convert_response(openai_response, tool_name_mapping=tool_name_mapping)

    @staticmethod
    def convert_stream(
        openai_chunks: Iterable[Dict[str, Any]],
        *,
        model: str = "unknown",
        tool_name_mapping: Optional[Dict[str, str]] = None,
    ) -> Iterator[Dict[str, Any]]:
        """
        Convert a stream of OpenAI chunks to Anthropic SSE events.

        Generates the full Anthropic event sequence:
        1. message_start (with initial usage)
        2. content_block_start (text/tool_use/thinking)
        3. content_block_delta* (text_delta/input_json_delta/thinking_delta)
        4. content_block_stop
        5. (repeat 2-4 for each content block)
        6. message_delta (with stop_reason and final usage)
        7. message_stop

        Args:
            openai_chunks: Iterable of OpenAI chat.completion.chunk dicts
            model: Model name for the message_start event
            tool_name_mapping: Mapping for name restoration

        Yields:
            Anthropic SSE event dicts
        """
        return _convert_stream(openai_chunks, model=model, tool_name_mapping=tool_name_mapping)

    @staticmethod
    async def aconvert_stream(
        openai_chunks: AsyncIterable[Dict[str, Any]],
        *,
        model: str = "unknown",
        tool_name_mapping: Optional[Dict[str, str]] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Async version of convert_stream."""
        async for event in _aconvert_stream(
            openai_chunks, model=model, tool_name_mapping=tool_name_mapping
        ):
            yield event
