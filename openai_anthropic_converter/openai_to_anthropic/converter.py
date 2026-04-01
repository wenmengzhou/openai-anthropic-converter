"""
OpenAIToAnthropicConverter: Main converter class.

Converts OpenAI ChatCompletion protocol <-> Anthropic Messages protocol.

Use case: You have OpenAI-format requests and want to call the Anthropic API,
then convert the Anthropic responses back to OpenAI format.
"""

from typing import Any, AsyncIterable, AsyncIterator, Dict, Iterable, Iterator

from ..constants import DEFAULT_MAX_TOKENS
from .request import convert_request as _convert_request
from .response import convert_response as _convert_response
from .stream import aconvert_stream as _aconvert_stream
from .stream import convert_stream as _convert_stream


class OpenAIToAnthropicConverter:
    """
    Bidirectional converter: OpenAI request -> Anthropic request,
    Anthropic response -> OpenAI response.

    All methods are static — the converter is stateless.

    Example usage (non-streaming):
        converter = OpenAIToAnthropicConverter()

        # Convert request
        anthropic_req = converter.convert_request(openai_request)

        # ... send anthropic_req to Anthropic API, get anthropic_resp ...

        # Convert response
        openai_resp = converter.convert_response(anthropic_resp)

    Example usage (streaming):
        anthropic_req = converter.convert_request(openai_request)

        # ... send anthropic_req to Anthropic API with stream=True ...

        for openai_chunk in converter.convert_stream(anthropic_sse_events):
            # process OpenAI-format chunk
            pass
    """

    @staticmethod
    def convert_request(
        openai_request: Dict[str, Any],
        *,
        default_max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> Dict[str, Any]:
        """
        Convert an OpenAI ChatCompletion request to an Anthropic Messages request.

        Handles:
        - System message extraction (inline -> top-level system param)
        - Message content conversion (image_url, tool_calls, tool results)
        - Parameter mapping (max_tokens, tools, tool_choice, stop, response_format, etc.)
        - Message alternation enforcement (Anthropic requires user/assistant alternation)

        Args:
            openai_request: OpenAI ChatCompletion request dict
            default_max_tokens: Default max_tokens if not specified

        Returns:
            Anthropic Messages API request dict
        """
        return _convert_request(openai_request, default_max_tokens=default_max_tokens)

    @staticmethod
    def convert_response(anthropic_response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert an Anthropic Messages response to an OpenAI ChatCompletion response.

        Handles:
        - Content blocks: text -> content, tool_use -> tool_calls, thinking -> thinking_blocks
        - Stop reason mapping: end_turn -> stop, max_tokens -> length, tool_use -> tool_calls
        - Usage conversion with cache token details
        - JSON mode detection (tool-based)

        Args:
            anthropic_response: Anthropic Messages API response dict

        Returns:
            OpenAI ChatCompletion response dict
        """
        return _convert_response(anthropic_response)

    @staticmethod
    def convert_stream(
        anthropic_events: Iterable[Dict[str, Any]],
    ) -> Iterator[Dict[str, Any]]:
        """
        Convert a stream of Anthropic SSE events to OpenAI streaming chunks.

        Event mapping:
        - message_start -> first chunk (role=assistant)
        - content_block_delta(text_delta) -> chunk(delta.content)
        - content_block_delta(input_json_delta) -> chunk(delta.tool_calls)
        - content_block_delta(thinking_delta) -> chunk(delta.thinking_blocks)
        - message_delta -> final chunk (finish_reason + usage)

        Args:
            anthropic_events: Iterable of Anthropic SSE event dicts

        Yields:
            OpenAI chat.completion.chunk dicts
        """
        return _convert_stream(anthropic_events)

    @staticmethod
    async def aconvert_stream(
        anthropic_events: AsyncIterable[Dict[str, Any]],
    ) -> AsyncIterator[Dict[str, Any]]:
        """Async version of convert_stream."""
        async for chunk in _aconvert_stream(anthropic_events):
            yield chunk
