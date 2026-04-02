"""
OpenAI -> Anthropic response conversion.

Converts OpenAI ChatCompletion responses to Anthropic Messages API format.
"""

import uuid
from typing import Any, Dict, List, Optional

from ..constants import OPENAI_TO_ANTHROPIC_STOP_REASON
from ..utils import safe_json_loads


def convert_openai_content_to_anthropic(
    choices: List[Dict[str, Any]],
    tool_name_mapping: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    """
    Convert OpenAI response choices to Anthropic content blocks.

    Handles:
    - thinking_blocks -> thinking/redacted_thinking blocks
    - reasoning_content -> thinking block (fallback)
    - message.content -> text block
    - tool_calls -> tool_use blocks (with name restoration from mapping)
    """
    content: List[Dict[str, Any]] = []

    for choice in choices:
        message = choice.get("message", {})

        # Handle thinking blocks first
        thinking_blocks = message.get("thinking_blocks")
        if thinking_blocks:
            for tb in thinking_blocks:
                tb_type = tb.get("type", "thinking")
                if tb_type == "thinking":
                    thinking_block: Dict[str, Any] = {
                        "type": "thinking",
                        "thinking": str(tb.get("thinking", "")),
                    }
                    sig = tb.get("signature")
                    if sig:
                        thinking_block["signature"] = str(sig)
                    content.append(thinking_block)
                elif tb_type == "redacted_thinking":
                    content.append(
                        {
                            "type": "redacted_thinking",
                            "data": str(tb.get("data", "")),
                        }
                    )
        elif message.get("reasoning_content"):
            # Fallback: use reasoning_content as a thinking block
            content.append(
                {
                    "type": "thinking",
                    "thinking": str(message["reasoning_content"]),
                }
            )

        # Handle text content
        text = message.get("content")
        if text is not None:
            content.append({"type": "text", "text": text})

        # Handle tool calls
        tool_calls = message.get("tool_calls", [])
        for tc in tool_calls:
            func = tc.get("function", {})

            # Restore original tool name if it was truncated
            truncated_name = func.get("name", "")
            original_name = (
                tool_name_mapping.get(truncated_name, truncated_name)
                if tool_name_mapping
                else truncated_name
            )

            # Parse arguments
            arguments = func.get("arguments", "{}")
            parsed_input = safe_json_loads(arguments)
            if isinstance(parsed_input, str):
                parsed_input = {}

            content.append(
                {
                    "type": "tool_use",
                    "id": tc.get("id", f"toolu_{uuid.uuid4().hex[:12]}"),
                    "name": original_name,
                    "input": parsed_input,
                }
            )

    return content


def map_finish_reason(openai_finish_reason: Optional[str]) -> str:
    """Map OpenAI finish_reason to Anthropic stop_reason."""
    if not openai_finish_reason:
        return "end_turn"
    return OPENAI_TO_ANTHROPIC_STOP_REASON.get(openai_finish_reason, "end_turn")


def convert_usage(openai_usage: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert OpenAI usage to Anthropic format.

    OpenAI: {prompt_tokens, completion_tokens, total_tokens, prompt_tokens_details, ...}
    Anthropic: {input_tokens, output_tokens, cache_creation_input_tokens, cache_read_input_tokens}
    """
    prompt_tokens = openai_usage.get("prompt_tokens", 0) or 0
    completion_tokens = openai_usage.get("completion_tokens", 0) or 0

    # Extract cache info from details
    prompt_details = openai_usage.get("prompt_tokens_details", {}) or {}
    cached_tokens = prompt_details.get("cached_tokens", 0) or 0
    cache_creation = prompt_details.get("cache_creation_tokens", 0) or 0

    # input_tokens excludes cached tokens in Anthropic format
    input_tokens = max(0, prompt_tokens - cached_tokens - cache_creation)

    usage: Dict[str, Any] = {
        "input_tokens": input_tokens,
        "output_tokens": completion_tokens,
    }

    if cache_creation > 0:
        usage["cache_creation_input_tokens"] = cache_creation
    if cached_tokens > 0:
        usage["cache_read_input_tokens"] = cached_tokens

    return usage


def convert_response(
    openai_response: Dict[str, Any],
    *,
    tool_name_mapping: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Convert a complete OpenAI ChatCompletion response to Anthropic Messages format.

    Args:
        openai_response: OpenAI ChatCompletion response dict
        tool_name_mapping: Mapping of truncated tool names to originals (from convert_request)

    Returns:
        Anthropic Messages API response dict
    """
    choices = openai_response.get("choices", [])

    # Convert content
    anthropic_content = convert_openai_content_to_anthropic(choices, tool_name_mapping)

    # Map finish reason
    finish_reason = choices[0].get("finish_reason", "stop") if choices else "stop"
    stop_reason = map_finish_reason(finish_reason)

    # Convert usage
    openai_usage = openai_response.get("usage", {})
    anthropic_usage = (
        convert_usage(openai_usage)
        if openai_usage
        else {
            "input_tokens": 0,
            "output_tokens": 0,
        }
    )

    response: Dict[str, Any] = {
        "id": openai_response.get("id", f"msg_{uuid.uuid4().hex[:12]}"),
        "type": "message",
        "role": "assistant",
        "model": openai_response.get("model", "unknown"),
        "content": anthropic_content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": anthropic_usage,
    }

    return response
