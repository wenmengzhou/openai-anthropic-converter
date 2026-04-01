"""
Anthropic -> OpenAI response conversion.

Converts Anthropic Messages API responses to OpenAI ChatCompletion format.
"""

import json
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from ..constants import ANTHROPIC_TO_OPENAI_FINISH_REASON, RESPONSE_FORMAT_TOOL_NAME


def extract_response_content(
    content_blocks: List[Dict[str, Any]],
) -> Tuple[
    str,  # text_content
    List[Dict[str, Any]],  # tool_calls (OpenAI format)
    Optional[List[Dict[str, Any]]],  # thinking_blocks
    Optional[str],  # reasoning_content
    Optional[List[Any]],  # citations
    Optional[List[Any]],  # web_search_results
    Optional[List[Any]],  # tool_results
]:
    """
    Parse Anthropic response content blocks into OpenAI-compatible components.
    """
    text_content = ""
    tool_calls: List[Dict[str, Any]] = []
    thinking_blocks: Optional[List[Dict[str, Any]]] = None
    reasoning_content: Optional[str] = None
    citations: Optional[List[Any]] = None
    web_search_results: Optional[List[Any]] = None
    tool_results: Optional[List[Any]] = None

    for idx, block in enumerate(content_blocks):
        block_type = block.get("type", "")

        if block_type == "text":
            text_content += block.get("text", "")

        elif block_type in ("tool_use", "server_tool_use"):
            tool_call = convert_tool_use_to_openai(block, idx)
            tool_calls.append(tool_call)

        elif block_type == "thinking" or block.get("thinking") is not None:
            if thinking_blocks is None:
                thinking_blocks = []
            thinking_blocks.append(block)

        elif block_type == "redacted_thinking":
            if thinking_blocks is None:
                thinking_blocks = []
            thinking_blocks.append(block)

        elif block_type.endswith("_tool_result"):
            if block_type == "web_search_tool_result" or block_type == "web_fetch_tool_result":
                if web_search_results is None:
                    web_search_results = []
                web_search_results.append(block)
            elif block_type != "tool_search_tool_result":
                if tool_results is None:
                    tool_results = []
                tool_results.append(block)

        # Handle citations
        if block.get("citations") is not None:
            if citations is None:
                citations = []
            citations.append(
                [
                    {**citation, "supported_text": block.get("text", "")}
                    for citation in block["citations"]
                ]
            )

    # Build reasoning_content from thinking blocks
    if thinking_blocks:
        reasoning_content = ""
        for tb in thinking_blocks:
            thinking_text = tb.get("thinking")
            if thinking_text:
                reasoning_content += thinking_text

    return (
        text_content,
        tool_calls,
        thinking_blocks,
        reasoning_content,
        citations,
        web_search_results,
        tool_results,
    )


def convert_tool_use_to_openai(
    block: Dict[str, Any],
    index: int,
) -> Dict[str, Any]:
    """
    Convert an Anthropic tool_use block to OpenAI tool call format.

    Anthropic: {type: "tool_use", id: "...", name: "...", input: {...}}
    OpenAI: {id: "...", type: "function", function: {name: "...", arguments: "..."}, index: N}
    """
    return {
        "id": block.get("id", ""),
        "type": "function",
        "function": {
            "name": block.get("name", ""),
            "arguments": json.dumps(block.get("input", {})),
        },
        "index": index,
    }


def convert_usage(
    usage_obj: Dict[str, Any],
    reasoning_content: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Convert Anthropic usage to OpenAI format.

    Anthropic: {input_tokens, output_tokens, cache_creation_input_tokens, cache_read_input_tokens}
    OpenAI: {prompt_tokens, completion_tokens, total_tokens, prompt_tokens_details, completion_tokens_details}
    """
    input_tokens = usage_obj.get("input_tokens", 0) or 0
    output_tokens = usage_obj.get("output_tokens", 0) or 0

    cache_creation = usage_obj.get("cache_creation_input_tokens", 0) or 0
    cache_read = usage_obj.get("cache_read_input_tokens", 0) or 0

    prompt_tokens = input_tokens + cache_creation + cache_read
    completion_tokens = output_tokens
    total_tokens = prompt_tokens + completion_tokens

    # Estimate reasoning tokens from reasoning_content length
    reasoning_tokens = 0
    if reasoning_content:
        # Rough estimate: ~4 chars per token
        reasoning_tokens = max(1, len(reasoning_content) // 4)

    usage: Dict[str, Any] = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "prompt_tokens_details": {
            "cached_tokens": cache_read,
            "cache_creation_tokens": cache_creation,
        },
        "completion_tokens_details": {
            "reasoning_tokens": reasoning_tokens,
            "text_tokens": max(0, completion_tokens - reasoning_tokens),
        },
    }

    return usage


def map_finish_reason(anthropic_stop_reason: Optional[str]) -> str:
    """Map Anthropic stop_reason to OpenAI finish_reason."""
    if not anthropic_stop_reason:
        return "stop"
    return ANTHROPIC_TO_OPENAI_FINISH_REASON.get(anthropic_stop_reason, "stop")


def convert_response(anthropic_response: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a complete Anthropic Messages response to OpenAI ChatCompletion format.

    Args:
        anthropic_response: Anthropic Messages API response dict

    Returns:
        OpenAI ChatCompletion response dict
    """
    content_blocks = anthropic_response.get("content", [])

    (
        text_content,
        tool_calls,
        thinking_blocks,
        reasoning_content,
        citations,
        web_search_results,
        tool_results,
    ) = extract_response_content(content_blocks)

    # Check for JSON mode (tool-based)
    json_mode_content = _check_json_mode(tool_calls)
    if json_mode_content is not None:
        text_content = json_mode_content
        tool_calls = []

    # Build message
    message: Dict[str, Any] = {
        "role": "assistant",
        "content": text_content if text_content else None,
    }

    if tool_calls:
        message["tool_calls"] = tool_calls
    if thinking_blocks:
        message["thinking_blocks"] = thinking_blocks
    if reasoning_content:
        message["reasoning_content"] = reasoning_content

    # Build provider_specific_fields
    provider_fields: Dict[str, Any] = {}
    if citations:
        provider_fields["citations"] = citations
    if web_search_results:
        provider_fields["web_search_results"] = web_search_results
    if tool_results:
        provider_fields["tool_results"] = tool_results
    if provider_fields:
        message["provider_specific_fields"] = provider_fields

    # Build usage
    usage = convert_usage(
        anthropic_response.get("usage", {}),
        reasoning_content,
    )

    # Build response
    finish_reason = map_finish_reason(anthropic_response.get("stop_reason"))

    response: Dict[str, Any] = {
        "id": anthropic_response.get("id", f"chatcmpl-{uuid.uuid4().hex[:12]}"),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": anthropic_response.get("model", ""),
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
        "usage": usage,
    }

    return response


def _check_json_mode(tool_calls: List[Dict[str, Any]]) -> Optional[str]:
    """
    Check if this is a JSON mode response (tool call with RESPONSE_FORMAT_TOOL_NAME).
    Returns the content string if it is, None otherwise.
    """
    if len(tool_calls) == 1:
        func = tool_calls[0].get("function", {})
        if func.get("name") == RESPONSE_FORMAT_TOOL_NAME:
            return func.get("arguments")
    return None
