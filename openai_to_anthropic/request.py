"""
OpenAI -> Anthropic request conversion.

Converts OpenAI ChatCompletion request format to Anthropic Messages API format.
"""

import copy
import json
from typing import Any, Dict, List, Optional, Tuple, Union

from ..constants import (
    DEFAULT_MAX_TOKENS,
    OUTPUT_FORMAT_SUPPORTED_MODEL_SUBSTRINGS,
    REASONING_EFFORT_TO_BUDGET_TOKENS,
    RESPONSE_FORMAT_TOOL_NAME,
)
from ..utils import filter_schema_for_anthropic, safe_json_loads, unpack_defs


def extract_system_messages(
    messages: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Extract system messages from message list and convert to Anthropic system format.

    Returns:
        (anthropic_system_blocks, remaining_messages)
    """
    system_blocks: List[Dict[str, Any]] = []
    remaining: List[Dict[str, Any]] = []

    for msg in messages:
        if msg.get("role") != "system":
            remaining.append(msg)
            continue

        content = msg.get("content")
        if isinstance(content, str):
            if not content:
                continue
            block: Dict[str, Any] = {"type": "text", "text": content}
            if "cache_control" in msg:
                block["cache_control"] = msg["cache_control"]
            system_blocks.append(block)
        elif isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    continue
                text_val = item.get("text")
                if item.get("type") == "text" and not text_val:
                    continue
                block = {"type": item.get("type", "text"), "text": text_val}
                if "cache_control" in item:
                    block["cache_control"] = item["cache_control"]
                system_blocks.append(block)

    return system_blocks, remaining


def convert_openai_message_to_anthropic(
    msg: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Convert a single OpenAI message to one or more Anthropic messages.

    Handles role mapping:
    - user -> user (with content block conversion)
    - assistant -> assistant (with tool_calls -> tool_use conversion)
    - tool -> user (with tool_result content block)
    """
    role = msg.get("role")
    content = msg.get("content")
    result: List[Dict[str, Any]] = []

    if role == "user":
        anthropic_content = _convert_user_content(content)
        if anthropic_content:
            result.append({"role": "user", "content": anthropic_content})

    elif role == "assistant":
        anthropic_content = _convert_assistant_content(msg)
        if anthropic_content:
            result.append({"role": "assistant", "content": anthropic_content})

    elif role == "tool":
        tool_result = _convert_tool_message(msg)
        result.append({"role": "user", "content": [tool_result]})

    return result


def _convert_user_content(
    content: Any,
) -> Union[str, List[Dict[str, Any]]]:
    """Convert OpenAI user message content to Anthropic format."""
    if isinstance(content, str):
        return content

    if not isinstance(content, list):
        return str(content) if content else ""

    blocks: List[Dict[str, Any]] = []
    for item in content:
        if isinstance(item, str):
            blocks.append({"type": "text", "text": item})
        elif isinstance(item, dict):
            item_type = item.get("type", "")
            if item_type == "text":
                block: Dict[str, Any] = {"type": "text", "text": item.get("text", "")}
                if "cache_control" in item:
                    block["cache_control"] = item["cache_control"]
                blocks.append(block)
            elif item_type == "image_url":
                image_url = item.get("image_url", {})
                url = image_url.get("url", "") if isinstance(image_url, dict) else ""
                blocks.append(_convert_image_url_to_anthropic(url))
            else:
                # Pass through unknown content types
                blocks.append(item)

    return blocks if blocks else ""


def _convert_image_url_to_anthropic(url: str) -> Dict[str, Any]:
    """Convert an OpenAI image_url to Anthropic image source format."""
    if url.startswith("data:"):
        # Parse data URI: data:image/jpeg;base64,<data>
        try:
            header, data = url.split(",", 1)
            media_type = header.split(":")[1].split(";")[0]
            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": data,
                },
            }
        except (IndexError, ValueError):
            pass

    # URL-based image
    return {
        "type": "image",
        "source": {"type": "url", "url": url},
    }


def _convert_assistant_content(msg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert OpenAI assistant message to Anthropic content blocks."""
    blocks: List[Dict[str, Any]] = []
    content = msg.get("content")

    # Add thinking blocks first
    thinking_blocks = msg.get("thinking_blocks", [])
    if thinking_blocks:
        for tb in thinking_blocks:
            tb_type = tb.get("type", "thinking")
            if tb_type == "thinking":
                block: Dict[str, Any] = {
                    "type": "thinking",
                    "thinking": tb.get("thinking", ""),
                }
                sig = tb.get("signature")
                if sig:
                    block["signature"] = sig
                blocks.append(block)
            elif tb_type == "redacted_thinking":
                blocks.append({
                    "type": "redacted_thinking",
                    "data": tb.get("data", ""),
                })

    # Add text content
    if isinstance(content, str) and content:
        blocks.append({"type": "text", "text": content})
    elif isinstance(content, list):
        for item in content:
            if isinstance(item, str):
                blocks.append({"type": "text", "text": item})
            elif isinstance(item, dict) and item.get("type") == "text":
                block = {"type": "text", "text": item.get("text", "")}
                if "cache_control" in item:
                    block["cache_control"] = item["cache_control"]
                blocks.append(block)

    # Convert tool_calls to tool_use blocks
    tool_calls = msg.get("tool_calls", [])
    for tc in tool_calls:
        func = tc.get("function", {})
        arguments = func.get("arguments", "{}")
        parsed_args = safe_json_loads(arguments)
        if isinstance(parsed_args, str):
            # Try wrapping as a dict if parse fails
            parsed_args = {"raw": parsed_args}

        blocks.append({
            "type": "tool_use",
            "id": tc.get("id", ""),
            "name": func.get("name", ""),
            "input": parsed_args,
        })

    return blocks


def _convert_tool_message(msg: Dict[str, Any]) -> Dict[str, Any]:
    """Convert OpenAI tool message to Anthropic tool_result content block."""
    content = msg.get("content", "")
    if isinstance(content, list):
        # Convert list content items
        result_content = []
        for item in content:
            if isinstance(item, str):
                result_content.append({"type": "text", "text": item})
            elif isinstance(item, dict):
                result_content.append(item)
        content = result_content

    return {
        "type": "tool_result",
        "tool_use_id": msg.get("tool_call_id", ""),
        "content": content,
    }


def convert_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert a list of OpenAI messages to Anthropic format.

    Handles:
    - Role conversion (tool -> user with tool_result)
    - Merging consecutive same-role messages (Anthropic requires user/assistant alternation)
    - Content block conversion (image_url, tool_calls, etc.)
    """
    anthropic_messages: List[Dict[str, Any]] = []

    for msg in messages:
        converted = convert_openai_message_to_anthropic(msg)
        anthropic_messages.extend(converted)

    # Merge consecutive same-role messages (Anthropic requires strict alternation)
    merged = _merge_consecutive_messages(anthropic_messages)
    # Ensure conversation starts with user and alternates
    merged = _ensure_alternation(merged)

    return merged


def _merge_consecutive_messages(
    messages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Merge consecutive messages with the same role."""
    if not messages:
        return []

    merged: List[Dict[str, Any]] = []
    for msg in messages:
        if merged and merged[-1]["role"] == msg["role"]:
            # Merge content into previous message
            prev_content = merged[-1]["content"]
            curr_content = msg["content"]

            prev_list = _to_content_list(prev_content)
            curr_list = _to_content_list(curr_content)
            merged[-1]["content"] = prev_list + curr_list
        else:
            merged.append(msg)

    return merged


def _to_content_list(content: Any) -> List[Dict[str, Any]]:
    """Normalize content to a list of content blocks."""
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if isinstance(content, list):
        return content
    return [{"type": "text", "text": str(content)}]


def _ensure_alternation(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Ensure messages alternate between user and assistant.
    Insert placeholder messages if needed.
    """
    if not messages:
        return messages

    result: List[Dict[str, Any]] = []
    for msg in messages:
        if result and result[-1]["role"] == msg["role"]:
            # Insert a placeholder of the opposite role
            if msg["role"] == "user":
                result.append({"role": "assistant", "content": [{"type": "text", "text": "."}]})
            else:
                result.append({"role": "user", "content": [{"type": "text", "text": "."}]})
        result.append(msg)

    # Anthropic requires the first message to be from user
    if result and result[0]["role"] != "user":
        result.insert(0, {"role": "user", "content": [{"type": "text", "text": "."}]})

    return result


def convert_tools(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert OpenAI tool definitions to Anthropic format.

    OpenAI: {type: "function", function: {name, description, parameters}}
    Anthropic: {name, description, input_schema, type: "custom"}
    """
    anthropic_tools: List[Dict[str, Any]] = []
    for tool in tools:
        tool_type = tool.get("type", "function")
        if tool_type == "function":
            func = tool.get("function", {})
            input_schema = func.get("parameters", {
                "type": "object",
                "properties": {},
            })
            # Ensure input_schema.type is "object" (Anthropic requirement)
            if input_schema.get("type") != "object":
                input_schema = dict(input_schema)
                input_schema["type"] = "object"
                if "properties" not in input_schema:
                    input_schema["properties"] = {}

            anthropic_tool: Dict[str, Any] = {
                "name": func.get("name", ""),
                "input_schema": input_schema,
            }
            desc = func.get("description")
            if desc is not None:
                anthropic_tool["description"] = desc

            # Preserve cache_control
            cache_ctrl = tool.get("cache_control") or func.get("cache_control")
            if cache_ctrl:
                anthropic_tool["cache_control"] = cache_ctrl

            anthropic_tools.append(anthropic_tool)
        else:
            # Pass through non-function tools (e.g., already in Anthropic format)
            anthropic_tools.append(tool)

    return anthropic_tools


def convert_tool_choice(
    tool_choice: Any,
    parallel_tool_calls: Optional[bool] = None,
) -> Optional[Dict[str, Any]]:
    """
    Convert OpenAI tool_choice to Anthropic format.

    OpenAI: "auto" | "required" | "none" | {type:"function", function:{name:...}}
    Anthropic: {type: "auto"/"any"/"tool"/"none", name?: str, disable_parallel_tool_use?: bool}
    """
    result: Optional[Dict[str, Any]] = None

    if tool_choice == "auto":
        result = {"type": "auto"}
    elif tool_choice == "required":
        result = {"type": "any"}
    elif tool_choice == "none":
        result = {"type": "none"}
    elif isinstance(tool_choice, dict):
        if "type" in tool_choice and "function" not in tool_choice:
            tc_type = tool_choice.get("type")
            if tc_type == "auto":
                result = {"type": "auto"}
            elif tc_type in ("required", "any"):
                result = {"type": "any"}
            elif tc_type == "none":
                result = {"type": "none"}
        else:
            func_name = tool_choice.get("function", {}).get("name")
            if func_name:
                result = {"type": "tool", "name": func_name}

    if parallel_tool_calls is not None and tool_choice != "none":
        if result is not None:
            result["disable_parallel_tool_use"] = not parallel_tool_calls
        else:
            result = {
                "type": "auto",
                "disable_parallel_tool_use": not parallel_tool_calls,
            }

    return result


def convert_stop_sequences(stop: Any) -> Optional[List[str]]:
    """Convert OpenAI stop parameter to Anthropic stop_sequences."""
    if isinstance(stop, str):
        return [stop] if stop.strip() else None
    if isinstance(stop, list):
        filtered = [s for s in stop if isinstance(s, str) and s.strip()]
        return filtered if filtered else None
    return None


def convert_response_format(
    value: Dict[str, Any],
    model: str,
) -> Dict[str, Any]:
    """
    Convert OpenAI response_format to Anthropic output_format or JSON tool.

    For newer models: returns {"output_format": {...}}
    For older models: returns {"tools": [...], "tool_choice": {...}}
    """
    result: Dict[str, Any] = {}

    if value.get("type") in ("text", None):
        return result

    json_schema = _extract_json_schema(value)
    if json_schema is None:
        return result

    # Check if model supports native output_format
    supports_output_format = any(
        sub in model for sub in OUTPUT_FORMAT_SUPPORTED_MODEL_SUBSTRINGS
    )

    if supports_output_format:
        # Resolve $ref/$defs and filter schema
        schema = copy.deepcopy(json_schema)
        defs = schema.pop("$defs", schema.pop("definitions", {}))
        if defs:
            unpack_defs(schema, defs)
        filtered = filter_schema_for_anthropic(schema)
        result["output_format"] = {"type": "json_schema", "schema": filtered}
    else:
        # Use tool-based JSON mode
        tool = {
            "name": RESPONSE_FORMAT_TOOL_NAME,
            "input_schema": json_schema if json_schema else {
                "type": "object",
                "properties": {},
                "additionalProperties": True,
            },
        }
        result["json_mode_tool"] = tool
        result["tool_choice"] = {"name": RESPONSE_FORMAT_TOOL_NAME, "type": "tool"}

    result["json_mode"] = True
    return result


def _extract_json_schema(value: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Extract the JSON schema from a response_format value."""
    if "response_schema" in value:
        return value["response_schema"]
    if "json_schema" in value:
        return value["json_schema"].get("schema")
    return None


def convert_reasoning_effort(
    reasoning_effort: str,
    model: str,
) -> Dict[str, Any]:
    """
    Convert OpenAI reasoning_effort to Anthropic thinking param.

    Returns dict with 'thinking' key and optionally 'output_config'.
    """
    result: Dict[str, Any] = {}

    if reasoning_effort == "none":
        return result

    budget = REASONING_EFFORT_TO_BUDGET_TOKENS.get(reasoning_effort, 10000)
    result["thinking"] = {"type": "enabled", "budget_tokens": budget}

    return result


def convert_context_management(
    context_management: Any,
) -> Optional[Dict[str, Any]]:
    """
    Convert OpenAI context_management to Anthropic format.

    OpenAI: [{"type": "compaction", "compact_threshold": 200000}]
    Anthropic: {"edits": [{"type": "compact_20260112", "trigger": {...}}]}
    """
    # Already in Anthropic format
    if isinstance(context_management, dict) and "edits" in context_management:
        return context_management

    if not isinstance(context_management, list):
        return None

    edits = []
    for entry in context_management:
        if not isinstance(entry, dict):
            continue
        if entry.get("type") == "compaction":
            edit: Dict[str, Any] = {"type": "compact_20260112"}
            threshold = entry.get("compact_threshold")
            if threshold is not None and isinstance(threshold, (int, float)):
                edit["trigger"] = {"type": "input_tokens", "value": int(threshold)}
            edits.append(edit)

    return {"edits": edits} if edits else None


def convert_request(
    openai_request: Dict[str, Any],
    *,
    default_max_tokens: int = DEFAULT_MAX_TOKENS,
) -> Dict[str, Any]:
    """
    Convert a complete OpenAI ChatCompletion request to Anthropic Messages format.

    Args:
        openai_request: OpenAI-format request dict
        default_max_tokens: Default max_tokens if not specified (Anthropic requires it)

    Returns:
        Anthropic Messages API request dict
    """
    request = dict(openai_request)  # shallow copy
    messages = request.pop("messages", [])
    model = request.pop("model", "")

    # Extract system messages
    system_blocks, remaining_messages = extract_system_messages(messages)

    # Convert messages
    anthropic_messages = convert_messages(remaining_messages)

    # Build result
    result: Dict[str, Any] = {
        "model": model,
        "messages": anthropic_messages,
    }

    if system_blocks:
        result["system"] = system_blocks

    # max_tokens (required by Anthropic)
    max_tokens = request.pop("max_tokens", None) or request.pop("max_completion_tokens", None)
    result["max_tokens"] = max_tokens if max_tokens else default_max_tokens

    # tools
    tools = request.pop("tools", None)
    if tools:
        result["tools"] = convert_tools(tools)

    # tool_choice + parallel_tool_calls
    tool_choice_val = request.pop("tool_choice", None)
    parallel_tool_calls = request.pop("parallel_tool_calls", None)
    if tool_choice_val is not None or parallel_tool_calls is not None:
        tc = convert_tool_choice(tool_choice_val, parallel_tool_calls)
        if tc:
            result["tool_choice"] = tc

    # stop -> stop_sequences
    stop = request.pop("stop", None)
    if stop is not None:
        seqs = convert_stop_sequences(stop)
        if seqs:
            result["stop_sequences"] = seqs

    # response_format -> output_format or json tool
    response_format = request.pop("response_format", None)
    if response_format and isinstance(response_format, dict):
        fmt_result = convert_response_format(response_format, model)
        if "output_format" in fmt_result:
            result["output_format"] = fmt_result["output_format"]
        if "json_mode_tool" in fmt_result:
            tools_list = result.get("tools", [])
            tools_list.append(fmt_result["json_mode_tool"])
            result["tools"] = tools_list
        if "tool_choice" in fmt_result and "tool_choice" not in result:
            result["tool_choice"] = fmt_result["tool_choice"]

    # reasoning_effort -> thinking
    reasoning_effort = request.pop("reasoning_effort", None)
    if reasoning_effort and isinstance(reasoning_effort, str):
        effort_result = convert_reasoning_effort(reasoning_effort, model)
        result.update(effort_result)

    # thinking (passthrough)
    thinking = request.pop("thinking", None)
    if thinking:
        result["thinking"] = thinking

    # user -> metadata.user_id
    user = request.pop("user", None)
    if user and isinstance(user, str):
        result["metadata"] = {"user_id": user}

    # context_management
    ctx_mgmt = request.pop("context_management", None)
    if ctx_mgmt:
        converted = convert_context_management(ctx_mgmt)
        if converted:
            result["context_management"] = converted

    # cache_control (passthrough)
    cache_control = request.pop("cache_control", None)
    if cache_control:
        result["cache_control"] = cache_control

    # Pass through simple params
    for param in ("temperature", "top_p", "top_k", "stream"):
        val = request.pop(param, None)
        if val is not None:
            result[param] = val

    return result
