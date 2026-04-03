"""
Anthropic -> OpenAI request conversion.

Converts Anthropic Messages API requests to OpenAI ChatCompletion format.
"""

import json
from typing import Any, Dict, List, Optional, Tuple

from ..utils import translate_anthropic_image_to_openai, truncate_tool_name


def convert_messages(
    messages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Convert Anthropic message list to OpenAI format.

    Handles:
    - User text/image/document content -> OpenAI text/image_url content
    - User tool_result -> OpenAI tool role message
    - Assistant text -> OpenAI assistant message
    - Assistant tool_use -> OpenAI tool_calls
    - Assistant thinking -> OpenAI thinking_blocks
    """
    new_messages: List[Dict[str, Any]] = []

    for msg in messages:
        role = msg.get("role")

        if role == "user":
            _convert_user_message(msg, new_messages)
        elif role == "assistant":
            _convert_assistant_message(msg, new_messages)

    return new_messages


def _convert_user_message(
    msg: Dict[str, Any],
    result: List[Dict[str, Any]],
) -> None:
    """Convert an Anthropic user message to OpenAI format messages."""
    content = msg.get("content")
    tool_messages: List[Dict[str, Any]] = []
    user_content_list: List[Dict[str, Any]] = []

    if isinstance(content, str):
        result.append({"role": "user", "content": content})
        return

    if not isinstance(content, list):
        result.append({"role": "user", "content": str(content) if content else ""})
        return

    for block in content:
        if not isinstance(block, dict):
            continue

        block_type = block.get("type", "")

        if block_type == "text":
            text_obj: Dict[str, Any] = {"type": "text", "text": block.get("text", "")}
            if "cache_control" in block:
                text_obj["cache_control"] = block["cache_control"]
            user_content_list.append(text_obj)

        elif block_type == "image":
            source = block.get("source", {})
            openai_url = translate_anthropic_image_to_openai(source)
            if openai_url:
                img_obj: Dict[str, Any] = {
                    "type": "image_url",
                    "image_url": {"url": openai_url},
                }
                if "cache_control" in block:
                    img_obj["cache_control"] = block["cache_control"]
                user_content_list.append(img_obj)

        elif block_type == "document":
            source = block.get("source", {})
            openai_url = translate_anthropic_image_to_openai(source)
            if openai_url:
                doc_obj: Dict[str, Any] = {
                    "type": "image_url",
                    "image_url": {"url": openai_url},
                }
                if "cache_control" in block:
                    doc_obj["cache_control"] = block["cache_control"]
                user_content_list.append(doc_obj)

        elif block_type == "tool_result":
            tool_msg = _convert_tool_result(block)
            tool_messages.append(tool_msg)

    # Tool messages go first (before user content)
    if tool_messages:
        result.extend(tool_messages)

    if user_content_list:
        result.append({"role": "user", "content": user_content_list})


def _convert_tool_result(block: Dict[str, Any]) -> Dict[str, Any]:
    """Convert an Anthropic tool_result block to an OpenAI tool message."""
    tool_call_id = block.get("tool_use_id", "")
    content = block.get("content")

    if content is None:
        return {"role": "tool", "tool_call_id": tool_call_id, "content": ""}

    if isinstance(content, str):
        return {"role": "tool", "tool_call_id": tool_call_id, "content": content}

    if isinstance(content, list):
        if len(content) == 0:
            return {"role": "tool", "tool_call_id": tool_call_id, "content": ""}
        if len(content) == 1:
            item = content[0]
            if isinstance(item, str):
                return {"role": "tool", "tool_call_id": tool_call_id, "content": item}
            if isinstance(item, dict):
                if item.get("type") == "text":
                    return {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": item.get("text", ""),
                    }
                elif item.get("type") == "image":
                    source = item.get("source", {})
                    url = translate_anthropic_image_to_openai(source) or ""
                    return {"role": "tool", "tool_call_id": tool_call_id, "content": url}
        else:
            # Multiple content items -> list content
            parts: List[Dict[str, Any]] = []
            for item in content:
                if isinstance(item, str):
                    parts.append({"type": "text", "text": item})
                elif isinstance(item, dict):
                    if item.get("type") == "text":
                        parts.append({"type": "text", "text": item.get("text", "")})
                    elif item.get("type") == "image":
                        source = item.get("source", {})
                        url = translate_anthropic_image_to_openai(source) or ""
                        if url:
                            parts.append(
                                {
                                    "type": "image_url",
                                    "image_url": {"url": url},
                                }
                            )
            if parts:
                return {"role": "tool", "tool_call_id": tool_call_id, "content": parts}

    return {"role": "tool", "tool_call_id": tool_call_id, "content": str(content)}


def _convert_assistant_message(
    msg: Dict[str, Any],
    result: List[Dict[str, Any]],
) -> None:
    """Convert an Anthropic assistant message to OpenAI format."""
    content = msg.get("content")
    text_parts: List[str] = []
    text_blocks_with_cache: List[Dict[str, Any]] = []
    has_cache_control = False
    tool_calls: List[Dict[str, Any]] = []
    thinking_blocks: List[Dict[str, Any]] = []

    if isinstance(content, str):
        text_parts.append(content)
    elif isinstance(content, list):
        for block in content:
            if isinstance(block, str):
                text_parts.append(block)
            elif isinstance(block, dict):
                block_type = block.get("type", "")

                if block_type == "text":
                    text_block: Dict[str, Any] = {
                        "type": "text",
                        "text": block.get("text", ""),
                    }
                    if "cache_control" in block:
                        text_block["cache_control"] = block["cache_control"]
                        has_cache_control = True
                    text_blocks_with_cache.append(text_block)
                    text_parts.append(block.get("text", ""))

                elif block_type in ("tool_use", "server_tool_use", "mcp_tool_use"):
                    tool_name = truncate_tool_name(block.get("name", ""))
                    tc: Dict[str, Any] = {
                        "id": block.get("id", ""),
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": json.dumps(block.get("input", {})),
                        },
                    }
                    tool_calls.append(tc)

                elif block_type == "thinking":
                    tb_entry: Dict[str, Any] = {
                        "type": "thinking",
                        "thinking": block.get("thinking", ""),
                    }
                    sig = block.get("signature")
                    if sig:
                        tb_entry["signature"] = sig
                    thinking_blocks.append(tb_entry)

                elif block_type == "redacted_thinking":
                    thinking_blocks.append(
                        {
                            "type": "redacted_thinking",
                            "data": block.get("data", ""),
                        }
                    )

    if not text_parts and not tool_calls and not thinking_blocks:
        return

    assistant_msg: Dict[str, Any] = {"role": "assistant"}

    # Use list format if cache_control present, otherwise string
    if has_cache_control and text_blocks_with_cache:
        assistant_msg["content"] = text_blocks_with_cache
    elif text_parts:
        assistant_msg["content"] = "".join(text_parts)
    else:
        assistant_msg["content"] = None

    if tool_calls:
        assistant_msg["tool_calls"] = tool_calls
    if thinking_blocks:
        assistant_msg["thinking_blocks"] = thinking_blocks

    result.append(assistant_msg)


def convert_system_to_messages(
    system: Any,
) -> List[Dict[str, Any]]:
    """
    Convert Anthropic system parameter to OpenAI system messages.

    Anthropic: string or list of {type: "text", text: "..."}
    OpenAI: [{role: "system", content: "..."}]
    """
    if isinstance(system, str):
        return [{"role": "system", "content": system}]

    if isinstance(system, list):
        content_blocks: List[Dict[str, Any]] = []
        for block in system:
            if isinstance(block, dict) and block.get("type") == "text":
                text_block: Dict[str, Any] = {
                    "type": "text",
                    "text": block.get("text", ""),
                }
                if "cache_control" in block:
                    text_block["cache_control"] = block["cache_control"]
                content_blocks.append(text_block)
        if content_blocks:
            return [{"role": "system", "content": content_blocks}]

    return []


def convert_tools(
    tools: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """
    Convert Anthropic tool definitions to OpenAI format.

    Returns:
        (openai_tools, tool_name_mapping) where tool_name_mapping maps
        truncated names back to originals for names > 64 chars.
    """
    openai_tools: List[Dict[str, Any]] = []
    name_mapping: Dict[str, str] = {}

    for tool in tools:
        tool_type = tool.get("type", "")

        # Skip Anthropic server/hosted tools that have no OpenAI equivalent
        # Regular tools have no type or type="custom"; server tools have specific types
        if tool_type and tool_type != "custom":
            continue

        original_name = tool.get("name", "")
        truncated_name = truncate_tool_name(original_name)

        if truncated_name != original_name:
            name_mapping[truncated_name] = original_name

        func_def: Dict[str, Any] = {"name": truncated_name}
        if "input_schema" in tool:
            func_def["parameters"] = tool["input_schema"]
        if "description" in tool:
            func_def["description"] = tool["description"]

        openai_tools.append({"type": "function", "function": func_def})

    return openai_tools, name_mapping


def convert_tool_choice(
    tool_choice: Dict[str, Any],
) -> Any:
    """
    Convert Anthropic tool_choice to OpenAI format.

    Anthropic: {type: "auto"/"any"/"tool"/"none", name?: str}
    OpenAI: "auto" | "required" | "none" | {type: "function", function: {name: "..."}}
    """
    tc_type = tool_choice.get("type", "auto")

    if tc_type == "any":
        return "required"
    elif tc_type == "auto":
        return "auto"
    elif tc_type == "none":
        return "none"
    elif tc_type == "tool":
        name = tool_choice.get("name", "")
        truncated = truncate_tool_name(name)
        return {"type": "function", "function": {"name": truncated}}

    return "auto"


def convert_output_format_to_response_format(
    output_format: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """
    Convert Anthropic output_format to OpenAI response_format.

    Anthropic: {"type": "json_schema", "schema": {...}}
    OpenAI: {"type": "json_schema", "json_schema": {"name": "...", "schema": {...}, "strict": true}}
    """
    if not isinstance(output_format, dict):
        return None
    if output_format.get("type") != "json_schema":
        return None
    schema = output_format.get("schema")
    if not schema:
        return None

    return {
        "type": "json_schema",
        "json_schema": {
            "name": "structured_output",
            "schema": schema,
            "strict": True,
        },
    }


def convert_request(
    anthropic_request: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """
    Convert a complete Anthropic Messages request to OpenAI ChatCompletion format.

    Args:
        anthropic_request: Anthropic Messages API request dict

    Returns:
        (openai_request, tool_name_mapping) where tool_name_mapping maps
        truncated tool names back to original names.
    """
    request = dict(anthropic_request)
    model = request.pop("model", "")
    messages_raw = request.pop("messages", [])
    tool_name_mapping: Dict[str, str] = {}

    # Convert messages
    openai_messages = convert_messages(messages_raw)

    # Convert system -> system messages (prepend)
    system = request.pop("system", None)
    if system:
        system_messages = convert_system_to_messages(system)
        openai_messages = system_messages + openai_messages

    result: Dict[str, Any] = {
        "model": model,
        "messages": openai_messages,
    }

    # max_tokens
    max_tokens = request.pop("max_tokens", None)
    if max_tokens:
        result["max_tokens"] = max_tokens

    # tools
    tools = request.pop("tools", None)
    if tools:
        # Detect web search tools
        has_web_search = any(
            isinstance(t, dict) and t.get("type", "").startswith("web_search") for t in tools
        )
        if has_web_search:
            result["web_search_options"] = {}

        # convert_tools filters out all server tools (web_search, text_editor, etc.)
        openai_tools, tool_name_mapping = convert_tools(tools)
        if openai_tools:
            result["tools"] = openai_tools

    # tool_choice
    tool_choice = request.pop("tool_choice", None)
    if tool_choice:
        result["tool_choice"] = convert_tool_choice(tool_choice)

    # thinking -> enable_thinking + thinking_budget (Bailian/DashScope compat)
    # Standard OpenAI uses reasoning_effort, but DashScope/Bailian uses enable_thinking.
    # We output enable_thinking for broad compatibility.
    thinking = request.pop("thinking", None)
    if thinking and isinstance(thinking, dict):
        thinking_type = thinking.get("type", "disabled")
        if thinking_type in ("enabled", "adaptive"):
            result["enable_thinking"] = True
            budget = thinking.get("budget_tokens")
            if budget:
                result["thinking_budget"] = budget

    # output_config.format or output_format (legacy) -> response_format
    output_config = request.pop("output_config", None)
    output_format = request.pop("output_format", None)
    fmt = None
    if output_config and isinstance(output_config, dict):
        fmt = output_config.get("format")
    if not fmt and output_format:
        fmt = output_format
    if fmt:
        resp_format = convert_output_format_to_response_format(fmt)
        if resp_format:
            result["response_format"] = resp_format

    # metadata.user_id -> user
    metadata = request.pop("metadata", None)
    if metadata and isinstance(metadata, dict) and "user_id" in metadata:
        result["user"] = metadata["user_id"]

    # stop_sequences -> stop
    stop_seqs = request.pop("stop_sequences", None)
    if stop_seqs:
        result["stop"] = stop_seqs

    # Pass through simple params
    for param in ("temperature", "top_p", "stream"):
        val = request.pop(param, None)
        if val is not None:
            result[param] = val

    # Silently drop Anthropic-only params that have no OpenAI equivalent
    for param in ("top_k", "context_management", "cache_control"):
        request.pop(param, None)

    return result, tool_name_mapping
