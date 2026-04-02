"""
Shared utilities for protocol conversion.
"""

import copy
import hashlib
import json
from typing import Any, Dict, List, Optional

from .constants import (
    OPENAI_MAX_TOOL_NAME_LENGTH,
    TOOL_NAME_HASH_LENGTH,
    TOOL_NAME_PREFIX_LENGTH,
)


def truncate_tool_name(name: str) -> str:
    """
    Truncate tool names that exceed OpenAI's 64-character limit.
    Uses format: {55-char-prefix}_{8-char-hash} to avoid collisions.
    """
    if len(name) <= OPENAI_MAX_TOOL_NAME_LENGTH:
        return name
    name_hash = hashlib.sha256(name.encode()).hexdigest()[:TOOL_NAME_HASH_LENGTH]
    return f"{name[:TOOL_NAME_PREFIX_LENGTH]}_{name_hash}"


def create_tool_name_mapping(tools: List[Dict[str, Any]]) -> Dict[str, str]:
    """
    Create a mapping of truncated tool names to original names.
    Only includes entries for tools that were actually truncated.
    """
    mapping: Dict[str, str] = {}
    for tool in tools:
        original_name = tool.get("name", "")
        truncated_name = truncate_tool_name(original_name)
        if truncated_name != original_name:
            mapping[truncated_name] = original_name
    return mapping


def safe_json_loads(s: str) -> Any:
    """Safely parse JSON string, returning the raw string on failure."""
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return s


def safe_json_dumps(obj: Any) -> str:
    """Safely serialize to JSON string."""
    if isinstance(obj, str):
        return obj
    try:
        return json.dumps(obj)
    except (TypeError, ValueError):
        return str(obj)


def filter_schema_for_anthropic(schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Filter out unsupported fields from JSON schema for Anthropic's output_format API.

    Anthropic doesn't support: maxItems/minItems, minimum/maximum,
    exclusiveMinimum/exclusiveMaximum, minLength/maxLength.
    Adds removed constraint info to description.
    """
    if not isinstance(schema, dict):
        return schema

    unsupported_fields = {
        "maxItems",
        "minItems",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "minLength",
        "maxLength",
    }

    constraint_labels = {
        "minItems": "minimum number of items: {}",
        "maxItems": "maximum number of items: {}",
        "minimum": "minimum value: {}",
        "maximum": "maximum value: {}",
        "exclusiveMinimum": "exclusive minimum value: {}",
        "exclusiveMaximum": "exclusive maximum value: {}",
        "minLength": "minimum length: {}",
        "maxLength": "maximum length: {}",
    }

    constraint_descriptions = []
    for field in sorted(unsupported_fields):
        if field in schema:
            constraint_descriptions.append(constraint_labels[field].format(schema[field]))

    result: Dict[str, Any] = {}

    if constraint_descriptions:
        existing_desc = schema.get("description", "")
        constraint_note = "Note: " + ", ".join(constraint_descriptions) + "."
        result["description"] = (
            f"{existing_desc} {constraint_note}" if existing_desc else constraint_note
        )

    for key, value in schema.items():
        if key in unsupported_fields:
            continue
        if key == "description" and "description" in result:
            continue

        if key == "properties" and isinstance(value, dict):
            result[key] = {k: filter_schema_for_anthropic(v) for k, v in value.items()}
        elif key == "items" and isinstance(value, dict):
            result[key] = filter_schema_for_anthropic(value)
        elif key == "$defs" and isinstance(value, dict):
            result[key] = {k: filter_schema_for_anthropic(v) for k, v in value.items()}
        elif key in ("anyOf", "allOf", "oneOf") and isinstance(value, list):
            result[key] = [filter_schema_for_anthropic(item) for item in value]
        else:
            result[key] = value

    # Anthropic requires additionalProperties=false for object schemas
    if result.get("type") == "object" and "additionalProperties" not in result:
        result["additionalProperties"] = False

    return result


def unpack_defs(
    schema: Dict[str, Any],
    defs: Dict[str, Any],
    _expanding: Optional[set] = None,
) -> None:
    """
    Recursively resolve $ref references in a JSON schema using provided $defs.
    Modifies the schema in-place. Handles circular references by stopping
    expansion when a definition is already being expanded in the current path.
    """
    if not isinstance(schema, dict):
        return

    if _expanding is None:
        _expanding = set()

    if "$ref" in schema:
        ref_path = schema.pop("$ref")
        # Extract the definition name from the ref path (e.g., "/$defs/Foo" -> "Foo")
        ref_name = ref_path.rsplit("/", 1)[-1]
        if ref_name in defs:
            if ref_name in _expanding:
                # Circular reference — stop expanding to avoid infinite recursion
                return
            _expanding.add(ref_name)
            # Deep copy to prevent mutation of defs when processing
            # nested $refs in the expanded content
            schema.update(copy.deepcopy(defs[ref_name]))

    for key, value in list(schema.items()):
        if isinstance(value, dict):
            unpack_defs(value, defs, set(_expanding))
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    unpack_defs(item, defs, set(_expanding))


def translate_anthropic_image_to_openai(image_source: dict) -> Optional[str]:
    """
    Translate Anthropic image source format to OpenAI-compatible image URL.

    Anthropic formats:
    1. Base64: {"type": "base64", "media_type": "image/jpeg", "data": "..."}
    2. URL: {"type": "url", "url": "https://..."}
    """
    if not isinstance(image_source, dict):
        return None

    source_type = image_source.get("type")
    if source_type == "base64":
        media_type = image_source.get("media_type", "image/jpeg")
        image_data = image_source.get("data", "")
        if image_data:
            return f"data:{media_type};base64,{image_data}"
    elif source_type == "url":
        return image_source.get("url", "")

    return None
