"""
Constants for Anthropic <-> OpenAI protocol conversion.
"""

# Default max_tokens when not specified (Anthropic requires it)
DEFAULT_MAX_TOKENS = 4096

# Tool name used for JSON mode via tool calling (older Anthropic models)
RESPONSE_FORMAT_TOOL_NAME = "json_tool_call"

# OpenAI has a 64-character limit for function/tool names
OPENAI_MAX_TOOL_NAME_LENGTH = 64
TOOL_NAME_HASH_LENGTH = 8
TOOL_NAME_PREFIX_LENGTH = OPENAI_MAX_TOOL_NAME_LENGTH - TOOL_NAME_HASH_LENGTH - 1  # 55

# Anthropic -> OpenAI stop reason mapping
ANTHROPIC_TO_OPENAI_FINISH_REASON = {
    "end_turn": "stop",
    "max_tokens": "length",
    "stop_sequence": "stop",
    "tool_use": "tool_calls",
}

# OpenAI -> Anthropic stop reason mapping
OPENAI_TO_ANTHROPIC_STOP_REASON = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "content_filter": "end_turn",
}

# Reasoning effort -> thinking budget_tokens mapping
REASONING_EFFORT_TO_BUDGET_TOKENS = {
    "minimal": 1024,
    "low": 2048,
    "medium": 5000,
    "high": 10000,
}

# Thinking budget_tokens -> reasoning effort mapping (reverse)
BUDGET_TOKENS_THRESHOLDS = [
    (10000, "high"),
    (5000, "medium"),
    (2000, "low"),
    (0, "minimal"),
]

# Models that support native output_format (instead of tool-based JSON mode)
OUTPUT_FORMAT_SUPPORTED_MODEL_SUBSTRINGS = {
    "sonnet-4.5", "sonnet-4-5",
    "opus-4.1", "opus-4-1",
    "opus-4.5", "opus-4-5",
    "opus-4.6", "opus-4-6",
    "sonnet-4.6", "sonnet-4-6",
    "sonnet_4.6", "sonnet_4_6",
}
