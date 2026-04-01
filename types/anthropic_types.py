"""
Anthropic Messages API type definitions (TypedDicts).
These mirror the Anthropic wire format for /v1/messages.
"""

from typing import Any, Dict, List, Literal, Optional, Union

from typing_extensions import NotRequired, TypedDict


# ── Content Blocks ──────────────────────────────────────────────────────

class AnthropicTextBlock(TypedDict):
    type: Literal["text"]
    text: str
    citations: NotRequired[Optional[List[Any]]]
    cache_control: NotRequired[Optional[Dict[str, Any]]]


class AnthropicToolUseBlock(TypedDict):
    type: Literal["tool_use"]
    id: str
    name: str
    input: Dict[str, Any]
    cache_control: NotRequired[Optional[Dict[str, Any]]]


class AnthropicServerToolUseBlock(TypedDict):
    type: Literal["server_tool_use"]
    id: str
    name: str
    input: Dict[str, Any]


class AnthropicThinkingBlock(TypedDict):
    type: Literal["thinking"]
    thinking: str
    signature: NotRequired[Optional[str]]
    cache_control: NotRequired[Optional[Dict[str, Any]]]


class AnthropicRedactedThinkingBlock(TypedDict):
    type: Literal["redacted_thinking"]
    data: str
    cache_control: NotRequired[Optional[Dict[str, Any]]]


class AnthropicCompactionBlock(TypedDict):
    type: Literal["compaction"]


AnthropicContentBlock = Union[
    AnthropicTextBlock,
    AnthropicToolUseBlock,
    AnthropicServerToolUseBlock,
    AnthropicThinkingBlock,
    AnthropicRedactedThinkingBlock,
    AnthropicCompactionBlock,
    Dict[str, Any],  # fallback for tool_result blocks etc.
]


# ── Tool Definitions ────────────────────────────────────────────────────

class AnthropicInputSchema(TypedDict, total=False):
    type: str
    properties: Dict[str, Any]
    required: List[str]
    additionalProperties: bool


class AnthropicToolDef(TypedDict, total=False):
    name: str
    description: str
    input_schema: AnthropicInputSchema
    type: str
    cache_control: Dict[str, Any]


class AnthropicToolChoice(TypedDict, total=False):
    type: str  # "auto" | "any" | "tool" | "none"
    name: str
    disable_parallel_tool_use: bool


# ── System Message ──────────────────────────────────────────────────────

class AnthropicSystemMessageContent(TypedDict, total=False):
    type: str
    text: str
    cache_control: Dict[str, Any]


# ── Thinking Param ──────────────────────────────────────────────────────

class AnthropicThinkingParam(TypedDict, total=False):
    type: str  # "enabled" | "disabled" | "adaptive"
    budget_tokens: int


# ── Usage ───────────────────────────────────────────────────────────────

class AnthropicUsage(TypedDict, total=False):
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int
    cache_read_input_tokens: int
    server_tool_use: Dict[str, Any]
    inference_geo: str


# ── Messages ────────────────────────────────────────────────────────────

class AnthropicUserMessage(TypedDict):
    role: Literal["user"]
    content: Union[str, List[Any]]


class AnthropicAssistantMessage(TypedDict):
    role: Literal["assistant"]
    content: Union[str, List[Any]]


AnthropicMessage = Union[AnthropicUserMessage, AnthropicAssistantMessage]


# ── Metadata ────────────────────────────────────────────────────────────

class AnthropicMetadata(TypedDict, total=False):
    user_id: str


# ── Output Format ───────────────────────────────────────────────────────

class AnthropicOutputFormat(TypedDict, total=False):
    type: str  # "json_schema"
    schema: Dict[str, Any]


# ── Request ─────────────────────────────────────────────────────────────

class AnthropicRequest(TypedDict, total=False):
    model: str
    messages: List[AnthropicMessage]
    max_tokens: int
    system: Union[str, List[AnthropicSystemMessageContent]]
    tools: List[Union[AnthropicToolDef, Dict[str, Any]]]
    tool_choice: AnthropicToolChoice
    stop_sequences: List[str]
    temperature: float
    top_p: float
    top_k: int
    stream: bool
    metadata: AnthropicMetadata
    thinking: AnthropicThinkingParam
    output_format: AnthropicOutputFormat
    context_management: Dict[str, Any]
    cache_control: Dict[str, Any]


# ── Response ────────────────────────────────────────────────────────────

class AnthropicResponse(TypedDict, total=False):
    id: str
    type: str  # "message"
    role: str  # "assistant"
    model: str
    content: List[AnthropicContentBlock]
    stop_reason: Optional[str]  # "end_turn" | "max_tokens" | "stop_sequence" | "tool_use"
    stop_sequence: Optional[str]
    usage: AnthropicUsage
