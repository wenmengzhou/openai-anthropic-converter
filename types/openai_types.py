"""
OpenAI Chat Completion API type definitions (TypedDicts).
These mirror the OpenAI wire format for /v1/chat/completions.
"""

from typing import Any, Dict, List, Literal, Optional, Union

from typing_extensions import NotRequired, TypedDict


# ── Message Types ───────────────────────────────────────────────────────

class OpenAISystemMessage(TypedDict, total=False):
    role: Literal["system"]
    content: Union[str, List[Any]]


class OpenAIUserMessage(TypedDict, total=False):
    role: Literal["user"]
    content: Union[str, List[Any]]


class OpenAIFunctionCall(TypedDict, total=False):
    name: str
    arguments: str


class OpenAIToolCall(TypedDict, total=False):
    id: str
    type: Literal["function"]
    function: OpenAIFunctionCall
    index: int


class OpenAIThinkingBlock(TypedDict, total=False):
    type: str  # "thinking" | "redacted_thinking"
    thinking: str
    signature: str
    data: str
    cache_control: Dict[str, Any]


class OpenAIAssistantMessage(TypedDict, total=False):
    role: Literal["assistant"]
    content: Optional[str]
    tool_calls: List[OpenAIToolCall]
    thinking_blocks: List[OpenAIThinkingBlock]
    reasoning_content: Optional[str]


class OpenAIToolMessage(TypedDict, total=False):
    role: Literal["tool"]
    content: Union[str, List[Any]]
    tool_call_id: str


OpenAIMessage = Union[
    OpenAISystemMessage,
    OpenAIUserMessage,
    OpenAIAssistantMessage,
    OpenAIToolMessage,
    Dict[str, Any],
]


# ── Tool Definitions ────────────────────────────────────────────────────

class OpenAIFunctionDef(TypedDict, total=False):
    name: str
    description: str
    parameters: Dict[str, Any]


class OpenAIToolDef(TypedDict, total=False):
    type: Literal["function"]
    function: OpenAIFunctionDef


class OpenAIToolChoiceFunction(TypedDict, total=False):
    name: str


class OpenAIToolChoiceObject(TypedDict, total=False):
    type: Literal["function"]
    function: OpenAIToolChoiceFunction


OpenAIToolChoice = Union[str, OpenAIToolChoiceObject]  # "auto" | "required" | "none" | object


# ── Content Objects ─────────────────────────────────────────────────────

class OpenAITextContent(TypedDict, total=False):
    type: Literal["text"]
    text: str


class OpenAIImageUrlDetail(TypedDict, total=False):
    url: str


class OpenAIImageContent(TypedDict, total=False):
    type: Literal["image_url"]
    image_url: OpenAIImageUrlDetail


# ── Usage ───────────────────────────────────────────────────────────────

class OpenAIPromptTokensDetails(TypedDict, total=False):
    cached_tokens: int
    cache_creation_tokens: int


class OpenAICompletionTokensDetails(TypedDict, total=False):
    reasoning_tokens: int
    text_tokens: int


class OpenAIUsage(TypedDict, total=False):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_tokens_details: OpenAIPromptTokensDetails
    completion_tokens_details: OpenAICompletionTokensDetails


# ── Response Format ─────────────────────────────────────────────────────

class OpenAIJsonSchema(TypedDict, total=False):
    name: str
    schema: Dict[str, Any]
    strict: bool


class OpenAIResponseFormat(TypedDict, total=False):
    type: str  # "text" | "json_object" | "json_schema"
    json_schema: OpenAIJsonSchema


# ── Request ─────────────────────────────────────────────────────────────

class OpenAIRequest(TypedDict, total=False):
    model: str
    messages: List[OpenAIMessage]
    max_tokens: int
    max_completion_tokens: int
    tools: List[OpenAIToolDef]
    tool_choice: OpenAIToolChoice
    stop: Union[str, List[str]]
    temperature: float
    top_p: float
    stream: bool
    response_format: OpenAIResponseFormat
    reasoning_effort: str  # "low" | "medium" | "high" | "minimal"
    user: str
    parallel_tool_calls: bool
    thinking: Dict[str, Any]  # passthrough for Claude models


# ── Response ────────────────────────────────────────────────────────────

class OpenAIResponseMessage(TypedDict, total=False):
    role: str
    content: Optional[str]
    tool_calls: List[OpenAIToolCall]
    thinking_blocks: List[OpenAIThinkingBlock]
    reasoning_content: Optional[str]


class OpenAIChoice(TypedDict, total=False):
    index: int
    message: OpenAIResponseMessage
    finish_reason: str  # "stop" | "length" | "tool_calls" | "content_filter"


class OpenAIResponse(TypedDict, total=False):
    id: str
    object: str  # "chat.completion"
    created: int
    model: str
    choices: List[OpenAIChoice]
    usage: OpenAIUsage


# ── Streaming ───────────────────────────────────────────────────────────

class OpenAIDelta(TypedDict, total=False):
    role: str
    content: Optional[str]
    tool_calls: List[OpenAIToolCall]
    thinking_blocks: List[OpenAIThinkingBlock]
    reasoning_content: Optional[str]


class OpenAIStreamChoice(TypedDict, total=False):
    index: int
    delta: OpenAIDelta
    finish_reason: Optional[str]


class OpenAIStreamChunk(TypedDict, total=False):
    id: str
    object: str  # "chat.completion.chunk"
    created: int
    model: str
    choices: List[OpenAIStreamChoice]
    usage: OpenAIUsage
