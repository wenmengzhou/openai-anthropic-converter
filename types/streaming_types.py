"""
Anthropic streaming (SSE) event type definitions.
"""

from typing import Any, Dict, List, Literal, Optional, Union

from typing_extensions import NotRequired, TypedDict

# ── Delta Types ─────────────────────────────────────────────────────────


class AnthropicTextDelta(TypedDict):
    type: Literal["text_delta"]
    text: str


class AnthropicInputJsonDelta(TypedDict):
    type: Literal["input_json_delta"]
    partial_json: str


class AnthropicThinkingDelta(TypedDict):
    type: Literal["thinking_delta"]
    thinking: str


class AnthropicSignatureDelta(TypedDict):
    type: Literal["signature_delta"]
    signature: str


AnthropicDelta = Union[
    AnthropicTextDelta,
    AnthropicInputJsonDelta,
    AnthropicThinkingDelta,
    AnthropicSignatureDelta,
]


# ── Content Block Types for Start Events ────────────────────────────────


class AnthropicTextBlockStart(TypedDict):
    type: Literal["text"]
    text: str


class AnthropicToolUseBlockStart(TypedDict):
    type: Literal["tool_use"]
    id: str
    name: str
    input: Dict[str, Any]


class AnthropicThinkingBlockStart(TypedDict):
    type: Literal["thinking"]
    thinking: str
    signature: NotRequired[str]


AnthropicContentBlockStart = Union[
    AnthropicTextBlockStart,
    AnthropicToolUseBlockStart,
    AnthropicThinkingBlockStart,
]


# ── SSE Event Types ─────────────────────────────────────────────────────


class AnthropicUsageDelta(TypedDict, total=False):
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int
    cache_read_input_tokens: int


class AnthropicMessageStartBody(TypedDict, total=False):
    id: str
    type: str
    role: str
    model: str
    content: List[Any]
    stop_reason: Optional[str]
    stop_sequence: Optional[str]
    usage: AnthropicUsageDelta


class AnthropicMessageStartEvent(TypedDict):
    type: Literal["message_start"]
    message: AnthropicMessageStartBody


class AnthropicContentBlockStartEvent(TypedDict):
    type: Literal["content_block_start"]
    index: int
    content_block: AnthropicContentBlockStart


class AnthropicContentBlockDeltaEvent(TypedDict):
    type: Literal["content_block_delta"]
    index: int
    delta: AnthropicDelta


class AnthropicContentBlockStopEvent(TypedDict):
    type: Literal["content_block_stop"]
    index: int


class AnthropicMessageDeltaBody(TypedDict, total=False):
    stop_reason: Optional[str]
    stop_sequence: Optional[str]


class AnthropicMessageDeltaEvent(TypedDict, total=False):
    type: Literal["message_delta"]
    delta: AnthropicMessageDeltaBody
    usage: AnthropicUsageDelta


class AnthropicMessageStopEvent(TypedDict):
    type: Literal["message_stop"]


AnthropicSSEEvent = Union[
    AnthropicMessageStartEvent,
    AnthropicContentBlockStartEvent,
    AnthropicContentBlockDeltaEvent,
    AnthropicContentBlockStopEvent,
    AnthropicMessageDeltaEvent,
    AnthropicMessageStopEvent,
]
