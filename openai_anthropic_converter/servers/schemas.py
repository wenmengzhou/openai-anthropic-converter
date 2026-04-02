"""
Pydantic models for OpenAPI schema generation.

These models define the API contracts for both the OpenAI-compatible and
Anthropic-compatible proxy servers. They drive:
  - Swagger UI at /docs (interactive documentation)
  - ReDoc at /redoc (reference documentation)
  - OpenAPI JSON at /openapi.json (machine-readable schema)

Note: The actual conversion logic uses raw dicts internally.
These models are the documented contract, not the validation layer —
extra fields are allowed and passed through to the converters.
"""

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field

# ═══════════════════════════════════════════════════════════════════════════
# OpenAI Chat Completion API Models (for openai_server)
# ═══════════════════════════════════════════════════════════════════════════


class OpenAIFunctionCallSchema(BaseModel):
    name: str = Field(..., description="The name of the function to call")
    arguments: str = Field(..., description="JSON string of function arguments")


class OpenAIToolCallSchema(BaseModel):
    id: str = Field(..., description="Tool call ID")
    type: Literal["function"] = "function"
    function: OpenAIFunctionCallSchema
    index: Optional[int] = Field(None, description="Index of the tool call")


class OpenAIThinkingBlockSchema(BaseModel):
    type: str = Field(..., description="'thinking' or 'redacted_thinking'")
    thinking: Optional[str] = None
    signature: Optional[str] = None
    data: Optional[str] = Field(None, description="Encrypted data for redacted_thinking")

    model_config = {"extra": "allow"}


class OpenAITextContentSchema(BaseModel):
    type: Literal["text"] = "text"
    text: str


class OpenAIImageUrlDetailSchema(BaseModel):
    url: str = Field(..., description="Image URL or data URI (data:image/jpeg;base64,...)")
    detail: Optional[str] = Field(None, description="Image detail level: 'auto', 'low', 'high'")


class OpenAIImageContentSchema(BaseModel):
    type: Literal["image_url"] = "image_url"
    image_url: OpenAIImageUrlDetailSchema


class OpenAIMessageSchema(BaseModel):
    """A message in the conversation. Supports system, user, assistant, and tool roles."""

    role: str = Field(
        ...,
        description="Message role: 'system', 'user', 'assistant', or 'tool'",
    )
    content: Optional[Union[str, List[Dict[str, Any]]]] = Field(
        None,
        description="Message content. String for text, list for multi-modal content blocks",
    )
    tool_calls: Optional[List[OpenAIToolCallSchema]] = Field(
        None, description="Tool calls made by the assistant"
    )
    tool_call_id: Optional[str] = Field(
        None, description="ID of the tool call this message responds to (role=tool)"
    )
    thinking_blocks: Optional[List[OpenAIThinkingBlockSchema]] = Field(
        None, description="Thinking/reasoning blocks from extended thinking"
    )
    reasoning_content: Optional[str] = Field(
        None, description="[Bailian compat] Reasoning content string"
    )

    model_config = {"extra": "allow"}


class OpenAIFunctionDefSchema(BaseModel):
    name: str = Field(..., description="Function name (max 64 chars)")
    description: Optional[str] = Field(None, description="Function description")
    parameters: Optional[Dict[str, Any]] = Field(
        None, description="JSON Schema for function parameters"
    )
    strict: Optional[bool] = Field(None, description="Enable strict schema adherence")


class OpenAIToolDefSchema(BaseModel):
    type: Literal["function"] = "function"
    function: OpenAIFunctionDefSchema


class OpenAIToolChoiceFunctionSchema(BaseModel):
    name: str


class OpenAIToolChoiceObjectSchema(BaseModel):
    type: Literal["function"] = "function"
    function: OpenAIToolChoiceFunctionSchema


class OpenAIJsonSchemaSchema(BaseModel):
    name: Optional[str] = None
    schema_: Optional[Dict[str, Any]] = Field(None, alias="schema")
    strict: Optional[bool] = None


class OpenAIResponseFormatSchema(BaseModel):
    type: str = Field(
        ..., description="Response format type: 'text', 'json_object', or 'json_schema'"
    )
    json_schema: Optional[OpenAIJsonSchemaSchema] = None
    response_schema: Optional[Dict[str, Any]] = Field(
        None, description="Alternative schema field (some SDKs use this)"
    )


class OpenAIStreamOptionsSchema(BaseModel):
    include_usage: Optional[bool] = Field(None, description="Include usage in stream chunks")


# -- OpenAI Chat Completion Request --


class OpenAIChatCompletionRequest(BaseModel):
    """
    OpenAI Chat Completion request body.

    This server converts this request to Anthropic Messages API format,
    forwards it to the configured Anthropic backend, and converts the
    response back to OpenAI format.
    """

    model: str = Field(
        ...,
        description="Model ID (e.g. 'claude-sonnet-4-20250514', 'claude-opus-4-20250514')",
        examples=["claude-sonnet-4-20250514"],
    )
    messages: List[OpenAIMessageSchema] = Field(
        ..., description="Conversation messages in chronological order"
    )
    max_tokens: Optional[int] = Field(
        None,
        description="Maximum tokens to generate. Defaults to 4096 if not specified",
    )
    max_completion_tokens: Optional[int] = Field(
        None, description="Alternative to max_tokens (newer OpenAI SDK name)"
    )
    temperature: Optional[float] = Field(None, ge=0, le=2, description="Sampling temperature")
    top_p: Optional[float] = Field(None, ge=0, le=1, description="Nucleus sampling threshold")
    top_k: Optional[int] = Field(None, description="Top-K sampling (Anthropic-specific)")
    stop: Optional[Union[str, List[str]]] = Field(None, description="Stop sequences")
    stream: Optional[bool] = Field(False, description="Enable streaming response")
    tools: Optional[List[OpenAIToolDefSchema]] = Field(None, description="Tool definitions")
    tool_choice: Optional[Union[str, OpenAIToolChoiceObjectSchema]] = Field(
        None,
        description="Tool choice: 'auto', 'required', 'none', or specific function",
    )
    parallel_tool_calls: Optional[bool] = Field(
        None,
        description="Allow parallel tool calls. Maps to Anthropic disable_parallel_tool_use",
    )
    response_format: Optional[OpenAIResponseFormatSchema] = Field(
        None,
        description="Response format. json_schema uses native output_format on Claude 4.5+",
    )
    reasoning_effort: Optional[str] = Field(
        None,
        description="Reasoning effort: 'minimal', 'low', 'medium', 'high'. "
        "Maps to Anthropic thinking.budget_tokens",
    )
    thinking: Optional[Dict[str, Any]] = Field(
        None,
        description="Direct Anthropic thinking parameter passthrough. "
        "Example: {type: 'enabled', budget_tokens: 10000}",
    )
    user: Optional[str] = Field(None, description="End-user ID. Maps to Anthropic metadata.user_id")
    # [Bailian compat]
    enable_thinking: Optional[bool] = Field(
        None,
        description="[Bailian/DashScope] Enable thinking mode. Maps to Anthropic thinking param",
    )
    thinking_budget: Optional[int] = Field(
        None,
        description="[Bailian/DashScope] Token budget for thinking. Default 10000",
    )
    enable_search: Optional[bool] = Field(
        None,
        description="[Bailian/DashScope] Enable web search. Maps to Anthropic web_search tool",
    )
    stream_options: Optional[OpenAIStreamOptionsSchema] = Field(
        None, description="Stream options (dropped — Anthropic handles usage differently)"
    )

    model_config = {"extra": "allow"}


# -- OpenAI Chat Completion Response --


class OpenAIPromptTokensDetailsSchema(BaseModel):
    cached_tokens: Optional[int] = 0
    cache_creation_tokens: Optional[int] = 0


class OpenAICompletionTokensDetailsSchema(BaseModel):
    reasoning_tokens: Optional[int] = 0
    text_tokens: Optional[int] = 0


class OpenAIUsageSchema(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    prompt_tokens_details: Optional[OpenAIPromptTokensDetailsSchema] = None
    completion_tokens_details: Optional[OpenAICompletionTokensDetailsSchema] = None


class OpenAIResponseMessageSchema(BaseModel):
    role: str = "assistant"
    content: Optional[str] = None
    tool_calls: Optional[List[OpenAIToolCallSchema]] = None
    thinking_blocks: Optional[List[OpenAIThinkingBlockSchema]] = None
    reasoning_content: Optional[str] = None

    model_config = {"extra": "allow"}


class OpenAIChoiceSchema(BaseModel):
    index: int = 0
    message: OpenAIResponseMessageSchema
    finish_reason: Optional[str] = Field(
        None,
        description="'stop', 'length', 'tool_calls', or 'content_filter'",
    )


class OpenAIChatCompletionResponse(BaseModel):
    """OpenAI Chat Completion response."""

    id: str = Field(..., description="Response ID")
    object: str = "chat.completion"
    created: int = Field(..., description="Unix timestamp")
    model: str
    choices: List[OpenAIChoiceSchema]
    usage: OpenAIUsageSchema

    model_config = {"extra": "allow"}


# -- OpenAI Models Response --


class OpenAIModelSchema(BaseModel):
    id: str
    object: str = "model"
    owned_by: str = "anthropic"


class OpenAIModelsResponse(BaseModel):
    """Response for /v1/models endpoint."""

    object: str = "list"
    data: List[OpenAIModelSchema]


# ═══════════════════════════════════════════════════════════════════════════
# Anthropic Messages API Models (for anthropic_server)
# ═══════════════════════════════════════════════════════════════════════════


class AnthropicTextBlockSchema(BaseModel):
    type: Literal["text"] = "text"
    text: str
    cache_control: Optional[Dict[str, Any]] = None
    citations: Optional[List[Any]] = None


class AnthropicImageSourceSchema(BaseModel):
    type: str = Field(..., description="'base64' or 'url'")
    media_type: Optional[str] = Field(None, description="MIME type (e.g. 'image/jpeg')")
    data: Optional[str] = Field(None, description="Base64 data (when type=base64)")
    url: Optional[str] = Field(None, description="URL (when type=url)")


class AnthropicImageBlockSchema(BaseModel):
    type: Literal["image"] = "image"
    source: AnthropicImageSourceSchema
    cache_control: Optional[Dict[str, Any]] = None


class AnthropicToolUseBlockSchema(BaseModel):
    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: Dict[str, Any] = {}
    cache_control: Optional[Dict[str, Any]] = None


class AnthropicToolResultBlockSchema(BaseModel):
    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: Optional[Union[str, List[Dict[str, Any]]]] = None
    is_error: Optional[bool] = None
    cache_control: Optional[Dict[str, Any]] = None


class AnthropicThinkingBlockSchema(BaseModel):
    type: Literal["thinking"] = "thinking"
    thinking: str = ""
    signature: Optional[str] = None
    cache_control: Optional[Dict[str, Any]] = None


class AnthropicRedactedThinkingBlockSchema(BaseModel):
    type: Literal["redacted_thinking"] = "redacted_thinking"
    data: str = ""


class AnthropicSystemContentBlockSchema(BaseModel):
    type: str = "text"
    text: str
    cache_control: Optional[Dict[str, Any]] = None


class AnthropicMessageSchema(BaseModel):
    """A message in the conversation. Anthropic requires strict user/assistant alternation."""

    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: Union[str, List[Dict[str, Any]]] = Field(
        ...,
        description="String for simple text, or list of content blocks "
        "(text, image, tool_use, tool_result, thinking)",
    )

    model_config = {"extra": "allow"}


class AnthropicInputSchemaSchema(BaseModel):
    type: str = "object"
    properties: Optional[Dict[str, Any]] = None
    required: Optional[List[str]] = None
    additionalProperties: Optional[bool] = None

    model_config = {"extra": "allow"}


class AnthropicToolDefSchema(BaseModel):
    name: Optional[str] = Field(None, description="Tool name (no length limit in Anthropic)")
    description: Optional[str] = None
    input_schema: Optional[AnthropicInputSchemaSchema] = None
    type: Optional[str] = Field(
        None,
        description="Tool type. Omit or 'custom' for regular tools. "
        "Server tools: 'web_search_20250305', 'text_editor_20250124', etc.",
    )
    cache_control: Optional[Dict[str, Any]] = None

    model_config = {"extra": "allow"}


class AnthropicToolChoiceSchema(BaseModel):
    type: str = Field(..., description="Tool choice type: 'auto', 'any', 'tool', or 'none'")
    name: Optional[str] = Field(None, description="Specific tool name (when type='tool')")
    disable_parallel_tool_use: Optional[bool] = None


class AnthropicThinkingParamSchema(BaseModel):
    type: str = Field(..., description="'enabled', 'disabled', or 'adaptive'")
    budget_tokens: Optional[int] = Field(None, description="Max tokens for thinking/reasoning")


class AnthropicMetadataSchema(BaseModel):
    user_id: Optional[str] = None

    model_config = {"extra": "allow"}


class AnthropicOutputFormatSchema(BaseModel):
    type: str = Field("json_schema", description="Currently only 'json_schema' supported")
    schema_: Optional[Dict[str, Any]] = Field(None, alias="schema")


# -- Anthropic Messages Request --


class AnthropicMessagesRequest(BaseModel):
    """
    Anthropic Messages API request body.

    This server converts this request to OpenAI ChatCompletion format,
    forwards it to the configured OpenAI-compatible backend, and converts
    the response back to Anthropic format.
    """

    model: str = Field(
        ...,
        description="Model ID (e.g. 'gpt-4o', 'gpt-4o-mini')",
        examples=["gpt-4o"],
    )
    messages: List[AnthropicMessageSchema] = Field(
        ...,
        description="Conversation messages. Must alternate user/assistant. "
        "First message must be user role",
    )
    max_tokens: int = Field(
        ...,
        description="Maximum tokens to generate (required in Anthropic API)",
    )
    system: Optional[Union[str, List[AnthropicSystemContentBlockSchema]]] = Field(
        None,
        description="System prompt. String or list of text blocks with optional cache_control",
    )
    tools: Optional[List[AnthropicToolDefSchema]] = Field(
        None,
        description="Tool definitions. Names > 64 chars are truncated for OpenAI backend",
    )
    tool_choice: Optional[AnthropicToolChoiceSchema] = Field(
        None,
        description="Tool choice constraint",
    )
    stop_sequences: Optional[List[str]] = Field(None, description="Custom stop sequences")
    temperature: Optional[float] = Field(None, ge=0, le=1, description="Sampling temperature")
    top_p: Optional[float] = Field(None, ge=0, le=1, description="Nucleus sampling")
    top_k: Optional[int] = Field(
        None,
        description="Top-K sampling (dropped — no OpenAI equivalent)",
    )
    stream: Optional[bool] = Field(False, description="Enable SSE streaming response")
    metadata: Optional[AnthropicMetadataSchema] = Field(
        None,
        description="Request metadata. user_id maps to OpenAI 'user' param",
    )
    thinking: Optional[AnthropicThinkingParamSchema] = Field(
        None,
        description="Extended thinking config. Maps to OpenAI reasoning_effort",
    )
    output_format: Optional[AnthropicOutputFormatSchema] = Field(
        None,
        description="Structured output format. Maps to OpenAI response_format",
    )
    context_management: Optional[Dict[str, Any]] = Field(
        None,
        description="Context management config (dropped — no OpenAI equivalent)",
    )
    cache_control: Optional[Dict[str, Any]] = Field(
        None,
        description="Prompt caching config (dropped — no OpenAI equivalent)",
    )

    model_config = {"extra": "allow"}


# -- Anthropic Messages Response --


class AnthropicUsageSchema(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: Optional[int] = None
    cache_read_input_tokens: Optional[int] = None

    model_config = {"extra": "allow"}


class AnthropicMessagesResponse(BaseModel):
    """Anthropic Messages API response."""

    id: str = Field(..., description="Message ID (msg_...)")
    type: str = "message"
    role: str = "assistant"
    model: str
    content: List[Dict[str, Any]] = Field(
        ...,
        description="Content blocks: text, tool_use, thinking, redacted_thinking",
    )
    stop_reason: Optional[str] = Field(
        None,
        description="'end_turn', 'max_tokens', 'stop_sequence', or 'tool_use'",
    )
    stop_sequence: Optional[str] = None
    usage: AnthropicUsageSchema

    model_config = {"extra": "allow"}


class AnthropicErrorDetailSchema(BaseModel):
    type: str = Field(
        ...,
        description="Error type: 'invalid_request_error', 'authentication_error', "
        "'permission_error', 'not_found_error', 'rate_limit_error', 'api_error', "
        "'overloaded_error'",
    )
    message: str


class AnthropicErrorResponse(BaseModel):
    """Anthropic error response format."""

    type: str = "error"
    error: AnthropicErrorDetailSchema


# -- Anthropic Count Tokens --


class AnthropicCountTokensRequest(BaseModel):
    """Token counting request. Same shape as Messages request."""

    model: str = Field(..., description="Model ID")
    messages: List[AnthropicMessageSchema] = Field(..., description="Messages to count")
    system: Optional[Union[str, List[AnthropicSystemContentBlockSchema]]] = None
    tools: Optional[List[AnthropicToolDefSchema]] = None
    thinking: Optional[AnthropicThinkingParamSchema] = None

    model_config = {"extra": "allow"}


class AnthropicCountTokensResponse(BaseModel):
    """Token counting response."""

    input_tokens: int = Field(..., description="Estimated input token count")


# ═══════════════════════════════════════════════════════════════════════════
# Shared
# ═══════════════════════════════════════════════════════════════════════════


class HealthResponse(BaseModel):
    status: str = "ok"
