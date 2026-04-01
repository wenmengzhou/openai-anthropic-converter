"""
openai_anthropic_converter: Standalone bidirectional converter between
OpenAI ChatCompletion and Anthropic Messages API protocols.

Two converter classes:

- OpenAIToAnthropicConverter: OpenAI request -> Anthropic request,
  Anthropic response -> OpenAI response (for calling Anthropic API)

- AnthropicToOpenAIConverter: Anthropic request -> OpenAI request,
  OpenAI response -> Anthropic response (for proxying Anthropic to OpenAI backends)

Both support streaming and non-streaming conversion.
"""

from .anthropic_to_openai import AnthropicToOpenAIConverter
from .openai_to_anthropic import OpenAIToAnthropicConverter

__all__ = [
    "OpenAIToAnthropicConverter",
    "AnthropicToOpenAIConverter",
]
