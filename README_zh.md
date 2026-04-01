# OpenAI-Anthropic Protocol Converter

独立的双向协议转换器，实现 OpenAI ChatCompletion 和 Anthropic Messages API 之间的协议互转。支持流式和非流式请求/响应。

## 安装

零外部依赖（仅需 `typing_extensions`），直接复制 `openai_anthropic_converter/` 目录即可使用。

## 核心 API

### Converter 1: `OpenAIToAnthropicConverter`

将 OpenAI 格式请求转为 Anthropic 格式发送，将 Anthropic 响应转回 OpenAI 格式。

```python
from openai_anthropic_converter import OpenAIToAnthropicConverter

# 非流式
anthropic_req = OpenAIToAnthropicConverter.convert_request(openai_request)
# ... 发送到 Anthropic API ...
openai_resp = OpenAIToAnthropicConverter.convert_response(anthropic_response)

# 流式
for chunk in OpenAIToAnthropicConverter.convert_stream(anthropic_sse_events):
    print(chunk)  # OpenAI chat.completion.chunk 格式
```

### Converter 2: `AnthropicToOpenAIConverter`

将 Anthropic /v1/messages 请求转为 OpenAI 格式转发，将 OpenAI 响应转回 Anthropic 格式。

```python
from openai_anthropic_converter import AnthropicToOpenAIConverter

# 非流式 (注意: convert_request 返回 tuple)
openai_req, tool_name_mapping = AnthropicToOpenAIConverter.convert_request(anthropic_request)
# ... 发送到 OpenAI API ...
anthropic_resp = AnthropicToOpenAIConverter.convert_response(
    openai_response, tool_name_mapping=tool_name_mapping
)

# 流式
for event in AnthropicToOpenAIConverter.convert_stream(
    openai_chunks, tool_name_mapping=tool_name_mapping
):
    # event 是 Anthropic SSE 事件 dict (message_start, content_block_delta, etc.)
    print(f"event: {event['type']}\ndata: {json.dumps(event)}\n")
```

## 转换覆盖范围

| 特性 | 支持 |
|-----|------|
| 文本消息 | ✅ |
| System 消息 | ✅ (inline ↔ 独立参数) |
| Tool/Function calling | ✅ (定义 + 调用 + 结果) |
| 工具名 >64 字符截断与恢复 | ✅ |
| 图片 (base64 + URL) | ✅ |
| Thinking/Reasoning | ✅ (thinking ↔ reasoning_effort) |
| 流式请求 | ✅ (sync + async) |
| Usage/缓存 token 统计 | ✅ |
| response_format ↔ output_format | ✅ |
| stop ↔ stop_sequences | ✅ |
| context_management | ✅ |
| 消息交替合并 | ✅ |

## 运行测试

```bash
python -m pytest openai_anthropic_converter/tests/ -v
```
