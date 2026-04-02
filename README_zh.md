# OpenAI ↔ Anthropic Protocol Converter

[English](README.md) | 中文

OpenAI 和 Anthropic 是当前最主流的两套 LLM API 协议。许多模型服务商只兼容其中一种，而客户端/框架往往只支持另一种。本项目提供**双向、完整**的协议转换，让任何 OpenAI 客户端可以调用 Anthropic 后端，反之亦然。

**设计哲学：** 核心库是一个纯粹的转换层 — **无状态、零依赖、dict 进 dict 出**。不绑定 HTTP 客户端，不耦合任何框架，没有全局状态。转换函数接收一个普通 dict，返回一个普通 dict。这使得它可以轻松嵌入任何 Python 项目、无需 mock 即可测试、与任意传输层自由组合。可选的代理服务器只是上层的薄封装，而非反过来。

---

## 两种使用方式

### 方式一：代码接口 — 嵌入你的应用

将转换器作为库引入，在你自己的代码中完成协议转换。适合需要**自定义请求/响应处理逻辑**的场景。

```bash
pip install -e .
```

**OpenAI 格式 → Anthropic 格式（调用 Anthropic 后端）**

```python
from openai_anthropic_converter import OpenAIToAnthropicConverter

# 转换请求
anthropic_req = OpenAIToAnthropicConverter.convert_request(openai_request)
# ... 发送到 Anthropic API ...
openai_resp = OpenAIToAnthropicConverter.convert_response(anthropic_response)

# 异步流式
async for chunk in OpenAIToAnthropicConverter.aconvert_stream(anthropic_sse_events):
    print(chunk)  # OpenAI chat.completion.chunk 格式
```

**Anthropic 格式 → OpenAI 格式（调用 OpenAI 后端）**

```python
import json
from openai_anthropic_converter import AnthropicToOpenAIConverter

# 转换请求（返回 tuple，包含工具名映射）
openai_req, tool_name_mapping = AnthropicToOpenAIConverter.convert_request(anthropic_request)
# ... 发送到 OpenAI API ...
anthropic_resp = AnthropicToOpenAIConverter.convert_response(
    openai_response, tool_name_mapping=tool_name_mapping
)

# 异步流式
async for event in AnthropicToOpenAIConverter.aconvert_stream(
    openai_chunks, tool_name_mapping=tool_name_mapping
):
    print(f"event: {event['type']}\ndata: {json.dumps(event)}\n")
```

> **注意：** `AnthropicToOpenAIConverter.convert_request()` 返回 `(request, tool_name_mapping)`。OpenAI 对工具名有 64 字符限制，转换器会自动截断并以 SHA-256 哈希后缀生成映射表，响应转换时传入此映射即可恢复原始名称。

---

### 方式二：HTTP 代理服务器 — 开箱即用

启动一个代理服务器，对外暴露一种协议的接口，自动将请求转发到另一种协议的后端。适合**无需修改客户端代码**的场景。

```bash
pip install -e ".[server]"
```

**OpenAI 兼容服务器**（将 Anthropic 后端伪装为 OpenAI API）

```bash
# 启动服务器
./start_openai_server.sh
# 或手动指定参数：
python -m openai_anthropic_converter.servers.openai_server \
    --backend-url https://api.anthropic.com/v1 \
    --backend-api-key $ANTHROPIC_API_KEY \
    --port 8001

# 用任何 OpenAI 客户端调用
curl http://localhost:8001/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"claude-sonnet-4-20250514","messages":[{"role":"user","content":"Hello!"}],"max_tokens":1024}'
```

**Anthropic 兼容服务器**（将 OpenAI 后端伪装为 Anthropic API）

```bash
# 启动服务器
./start_anthropic_server.sh
# 或手动指定参数：
python -m openai_anthropic_converter.servers.anthropic_server \
    --backend-url https://api.openai.com/v1 \
    --backend-api-key $OPENAI_API_KEY \
    --port 8002

# 用任何 Anthropic 客户端调用
curl http://localhost:8002/v1/messages \
    -H "Content-Type: application/json" \
    -H "x-api-key: anything" \
    -H "anthropic-version: 2023-06-01" \
    -d '{"model":"gpt-4o","messages":[{"role":"user","content":"Hello!"}],"max_tokens":1024}'
```

两个服务器均提供：Swagger UI (`/docs`)、ReDoc (`/redoc`)、OpenAPI JSON (`/openapi.json`)、交互式调试页面 (`/debug`)、健康检查 (`/health`)。

配置通过 `.env` 文件自动加载，也可通过环境变量或 CLI 参数覆盖。

---

## 功能覆盖

| 功能 | 支持 |
|------|------|
| 文本消息 | ✅ |
| System 消息 | ✅（inline ↔ 独立参数） |
| Tool/Function calling | ✅（定义 + 调用 + 结果） |
| 工具名 >64 字符截断与恢复 | ✅ |
| 图片（base64 + URL） | ✅ |
| Thinking/Reasoning | ✅（thinking ↔ reasoning_effort） |
| 流式响应 | ✅（同步 + 异步） |
| Usage / cache token 统计 | ✅ |
| response_format ↔ output_format | ✅ |
| stop ↔ stop_sequences | ✅ |
| context_management | ✅ |
| 消息交替合并 | ✅（Anthropic 要求严格 user/assistant 交替） |

## 设计要点

- **零外部依赖** — 核心库纯 dict 输入/输出，TypedDict 仅用于类型标注
- **流式转换基于状态机** — `AnthropicSSEToOpenAIStream` / `OpenAIToAnthropicSSEStream` 内部跟踪 content block 类型与索引
- **前向兼容** — 不识别的参数静默丢弃，不会报错

## 安装与开发

```bash
pip install -e .              # 仅核心库
pip install -e ".[server]"    # 含代理服务器
pip install -e ".[dev]"       # 含开发工具（pytest, ruff, mypy）

python -m pytest tests/ -v    # 运行测试
ruff check . && ruff format --check .    # 代码检查
mypy --ignore-missing-imports openai_anthropic_converter/
```

## 项目结构

```
openai_anthropic_converter/
├── __init__.py                    # 导出两个 Converter 类
├── constants.py                   # 常量（stop reason 映射、模型列表）
├── utils.py                       # 工具函数（名称截断、JSON schema 过滤）
├── types/                         # TypedDict 类型定义
├── openai_to_anthropic/           # 方向一：OpenAI → Anthropic → OpenAI
│   ├── converter.py               # OpenAIToAnthropicConverter 入口
│   ├── request.py                 # 请求转换
│   ├── response.py                # 响应转换
│   └── stream.py                  # 流式转换（状态机）
├── anthropic_to_openai/           # 方向二：Anthropic → OpenAI → Anthropic
│   ├── converter.py               # AnthropicToOpenAIConverter 入口
│   ├── request.py                 # 请求转换
│   ├── response.py                # 响应转换
│   └── stream.py                  # 流式转换（状态机）
└── servers/                       # HTTP 代理服务器（需 [server] 依赖）
    ├── openai_server.py           # /v1/chat/completions 端点
    ├── anthropic_server.py        # /v1/messages 端点
    ├── schemas.py                 # Pydantic 模型（OpenAPI 文档）
    └── debug_page.py              # 交互式调试页面
```
