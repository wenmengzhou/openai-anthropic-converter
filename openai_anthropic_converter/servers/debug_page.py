"""
Debug playground HTML page for interactive API testing.

Serves a self-contained HTML page at /debug with:
- Endpoint selector
- JSON request editor with example templates
- Send button with streaming support
- Response display with timing info
"""

OPENAI_EXAMPLES = {
    "Basic Chat": {
        "model": "__MODEL__",
        "messages": [
            {"role": "user", "content": "Hello! What's 2+2?"}
        ],
        "max_tokens": 256,
        "stream": True,
    },
    "System + Multi-turn": {
        "model": "__MODEL__",
        "messages": [
            {"role": "system", "content": "You are a helpful math tutor. Be concise."},
            {"role": "user", "content": "What is the derivative of x^2?"},
            {"role": "assistant", "content": "The derivative of x² is 2x."},
            {"role": "user", "content": "And the integral?"},
        ],
        "max_tokens": 256,
    },
    "Tool Use": {
        "model": "__MODEL__",
        "messages": [
            {"role": "user", "content": "What's the weather in Tokyo?"}
        ],
        "max_tokens": 1024,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather for a city",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string", "description": "City name"},
                            "unit": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                                "description": "Temperature unit",
                            },
                        },
                        "required": ["city"],
                    },
                },
            }
        ],
    },
    "JSON Schema Output": {
        "model": "__MODEL__",
        "messages": [
            {"role": "user", "content": "List 3 programming languages with their year of creation"}
        ],
        "max_tokens": 1024,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "languages",
                "schema": {
                    "type": "object",
                    "properties": {
                        "languages": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "year": {"type": "integer"},
                                },
                                "required": ["name", "year"],
                            },
                        }
                    },
                    "required": ["languages"],
                },
            },
        },
    },
    "Streaming": {
        "model": "__MODEL__",
        "messages": [
            {"role": "user", "content": "Write a haiku about programming."}
        ],
        "max_tokens": 256,
        "stream": True,
    },
    "Extended Thinking": {
        "model": "__MODEL__",
        "messages": [
            {"role": "user", "content": "What is 127 * 389? Think step by step."}
        ],
        "max_tokens": 4096,
        "reasoning_effort": "high",
    },
    "[Bailian] Thinking + Search": {
        "model": "__MODEL__",
        "messages": [
            {"role": "user", "content": "What happened in tech news today?"}
        ],
        "max_tokens": 2048,
        "enable_thinking": True,
        "thinking_budget": 5000,
        "enable_search": True,
    },
}

# Note: model names are just examples. The server passes the model name through
# to the backend as-is. Use whatever model name your backend supports, e.g.:
#   - Anthropic: claude-sonnet-4-20250514, claude-opus-4-20250514
#   - Bailian/DashScope Anthropic-native: claude-sonnet-4-20250514
#   - OpenAI: gpt-4o, gpt-4o-mini
#   - Bailian/DashScope OpenAI-compat: qwen-plus, qwen-max, qwen-turbo

ANTHROPIC_EXAMPLES = {
    "Basic Message": {
        "model": "__MODEL__",
        "messages": [
            {"role": "user", "content": "Hello! What's 2+2?"}
        ],
        "max_tokens": 256,
        "stream": True,
    },
    "System Prompt": {
        "model": "__MODEL__",
        "system": "You are a helpful math tutor. Be concise.",
        "messages": [
            {"role": "user", "content": "What is the derivative of x^2?"}
        ],
        "max_tokens": 256,
    },
    "Multi-modal (Image URL)": {
        "model": "__MODEL__",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What do you see in this image?"},
                    {
                        "type": "image",
                        "source": {
                            "type": "url",
                            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg",
                        },
                    },
                ],
            }
        ],
        "max_tokens": 1024,
    },
    "Tool Use": {
        "model": "__MODEL__",
        "messages": [
            {"role": "user", "content": "What's the weather in Tokyo?"}
        ],
        "max_tokens": 1024,
        "tools": [
            {
                "name": "get_weather",
                "description": "Get current weather for a city",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "City name"},
                    },
                    "required": ["city"],
                },
            }
        ],
    },
    "Extended Thinking": {
        "model": "__MODEL__",
        "messages": [
            {"role": "user", "content": "What is 127 * 389? Think step by step."}
        ],
        "max_tokens": 4096,
        "thinking": {"type": "enabled", "budget_tokens": 10000},
    },
    "Streaming": {
        "model": "__MODEL__",
        "messages": [
            {"role": "user", "content": "Write a haiku about programming."}
        ],
        "max_tokens": 256,
        "stream": True,
    },
    "Count Tokens": {
        "model": "__MODEL__",
        "messages": [
            {"role": "user", "content": "Hello, how are you?"}
        ],
    },
}


def get_debug_html(server_type: str, models: list[str] | None = None) -> str:
    """
    Generate the debug playground HTML page.

    Args:
        server_type: "openai" or "anthropic"
        models: list of available model names for the model selector
    """
    import json

    if not models:
        models = ["claude-sonnet-4-20250514"] if server_type == "openai" else ["gpt-4o"]

    if server_type == "openai":
        title = "OpenAI-Compatible Server Debug Playground"
        examples = OPENAI_EXAMPLES
        endpoints_js = json.dumps([
            {"path": "/v1/chat/completions", "method": "POST", "label": "Chat Completions"},
            {"path": "/v1/models", "method": "GET", "label": "List Models"},
            {"path": "/health", "method": "GET", "label": "Health Check"},
        ])
        default_endpoint = "/v1/chat/completions"
    else:
        title = "Anthropic-Compatible Server Debug Playground"
        examples = ANTHROPIC_EXAMPLES
        endpoints_js = json.dumps([
            {"path": "/v1/messages", "method": "POST", "label": "Messages"},
            {"path": "/v1/messages/count_tokens", "method": "POST", "label": "Count Tokens"},
            {"path": "/health", "method": "GET", "label": "Health Check"},
        ])
        default_endpoint = "/v1/messages"

    examples_js = json.dumps(examples, indent=2)
    models_js = json.dumps(models)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
  :root {{
    --bg: #0d1117; --surface: #161b22; --border: #30363d;
    --text: #e6edf3; --muted: #8b949e; --accent: #58a6ff;
    --green: #3fb950; --red: #f85149; --orange: #d29922;
  }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
    background: var(--bg); color: var(--text); line-height: 1.5;
  }}
  .container {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}
  h1 {{ font-size: 1.5rem; margin-bottom: 4px; }}
  .subtitle {{ color: var(--muted); font-size: 0.875rem; margin-bottom: 20px; }}
  .subtitle a {{ color: var(--accent); text-decoration: none; }}
  .subtitle a:hover {{ text-decoration: underline; }}
  .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
  @media (max-width: 800px) {{ .grid {{ grid-template-columns: 1fr; }} }}
  .panel {{
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 8px; overflow: hidden;
  }}
  .panel-header {{
    padding: 10px 16px; border-bottom: 1px solid var(--border);
    display: flex; align-items: center; gap: 10px; flex-wrap: wrap;
  }}
  .panel-header h2 {{ font-size: 0.875rem; font-weight: 600; }}
  select, button {{
    background: var(--bg); color: var(--text); border: 1px solid var(--border);
    border-radius: 6px; padding: 4px 10px; font-size: 0.8125rem; cursor: pointer;
  }}
  select:focus, button:focus {{ outline: 2px solid var(--accent); outline-offset: -1px; }}
  button:hover {{ border-color: var(--accent); }}
  button.primary {{
    background: var(--accent); color: #fff; border-color: var(--accent); font-weight: 600;
  }}
  button.primary:hover {{ opacity: 0.9; }}
  button.danger {{ color: var(--red); border-color: var(--red); }}
  textarea {{
    width: 100%; min-height: 400px; background: var(--bg); color: var(--text);
    border: none; padding: 16px; font-family: 'SF Mono', 'Fira Code', monospace;
    font-size: 0.8125rem; resize: vertical; outline: none; tab-size: 2;
  }}
  .response-area {{
    padding: 16px; font-family: 'SF Mono', 'Fira Code', monospace;
    font-size: 0.8125rem; min-height: 400px; overflow-y: auto;
    white-space: pre-wrap; word-break: break-word;
  }}
  .status-bar {{
    padding: 6px 16px; border-top: 1px solid var(--border);
    font-size: 0.75rem; color: var(--muted); display: flex; gap: 16px;
  }}
  .status-bar .status {{ font-weight: 600; }}
  .status-bar .ok {{ color: var(--green); }}
  .status-bar .err {{ color: var(--red); }}
  .status-bar .pending {{ color: var(--orange); }}
  .jt {{ font-family: 'SF Mono', 'Fira Code', monospace; font-size: 0.8125rem; line-height: 1.5; }}
  .jt .k {{ color: #79c0ff; }}
  .jt .s {{ color: #a5d6ff; }}
  .jt .n {{ color: #d2a8ff; }}
  .jt .b {{ color: #ff7b72; }}
  .jt .nl {{ color: var(--muted); }}
  .jt .tog {{
    display: inline-block; width: 1em; text-align: center; cursor: pointer;
    color: var(--muted); user-select: none; font-size: 0.75rem;
  }}
  .jt .tog:hover {{ color: var(--accent); }}
  .jt .collapsed > .inner {{ display: none; }}
  .jt .collapsed > .close-bracket {{ display: none; }}
  .jt .collapsed > .ellipsis {{ display: inline; }}
  .jt .ellipsis {{ display: none; color: var(--muted); }}
  .tab {{ border-radius: 4px 4px 0 0; padding: 3px 12px; font-size: 0.75rem; border-bottom: 2px solid transparent; }}
  .tab.active {{ color: var(--accent); border-bottom-color: var(--accent); background: var(--bg); }}
  .stream-chunk {{ border-bottom: 1px dashed var(--border); padding-bottom: 4px; margin-bottom: 4px; }}
  .stream-text {{ white-space: pre-wrap; line-height: 1.6; font-size: 0.875rem; }}
  .stream-text .thinking-block {{
    background: #1c2333; border-left: 3px solid var(--orange); padding: 8px 12px;
    margin: 8px 0; border-radius: 0 6px 6px 0; color: var(--muted); font-size: 0.8125rem;
  }}
  .stream-text .thinking-label {{ color: var(--orange); font-weight: 600; font-size: 0.75rem; margin-bottom: 4px; }}
  .stream-text .tool-block {{
    background: #1c2333; border-left: 3px solid var(--accent); padding: 8px 12px;
    margin: 8px 0; border-radius: 0 6px 6px 0; font-size: 0.8125rem;
  }}
  .stream-text .tool-label {{ color: var(--accent); font-weight: 600; font-size: 0.75rem; margin-bottom: 4px; }}
  .stream-cursor {{ display: inline-block; width: 2px; height: 1em; background: var(--accent); animation: blink 0.8s infinite; vertical-align: text-bottom; margin-left: 1px; }}
  @keyframes blink {{ 0%,50% {{ opacity: 1; }} 51%,100% {{ opacity: 0; }} }}
</style>
</head>
<body>
<div class="container">
  <h1>{title}</h1>
  <p class="subtitle">
    <a href="/docs">Swagger UI</a> &middot;
    <a href="/redoc">ReDoc</a> &middot;
    <a href="/openapi.json">OpenAPI JSON</a>
  </p>
  <div class="grid">
    <!-- Request Panel -->
    <div class="panel">
      <div class="panel-header">
        <h2>Request</h2>
        <select id="modelSel" title="Select model">
        </select>
        <select id="endpoint">
        </select>
        <select id="examples" title="Load example">
          <option value="">— examples —</option>
        </select>
        <button class="primary" id="sendBtn" onclick="sendRequest()">Send</button>
        <button class="danger" id="cancelBtn" onclick="cancelRequest()" style="display:none">Cancel</button>
      </div>
      <textarea id="requestBody" spellcheck="false" placeholder="Enter JSON request body..."></textarea>
    </div>
    <!-- Response Panel -->
    <div class="panel">
      <div class="panel-header">
        <h2>Response</h2>
        <div class="view-tabs" style="display:flex;gap:0;margin-left:8px;">
          <button class="tab active" id="tabText" onclick="switchView('text')">Text</button>
          <button class="tab" id="tabJson" onclick="switchView('json')">JSON</button>
        </div>
        <button onclick="copyResponse()" style="margin-left:auto;">Copy</button>
        <button onclick="clearResponse()">Clear</button>
      </div>
      <div class="response-area" id="responseArea"></div>
      <div class="response-area" id="responseJsonArea" style="display:none;"></div>
      <div class="status-bar">
        <span>Status: <span id="statusText" class="status">Ready</span></span>
        <span id="timingText"></span>
        <span id="tokenText"></span>
      </div>
    </div>
  </div>
</div>

<script>
const ENDPOINTS = {endpoints_js};
const EXAMPLES = {examples_js};
const MODELS = {models_js};
let abortController = null;
let rawResponse = '';
let rawTextContent = '';
let currentView = 'text';

function getSelectedModel() {{
  return document.getElementById('modelSel').value;
}}

function applyModel(jsonStr) {{
  return jsonStr.replace(/"__MODEL__"/g, JSON.stringify(getSelectedModel()));
}}

function loadExample(name) {{
  if (!name || !EXAMPLES[name]) return;
  const json = JSON.stringify(EXAMPLES[name], null, 2);
  document.getElementById('requestBody').value = applyModel(json);
}}

// Initialize
(function init() {{
  // Populate model selector
  const modelSel = document.getElementById('modelSel');
  MODELS.forEach(m => {{
    const opt = document.createElement('option');
    opt.value = m;
    opt.textContent = m;
    modelSel.appendChild(opt);
  }});
  modelSel.addEventListener('change', () => {{
    // Replace model in current request body
    try {{
      const body = JSON.parse(document.getElementById('requestBody').value);
      body.model = getSelectedModel();
      document.getElementById('requestBody').value = JSON.stringify(body, null, 2);
    }} catch {{}}
  }});

  const epSel = document.getElementById('endpoint');
  ENDPOINTS.forEach(ep => {{
    const opt = document.createElement('option');
    opt.value = JSON.stringify(ep);
    opt.textContent = ep.label + ' (' + ep.method + ' ' + ep.path + ')';
    epSel.appendChild(opt);
  }});
  const exSel = document.getElementById('examples');
  Object.keys(EXAMPLES).forEach(name => {{
    const opt = document.createElement('option');
    opt.value = name;
    opt.textContent = name;
    exSel.appendChild(opt);
  }});
  exSel.addEventListener('change', () => {{
    loadExample(exSel.value);
  }});
  // Load first example
  const firstKey = Object.keys(EXAMPLES)[0];
  if (firstKey) {{
    loadExample(firstKey);
    exSel.value = firstKey;
  }}
}})();

function getEndpoint() {{
  return JSON.parse(document.getElementById('endpoint').value);
}}

async function sendRequest() {{
  const ep = getEndpoint();
  const area = document.getElementById('responseArea');
  const jsonArea = document.getElementById('responseJsonArea');
  const statusEl = document.getElementById('statusText');
  const timingEl = document.getElementById('timingText');
  const tokenEl = document.getElementById('tokenText');
  area.innerHTML = '';
  jsonArea.innerHTML = '';
  rawResponse = '';
  rawTextContent = '';
  switchView('text');
  statusEl.textContent = 'Sending...';
  statusEl.className = 'status pending';
  timingEl.textContent = '';
  tokenEl.textContent = '';
  document.getElementById('sendBtn').style.display = 'none';
  document.getElementById('cancelBtn').style.display = '';

  abortController = new AbortController();
  const startTime = performance.now();

  try {{
    const opts = {{ method: ep.method, signal: abortController.signal, headers: {{}}}};
    if (ep.method === 'POST') {{
      opts.headers['Content-Type'] = 'application/json';
      opts.body = document.getElementById('requestBody').value;
    }}

    const resp = await fetch(ep.path, opts);
    const elapsed = ((performance.now() - startTime) / 1000).toFixed(2);
    const ct = resp.headers.get('content-type') || '';

    if (ct.includes('text/event-stream')) {{
      // Streaming — always populate both Text and JSON views
      statusEl.textContent = resp.status + ' Streaming...';
      statusEl.className = 'status pending';
      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      // Text view state
      area.innerHTML = '<div class="stream-text" id="streamTextArea"></div>';
      const textEl = document.getElementById('streamTextArea');
      const cursorEl = document.createElement('span');
      cursorEl.className = 'stream-cursor';
      textEl.appendChild(cursorEl);
      let thinkEl = null;
      let toolEls = {{}};

      while (true) {{
        const {{ done, value }} = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, {{ stream: true }});
        const lines = buffer.split('\\n');
        buffer = lines.pop() || '';
        for (const line of lines) {{
          const trimmed = line.trim();
          if (!trimmed) continue;
          if (trimmed === 'data: [DONE]') {{
            jsonArea.innerHTML += '<div class="stream-chunk" style="color:var(--green)">— stream done —</div>';
            continue;
          }}
          if (trimmed.startsWith('data: ')) {{
            const json = trimmed.slice(6);
            rawResponse += json + '\\n';

            // JSON view: append formatted chunk
            let display = json;
            try {{ display = JSON.stringify(JSON.parse(json), null, 2); }} catch {{}}
            jsonArea.innerHTML += '<div class="stream-chunk">' + renderJson(display) + '</div>';
            jsonArea.scrollTop = jsonArea.scrollHeight;

            // Text view: extract and append content with typewriter effect
            try {{
              const chunk = JSON.parse(json);
              // OpenAI format: choices[0].delta
              const delta = chunk.choices && chunk.choices[0] && chunk.choices[0].delta;
              if (delta) {{
                if (delta.content) {{
                  cursorEl.remove();
                  textEl.appendChild(document.createTextNode(delta.content));
                  textEl.appendChild(cursorEl);
                }}
                if (delta.reasoning_content) {{
                  if (!thinkEl) {{
                    cursorEl.remove();
                    thinkEl = document.createElement('div');
                    thinkEl.className = 'thinking-block';
                    thinkEl.innerHTML = '<div class="thinking-label">Thinking</div>';
                    thinkEl.appendChild(document.createElement('span'));
                    textEl.appendChild(thinkEl);
                    textEl.appendChild(cursorEl);
                  }}
                  thinkEl.querySelector('span').textContent += delta.reasoning_content;
                }}
                if (delta.thinking_blocks) {{
                  delta.thinking_blocks.forEach(tb => {{
                    if (tb.thinking) {{
                      if (!thinkEl) {{
                        cursorEl.remove();
                        thinkEl = document.createElement('div');
                        thinkEl.className = 'thinking-block';
                        thinkEl.innerHTML = '<div class="thinking-label">Thinking</div>';
                        thinkEl.appendChild(document.createElement('span'));
                        textEl.appendChild(thinkEl);
                        textEl.appendChild(cursorEl);
                      }}
                      thinkEl.querySelector('span').textContent += tb.thinking;
                    }}
                  }});
                }}
                if (delta.tool_calls) {{
                  delta.tool_calls.forEach(tc => {{
                    const idx = tc.index || 0;
                    if (!toolEls[idx]) {{
                      thinkEl = null;
                      cursorEl.remove();
                      const el = document.createElement('div');
                      el.className = 'tool-block';
                      const name = (tc.function && tc.function.name) || 'tool_call';
                      el.innerHTML = '<div class="tool-label">Tool: ' + name + '</div>';
                      el.appendChild(document.createElement('span'));
                      textEl.appendChild(el);
                      textEl.appendChild(cursorEl);
                      toolEls[idx] = el;
                    }}
                    if (tc.function && tc.function.arguments) {{
                      toolEls[idx].querySelector('span').textContent += tc.function.arguments;
                    }}
                  }});
                }}
                if (chunk.choices[0].finish_reason) {{
                  thinkEl = null; toolEls = {{}};
                }}
              }}
              // Anthropic format
              if (chunk.type === 'content_block_delta' && chunk.delta) {{
                cursorEl.remove();
                if (chunk.delta.type === 'text_delta' && chunk.delta.text) {{
                  textEl.appendChild(document.createTextNode(chunk.delta.text));
                }} else if (chunk.delta.type === 'thinking_delta' && chunk.delta.thinking) {{
                  if (!thinkEl) {{
                    thinkEl = document.createElement('div');
                    thinkEl.className = 'thinking-block';
                    thinkEl.innerHTML = '<div class="thinking-label">Thinking</div>';
                    thinkEl.appendChild(document.createElement('span'));
                    textEl.appendChild(thinkEl);
                  }}
                  thinkEl.querySelector('span').textContent += chunk.delta.thinking;
                }} else if (chunk.delta.type === 'input_json_delta' && chunk.delta.partial_json) {{
                  const lastTool = textEl.querySelector('.tool-block:last-of-type span');
                  if (lastTool) lastTool.textContent += chunk.delta.partial_json;
                }}
                textEl.appendChild(cursorEl);
              }}
              if (chunk.type === 'content_block_start' && chunk.content_block) {{
                if (chunk.content_block.type === 'tool_use') {{
                  cursorEl.remove();
                  thinkEl = null;
                  const el = document.createElement('div');
                  el.className = 'tool-block';
                  el.innerHTML = '<div class="tool-label">Tool: ' + (chunk.content_block.name || 'tool_call') + '</div>';
                  el.appendChild(document.createElement('span'));
                  textEl.appendChild(el);
                  textEl.appendChild(cursorEl);
                }}
              }}
              // Extract usage
              const usage = chunk.usage || (chunk.choices && chunk.choices[0] && chunk.choices[0].usage);
              if (usage) {{
                const parts = [];
                if (usage.prompt_tokens !== undefined) parts.push('in:' + usage.prompt_tokens);
                if (usage.completion_tokens !== undefined) parts.push('out:' + usage.completion_tokens);
                if (usage.input_tokens !== undefined) parts.push('in:' + usage.input_tokens);
                if (usage.output_tokens !== undefined) parts.push('out:' + usage.output_tokens);
                if (parts.length) tokenEl.textContent = 'Tokens: ' + parts.join(' ');
              }}
            }} catch {{}}
          }} else if (trimmed.startsWith('event: ')) {{
            jsonArea.innerHTML += '<div style="color:var(--muted)">event: ' + trimmed.slice(7) + '</div>';
          }}
          area.scrollTop = area.scrollHeight;
        }}
      }}
      cursorEl.remove();
      rawTextContent = textEl.textContent;
      const totalElapsed = ((performance.now() - startTime) / 1000).toFixed(2);
      statusEl.textContent = resp.status + ' Done';
      statusEl.className = 'status ok';
      timingEl.textContent = 'Time: ' + totalElapsed + 's';
    }} else {{
      // Non-streaming
      const text = await resp.text();
      rawResponse = text;
      statusEl.textContent = resp.status + ' ' + (resp.ok ? 'OK' : 'Error');
      statusEl.className = 'status ' + (resp.ok ? 'ok' : 'err');
      timingEl.textContent = 'Time: ' + elapsed + 's';

      // JSON view
      try {{
        jsonArea.innerHTML = renderJson(text);
      }} catch {{
        jsonArea.textContent = text;
      }}

      // Text view: extract readable content
      try {{
        const data = JSON.parse(text);
        const parts = extractTextContent(data);
        if (parts.length > 0) {{
          renderTextView(parts, area);
        }} else {{
          area.innerHTML = renderJson(text);
        }}
        // Extract token info
        const usage = data.usage;
        if (usage) {{
          const tp = [];
          if (usage.prompt_tokens !== undefined) tp.push('in:' + usage.prompt_tokens);
          if (usage.completion_tokens !== undefined) tp.push('out:' + usage.completion_tokens);
          if (usage.input_tokens !== undefined) tp.push('in:' + usage.input_tokens);
          if (usage.output_tokens !== undefined) tp.push('out:' + usage.output_tokens);
          tokenEl.textContent = 'Tokens: ' + tp.join(' ');
        }}
      }} catch {{
        area.textContent = text;
      }}
    }}
  }} catch (e) {{
    if (e.name === 'AbortError') {{
      statusEl.textContent = 'Cancelled';
      statusEl.className = 'status err';
    }} else {{
      statusEl.textContent = 'Error';
      statusEl.className = 'status err';
      area.textContent = e.message;
    }}
  }} finally {{
    document.getElementById('sendBtn').style.display = '';
    document.getElementById('cancelBtn').style.display = 'none';
    abortController = null;
  }}
}}

function cancelRequest() {{
  if (abortController) abortController.abort();
}}

function switchView(view) {{
  currentView = view;
  document.getElementById('tabText').className = 'tab' + (view === 'text' ? ' active' : '');
  document.getElementById('tabJson').className = 'tab' + (view === 'json' ? ' active' : '');
  document.getElementById('responseArea').style.display = view === 'text' ? '' : 'none';
  document.getElementById('responseJsonArea').style.display = view === 'json' ? '' : 'none';
}}

function copyResponse() {{
  if (currentView === 'text') {{
    navigator.clipboard.writeText(rawTextContent || document.getElementById('responseArea').textContent);
  }} else {{
    navigator.clipboard.writeText(rawResponse || document.getElementById('responseJsonArea').textContent);
  }}
}}

function clearResponse() {{
  document.getElementById('responseArea').innerHTML = '';
  document.getElementById('responseJsonArea').innerHTML = '';
  rawResponse = '';
  rawTextContent = '';
  document.getElementById('statusText').textContent = 'Ready';
  document.getElementById('statusText').className = 'status';
  document.getElementById('timingText').textContent = '';
  document.getElementById('tokenText').textContent = '';
}}

function extractTextContent(data) {{
  // Extract readable text from OpenAI or Anthropic response
  let parts = [];
  // OpenAI format
  if (data.choices && data.choices[0] && data.choices[0].message) {{
    const msg = data.choices[0].message;
    if (msg.thinking_blocks) {{
      msg.thinking_blocks.forEach(tb => {{
        if (tb.thinking) parts.push({{ type: 'thinking', text: tb.thinking }});
      }});
    }}
    if (msg.content) parts.push({{ type: 'text', text: msg.content }});
    if (msg.tool_calls) {{
      msg.tool_calls.forEach(tc => {{
        const name = tc.function ? tc.function.name : 'tool';
        const args = tc.function ? tc.function.arguments : '';
        parts.push({{ type: 'tool', name: name, text: args }});
      }});
    }}
  }}
  // Anthropic format
  if (data.content && Array.isArray(data.content)) {{
    data.content.forEach(block => {{
      if (block.type === 'thinking' && block.thinking) {{
        parts.push({{ type: 'thinking', text: block.thinking }});
      }} else if (block.type === 'text' && block.text) {{
        parts.push({{ type: 'text', text: block.text }});
      }} else if (block.type === 'tool_use') {{
        parts.push({{ type: 'tool', name: block.name || 'tool', text: JSON.stringify(block.input || {{}}, null, 2) }});
      }}
    }});
  }}
  return parts;
}}

function renderTextView(parts, area) {{
  area.innerHTML = '';
  const container = document.createElement('div');
  container.className = 'stream-text';
  parts.forEach(p => {{
    if (p.type === 'thinking') {{
      const el = document.createElement('div');
      el.className = 'thinking-block';
      el.innerHTML = '<div class="thinking-label">Thinking</div>';
      const span = document.createElement('span');
      span.textContent = p.text;
      el.appendChild(span);
      container.appendChild(el);
    }} else if (p.type === 'tool') {{
      const el = document.createElement('div');
      el.className = 'tool-block';
      el.innerHTML = '<div class="tool-label">Tool: ' + esc(p.name) + '</div>';
      const span = document.createElement('span');
      span.textContent = p.text;
      el.appendChild(span);
      container.appendChild(el);
    }} else {{
      container.appendChild(document.createTextNode(p.text));
    }}
  }});
  area.appendChild(container);
  rawTextContent = container.textContent;
}}

function esc(s) {{ return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }}

function renderJsonToHtml(val, indent) {{
  indent = indent || 0;
  const pad = '  '.repeat(indent);
  const pad1 = '  '.repeat(indent + 1);
  if (val === null) return '<span class="nl">null</span>';
  if (typeof val === 'boolean') return '<span class="b">' + val + '</span>';
  if (typeof val === 'number') return '<span class="n">' + val + '</span>';
  if (typeof val === 'string') return '<span class="s">"' + esc(val) + '"</span>';
  if (Array.isArray(val)) {{
    if (val.length === 0) return '[]';
    const items = val.map((v, i) => {{
      const comma = i < val.length - 1 ? ',' : '';
      return pad1 + renderJsonToHtml(v, indent + 1) + comma;
    }});
    const id = 'jn' + (++renderJsonToHtml._id);
    return '<span id="' + id + '"><span class="tog" onclick="toggleJson(\\''+id+'\\')">-</span>[\\n'
      + '<span class="inner">' + items.join('\\n') + '\\n' + pad + '</span>'
      + '<span class="ellipsis"> ...' + val.length + ' items ]</span>'
      + '<span class="close-bracket">]</span></span>';
  }}
  if (typeof val === 'object') {{
    const keys = Object.keys(val);
    if (keys.length === 0) return '{{}}';
    const items = keys.map((k, i) => {{
      const comma = i < keys.length - 1 ? ',' : '';
      return pad1 + '<span class="k">"' + esc(k) + '"</span>: ' + renderJsonToHtml(val[k], indent + 1) + comma;
    }});
    const id = 'jn' + (++renderJsonToHtml._id);
    return '<span id="' + id + '"><span class="tog" onclick="toggleJson(\\''+id+'\\')">-</span>{{\\n'
      + '<span class="inner">' + items.join('\\n') + '\\n' + pad + '</span>'
      + '<span class="ellipsis"> ...' + keys.length + ' keys }}</span>'
      + '<span class="close-bracket">}}</span></span>';
  }}
  return esc(String(val));
}}
renderJsonToHtml._id = 0;

function toggleJson(id) {{
  const el = document.getElementById(id);
  if (!el) return;
  const tog = el.querySelector(':scope > .tog');
  if (el.classList.contains('collapsed')) {{
    el.classList.remove('collapsed');
    tog.textContent = '-';
  }} else {{
    el.classList.add('collapsed');
    tog.textContent = '+';
  }}
}}

function renderJson(text) {{
  try {{
    const obj = typeof text === 'string' ? JSON.parse(text) : text;
    return '<div class="jt"><pre style="margin:0;white-space:pre-wrap;">' + renderJsonToHtml(obj, 0) + '</pre></div>';
  }} catch {{
    return '<pre style="margin:0;">' + esc(typeof text === 'string' ? text : JSON.stringify(text)) + '</pre>';
  }}
}}

// Ctrl+Enter to send
document.getElementById('requestBody').addEventListener('keydown', (e) => {{
  if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {{
    e.preventDefault();
    sendRequest();
  }}
}});
</script>
</body>
</html>"""
