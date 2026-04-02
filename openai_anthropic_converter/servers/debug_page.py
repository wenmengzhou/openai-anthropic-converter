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
        "model": "claude-sonnet-4-20250514",
        "messages": [
            {"role": "user", "content": "Hello! What's 2+2?"}
        ],
        "max_tokens": 256,
    },
    "System + Multi-turn": {
        "model": "claude-sonnet-4-20250514",
        "messages": [
            {"role": "system", "content": "You are a helpful math tutor. Be concise."},
            {"role": "user", "content": "What is the derivative of x^2?"},
            {"role": "assistant", "content": "The derivative of x² is 2x."},
            {"role": "user", "content": "And the integral?"},
        ],
        "max_tokens": 256,
    },
    "Tool Use": {
        "model": "claude-sonnet-4-20250514",
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
        "model": "claude-sonnet-4-20250514",
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
        "model": "claude-sonnet-4-20250514",
        "messages": [
            {"role": "user", "content": "Write a haiku about programming."}
        ],
        "max_tokens": 256,
        "stream": True,
    },
    "Extended Thinking": {
        "model": "claude-sonnet-4-20250514",
        "messages": [
            {"role": "user", "content": "What is 127 * 389? Think step by step."}
        ],
        "max_tokens": 4096,
        "reasoning_effort": "high",
    },
    "[Bailian] Thinking + Search": {
        "model": "claude-sonnet-4-20250514",
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
        "model": "gpt-4o",
        "messages": [
            {"role": "user", "content": "Hello! What's 2+2?"}
        ],
        "max_tokens": 256,
    },
    "System Prompt": {
        "model": "gpt-4o",
        "system": "You are a helpful math tutor. Be concise.",
        "messages": [
            {"role": "user", "content": "What is the derivative of x^2?"}
        ],
        "max_tokens": 256,
    },
    "Multi-modal (Image URL)": {
        "model": "gpt-4o",
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
        "model": "gpt-4o",
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
        "model": "gpt-4o",
        "messages": [
            {"role": "user", "content": "What is 127 * 389? Think step by step."}
        ],
        "max_tokens": 4096,
        "thinking": {"type": "enabled", "budget_tokens": 10000},
    },
    "Streaming": {
        "model": "gpt-4o",
        "messages": [
            {"role": "user", "content": "Write a haiku about programming."}
        ],
        "max_tokens": 256,
        "stream": True,
    },
    "Count Tokens": {
        "model": "gpt-4o",
        "messages": [
            {"role": "user", "content": "Hello, how are you?"}
        ],
    },
}


def get_debug_html(server_type: str) -> str:
    """
    Generate the debug playground HTML page.

    Args:
        server_type: "openai" or "anthropic"
    """
    import json

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
  .json-key {{ color: #79c0ff; }}
  .json-str {{ color: #a5d6ff; }}
  .json-num {{ color: #d2a8ff; }}
  .json-bool {{ color: #ff7b72; }}
  .json-null {{ color: var(--muted); }}
  .stream-chunk {{ border-bottom: 1px dashed var(--border); padding-bottom: 4px; margin-bottom: 4px; }}
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
        <button onclick="copyResponse()">Copy</button>
        <button onclick="clearResponse()">Clear</button>
        <label style="font-size:0.8125rem; color:var(--muted); margin-left:auto;">
          <input type="checkbox" id="prettyPrint" checked onchange="reformatResponse()"> Pretty
        </label>
      </div>
      <div class="response-area" id="responseArea"></div>
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
let abortController = null;
let rawResponse = '';

// Initialize
(function init() {{
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
    if (exSel.value && EXAMPLES[exSel.value]) {{
      document.getElementById('requestBody').value = JSON.stringify(EXAMPLES[exSel.value], null, 2);
    }}
  }});
  // Load first example
  const firstKey = Object.keys(EXAMPLES)[0];
  if (firstKey) {{
    document.getElementById('requestBody').value = JSON.stringify(EXAMPLES[firstKey], null, 2);
    exSel.value = firstKey;
  }}
}})();

function getEndpoint() {{
  return JSON.parse(document.getElementById('endpoint').value);
}}

async function sendRequest() {{
  const ep = getEndpoint();
  const area = document.getElementById('responseArea');
  const statusEl = document.getElementById('statusText');
  const timingEl = document.getElementById('timingText');
  const tokenEl = document.getElementById('tokenText');
  area.innerHTML = '';
  rawResponse = '';
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
      // Streaming
      statusEl.textContent = resp.status + ' Streaming...';
      statusEl.className = 'status pending';
      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

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
            area.innerHTML += '<div class="stream-chunk" style="color:var(--green)">— stream done —</div>';
            continue;
          }}
          if (trimmed.startsWith('data: ')) {{
            const json = trimmed.slice(6);
            rawResponse += json + '\\n';
            area.innerHTML += '<div class="stream-chunk">' + syntaxHighlight(json) + '</div>';
          }} else if (trimmed.startsWith('event: ')) {{
            area.innerHTML += '<div style="color:var(--muted)">event: ' + trimmed.slice(7) + '</div>';
          }}
          area.scrollTop = area.scrollHeight;
        }}
      }}
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

      if (document.getElementById('prettyPrint').checked) {{
        try {{
          area.innerHTML = syntaxHighlight(JSON.stringify(JSON.parse(text), null, 2));
        }} catch {{
          area.textContent = text;
        }}
      }} else {{
        area.textContent = text;
      }}

      // Extract token info
      try {{
        const data = JSON.parse(text);
        const usage = data.usage;
        if (usage) {{
          const parts = [];
          if (usage.prompt_tokens !== undefined) parts.push('in:' + usage.prompt_tokens);
          if (usage.completion_tokens !== undefined) parts.push('out:' + usage.completion_tokens);
          if (usage.input_tokens !== undefined) parts.push('in:' + usage.input_tokens);
          if (usage.output_tokens !== undefined) parts.push('out:' + usage.output_tokens);
          tokenEl.textContent = 'Tokens: ' + parts.join(' ');
        }}
      }} catch {{}}
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

function copyResponse() {{
  navigator.clipboard.writeText(rawResponse || document.getElementById('responseArea').textContent);
}}

function clearResponse() {{
  document.getElementById('responseArea').innerHTML = '';
  rawResponse = '';
  document.getElementById('statusText').textContent = 'Ready';
  document.getElementById('statusText').className = 'status';
  document.getElementById('timingText').textContent = '';
  document.getElementById('tokenText').textContent = '';
}}

function reformatResponse() {{
  if (!rawResponse) return;
  const area = document.getElementById('responseArea');
  if (document.getElementById('prettyPrint').checked) {{
    try {{
      area.innerHTML = syntaxHighlight(JSON.stringify(JSON.parse(rawResponse), null, 2));
    }} catch {{
      area.textContent = rawResponse;
    }}
  }} else {{
    area.textContent = rawResponse;
  }}
}}

function syntaxHighlight(json) {{
  if (typeof json !== 'string') json = JSON.stringify(json, null, 2);
  return json
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"([^"\\\\]*(?:\\\\.[^"\\\\]*)*)"\\s*:/g, '<span class="json-key">"$1"</span>:')
    .replace(/"([^"\\\\]*(?:\\\\.[^"\\\\]*)*)"/g, '<span class="json-str">"$1"</span>')
    .replace(/\\b(true|false)\\b/g, '<span class="json-bool">$1</span>')
    .replace(/\\bnull\\b/g, '<span class="json-null">null</span>')
    .replace(/\\b(-?\\d+\\.?\\d*(?:[eE][+-]?\\d+)?)\\b/g, '<span class="json-num">$1</span>');
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
