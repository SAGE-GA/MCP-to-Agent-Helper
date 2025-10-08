# MCP Tool Helper

 MCP Tool Helper bridges [FastMCP](https://github.com/mom1/fastmcp)'s tooling ecosystem with OpenAI's Agent Framework. It discovers tools exposed by an MCP server, wraps them as OpenAI `FunctionTool`s, and forwards FastMCP progress and log events so agent UIs can show real-time status updates. This fills the gap in the Agent Framework, which does not natively stream tool status back to the client.

## Features

- Automatically discovers MCP tools and exposes them as OpenAI function tools
- Streams FastMCP progress and log messages back into your agent runtime
- Provides optional caching and cache refresh of tool metadata
- Defaults to SSE transport and still supports Streamable HTTP or any transport that `fastmcp.infer_transport` understands
- Supports bearer tokens via environment variables or explicit headers
- Offers ready-made transport factory helpers for common deployment targets

## Installation

```bash
pip install fastmcp openai
```

Clone or copy `utility/mcp_tool_helper.py` into your project. (A packaged release is planned.)

## Quick Start

```python
from utility.mcp_tool_helper import MCPToolkit, UserContext

# Option A: connect to an SSE-enabled FastMCP server
toolkit = MCPToolkit(host="127.0.0.1", port=8010, sse_path="/sse/")

# Option B: launch tools via stdio (Python process)
# toolkit = MCPToolkit(
#     server_cmd=".venv/bin/python",
#     server_script="path/to/mcp_tools.py",
#     server_args=("--flag",),
# )

# Discover tools and hand them to an OpenAI agent
tools = toolkit.get_function_tools()

agent = Agent(
    name="Generate Course Content",
    instructions="You are a course builder...",
    tools=tools,
    model="gpt-5",
)

context = UserContext(
    emit_progress=lambda tool, progress, total, msg: print(f"[{tool}] {progress}/{total}: {msg}"),
    emit_log=lambda tool, level, data: print(f"[{tool}] {level}: {data}"),
)

result = Runner.run_streamed(
    agent,
    input="Create a lesson plan for AI ethics.",
    context=context,
)
```

## Progress and Log Streaming

`UserContext` supplies callbacks for progress and log messages. During each tool invocation `_build_function_tool` wires FastMCP's native callbacks into these handlers. Provide your own functions to surface status in a UI; omit them to fall back to stdout logging.

```python
def handle_progress(tool, progress, total, message):
    update_ui_progress(tool, progress, total, message)

def handle_log(tool, level, data):
    append_log_entry(tool, level, data)

context = UserContext(
    emit_progress=handle_progress,
    emit_log=handle_log,
)
```

## Choosing a Transport

### Stdio

```python
toolkit = MCPToolkit(
    server_cmd=".venv/bin/python",
    server_script="path/to/mcp_tools.py",
    server_args=("--flag",),
)
```

### SSE (default)

```python
toolkit = MCPToolkit()  # reads MCP_* env vars or falls back to http://127.0.0.1:8010/sse/
```

Override parts of the connection as needed:

```python
toolkit = MCPToolkit(
    base_url="https://mcp.example.com",
    sse_path="/custom/sse/",
    auth_token_env="MCP_AUTH_TOKEN",
)

# Tip: pass headers per request when your app already has the bearer token.
toolkit = MCPToolkit(
    base_url="https://mcp.example.com",
    headers={"Authorization": f"Bearer {token}"},
)
```

### Streamable HTTP

```python
from utility.mcp_tool_helper import build_streamable_http_transport_factory

toolkit = MCPToolkit(
    transport_factory=build_streamable_http_transport_factory(
        url="https://mcp.example.com/mcp",
        headers={"Authorization": "Bearer TOKEN"},
    ),
)
```

### Custom SSE Transport

```python
from utility.mcp_tool_helper import build_sse_transport_factory

toolkit = MCPToolkit(
    transport_factory=build_sse_transport_factory(
        url="https://mcp.example.com/mcp/sse/",
        headers={"X-Custom": "value"},
    ),
)
```

### Auto-Inferred Transport

```python
# Accepts URLs, paths, ClientTransport instances, or MCP config dicts
toolkit = MCPToolkit(transport="https://mcp.example.com/mcp")
```

## Caching and Refresh

`MCPToolkit` caches discovered `FunctionTool`s on first use. Call `refresh_cache()` to force a reload, or construct the toolkit with `preload=False` to defer discovery until the first lookup.

```python
toolkit = MCPToolkit(..., preload=False)
tools = toolkit.get_function_tools()  # loads on demand
toolkit.refresh_cache()               # refreshes immediately
```

## Authentication

Set `MCP_TEST_TOKEN` (or supply `auth_token_env=`) when you already have a static bearer token.  
Alternatively provide explicit headers via the `headers=` argument on `MCPToolkit` or `get_mcp_function_tools`, or ensure your custom transport factory attaches whatever auth scheme your deployment expects.

### Practical Environment Variables

- `MCP_HOST` (default `127.0.0.1`; also accepts legacy `MCP_HOST`)
- `MCP_PORT` (default `8010`; legacy `MCP_PORT`)
- `MCP_SSE_PATH` (default `/sse/`; legacy `MCP_SSE_PATH`)
- `MCP_SCHEME` (default `http`; legacy `MCP_SCHEME`)
- `MCP_AUTH_TOKEN` (bearer token; legacy `MCP_TEST_TOKEN`)

Provide explicit arguments to `MCPToolkit` / `get_mcp_function_tools` to override any of these or to supply stdio settings such as `server_cmd` and `server_script`.

### Typical SSE Deployment Flow

1. Deploy your FastMCP server behind HTTPS and expose its SSE endpoint (e.g., `/sse/`).
2. Issue time-limited access tokens (e.g., Azure AD) and hand them to clients.
3. On the client, call `MCPToolkit(base_url="https://...", headers={"Authorization": f"Bearer {token}"})`.
4. Optionally reuse the same toolkit across requests; call `refresh_cache()` if tools change server-side.

### Local Development (stdio > SSE migration)

1. Start with stdio: `MCPToolkit(server_cmd="python", server_script="local_server.py")`.
2. Switch to SSE by providing host/port/path without touching call sites:

```python
toolkit = MCPToolkit(
    base_url="http://127.0.0.1:8010",
    sse_path="/sse/",
)
```

3. If both stdio and SSE are available, instantiate two toolkits and select based on environment.

## Convenience Helper

The top-level helper mirrors the constructor and returns a list of `FunctionTool`s in one call.

```python
from utility.mcp_tool_helper import get_mcp_function_tools

tools = get_mcp_function_tools(
    host="localhost",
    port=8010,
    auth_token_env="MCP_TEST_TOKEN",
    transport="https://mcp.example.com/mcp",  # optional override
)

# or via stdio
# tools = get_mcp_function_tools(
#     server_cmd=".venv/bin/python",
#     server_script="path/to/mcp_tools.py",
# )
```

## Development Notes

- Requires FastMCP 2.9 or later and OpenAI's Agent SDK.
- Invalid JSON input for tool invocations raises `ModelBehaviorError` to signal agent-side issues.
- Synchronous callers are supported: the helper executes asynchronous discovery inside `asyncio.run`, and falls back to a worker thread if an event loop is already running.

## License

GPL-3.0.
