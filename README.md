# SAGE MCP Tool Helper

SAGE MCP Tool Helper bridges [FastMCP](https://github.com/mom1/fastmcp)'s tooling ecosystem with OpenAI's Agent Framework. It discovers tools exposed by an MCP server, wraps them as OpenAI `FunctionTool`s, and forwards FastMCP progress and log events so agent UIs can show real-time status updates. This fills the gap in the Agent Framework, which does not natively stream tool status back to the client.

## Features

- Automatically discovers MCP tools and exposes them as OpenAI function tools
- Streams FastMCP progress and log messages back into your agent runtime
- Provides optional caching and cache refresh of tool metadata
- Supports stdio, Streamable HTTP, SSE, and any transport that `fastmcp.infer_transport` understands
- Offers ready-made transport factory helpers for common deployment targets

## Installation

```bash
pip install fastmcp openai
```

Clone or copy `utility/mcp_tool_helper.py` into your project. (A packaged release is planned.)

## Quick Start

```python
from utility.mcp_tool_helper import MCPToolkit, UserContext

# Launch a local MCP server via stdio
toolkit = MCPToolkit(
    server_cmd="../SAGE-H5P-Activity-Generator/.venv/bin/python",
    server_script="../SAGE-H5P-Activity-Generator/mcp_tools.py",
    server_args=("--option",),
)

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

### Stdio (default)

```python
toolkit = MCPToolkit(
    server_cmd="/path/to/python",
    server_script="/path/to/mcp_tools.py",
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

### Server-Sent Events (SSE)

```python
from utility.mcp_tool_helper import build_sse_transport_factory

toolkit = MCPToolkit(
    transport_factory=build_sse_transport_factory(
        url="https://mcp.example.com/mcp/sse",
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

## Convenience Helper

The top-level helper mirrors the constructor and returns a list of `FunctionTool`s in one call.

```python
from utility.mcp_tool_helper import get_mcp_function_tools

tools = get_mcp_function_tools(
    server_cmd="python",
    server_script="path/to/mcp_server.py",
    server_args=("--flag",),
    transport="https://mcp.example.com/mcp",  # optional
)
```

## Development Notes

- Requires FastMCP 2.9 or later and OpenAI's Agent SDK.
- Invalid JSON input for tool invocations raises `ModelBehaviorError` to signal agent-side issues.
- Synchronous callers are supported: the helper executes asynchronous discovery inside `asyncio.run`, and falls back to a worker thread if an event loop is already running.

## License

GPL-3.0.
