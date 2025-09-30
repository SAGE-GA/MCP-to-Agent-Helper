from __future__ import annotations

import asyncio
import json
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from agents.exceptions import ModelBehaviorError
from agents.tool import FunctionTool
from agents.tool_context import ToolContext
from fastmcp import Client
from fastmcp.client.logging import LogMessage
from fastmcp.client.transports import (
    ClientTransport,
    PythonStdioTransport,
    SSETransport,
    StreamableHttpTransport,
    infer_transport,
)


@dataclass
class UserContext:
    emit_progress: Callable[[str, float, Optional[float], Optional[str]], None]
    emit_log: Optional[Callable[[str, str, Any], None]] = None


@dataclass(frozen=True)
class MCPClientConfig:
    server_cmd: str | None
    server_script: Path | None
    server_args: tuple[str, ...] = ()


TransportFactory = Callable[[MCPClientConfig], ClientTransport]


@dataclass
class MCPToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]


def _stdout_emit_progress(
    tool_name: str, progress: float, total: Optional[float], message: Optional[str]
) -> None:
    if total:
        pct = (progress / total) * 100
        print(f"[{tool_name}] {pct:.1f}% {message or ''}")
    else:
        print(f"[{tool_name}] {progress} {message or ''}")


def _stdout_emit_log(tool_name: str, level: str, data: Any) -> None:
    print(f"[{tool_name}] {level.upper()} {data}")


def emit_progress(
    tool_name: str, progress: float, total: Optional[float], message: Optional[str]
) -> None:
    """Default progress emitter used when no custom handler is supplied."""

    _stdout_emit_progress(tool_name, progress, total, message)


def emit_log(tool_name: str, level: str, data: Any) -> None:
    """Default log emitter used when no custom handler is supplied."""

    _stdout_emit_log(tool_name, level, data)


def _normalize_args(config: MCPClientConfig) -> tuple[str, ...]:
    if not config.server_args:
        return ()

    script_path = config.server_script
    if script_path is None:
        return config.server_args

    filtered: list[str] = []
    for arg in config.server_args:
        arg_path = Path(arg)
        if arg_path.resolve() == script_path:
            continue
        filtered.append(arg)
    return tuple(filtered)


def _build_stdio_transport(config: MCPClientConfig) -> PythonStdioTransport:
    if config.server_script is None:
        raise ValueError("server_script must be provided for stdio transport")
    if config.server_cmd is None:
        raise ValueError("server_cmd must be provided for stdio transport")

    return PythonStdioTransport(
        config.server_script,
        python_cmd=config.server_cmd,
        args=list(_normalize_args(config)),
    )


def _transport_factory_from_spec(transport_spec: Any) -> TransportFactory:
    def _factory(_: MCPClientConfig) -> ClientTransport:
        if isinstance(transport_spec, ClientTransport):
            return transport_spec
        return infer_transport(transport_spec)

    return _factory


def build_streamable_http_transport_factory(
    *,
    url: str,
    headers: Optional[dict[str, str]] = None,
    auth: Any = None,
    sse_read_timeout: Any = None,
    httpx_client_factory: Optional[Callable[[], Any]] = None,
) -> TransportFactory:
    def _factory(_: MCPClientConfig) -> ClientTransport:
        return StreamableHttpTransport(
            url=url,
            headers=headers,
            auth=auth,
            sse_read_timeout=sse_read_timeout,
            httpx_client_factory=httpx_client_factory,
        )

    return _factory


def build_sse_transport_factory(
    *,
    url: str,
    headers: Optional[dict[str, str]] = None,
    auth: Any = None,
    sse_read_timeout: Any = None,
    httpx_client_factory: Optional[Callable[[], Any]] = None,
) -> TransportFactory:
    def _factory(_: MCPClientConfig) -> ClientTransport:
        return SSETransport(
            url=url,
            headers=headers,
            auth=auth,
            sse_read_timeout=sse_read_timeout,
            httpx_client_factory=httpx_client_factory,
        )

    return _factory


def build_inferred_transport_factory(transport_spec: Any) -> TransportFactory:
    return _transport_factory_from_spec(transport_spec)


def _format_tool_output(content: Any) -> str:
    if not content:
        return ""

    if hasattr(content, "output"):
        output_value = getattr(content, "output")
        if isinstance(output_value, str):
            return output_value
        if output_value is not None:
            try:
                return json.dumps(output_value, indent=2)
            except TypeError:
                return str(output_value)

        items = getattr(content, "items", None)
        if items is not None:
            content = items

    if isinstance(content, str):
        return content

    try:
        iterator = iter(content)
    except TypeError:
        return str(content)

    parts: list[str] = []
    for item in iterator:
        text = getattr(item, "text", None)
        parts.append(text if text is not None else str(item))
    return "\n".join(parts)


def _ensure_config(
    *,
    server_cmd: str | None = None,
    server_script: str | Path | None,
    server_args: Sequence[str] | None = None,
    allow_missing_script: bool = False,
) -> MCPClientConfig:
    script_path: Path | None
    if server_script is None:
        if not allow_missing_script:
            raise ValueError("server_script must be provided")
        script_path = None
    else:
        script_path = Path(server_script).resolve()

    if server_cmd is not None:
        cmd = server_cmd
    elif script_path is not None:
        cmd = sys.executable
    else:
        cmd = None
    args_tuple: tuple[str, ...] = tuple(server_args) if server_args else ()
    return MCPClientConfig(
        server_cmd=cmd,
        server_script=script_path,
        server_args=args_tuple,
    )


class MCPToolkit:
    """Factory for wrapping FastMCP tools as OpenAI Agent FunctionTools."""

    def __init__(
        self,
        *,
        server_cmd: str | None = None,
        server_script: str | Path | None = None,
        server_args: Sequence[str] | None = None,
        emit_progress: Optional[
            Callable[[str, float, Optional[float], Optional[str]], None]
        ] = None,
        emit_log: Optional[Callable[[str, str, Any], None]] = None,
        preload: bool = True,
        transport: ClientTransport | str | Path | dict[str, Any] | None = None,
        transport_factory: TransportFactory | None = None,
    ) -> None:
        if transport is not None and transport_factory is not None:
            raise ValueError("Provide either transport or transport_factory, not both")

        if transport is not None:
            transport_factory = _transport_factory_from_spec(transport)

        allow_missing_script = transport_factory is not None
        self.config = _ensure_config(
            server_cmd=server_cmd,
            server_script=server_script,
            server_args=server_args,
            allow_missing_script=allow_missing_script,
        )
        self._default_emit_progress = emit_progress or _stdout_emit_progress
        self._default_emit_log = emit_log or _stdout_emit_log
        self._tools_cache: list[FunctionTool] | None = None
        self._transport_factory = transport_factory or _build_stdio_transport

        if preload:
            self._tools_cache = self._build_tools_sync()

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def get_function_tools(self, *, use_cache: bool = True) -> list[FunctionTool]:
        if use_cache and self._tools_cache is not None:
            return list(self._tools_cache)

        tools = self._build_tools_sync()
        if use_cache:
            self._tools_cache = list(tools)
        return tools

    def refresh_cache(self) -> list[FunctionTool]:
        tools = self._build_tools_sync()
        self._tools_cache = list(tools)
        return tools

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _create_client(
        self,
        *,
        progress_handler: Optional[
            Callable[[float, Optional[float], Optional[str]], Any]
        ] = None,
        log_handler: Optional[Callable[[LogMessage], Any]] = None,
    ) -> Client:
        transport = self._transport_factory(self.config)
        return Client(
            transport,
            progress_handler=progress_handler,
            log_handler=log_handler,
        )

    async def _list_tool_specs(self) -> list[MCPToolSpec]:
        client = self._create_client()
        async with client:
            tools = await client.list_tools()

        specs: list[MCPToolSpec] = []
        for tool in tools:
            schema = tool.inputSchema or {"type": "object", "properties": {}}
            specs.append(
                MCPToolSpec(
                    name=tool.name,
                    description=tool.description or "",
                    input_schema=json.loads(json.dumps(schema)),
                )
            )
        return specs

    def _build_function_tool(self, spec: MCPToolSpec) -> FunctionTool:
        async def _on_invoke(ctx: ToolContext[Any], raw_input: str) -> str:
            try:
                arguments = json.loads(raw_input) if raw_input else {}
            except json.JSONDecodeError as exc:
                raise ModelBehaviorError(
                    f"Invalid JSON input for tool {spec.name}: {raw_input}"
                ) from exc

            context = getattr(ctx, "context", None)
            emit_progress = (
                getattr(context, "emit_progress", None) or self._default_emit_progress
            )
            emit_log = getattr(context, "emit_log", None) or self._default_emit_log

            async def progress_handler(
                progress: float, total: Optional[float], message: Optional[str]
            ) -> None:
                if emit_progress:
                    emit_progress(spec.name, progress, total, message)

            async def log_handler(message: LogMessage) -> None:
                if emit_log:
                    emit_log(spec.name, message.level, message.data)

            client = self._create_client(
                progress_handler=progress_handler,
                log_handler=log_handler,
            )

            async with client:
                result = await client.call_tool(spec.name, arguments)
            return _format_tool_output(result)

        return FunctionTool(
            name=spec.name,
            description=spec.description,
            params_json_schema=json.loads(json.dumps(spec.input_schema)),
            on_invoke_tool=_on_invoke,
            strict_json_schema=False,
        )

    async def _build_tools_async(self) -> list[FunctionTool]:
        specs = await self._list_tool_specs()
        return [self._build_function_tool(spec) for spec in specs]

    def _build_tools_sync(self) -> list[FunctionTool]:
        try:
            return asyncio.run(self._build_tools_async())
        except RuntimeError as exc:
            message = str(exc)
            if (
                "asyncio.run() cannot be called" not in message
                and "Cannot run the event loop" not in message
            ):
                raise

            result: list[FunctionTool] | None = None
            error: Exception | None = None

            def _worker() -> None:
                nonlocal result, error
                loop = asyncio.new_event_loop()
                try:
                    asyncio.set_event_loop(loop)
                    result = loop.run_until_complete(self._build_tools_async())
                except Exception as worker_exc:  # pragma: no cover
                    error = worker_exc
                finally:
                    asyncio.set_event_loop(None)
                    loop.close()

            thread = threading.Thread(target=_worker, daemon=True)
            thread.start()
            thread.join()

            if error:
                raise error
            assert result is not None
            return result


def get_mcp_function_tools(
    *,
    server_cmd: str | None = None,
    server_script: str | Path | None = None,
    server_args: Sequence[str] | None = None,
    emit_progress_handler: Optional[
        Callable[[str, float, Optional[float], Optional[str]], None]
    ] = None,
    emit_log_handler: Optional[Callable[[str, str, Any], None]] = None,
    preload: bool = True,
    transport: ClientTransport | str | Path | dict[str, Any] | None = None,
    transport_factory: TransportFactory | None = None,
) -> list[FunctionTool]:
    """Convenience helper to build FunctionTool proxies for the given MCP server."""

    toolkit = MCPToolkit(
        server_cmd=server_cmd,
        server_script=server_script,
        server_args=server_args,
        emit_progress=emit_progress_handler,
        emit_log=emit_log_handler,
        preload=preload,
        transport=transport,
        transport_factory=transport_factory,
    )
    return toolkit.get_function_tools()
