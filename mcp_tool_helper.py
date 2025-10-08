from __future__ import annotations

import asyncio
import inspect
import json
import os
import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

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


def _coerce_env_names(
    names: str | Sequence[str] | None,
) -> tuple[str, ...]:
    if names is None:
        return ()
    if isinstance(names, str):
        return (names,)
    return tuple(name for name in names if name)


def _first_env_value(names: tuple[str, ...]) -> str | None:
    for name in names:
        if not name:
            continue
        value = os.getenv(name)
        if value:
            return value
    return None


@dataclass(frozen=True)
class MCPClientConfig:
    base_url: str
    sse_path: str
    auth_token_env: tuple[str, ...] = ()
    connection_retries: int = 5
    retry_interval: float = 0.5
    headers: tuple[tuple[str, str], ...] = ()
    server_cmd: str | None = None
    server_script: Path | None = None
    server_args: tuple[str, ...] = ()

    def build_headers(
        self,
        overrides: Optional[Mapping[str, str]] = None,
    ) -> dict[str, str]:
        headers = {name: value for name, value in self.headers}
        for name in self.auth_token_env:
            token = os.getenv(name)
            if token:
                headers.setdefault("Authorization", f"Bearer {token}")
                break
        if overrides:
            headers.update(overrides)
        return headers


TransportFactory = Callable[
    [MCPClientConfig, Optional[Mapping[str, str]]], ClientTransport
]


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


def _merge_headers(
    *sources: Optional[Mapping[str, str]]
) -> dict[str, str]:
    merged: dict[str, str] = {}
    for source in sources:
        if source:
            merged.update(source)
    return merged


def _normalize_path(path: str) -> str:
    if not path.startswith("/"):
        path = f"/{path}"
    if not path.endswith("/"):
        path = f"{path}/"
    return path


def _build_sse_url(config: MCPClientConfig) -> str:
    return config.base_url.rstrip("/") + config.sse_path


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


def _build_default_transport(
    config: MCPClientConfig, headers: Optional[Mapping[str, str]]
) -> ClientTransport:
    if config.server_script is not None:
        if config.server_cmd is None:
            raise ValueError(
                "server_cmd must be provided when using stdio transport."
            )
        return PythonStdioTransport(
            config.server_script,
            python_cmd=config.server_cmd,
            args=list(_normalize_args(config)),
        )

    header_dict = _merge_headers(headers)
    return SSETransport(
        url=_build_sse_url(config),
        headers=header_dict or None,
    )


def _transport_factory_from_spec(transport_spec: Any) -> TransportFactory:
    def _factory(
        config: MCPClientConfig, headers: Optional[Mapping[str, str]]
    ) -> ClientTransport:
        if isinstance(transport_spec, ClientTransport):
            return transport_spec

        transport = infer_transport(transport_spec)
        if headers:
            header_dict = _merge_headers(headers)
            if isinstance(transport, SSETransport):
                transport.headers = header_dict
            elif isinstance(transport, StreamableHttpTransport):
                transport.headers = header_dict
            else:
                raise ValueError(
                    "Cannot merge headers into inferred transport. "
                    "Provide a transport_factory instead."
                )
        return transport

    return _factory


def _normalize_transport_factory(
    factory: Callable[..., ClientTransport] | None,
) -> TransportFactory:
    if factory is None:
        return _build_default_transport

    try:
        signature = inspect.signature(factory)
    except (TypeError, ValueError):
        signature = None

    if signature is not None and len(signature.parameters) == 1:

        def _wrapped(
            config: MCPClientConfig, headers: Optional[Mapping[str, str]]
        ) -> ClientTransport:
            return factory(config)  # type: ignore[arg-type]

        return _wrapped

    return factory  # type: ignore[return-value]


def build_streamable_http_transport_factory(
    *,
    url: str,
    headers: Optional[Mapping[str, str]] = None,
    auth: Any = None,
    sse_read_timeout: Any = None,
    httpx_client_factory: Optional[Callable[[], Any]] = None,
) -> TransportFactory:
    def _factory(
        config: MCPClientConfig, base_headers: Optional[Mapping[str, str]]
    ) -> ClientTransport:
        merged_headers = dict(base_headers or {})
        merged_headers = _merge_headers(base_headers, headers)
        return StreamableHttpTransport(
            url=url,
            headers=merged_headers or None,
            auth=auth,
            sse_read_timeout=sse_read_timeout,
            httpx_client_factory=httpx_client_factory,
        )

    return _factory


def build_sse_transport_factory(
    *,
    url: Optional[str] = None,
    headers: Optional[Mapping[str, str]] = None,
    auth: Any = None,
    sse_read_timeout: Any = None,
    httpx_client_factory: Optional[Callable[[], Any]] = None,
) -> TransportFactory:
    def _factory(
        config: MCPClientConfig, base_headers: Optional[Mapping[str, str]]
    ) -> ClientTransport:
        resolved_url = url or _build_sse_url(config)
        merged_headers = _merge_headers(base_headers, headers)
        return SSETransport(
            url=resolved_url,
            headers=merged_headers or None,
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
    base_url: str | None = None,
    host: str | None = None,
    host_env: str | Sequence[str] = ("MCP_HOST"),
    port: int | None = None,
    port_env: str | Sequence[str] | None = ("MCP_PORT"),
    scheme: str | None = None,
    scheme_env: str | Sequence[str] | None = ("MCP_SCHEME"),
    sse_path: str | None = None,
    sse_path_env: str | Sequence[str] = ("MCP_SSE_PATH"),
    auth_token_env: str | Sequence[str] | None = ("MCP_AUTH_TOKEN"),
    connection_retries: int = 5,
    retry_interval: float = 0.5,
    headers: Optional[Mapping[str, str]] = None,
    default_port: int | None = None,
    server_cmd: str | None = None,
    server_script: str | Path | None = None,
    server_args: Optional[Sequence[str]] = None,
) -> MCPClientConfig:
    host_env_names = _coerce_env_names(host_env)
    port_env_names = _coerce_env_names(port_env)
    scheme_env_names = _coerce_env_names(scheme_env)
    sse_path_env_names = _coerce_env_names(sse_path_env)
    auth_token_env_names = _coerce_env_names(auth_token_env)

    if base_url is not None:
        normalized_base = base_url.rstrip("/")
    else:
        resolved_host = host or _first_env_value(host_env_names) or "127.0.0.1"
        if scheme is None:
            resolved_scheme = _first_env_value(scheme_env_names) or "http"
        else:
            resolved_scheme = scheme

        resolved_port = port
        if resolved_port is None and port_env_names:
            port_value = _first_env_value(port_env_names)
            if port_value:
                resolved_port = int(port_value)
        if resolved_port is None:
            resolved_port = default_port

        if resolved_port is None:
            normalized_base = f"{resolved_scheme}://{resolved_host}"
        else:
            normalized_base = f"{resolved_scheme}://{resolved_host}:{resolved_port}"

    sse_path_value = sse_path or _first_env_value(sse_path_env_names) or "/sse/"
    normalized_path = _normalize_path(sse_path_value)

    header_items: tuple[tuple[str, str], ...] = ()
    if headers:
        header_items = tuple(headers.items())

    script_path: Path | None = None
    if server_script is not None:
        script_path = Path(server_script).expanduser().resolve()

    args_tuple: tuple[str, ...] = ()
    if server_args:
        args_tuple = tuple(server_args)

    return MCPClientConfig(
        base_url=normalized_base.rstrip("/"),
        sse_path=normalized_path,
        auth_token_env=auth_token_env_names,
        connection_retries=connection_retries,
        retry_interval=retry_interval,
        headers=header_items,
        server_cmd=server_cmd,
        server_script=script_path,
        server_args=args_tuple,
    )


class MCPToolkit:
    """Factory for wrapping FastMCP tools as OpenAI Agent FunctionTools."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        host: str | None = None,
        host_env: str | Sequence[str] = ("MCP_HOST"),
        port: int | None = None,
        port_env: str | Sequence[str] | None = ("MCP_PORT"),
        scheme: str | None = None,
        scheme_env: str | Sequence[str] | None = ("MCP_SCHEME"),
        sse_path: str | None = None,
        sse_path_env: str | Sequence[str] = ("MCP_SSE_PATH"),
        auth_token_env: str | Sequence[str] | None = ("MCP_AUTH_TOKEN"),
        emit_progress: Optional[
            Callable[[str, float, Optional[float], Optional[str]], None]
        ] = None,
        emit_log: Optional[Callable[[str, str, Any], None]] = None,
        preload: bool = True,
        transport: ClientTransport
        | str
        | os.PathLike[str]
        | dict[str, Any]
        | None = None,
        transport_factory: TransportFactory | None = None,
        headers: Optional[Mapping[str, str]] = None,
        connection_retries: int = 5,
        retry_interval: float = 0.5,
        default_port: int | None = 8010,
        server_cmd: str | None = None,
        server_script: str | Path | None = None,
        server_args: Optional[Sequence[str]] = None,
    ) -> None:
        if transport is not None and transport_factory is not None:
            raise ValueError("Provide either transport or transport_factory, not both")

        if transport is not None:
            transport_factory = _transport_factory_from_spec(transport)

        self.config = _ensure_config(
            base_url=base_url,
            host=host,
            host_env=host_env,
            port=port,
            port_env=port_env,
            scheme=scheme,
            scheme_env=scheme_env,
            sse_path=sse_path,
            sse_path_env=sse_path_env,
            auth_token_env=auth_token_env,
            connection_retries=connection_retries,
            retry_interval=retry_interval,
            headers=headers,
            default_port=default_port,
            server_cmd=server_cmd,
            server_script=server_script,
            server_args=server_args,
        )
        self._default_emit_progress = emit_progress or _stdout_emit_progress
        self._default_emit_log = emit_log or _stdout_emit_log
        self._tools_cache: list[FunctionTool] | None = None
        normalized_factory = _normalize_transport_factory(transport_factory)
        self._transport_factory = normalized_factory

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
        headers = self.config.build_headers()
        transport = self._transport_factory(self.config, headers)
        return Client(
            transport,
            progress_handler=progress_handler,
            log_handler=log_handler,
        )

    @asynccontextmanager
    async def _client_session(
        self,
        *,
        progress_handler: Optional[
            Callable[[float, Optional[float], Optional[str]], Any]
        ] = None,
        log_handler: Optional[Callable[[LogMessage], Any]] = None,
    ):
        last_exc: Exception | None = None
        for attempt in range(self.config.connection_retries):
            client = self._create_client(
                progress_handler=progress_handler,
                log_handler=log_handler,
            )
            try:
                async with client:
                    yield client
                    return
            except Exception as exc:  # pragma: no cover - transient connection errors
                last_exc = exc
                if attempt == self.config.connection_retries - 1:
                    break
                await asyncio.sleep(self.config.retry_interval)
        raise RuntimeError(
            f"Unable to connect to {_describe_connection(self.config)}"
        ) from last_exc

    async def _list_tool_specs(self) -> list[MCPToolSpec]:
        async with self._client_session() as client:
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

            async with self._client_session(
                progress_handler=progress_handler,
                log_handler=log_handler,
            ) as client:
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
    base_url: str | None = None,
    host: str | None = None,
    host_env: str | Sequence[str] = ("MCP_HOST"),
    port: int | None = None,
    port_env: str | Sequence[str] | None = ("MCP_PORT"),
    scheme: str | None = None,
    scheme_env: str | Sequence[str] | None = ("MCP_SCHEME"),
    sse_path: str | None = None,
    sse_path_env: str | Sequence[str] = ("MCP_SSE_PATH"),
    auth_token_env: str | Sequence[str] | None = ("MCP_AUTH_TOKEN"),
    emit_progress_handler: Optional[
        Callable[[str, float, Optional[float], Optional[str]], None]
    ] = None,
    emit_log_handler: Optional[Callable[[str, str, Any], None]] = None,
    preload: bool = True,
    transport: ClientTransport
    | str
    | os.PathLike[str]
    | dict[str, Any]
    | None = None,
    transport_factory: TransportFactory | None = None,
    headers: Optional[Mapping[str, str]] = None,
    connection_retries: int = 5,
    retry_interval: float = 0.5,
    default_port: int | None = 8010,
    server_cmd: str | None = None,
    server_script: str | Path | None = None,
    server_args: Optional[Sequence[str]] = None,
) -> list[FunctionTool]:
    """Convenience helper to build FunctionTool proxies for the given MCP server."""

    toolkit = MCPToolkit(
        base_url=base_url,
        host=host,
        host_env=host_env,
        port=port,
        port_env=port_env,
        scheme=scheme,
        scheme_env=scheme_env,
        sse_path=sse_path,
        sse_path_env=sse_path_env,
        auth_token_env=auth_token_env,
        emit_progress=emit_progress_handler,
        emit_log=emit_log_handler,
        preload=preload,
        transport=transport,
        transport_factory=transport_factory,
        headers=headers,
        connection_retries=connection_retries,
        retry_interval=retry_interval,
        default_port=default_port,
        server_cmd=server_cmd,
        server_script=server_script,
        server_args=server_args,
    )
    return toolkit.get_function_tools()


def _describe_connection(config: MCPClientConfig) -> str:
    if config.server_script is not None:
        return f"FastMCP server via stdio at {config.server_script}"
    return f"FastMCP server at {_build_sse_url(config)}"
