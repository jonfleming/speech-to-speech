"""Server-side MCP client for the LLM handlers.

``MCPToolRuntime`` owns one asyncio loop on a daemon thread. Each configured
MCP server gets a long-lived task that connects (stdio subprocess or streamable
HTTP), discovers tools with ``list_tools``, and stays connected. The LLM handler
threads call :meth:`MCPToolRuntime.execute` synchronously; every call has a hard
timeout so a slow service cannot stall the conversation.

Servers that fail to start, or whose ``${VAR}`` secrets are missing, are logged
and skipped; the rest keep working, and a failed server is retried in the
background with backoff.
"""

from __future__ import annotations

import asyncio
import atexit
import concurrent.futures
import copy
import json
import logging
import os
import re
import shutil
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from speech_to_speech.LLM.mcp_tools.config import MCPConfig, ServerSpec, ToolSpec, load_config
from speech_to_speech.LLM.mcp_tools.confirm import (
    PendingAction,
    PendingActionStore,
    describe_action,
    is_affirmative,
    is_negative,
)
from speech_to_speech.LLM.mcp_tools.summarize import clip, looks_like_auth_prompt, result_text, summarize_result

logger = logging.getLogger(__name__)

ClientFactory = Callable[[ServerSpec], Any]

CONFIRM_PROMPT = (
    "NOT DONE YET - nothing has been saved. The user must confirm out loud first. "
    'Say this to the user in one short sentence and then stop: "Should I {description}? Say yes to confirm."'
)
TIMEOUT_TEXT = "{label} did not answer within {seconds:g} seconds. Tell the user you couldn't reach {label} right now."
UNREACHABLE_TEXT = "{label} is not reachable right now. Tell the user you couldn't reach {label}."
DONE_TEXT = "Done, it's saved."
CANCELLED_TEXT = "Okay, I won't do that."
FAILED_WRITE_TEXT = "Sorry, that didn't go through. {detail}"

_ISO_DT = re.compile(r"^\d{4}-\d{2}-\d{2}([T ]\d{2}:\d{2}(:\d{2}(\.\d+)?)?(Z|[+-]\d{2}:?\d{2})?)?$")


def _humanize_value(value: Any) -> Any:
    """'2026-11-21T15:00:00-08:00' -> 'Saturday, November 21 at 3:00 PM' for read-backs."""
    if not isinstance(value, str) or not _ISO_DT.match(value.strip()):
        return value
    raw = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return value
    day = f"{parsed:%A}, {parsed:%B} {parsed.day}"
    if len(raw) <= 10:
        return day
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone()
    hour = parsed.hour % 12 or 12
    return f"{day} at {hour}:{parsed:%M} {'AM' if parsed.hour < 12 else 'PM'}"


@dataclass
class ExposedTool:
    server: str
    label: str
    spec: ToolSpec
    definition: dict[str, Any]
    read_only: bool


@dataclass
class _ServerState:
    spec: ServerSpec
    client: Any = None
    tools: list[Any] = field(default_factory=list)
    error: str | None = None
    ready: asyncio.Event | None = None
    wake: asyncio.Event | None = None
    task: asyncio.Task[None] | None = None


def _default_client_factory(spec: ServerSpec) -> Any:
    """Build an ``mcp.Client`` for a stdio command or a streamable HTTP URL."""
    from mcp import Client, StdioServerParameters

    if spec.transport == "streamable_http":
        from mcp.client.streamable_http import create_mcp_http_client, streamable_http_client

        assert spec.url
        http_client = create_mcp_http_client(headers=spec.headers or None)
        return Client(streamable_http_client(spec.url, http_client=http_client))

    assert spec.command
    command = spec.command
    resolved = shutil.which(command)
    if resolved:
        # On Windows "npx" must resolve to npx.cmd for CreateProcess.
        command = resolved
    env: dict[str, str] | None
    if spec.inherit_env:
        env = {**os.environ, **spec.env}
    else:
        from mcp.client.stdio import get_default_environment

        env = {**get_default_environment(), **spec.env}
    return Client(StdioServerParameters(command=command, args=spec.args, env=env, cwd=spec.cwd))


def build_definition(spec: ToolSpec, tool: Any) -> dict[str, Any]:
    """Responses-style function tool for the model, derived from ``list_tools``."""
    if spec.parameters is not None:
        schema = copy.deepcopy(spec.parameters)
    else:
        schema = copy.deepcopy(getattr(tool, "input_schema", None) or {"type": "object", "properties": {}})
        props: dict[str, Any] = schema.setdefault("properties", {})
        hidden = set(spec.fixed_args) | set(spec.default_args)
        if spec.expose_params is not None:
            hidden |= set(props) - set(spec.expose_params)
        for name in hidden:
            props.pop(name, None)
        required = [r for r in schema.get("required") or [] if r in props]
        if required:
            schema["required"] = required
        else:
            schema.pop("required", None)
    schema.pop("$schema", None)
    schema.setdefault("type", "object")
    description = spec.description or getattr(tool, "description", None) or spec.expose_as
    description = clip(" ".join(description.split()), 500)
    if spec.kind == "write":
        description += " The server asks the user to confirm out loud before anything is saved."
    return {"type": "function", "name": spec.expose_as, "description": description, "parameters": schema}


def build_call_args(spec: ToolSpec, model_args: dict[str, Any]) -> dict[str, Any]:
    args = {k: v for k, v in (model_args or {}).items() if v is not None and k not in spec.fixed_args}
    if spec.wrap_in_list:
        args = {spec.wrap_in_list: [args]}
    return {**spec.default_args, **args, **spec.fixed_args}


class MCPToolRuntime:
    def __init__(self, config: MCPConfig, *, client_factory: ClientFactory | None = None) -> None:
        self.config = config
        self.client_factory = client_factory or _default_client_factory
        self.pending = PendingActionStore(ttl_s=config.confirm_ttl_s)
        self._states: dict[str, _ServerState] = {}
        self._exposed: dict[str, ExposedTool] = {}
        self._exposed_lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._closing = False

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def start(self, wait_s: float | None = None) -> MCPToolRuntime:
        """Connect to every enabled server; wait up to ``wait_s`` for discovery."""
        if self._thread is not None:
            return self
        runnable: list[ServerSpec] = []
        for spec in self.config.servers:
            if not spec.enabled:
                logger.info("MCP server %s is disabled in config; skipping", spec.name)
            elif spec.missing_env:
                logger.warning(
                    "MCP server %s skipped: missing %s (set in the environment or .env)",
                    spec.name,
                    ", ".join(spec.missing_env),
                )
            elif not spec.tools:
                logger.warning("MCP server %s has no allowlisted tools; skipping", spec.name)
            else:
                runnable.append(spec)
        if not runnable:
            logger.warning("No MCP servers to start; continuing without MCP tools")
            return self
        ready = threading.Event()

        def run_loop() -> None:
            loop = asyncio.new_event_loop()
            self._loop = loop
            asyncio.set_event_loop(loop)
            ready.set()
            loop.run_forever()

        self._thread = threading.Thread(target=run_loop, name="mcp-tools", daemon=True)
        self._thread.start()
        ready.wait()
        for spec in runnable:
            self._states[spec.name] = _ServerState(spec=spec)
        asyncio.run_coroutine_threadsafe(self._start_all(), self._loop).result(10)
        wait = self.config.startup_wait_s if wait_s is None else wait_s
        if wait > 0:
            try:
                asyncio.run_coroutine_threadsafe(self._wait_ready(), self._loop).result(wait)
            except concurrent.futures.TimeoutError:
                slow = [n for n, s in self._states.items() if s.client is None and s.error is None]
                logger.warning(
                    "MCP servers still starting after %.0fs: %s; continuing, their tools appear when ready",
                    wait,
                    ", ".join(slow),
                )
        self._log_summary()
        return self

    async def _start_all(self) -> None:
        for state in self._states.values():
            state.ready = asyncio.Event()
            state.wake = asyncio.Event()
            state.task = asyncio.get_running_loop().create_task(self._serve(state), name=f"mcp-{state.spec.name}")

    async def _wait_ready(self) -> None:
        await asyncio.gather(*(s.ready.wait() for s in self._states.values() if s.ready is not None))

    async def _list_all_tools(self, client: Any) -> list[Any]:
        tools: list[Any] = []
        cursor: str | None = None
        for _ in range(20):
            result = await client.list_tools(cursor=cursor) if cursor else await client.list_tools()
            tools.extend(result.tools)
            cursor = getattr(result, "next_cursor", None)
            if not cursor:
                break
        return tools

    async def _serve(self, state: _ServerState) -> None:
        assert state.ready is not None and state.wake is not None
        backoff = 5.0
        while not self._closing:
            try:
                async with self.client_factory(state.spec) as client:
                    tools = await self._list_all_tools(client)
                    state.client, state.tools, state.error = client, tools, None
                    self._rebuild()
                    state.ready.set()
                    backoff = 5.0
                    state.wake.clear()
                    await state.wake.wait()
            except asyncio.CancelledError:
                raise
            except BaseException as exc:  # noqa: BLE001 - anyio wraps failures in ExceptionGroups
                state.error = f"{type(exc).__name__}: {exc}"
                logger.warning("MCP server %s unavailable: %s", state.spec.name, clip(state.error, 400))
            finally:
                if state.client is not None:
                    state.client = None
                    self._rebuild()
            state.ready.set()
            if self._closing:
                return
            logger.info("Retrying MCP server %s in %.0fs", state.spec.name, backoff)
            try:
                await asyncio.wait_for(state.wake.wait(), timeout=backoff)
                state.wake.clear()
            except asyncio.TimeoutError:
                pass
            backoff = min(backoff * 2, 300.0)

    def _request_reconnect(self, server: str) -> None:
        state = self._states.get(server)
        if state is not None and state.wake is not None and self._loop is not None:
            self._loop.call_soon_threadsafe(state.wake.set)

    def close(self) -> None:
        if self._loop is None or self._closing:
            return
        self._closing = True

        async def shutdown() -> None:
            for state in self._states.values():
                if state.wake is not None:
                    state.wake.set()
            tasks = [s.task for s in self._states.values() if s.task is not None]
            if tasks:
                _done, still = await asyncio.wait(tasks, timeout=5)
                for task in still:
                    task.cancel()

        try:
            asyncio.run_coroutine_threadsafe(shutdown(), self._loop).result(8)
        except Exception:  # noqa: BLE001
            pass
        self._loop.call_soon_threadsafe(self._loop.stop)

    # ── discovery / allowlist ─────────────────────────────────────────────────

    def _rebuild(self) -> None:
        exposed: dict[str, ExposedTool] = {}
        for state in self._states.values():
            if state.client is None:
                continue
            discovered = {getattr(t, "name", None): t for t in state.tools}
            for mcp_name, spec in state.spec.tools.items():
                tool = discovered.get(mcp_name)
                if tool is None:
                    logger.warning(
                        "MCP server %s does not offer allowlisted tool %r (has: %s)",
                        state.spec.name,
                        mcp_name,
                        ", ".join(sorted(n for n in discovered if n)),
                    )
                    continue
                name = spec.expose_as
                if name in exposed:
                    name = f"{state.spec.name}__{name}"[:64]
                    spec = copy.copy(spec)
                    spec.expose_as = name
                annotations = getattr(tool, "annotations", None)
                exposed[name] = ExposedTool(
                    server=state.spec.name,
                    label=state.spec.label,
                    spec=spec,
                    definition=build_definition(spec, tool),
                    read_only=bool(getattr(annotations, "read_only_hint", False)),
                )
        with self._exposed_lock:
            self._exposed = exposed

    def _log_summary(self) -> None:
        for name, state in self._states.items():
            if state.client is not None:
                names = [
                    t.spec.expose_as + ("*" if t.spec.kind == "write" else "") for t in self.tools() if t.server == name
                ]
                logger.info("MCP server %s ready; tools: %s (* = needs spoken confirmation)", name, ", ".join(names))
            elif state.error:
                logger.warning("MCP server %s not available: %s", name, clip(state.error, 300))

    def tools(self) -> list[ExposedTool]:
        with self._exposed_lock:
            return list(self._exposed.values())

    def tool_definitions(self) -> list[dict[str, Any]]:
        return [copy.deepcopy(t.definition) for t in self.tools()]

    def owns(self, name: str) -> bool:
        with self._exposed_lock:
            return name in self._exposed

    def status(self) -> dict[str, Any]:
        return {
            name: {
                "connected": state.client is not None,
                "error": state.error,
                "tools": [t.spec.expose_as for t in self.tools() if t.server == name],
            }
            for name, state in self._states.items()
        }

    def prompt_note(self, now: datetime | None = None) -> str:
        now = (now or datetime.now()).astimezone()
        offset = now.strftime("%z")
        offset = f"UTC{offset[:3]}:{offset[3:]}" if offset else "local time"
        return (
            f"Current date and time: {now:%A, %B} {now.day}, {now.year}, {now:%I:%M %p} ({offset}). "
            "Use it for calendar and task dates, and send times in RFC3339 with that offset. "
            "When a tool returns results, speak only a brief summary of the top few items. "
            "Tools that add things never save immediately: say the read-back the tool gives you and wait for the user."
        )

    # ── execution ─────────────────────────────────────────────────────────────

    def _invoke(self, tool: ExposedTool, arguments: dict[str, Any]) -> tuple[bool, str]:
        state = self._states.get(tool.server)
        client = state.client if state is not None else None
        if client is None or self._loop is None:
            return False, UNREACHABLE_TEXT.format(label=tool.label)
        timeout = float(tool.spec.timeout_s or self.config.call_timeout_s)
        future = asyncio.run_coroutine_threadsafe(
            client.call_tool(tool.spec.mcp_name, arguments, read_timeout_seconds=timeout), self._loop
        )
        try:
            result = future.result(timeout=timeout + 0.5)
        except concurrent.futures.TimeoutError:
            future.cancel()
            logger.warning("MCP tool %s.%s timed out after %.1fs", tool.server, tool.spec.mcp_name, timeout)
            return False, TIMEOUT_TEXT.format(label=tool.label, seconds=timeout)
        except Exception as exc:  # noqa: BLE001
            detail = repr(exc).lower()
            if "timed out" in detail or "timeout" in detail:
                logger.warning("MCP tool %s.%s timed out after %.1fs", tool.server, tool.spec.mcp_name, timeout)
                return False, TIMEOUT_TEXT.format(label=tool.label, seconds=timeout)
            logger.warning("MCP tool %s.%s failed: %s", tool.server, tool.spec.mcp_name, clip(repr(exc), 300))
            if "closed" in detail or "connection" in detail:
                self._request_reconnect(tool.server)
            return False, UNREACHABLE_TEXT.format(label=tool.label)
        text = summarize_result(
            result,
            label=tool.label,
            max_items=tool.spec.max_items or self.config.max_items,
            max_chars=self.config.max_result_chars,
        )
        ok = not getattr(result, "is_error", False) and not looks_like_auth_prompt(result_text(result))
        return ok, text

    def execute(
        self,
        name: str,
        arguments_json: str | None,
        *,
        session_key: str,
        assistant_ids: Iterable[str] = (),
    ) -> str:
        """Run (read) or park (write) a model tool call; return text for the model."""
        with self._exposed_lock:
            tool = self._exposed.get(name)
        if tool is None:
            return f"The tool {name} is not available right now."
        try:
            model_args = json.loads(arguments_json or "{}")
            if not isinstance(model_args, dict):
                raise ValueError("arguments must be a JSON object")
        except ValueError:
            return "The tool arguments were not valid JSON. Try again with a JSON object."
        call_args = build_call_args(tool.spec, model_args)
        if tool.spec.kind == "write":
            readable = {k: _humanize_value(v) for k, v in model_args.items()}
            description = describe_action(tool.spec.confirm_template, tool.spec.expose_as.replace("_", " "), readable)
            self.pending.stage(
                session_key,
                PendingAction(
                    tool_name=name,
                    arguments=call_args,
                    description=description,
                    assistant_ids_before=frozenset(assistant_ids),
                ),
            )
            logger.info("MCP write %s parked for confirmation: %s", name, description)
            return CONFIRM_PROMPT.format(description=description)
        ok, text = self._invoke(tool, call_args)
        logger.info("MCP tool %s -> %s", name, "ok" if ok else "failed")
        return text

    def has_pending(self, session_key: str) -> bool:
        return self.pending.peek(session_key) is not None

    def resolve_pending(self, session_key: str, user_text: str | None, assistant_ids: Iterable[str]) -> str | None:
        """Handle the utterance after a read-back.

        Returns text to speak directly (write executed, failed, or declined), or
        ``None`` to continue with a normal model turn. The parked call is always
        consumed: only the very next utterance can confirm it.
        """
        action = self.pending.pop(session_key)
        if action is None:
            return None
        if not set(assistant_ids) - action.assistant_ids_before:
            logger.info("Dropping parked MCP write %s: read-back was never delivered", action.tool_name)
            return None
        if is_affirmative(user_text):
            with self._exposed_lock:
                tool = self._exposed.get(action.tool_name)
            if tool is None:
                return FAILED_WRITE_TEXT.format(detail="That service isn't connected right now.")
            ok, text = self._invoke(tool, action.arguments)
            logger.info("MCP write %s confirmed -> %s", action.tool_name, "ok" if ok else "failed")
            if ok:
                return DONE_TEXT.format(description=action.description)
            return FAILED_WRITE_TEXT.format(detail=clip(text, 200))
        logger.info("MCP write %s not confirmed; dropped", action.tool_name)
        if is_negative(user_text):
            return CANCELLED_TEXT
        return None


_RUNTIMES: dict[str, MCPToolRuntime | None] = {}
_RUNTIMES_LOCK = threading.Lock()


def get_runtime(config_path: str | None) -> MCPToolRuntime | None:
    """Process-wide runtime per config file (handlers may be built per pipeline)."""
    if not config_path:
        return None
    key = os.path.abspath(config_path)
    with _RUNTIMES_LOCK:
        if key in _RUNTIMES:
            return _RUNTIMES[key]
        runtime: MCPToolRuntime | None = None
        try:
            import mcp  # noqa: F401
        except ImportError:
            logger.error("MCP config %s given but the 'mcp' package is not installed; MCP tools disabled", key)
        else:
            try:
                runtime = MCPToolRuntime(load_config(config_path)).start()
                atexit.register(runtime.close)
            except Exception as exc:  # noqa: BLE001 - never block the voice server on MCP
                logger.error("MCP tools disabled: %s", exc)
                runtime = None
        _RUNTIMES[key] = runtime
        return runtime
