"""Configuration for server-side MCP tools.

The config is a JSON file (``mcp_servers.json`` by convention) passed with
``--mcp_config`` or the ``S2S_MCP_CONFIG`` environment variable. See
``mcp_servers.example.json`` and ``docs/mcp-tools.md`` for the full format.

Only tools listed under a server's ``tools`` mapping are exposed to the model
(the allowlist). Each entry can rename the tool, pin arguments, hide parameters,
and mark it as a ``write`` tool that needs spoken confirmation.

``${VAR}`` placeholders in strings are expanded from the process environment,
falling back to a ``.env`` file next to the config (or in the working
directory). A server whose placeholders resolve to an empty value is skipped
with a log message instead of being started with a missing secret.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

ENV_CONFIG_PATH = "S2S_MCP_CONFIG"
_PLACEHOLDER = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

ToolKind = Literal["read", "write"]


class MCPConfigError(ValueError):
    """Raised when the MCP config file is malformed."""


@dataclass
class ToolSpec:
    """One allowlisted MCP tool and how it is shown to the model."""

    mcp_name: str
    expose_as: str
    kind: ToolKind = "read"
    description: str | None = None
    # Arguments always sent to the MCP tool and removed from the model's schema.
    fixed_args: dict[str, Any] = field(default_factory=dict)
    # Arguments sent when the model omits them (kept out of the schema too).
    default_args: dict[str, Any] = field(default_factory=dict)
    # Keep only these top-level parameters in the model-facing schema.
    expose_params: list[str] | None = None
    # Replace the model-facing parameter schema entirely (JSON Schema object).
    parameters: dict[str, Any] | None = None
    # Wrap the model's arguments as a one-element list under this key
    # (e.g. Todoist add-tasks expects {"tasks": [{...}]}).
    wrap_in_list: str | None = None
    # Spoken read-back for write tools, formatted with the model's arguments.
    confirm_template: str | None = None
    max_items: int | None = None
    timeout_s: float | None = None


@dataclass
class ServerSpec:
    name: str
    transport: Literal["stdio", "streamable_http"]
    label: str
    command: str | None = None
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    cwd: str | None = None
    inherit_env: bool = False
    url: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    tools: dict[str, ToolSpec] = field(default_factory=dict)
    enabled: bool = True
    missing_env: list[str] = field(default_factory=list)


@dataclass
class MCPConfig:
    servers: list[ServerSpec]
    call_timeout_s: float = 8.0
    startup_wait_s: float = 15.0
    max_items: int = 5
    max_result_chars: int = 900
    max_tool_rounds: int = 3
    confirm_ttl_s: float = 180.0
    source_path: str | None = None


def read_dotenv(path: Path) -> dict[str, str]:
    """Minimal ``KEY=value`` parser (comments, ``export``, and quotes handled)."""
    values: dict[str, str] = {}
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return values
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        key, _, value = line.partition("=")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key.strip()] = value
    return values


def _expand(value: Any, env: dict[str, str], missing: set[str]) -> Any:
    if isinstance(value, str):

        def repl(match: re.Match[str]) -> str:
            name = match.group(1)
            resolved = env.get(name, "")
            if not resolved:
                missing.add(name)
            return resolved

        return _PLACEHOLDER.sub(repl, value)
    if isinstance(value, list):
        return [_expand(v, env, missing) for v in value]
    if isinstance(value, dict):
        return {k: _expand(v, env, missing) for k, v in value.items()}
    return value


_NAME_RE = re.compile(r"[^A-Za-z0-9_-]")


def _tool_name(name: str) -> str:
    cleaned = _NAME_RE.sub("_", name).strip("_")
    return cleaned[:64] or "tool"


def _parse_tool(server: str, mcp_name: str, raw: Any) -> ToolSpec:
    if raw is None or raw is True:
        raw = {}
    if not isinstance(raw, dict):
        raise MCPConfigError(f"servers.{server}.tools.{mcp_name} must be an object")
    kind = raw.get("kind", "read")
    if kind not in ("read", "write"):
        raise MCPConfigError(f"servers.{server}.tools.{mcp_name}.kind must be 'read' or 'write'")
    return ToolSpec(
        mcp_name=mcp_name,
        expose_as=_tool_name(raw.get("expose_as") or mcp_name),
        kind=kind,
        description=raw.get("description"),
        fixed_args=dict(raw.get("fixed_args") or {}),
        default_args=dict(raw.get("default_args") or {}),
        expose_params=list(raw["expose_params"]) if raw.get("expose_params") is not None else None,
        parameters=raw.get("parameters"),
        wrap_in_list=raw.get("wrap_in_list"),
        confirm_template=raw.get("confirm_template"),
        max_items=raw.get("max_items"),
        timeout_s=raw.get("timeout_s"),
    )


def parse_config(
    data: dict[str, Any], *, env: dict[str, str] | None = None, source_path: str | None = None
) -> MCPConfig:
    """Validate a decoded config dict. ``env`` defaults to ``os.environ``."""
    if not isinstance(data, dict):
        raise MCPConfigError("MCP config must be a JSON object")
    env = dict(os.environ) if env is None else env
    defaults = data.get("defaults") or {}
    servers_raw = data.get("servers") or {}
    if not isinstance(servers_raw, dict):
        raise MCPConfigError("'servers' must be an object keyed by server name")

    servers: list[ServerSpec] = []
    for name, raw in servers_raw.items():
        if not isinstance(raw, dict):
            raise MCPConfigError(f"servers.{name} must be an object")
        missing: set[str] = set()
        expanded = _expand({k: v for k, v in raw.items() if k != "tools"}, env, missing)
        transport = expanded.get("transport") or ("streamable_http" if expanded.get("url") else "stdio")
        transport = transport.replace("-", "_")
        if transport not in ("stdio", "streamable_http"):
            raise MCPConfigError(f"servers.{name}.transport must be 'stdio' or 'streamable_http'")
        if transport == "stdio" and not expanded.get("command"):
            raise MCPConfigError(f"servers.{name}: stdio servers need a 'command'")
        if transport == "streamable_http" and not expanded.get("url"):
            raise MCPConfigError(f"servers.{name}: streamable_http servers need a 'url'")
        tools_raw = raw.get("tools") or {}
        if isinstance(tools_raw, list):
            tools_raw = {t: {} for t in tools_raw}
        tools: dict[str, ToolSpec] = {}
        for mcp_name, tool_raw in tools_raw.items():
            tool_missing: set[str] = set()
            spec = _parse_tool(name, mcp_name, _expand(tool_raw, env, tool_missing))
            missing |= tool_missing
            tools[mcp_name] = spec
        servers.append(
            ServerSpec(
                name=name,
                transport=transport,
                label=expanded.get("label") or name,
                command=expanded.get("command"),
                args=[str(a) for a in expanded.get("args") or []],
                env={k: str(v) for k, v in (expanded.get("env") or {}).items()},
                cwd=expanded.get("cwd"),
                inherit_env=bool(expanded.get("inherit_env", False)),
                url=expanded.get("url"),
                headers={k: str(v) for k, v in (expanded.get("headers") or {}).items()},
                tools=tools,
                enabled=bool(expanded.get("enabled", True)),
                missing_env=sorted(missing),
            )
        )

    return MCPConfig(
        servers=servers,
        call_timeout_s=float(defaults.get("call_timeout_s", 8.0)),
        startup_wait_s=float(defaults.get("startup_wait_s", 15.0)),
        max_items=int(defaults.get("max_items", 5)),
        max_result_chars=int(defaults.get("max_result_chars", 900)),
        max_tool_rounds=int(defaults.get("max_tool_rounds", 3)),
        confirm_ttl_s=float(defaults.get("confirm_ttl_s", 180.0)),
        source_path=source_path,
    )


def resolve_config_path(explicit: str | None) -> str | None:
    """CLI value wins, then ``S2S_MCP_CONFIG``. Empty means MCP is off."""
    path = (explicit or "").strip() or os.environ.get(ENV_CONFIG_PATH, "").strip()
    return path or None


def load_config(path: str) -> MCPConfig:
    """Read the JSON config, expanding ``${VAR}`` from env, then ``.env``."""
    config_path = Path(path).expanduser()
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise MCPConfigError(f"MCP config not found: {config_path}") from exc
    except json.JSONDecodeError as exc:
        raise MCPConfigError(f"MCP config is not valid JSON ({config_path}): {exc}") from exc
    env: dict[str, str] = {}
    for candidate in (Path.cwd() / ".env", config_path.resolve().parent / ".env"):
        env.update(read_dotenv(candidate))
    env.update({k: v for k, v in os.environ.items() if v})
    return parse_config(data, env=env, source_path=str(config_path))
