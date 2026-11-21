"""Server-side MCP tools (calendar, mail, tasks, ...) for the voice LLM loop."""

from speech_to_speech.LLM.mcp_tools.config import MCPConfig, MCPConfigError, load_config, resolve_config_path
from speech_to_speech.LLM.mcp_tools.runtime import MCPToolRuntime, get_runtime

__all__ = ["MCPConfig", "MCPConfigError", "MCPToolRuntime", "get_runtime", "load_config", "resolve_config_path"]
