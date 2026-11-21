"""Check MCP servers without starting the voice pipeline.

    python -m speech_to_speech.LLM.mcp_tools --config mcp_servers.json list
    python -m speech_to_speech.LLM.mcp_tools --config mcp_servers.json call list_calendar_events "{}"

``call`` prints exactly what the model would receive (already trimmed). Write
tools are executed directly here, without the spoken confirmation gate, so only
use ``call`` on a write tool when you really want the item created.

If a Google tool reports that sign-in is needed, the authorization link is
printed in the log above. Open it on this machine, approve access, then press
Enter and the call is retried while the server (and its OAuth callback
listener) is still running.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from speech_to_speech.LLM.mcp_tools.config import load_config, resolve_config_path
from speech_to_speech.LLM.mcp_tools.runtime import MCPToolRuntime, build_call_args
from speech_to_speech.LLM.mcp_tools.summarize import AUTH_NEEDED


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m speech_to_speech.LLM.mcp_tools")
    parser.add_argument("--config", default=None, help="Path to mcp_servers.json (default: $S2S_MCP_CONFIG)")
    parser.add_argument("--wait", type=float, default=90.0, help="Seconds to wait for servers to start")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="Show servers and the tools exposed to the model")
    call = sub.add_parser("call", help="Call one exposed tool and print the trimmed result")
    call.add_argument("tool")
    call.add_argument("arguments", nargs="?", default="{}")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    path = resolve_config_path(args.config)
    if not path:
        parser.error("pass --config or set S2S_MCP_CONFIG")
    runtime = MCPToolRuntime(load_config(path)).start(wait_s=args.wait)
    try:
        if args.cmd == "list":
            print(json.dumps(runtime.status(), indent=2))
            for tool in runtime.tools():
                kind = "WRITE (needs spoken yes)" if tool.spec.kind == "write" else "read"
                print(f"\n{tool.spec.expose_as}  [{tool.server}:{tool.spec.mcp_name}, {kind}]")
                print(json.dumps(tool.definition["parameters"], indent=1)[:1500])
            return 0
        exposed = {t.spec.expose_as: t for t in runtime.tools()}
        tool = exposed.get(args.tool)
        if tool is None:
            print(f"Unknown or unavailable tool {args.tool!r}. Available: {', '.join(exposed) or 'none'}")
            return 2
        model_args = json.loads(args.arguments)
        auth_text = AUTH_NEEDED.format(label=tool.label)
        while True:
            _ok, text = runtime._invoke(tool, build_call_args(tool.spec, model_args))
            print("\n--- what the model would get ---\n" + text)
            if text != auth_text or not sys.stdin.isatty():
                return 0
            try:
                input("\nFinish the sign-in in your browser (link in the log above), then press Enter to retry...")
            except EOFError:
                return 0
    finally:
        runtime.close()


if __name__ == "__main__":
    sys.exit(main())
