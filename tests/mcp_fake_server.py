"""Tiny MCP server used by tests/test_mcp_tools.py.

Runs over stdio when executed as a script, or in-memory via ``build_server()``.
It mimics the shape of the real calendar / mail / task servers closely enough
to exercise discovery, allowlisting, trimming, timeouts, and write gating.
"""

from __future__ import annotations

import asyncio

from mcp.server import MCPServer

CALLS: list[tuple[str, dict]] = []


def build_server(calls: list[tuple[str, dict]] | None = None) -> MCPServer:
    calls = CALLS if calls is None else calls
    server = MCPServer("fake-workspace")

    @server.tool()
    def get_events(user_google_email: str, time_min: str | None = None, max_results: int = 25) -> str:
        """Retrieves events from a calendar."""
        calls.append(("get_events", {"user_google_email": user_google_email, "time_min": time_min}))
        lines = [f"Successfully retrieved 12 events for {user_google_email}:"]
        for i in range(12):
            lines.append(
                f'- "Meeting {i}" (Starts: 2026-11-2{i % 9}T1{i % 10}:00:00-08:00) ID: evt{i}abc | '
                f"Link: https://calendar.google.com/event?eid={i}"
            )
        return "\n".join(lines)

    @server.tool()
    def find_tasks(limit: int = 10) -> dict:
        """Find tasks."""
        calls.append(("find_tasks", {"limit": limit}))
        return {
            "tasks": [
                {"id": f"t{i}", "content": f"Task {i}", "dueString": "today", "priority": "p2", "url": "https://x"}
                for i in range(8)
            ],
            "totalCount": 8,
        }

    @server.tool()
    def add_event(user_google_email: str, action: str, summary: str, start_time: str) -> str:
        """Create an event."""
        calls.append(("add_event", {"action": action, "summary": summary, "start_time": start_time}))
        return f"Created event '{summary}' at {start_time}. Link: https://calendar.google.com/x"

    @server.tool()
    def add_tasks(tasks: list[dict]) -> str:
        """Add tasks."""
        calls.append(("add_tasks", {"tasks": tasks}))
        return f"Added {len(tasks)} task(s)."

    @server.tool()
    def delete_everything() -> str:
        """Not allowlisted: must never be exposed."""
        calls.append(("delete_everything", {}))
        return "deleted"

    @server.tool()
    async def slow_search(query: str) -> str:
        """Takes too long."""
        calls.append(("slow_search", {"query": query}))
        await asyncio.sleep(5)
        return "too late"

    @server.tool()
    def needs_auth(query: str) -> str:
        """Pretends Google sign-in is missing."""
        return (
            "ACTION REQUIRED: Google Authentication Needed. Authorization URL: "
            "https://accounts.google.com/o/oauth2/auth?client_id=abc&redirect_uri=http://localhost:8000/oauth2callback"
        )

    return server


if __name__ == "__main__":
    build_server().run("stdio")
