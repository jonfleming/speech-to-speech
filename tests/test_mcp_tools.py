"""Tests for server-side MCP tools (src/speech_to_speech/LLM/mcp_tools).

A fake MCP server (tests/mcp_fake_server.py) is used in-memory for most tests
and once over real stdio, so no network, Google, or Todoist access is needed.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from mcp import Client

from speech_to_speech.LLM.mcp_tools import config as cfg
from speech_to_speech.LLM.mcp_tools import runtime as rt
from speech_to_speech.LLM.mcp_tools.confirm import is_affirmative, is_negative
from speech_to_speech.LLM.mcp_tools.runtime import CANCELLED_TEXT, MCPToolRuntime, build_call_args
from speech_to_speech.LLM.mcp_tools.summarize import AUTH_NEEDED, summarize_result
from tests.mcp_fake_server import build_server

FAKE_SCRIPT = str(Path(__file__).with_name("mcp_fake_server.py"))


def _config_dict(**server_overrides):
    server = {
        "transport": "stdio",
        "command": "unused",
        "label": "Google",
        "tools": {
            "get_events": {
                "expose_as": "list_calendar_events",
                "fixed_args": {"user_google_email": "${GOOGLE_USER_EMAIL}"},
                "expose_params": ["time_min"],
            },
            "find_tasks": {"expose_as": "list_tasks"},
            "add_event": {
                "expose_as": "add_calendar_event",
                "kind": "write",
                "fixed_args": {"user_google_email": "${GOOGLE_USER_EMAIL}", "action": "create"},
                "confirm_template": 'add "{summary}" on {start_time}',
            },
            "add_tasks": {
                "expose_as": "add_task",
                "kind": "write",
                "wrap_in_list": "tasks",
                "parameters": {
                    "type": "object",
                    "properties": {"content": {"type": "string"}, "dueString": {"type": "string"}},
                    "required": ["content"],
                },
                "confirm_template": 'add the task "{content}" due {dueString}',
            },
            "slow_search": {"expose_as": "search_mail", "timeout_s": 0.5},
            "needs_auth": {},
            "not_on_server": {},
        },
    }
    server.update(server_overrides)
    return {"defaults": {"call_timeout_s": 3, "startup_wait_s": 10}, "servers": {"fake": server}}


@pytest.fixture
def calls():
    return []


@pytest.fixture
def runtime(calls):
    config = cfg.parse_config(_config_dict(), env={"GOOGLE_USER_EMAIL": "jon@example.com"})
    server = build_server(calls)
    rt_ = MCPToolRuntime(config, client_factory=lambda spec: Client(server)).start()
    yield rt_
    rt_.close()


def _session_with_readback(runtime, tool, args, key="s1"):
    """Park a write the way the tool loop does, then pretend the read-back was spoken."""
    prompt = runtime.execute(tool, json.dumps(args), session_key=key, assistant_ids={"a_old"})
    return prompt, {"a_old", "a_readback"}


# ── config ───────────────────────────────────────────────────────────────────


def test_config_expands_env_and_skips_server_with_missing_secret(caplog):
    data = {
        "servers": {
            "todoist": {
                "command": "npx",
                "args": ["-y", "@todoist/mcp"],
                "env": {"TODOIST_API_KEY": "${TODOIST_API_KEY}"},
                "tools": ["find-tasks-by-date"],
            },
            "web": {"url": "http://127.0.0.1:9/mcp", "headers": {"Authorization": "Bearer ${WEB_TOKEN}"}, "tools": {}},
        }
    }
    config = cfg.parse_config(data, env={"WEB_TOKEN": "t0k"})
    todoist, web = config.servers
    assert todoist.transport == "stdio" and todoist.missing_env == ["TODOIST_API_KEY"]
    assert list(todoist.tools) == ["find-tasks-by-date"]
    assert todoist.tools["find-tasks-by-date"].expose_as == "find-tasks-by-date"
    assert web.transport == "streamable_http" and web.headers["Authorization"] == "Bearer t0k"

    runtime = MCPToolRuntime(config).start(wait_s=0)
    assert runtime.tools() == []
    assert "missing TODOIST_API_KEY" in caplog.text
    runtime.close()


def test_config_rejects_bad_kind():
    with pytest.raises(cfg.MCPConfigError):
        cfg.parse_config({"servers": {"x": {"command": "c", "tools": {"t": {"kind": "delete"}}}}}, env={})


def test_load_config_reads_dotenv_next_to_config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("FAKE_TOKEN_FOR_TEST", raising=False)
    (tmp_path / ".env").write_text('FAKE_TOKEN_FOR_TEST="abc"\n# comment\n', encoding="utf-8")
    path = tmp_path / "mcp_servers.json"
    path.write_text(
        json.dumps({"servers": {"s": {"command": "c", "env": {"K": "${FAKE_TOKEN_FOR_TEST}"}, "tools": ["t"]}}})
    )
    config = cfg.load_config(str(path))
    assert config.servers[0].env == {"K": "abc"} and not config.servers[0].missing_env


# ── discovery / allowlist ────────────────────────────────────────────────────


def test_discovery_exposes_only_allowlisted_tools_with_trimmed_schemas(runtime, caplog):
    names = sorted(t["name"] for t in runtime.tool_definitions())
    assert names == [
        "add_calendar_event",
        "add_task",
        "list_calendar_events",
        "list_tasks",
        "needs_auth",
        "search_mail",
    ]
    assert not runtime.owns("delete_everything")
    # Allowlisted but missing on the server: logged at startup, skipped.
    assert any("not_on_server" in r.getMessage() for r in caplog.get_records("setup"))

    defs = {t["name"]: t for t in runtime.tool_definitions()}
    events = defs["list_calendar_events"]["parameters"]
    assert set(events["properties"]) == {"time_min"}  # fixed + non-exposed params hidden
    assert "required" not in events
    add = defs["add_calendar_event"]
    assert set(add["parameters"]["properties"]) == {"summary", "start_time"}
    assert "confirm out loud" in add["description"]
    assert defs["add_task"]["parameters"]["required"] == ["content"]


def test_build_call_args_pins_fixed_and_wraps():
    spec = cfg.ToolSpec(mcp_name="add_tasks", expose_as="add_task", wrap_in_list="tasks")
    assert build_call_args(spec, {"content": "Call mom", "dueString": None}) == {"tasks": [{"content": "Call mom"}]}
    spec = cfg.ToolSpec(mcp_name="x", expose_as="x", fixed_args={"action": "create"}, default_args={"n": 5})
    assert build_call_args(spec, {"action": "delete", "q": 1}) == {"n": 5, "q": 1, "action": "create"}


# ── trimming ─────────────────────────────────────────────────────────────────


def test_read_tool_result_is_trimmed_for_voice(runtime, calls):
    text = runtime.execute("list_calendar_events", "{}", session_key="s")
    assert calls == [("get_events", {"user_google_email": "jon@example.com", "time_min": None})]
    assert "Meeting 0" in text and "Meeting 4" in text and "Meeting 5" not in text
    assert "top 5 of 12" in text
    assert "http" not in text and "evt0abc" not in text

    tasks = runtime.execute("list_tasks", "{}", session_key="s")
    assert tasks.splitlines()[0] == "8 found, top 5:"
    assert "1. Task 0 — due today — priority p2" in tasks
    assert "Task 5" not in tasks and "https" not in tasks


def test_summarize_structured_and_error_results():
    result = SimpleNamespace(
        content=[SimpleNamespace(text="ignored")],
        structured_content={"events": [{"summary": "Dentist", "start": {"dateTime": "2026-11-21T15:00:00-08:00"}}]},
        is_error=False,
    )
    assert summarize_result(result, label="Google") == "1 found:\n1. Dentist — 2026-11-21T15:00:00-08:00"
    error = SimpleNamespace(
        content=[
            SimpleNamespace(text="Request failed (HTTP 401).\nMessage: Unauthorized https://x\nDetails: event_id: ab12")
        ],
        structured_content=None,
        is_error=True,
    )
    assert (
        summarize_result(error, label="Todoist")
        == "Todoist returned an error: Request failed (HTTP 401). Message: Unauthorized"
    )
    long_text = SimpleNamespace(content=[SimpleNamespace(text="word " * 500)], structured_content=None, is_error=False)
    assert len(summarize_result(long_text, label="X", max_chars=200)) <= 200


def test_auth_prompt_becomes_short_spoken_hint(runtime, caplog):
    text = runtime.execute("needs_auth", '{"query": "x"}', session_key="s")
    assert text == AUTH_NEEDED.format(label="Google")
    assert "accounts.google.com" in caplog.text  # full link goes to the log, not to the model


# ── timeout ──────────────────────────────────────────────────────────────────


def test_slow_tool_times_out_quickly(runtime):
    started = time.monotonic()
    text = runtime.execute("search_mail", '{"query": "invoice"}', session_key="s")
    assert time.monotonic() - started < 2.5
    assert "did not answer within 0.5 seconds" in text and "couldn't reach Google" in text
    # The server stays usable after a timed-out call.
    assert "found" in runtime.execute("list_tasks", "{}", session_key="s")


# ── confirmation gate ────────────────────────────────────────────────────────


def test_write_tool_is_parked_until_spoken_yes(runtime, calls):
    prompt, ids = _session_with_readback(
        runtime, "add_calendar_event", {"summary": "Dentist", "start_time": "2026-11-21T15:00:00-08:00"}
    )
    assert "NOT DONE YET" in prompt and 'Should I add "Dentist" on Saturday, November 21 at' in prompt
    assert calls == []  # nothing executed yet
    spoken = runtime.resolve_pending("s1", "Yes, please.", ids)
    assert spoken == "Done, it's saved."
    assert calls == [
        ("add_event", {"action": "create", "summary": "Dentist", "start_time": "2026-11-21T15:00:00-08:00"})
    ]
    # Consumed: a second "yes" does nothing.
    assert runtime.resolve_pending("s1", "yes", ids) is None
    assert len(calls) == 1


def test_write_tool_declined_or_ignored_is_dropped(runtime, calls):
    _, ids = _session_with_readback(runtime, "add_task", {"content": "Call mom", "dueString": "tomorrow"})
    assert runtime.resolve_pending("s1", "No, don't.", ids) == CANCELLED_TEXT
    _, ids = _session_with_readback(runtime, "add_task", {"content": "Call mom", "dueString": "tomorrow"})
    assert runtime.resolve_pending("s1", "What's the weather like?", ids) is None
    assert not runtime.has_pending("s1")
    assert calls == []


def test_write_needs_delivered_readback_and_expires(runtime, calls):
    runtime.execute("add_task", '{"content": "A"}', session_key="s1", assistant_ids={"a_old"})
    assert runtime.resolve_pending("s1", "yes", {"a_old"}) is None  # read-back never committed
    runtime.pending.ttl_s = -1.0  # already expired (Windows monotonic clock is coarse)
    _, ids = _session_with_readback(runtime, "add_task", {"content": "B"})
    assert runtime.resolve_pending("s1", "yes", ids) is None
    assert calls == []


def test_confirmed_task_is_wrapped_for_todoist(runtime, calls):
    _, ids = _session_with_readback(runtime, "add_task", {"content": "Buy milk", "dueString": "tomorrow"})
    assert runtime.resolve_pending("s1", "yep", ids) == "Done, it's saved."
    assert calls == [("add_tasks", {"tasks": [{"content": "Buy milk", "dueString": "tomorrow"}]})]


@pytest.mark.parametrize("text", ["yes", "Yeah.", "uh, yes please", "Sure, go ahead", "okay", "do it", "Yep!"])
def test_affirmatives(text):
    assert is_affirmative(text)


@pytest.mark.parametrize(
    "text",
    [
        "no",
        "yes but not at three",
        "actually make it four",
        "wait",
        "",
        None,
        "what time is it",
        "yes I think we should also talk about the budget for next quarter",
    ],
)
def test_not_affirmative(text):
    assert not is_affirmative(text)


def test_negatives():
    assert is_negative("no thanks") and is_negative("cancel that") and not is_negative("tell me a joke please now ok")


# ── graceful startup ─────────────────────────────────────────────────────────


def test_failed_server_does_not_block_others(calls, caplog):
    data = _config_dict()
    data["servers"]["broken"] = {"command": "definitely-not-a-real-mcp-binary", "tools": ["x"]}
    config = cfg.parse_config(data, env={"GOOGLE_USER_EMAIL": "jon@example.com"})
    server = build_server(calls)

    def factory(spec):
        if spec.name == "broken":
            return rt._default_client_factory(spec)
        return Client(server)

    runtime = MCPToolRuntime(config, client_factory=factory).start(wait_s=10)
    try:
        assert runtime.owns("list_tasks")
        status = runtime.status()
        assert status["broken"]["connected"] is False and status["broken"]["error"]
        assert "MCP server broken unavailable" in caplog.text
    finally:
        runtime.close()


def test_get_runtime_without_or_with_bad_config(tmp_path, caplog):
    assert rt.get_runtime(None) is None
    assert rt.get_runtime(str(tmp_path / "missing.json")) is None
    assert "MCP tools disabled" in caplog.text


def test_real_stdio_server_round_trip():
    data = {
        "defaults": {"startup_wait_s": 60},
        "servers": {
            "fake": {"command": sys.executable, "args": [FAKE_SCRIPT], "label": "Fake", "tools": ["find_tasks"]}
        },
    }
    runtime = MCPToolRuntime(cfg.parse_config(data, env={})).start()
    try:
        assert runtime.owns("find_tasks")
        assert runtime.execute("find_tasks", '{"limit": 3}', session_key="s").startswith("8 found, top 5:")
    finally:
        runtime.close()
