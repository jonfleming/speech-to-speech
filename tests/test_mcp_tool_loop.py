"""The LLM handler runs MCP tools server-side inside its tool-calling loop."""

from __future__ import annotations

import json

import pytest
from mcp import Client

from speech_to_speech.LLM.chat import Chat
from speech_to_speech.LLM.mcp_tools import config as cfg
from speech_to_speech.LLM.mcp_tools.runtime import MCPToolRuntime
from tests.mcp_fake_server import build_server
from tests.test_chat_completions_backend import _chunk, _drive, _FakeStream, _make_handler, _tc_delta

CONFIG = {
    "defaults": {"call_timeout_s": 3},
    "servers": {
        "fake": {
            "command": "unused",
            "label": "Todoist",
            "tools": {
                "find_tasks": {"expose_as": "list_tasks"},
                "add_tasks": {
                    "expose_as": "add_task",
                    "kind": "write",
                    "wrap_in_list": "tasks",
                    "parameters": {"type": "object", "properties": {"content": {"type": "string"}}},
                    "confirm_template": 'add the task "{content}"',
                },
            },
        }
    },
}


@pytest.fixture
def setup():
    calls: list = []
    server = build_server(calls)
    runtime = MCPToolRuntime(cfg.parse_config(CONFIG, env={}), client_factory=lambda spec: Client(server)).start()
    handler = _make_handler(stream=True)
    handler.mcp = runtime
    requests: list[dict] = []
    yield handler, calls, requests
    runtime.close()


def _scripted(requests, responses):
    def create(**kwargs):
        requests.append(json.loads(json.dumps(kwargs, default=str)))
        return _FakeStream(responses.pop(0))

    return create


def test_mcp_tool_call_runs_server_side_and_model_answers(setup):
    handler, calls, requests = setup
    handler.client.chat.completions.create = _scripted(
        requests,
        [
            [_chunk(tool_calls=[_tc_delta(0, id="x", name="list_tasks", arguments="{}")])],
            [_chunk(content="You have eight tasks. First is Task 0.")],
        ],
    )
    text, tools_out, _usage, chat, end = _drive(handler, user="What's on my list?")

    assert text == "You have eight tasks. First is Task 0."
    assert tools_out == []  # never forwarded to the realtime client / ESP32
    assert calls == [("find_tasks", {"limit": 10})]
    assert end.error is None
    first, second = requests
    assert {t["function"]["name"] for t in first["tools"]} == {"list_tasks", "add_task"}
    assert "Current date and time" in first["messages"][0]["content"]
    tool_msgs = [m for m in second["messages"] if m["role"] == "tool"]
    assert len(tool_msgs) == 1 and tool_msgs[0]["content"].startswith("8 found, top 5:")
    # Only the spoken answer is kept in history, no dangling tool calls.
    assert not chat._pending_tool_calls
    assert [i.role for i in chat.buffer] == ["user", "assistant"]


def test_client_tools_still_forwarded_alongside_mcp(setup):
    handler, calls, requests = setup
    handler.client.chat.completions.create = _scripted(
        requests, [[_chunk(tool_calls=[_tc_delta(0, id="x", name="recall", arguments='{"q": "car"}')])]]
    )
    client_tool = {"type": "function", "name": "recall", "parameters": {"type": "object"}}
    _text, tools_out, *_ = _drive(handler, tools=[client_tool])
    assert [t.name for t in tools_out] == ["recall"]
    assert [t["function"]["name"] for t in requests[0]["tools"]] == ["recall", "list_tasks", "add_task"]
    assert calls == []


def test_write_tool_waits_for_spoken_yes_on_next_turn(setup):
    handler, calls, requests = setup
    handler.client.chat.completions.create = _scripted(
        requests,
        [
            [_chunk(tool_calls=[_tc_delta(0, id="x", name="add_task", arguments='{"content": "Buy milk"}')])],
            [_chunk(content='Should I add the task "Buy milk"? Say yes to confirm.')],
        ],
    )
    chat = Chat(10)
    text, *_ = _drive(handler, user="Add buy milk to my tasks", chat=chat)
    assert text.startswith("Should I add the task")
    assert calls == []
    assert "NOT DONE YET" in [m for m in requests[1]["messages"] if m["role"] == "tool"][0]["content"]

    # Next utterance: "yes" executes the parked call without another LLM request.
    n_requests = len(requests)
    text, *_ = _drive(handler, user="Yes please", chat=chat)
    assert text == "Done, it's saved."
    assert len(requests) == n_requests
    assert calls == [("add_tasks", {"tasks": [{"content": "Buy milk"}]})]
    assert chat.buffer[-1].content[0].text == text


def test_write_tool_dropped_when_user_says_something_else(setup):
    handler, calls, requests = setup
    handler.client.chat.completions.create = _scripted(
        requests,
        [
            [_chunk(tool_calls=[_tc_delta(0, id="x", name="add_task", arguments='{"content": "Buy milk"}')])],
            [_chunk(content="Should I add it? Say yes to confirm.")],
            [_chunk(content="It's sunny.")],
        ],
    )
    chat = Chat(10)
    _drive(handler, user="Add buy milk", chat=chat)
    text, *_ = _drive(handler, user="What's the weather?", chat=chat)
    assert text == "It's sunny."
    assert calls == []
    assert not handler.mcp.has_pending("chat-%d" % id(chat))


def test_tool_choice_none_disables_mcp_tools(setup):
    handler, _calls, requests = setup
    handler.client.chat.completions.create = _scripted(requests, [[_chunk(content="Hi.")]])
    _drive(handler, tool_choice="none")
    assert "tools" not in requests[0]
