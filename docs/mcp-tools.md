# Server-side MCP tools

speech-to-speech can call tools from MCP servers itself, inside the LLM tool loop,
so voice clients that don't run tools (the ESP32 watch, Reachy Mini, AIPI-Lite)
can still ask "what's on my calendar?" or "add milk to my list".

```
user speech -> STT -> LLM handler ──request──> LLM (via obsidian-memory)
                          │  <── tool call: list_calendar_events
                          ├── MCP client ──> workspace-mcp (Calendar, Gmail)   stdio
                          │              └─> todoist-mcp (Todoist)            stdio
                          │  ──request + trimmed tool result──> LLM
                          └── spoken answer -> TTS
```

## How it works

- **Discovery.** At startup each configured server is launched (stdio) or
  connected (streamable HTTP), and its tools come from `list_tools`. Only tools
  named under that server's `tools` key are shown to the model (allowlist).
  Schemas are trimmed (`expose_params`, `fixed_args`) so a small local model
  sees 1-5 parameters instead of 40.
- **Tool loop.** `BaseOpenAICompatibleHandler` (both `chat-completions` and
  `responses-api` backends) merges MCP tools with any tools the client sent in
  `session.update`. When the model calls an MCP tool, the server runs it, adds
  the result to the request, and asks the model again (up to
  `max_tool_rounds`). Client tools (e.g. the watch's Hindsight `recall`) are
  still forwarded to the client exactly as before. MCP calls and results are
  not written to conversation history; only the spoken answer is. Follow-up
  rounds drop the obsidian-memory callback headers so the memory proxy doesn't
  post a second follow-up for the same turn.
- **Voice-sized results.** Results are cut to the top `max_items` (default 5)
  entries with essentials only (title, time, sender, due date). IDs and links
  are dropped and the text is clipped to `max_result_chars`.
- **Timeouts.** Each call has `call_timeout_s` (default 8 s). On timeout the
  model is told it couldn't reach the service and says so.
- **Writes need a spoken yes.** Tools with `"kind": "write"` never run when the
  model calls them. The call is parked for that conversation and the model gets
  a read-back to say ("Should I add "Dentist" to your calendar on Saturday,
  November 21 at 3:00 PM? Say yes to confirm."). The **next** utterance decides:
  a clear yes ("yes", "yeah", "sure, go ahead") runs it and the server says
  "Done, it's saved." without another LLM call; "no"/"cancel" gets "Okay, I won't do
  that."; anything else (including "yes, but at 4") drops it and the
  conversation continues normally. A parked call expires after `confirm_ttl_s`
  (180 s) and is only confirmable after the read-back was actually delivered.
- **Graceful startup.** Missing config, a missing `${SECRET}`, a server that
  won't start, or a missing allowlisted tool is logged and skipped. The voice
  server always starts; a failed server is retried in the background.
- The current date/time and a few voice rules are appended to the system
  prompt only while MCP tools are available.

Not covered: the local `transformers`/MLX LLM backends and `--stt none`
audio-input turns (no transcript to check for "yes").

## Enable it

```bat
copy mcp_servers.example.json mcp_servers.json
copy .env.example .env
rem fill in .env, then add the flag to the serve command (e.g. in speech.cmd):
speech-to-speech serve ... --mcp_config C:\Projects\ai-voice-assistant\speech-to-speech\mcp_servers.json
```

`S2S_MCP_CONFIG` works instead of `--mcp_config`. `${VAR}` values are read from
the environment first, then `.env` in the working directory and next to the
config file. In PowerShell on Omen, call `uv.exe` (the profile defines a `uv`
function that runs uvicorn).

Check servers without starting the pipeline:

```bat
.venv\Scripts\python -m speech_to_speech.LLM.mcp_tools --config mcp_servers.json list
.venv\Scripts\python -m speech_to_speech.LLM.mcp_tools --config mcp_servers.json call list_tasks "{}"
.venv\Scripts\python -m speech_to_speech.LLM.mcp_tools --config mcp_servers.json call list_calendar_events "{}"
```

`call` prints exactly what the model would get. It runs write tools without the
confirmation gate, so don't use it on `add_*` unless you want the item created.

## Config reference

```jsonc
{
  "defaults": {
    "call_timeout_s": 8,      // per tool call
    "startup_wait_s": 20,     // how long serve waits for servers before going on
    "max_items": 5,           // items kept per result
    "max_result_chars": 900,
    "max_tool_rounds": 3,     // model -> tool -> model rounds per turn
    "confirm_ttl_s": 180      // parked write lifetime
  },
  "servers": {
    "<name>": {
      "label": "Todoist",                 // used in spoken errors
      "transport": "stdio",               // or "streamable_http"
      "command": "npx", "args": [],       // stdio
      "env": {"KEY": "${KEY}"},           // added to a minimal default env
      "inherit_env": false,               // true = pass the whole parent env
      "url": "http://127.0.0.1:8000/mcp", // streamable_http
      "headers": {"Authorization": "Bearer ${TOKEN}"},
      "enabled": true,
      "tools": {                          // the allowlist
        "<mcp tool name>": {
          "expose_as": "list_tasks",      // name the model sees
          "kind": "read",                 // or "write" (spoken confirmation)
          "description": "...",           // override the server's text
          "fixed_args": {},               // always sent, hidden from the model
          "default_args": {},             // sent if missing, hidden
          "expose_params": ["a", "b"],    // keep only these parameters
          "parameters": {},               // or replace the schema entirely
          "wrap_in_list": "tasks",        // send {"tasks": [model_args]}
          "confirm_template": "add the task \"{content}\" {dueString}",
          "max_items": 5, "timeout_s": 8
        }
      }
    }
  }
}
```

## Chosen servers

| Need | Server | Why |
| --- | --- | --- |
| Calendar list/add, Gmail search | [`workspace-mcp`](https://github.com/taylorwilsdon/google_workspace_mcp) 2.6.0 via `uvx`, stdio, `--single-user` | Most complete maintained Google Workspace MCP server; one process covers Calendar and Gmail; `--permissions gmail:readonly calendar:full` asks for read-only Gmail (no send) plus calendar write; `search_gmail_messages` returns subject/sender in one call (`include_headers`); local OAuth with your own Desktop client, token stays on Omen. |
| Tasks list/add | Official Todoist MCP [`@todoist/mcp`](https://github.com/Doist/todoist-ai) 17.2.0 via `npx`, stdio | Maintained by Doist (formerly `@doist/todoist-ai`); only needs an API token; `find-tasks-by-date` matches "what's due today", `add-tasks` takes natural-language due dates. |

Both run as child processes of speech-to-speech over stdio, so there are no new
listening ports except workspace-mcp's OAuth callback on `localhost:8000`
while a Google sign-in is pending.

## One-time setup

### Google (Calendar + Gmail)

1. In [Google Cloud Console](https://console.cloud.google.com/) create or pick a project.
2. APIs & Services > Library: enable **Google Calendar API** and **Gmail API**.
3. OAuth consent screen (Google Auth Platform): audience **External**, app name
   e.g. "Voice Assistant", add your Gmail address as a **Test user**.
4. Clients (Credentials) > Create > **OAuth client ID** > type **Desktop app**.
   Put the client ID and secret in `.env` (`GOOGLE_OAUTH_CLIENT_ID`,
   `GOOGLE_OAUTH_CLIENT_SECRET`) and set `GOOGLE_USER_EMAIL`.
5. First sign-in, on Omen:
   `.venv\Scripts\python -m speech_to_speech.LLM.mcp_tools --config mcp_servers.json call list_calendar_events "{}"`.
   The log prints a Google authorization link. Open it on Omen, approve, and
   the browser lands on `http://localhost:8000/oauth2callback`. Press Enter in
   the terminal to retry; you should see your next events. Repeat with
   `call search_mail "{\"query\": \"is:unread\"}"` if Gmail asks separately.
   The token is stored under `%USERPROFILE%\.google_workspace_mcp\credentials`.
6. While the consent screen is in **Testing**, Google expires the refresh token
   after 7 days. When the assistant says Google needs sign-in, repeat step 5, or
   move the app to **In production** (for your own account you click through an
   "unverified app" warning once).

### Todoist

1. Todoist web > Settings > Integrations > Developer > copy the **API token**.
2. Put it in `.env` as `TODOIST_API_KEY`.
3. Check: `... call list_tasks "{}"`.

## Requirements on the host

- Python package `mcp` (in `pyproject.toml`; `uv sync` installs it).
- `uv`/`uvx` for workspace-mcp (uvx fetches Python >= 3.11 for it).
- Node.js with `npx` for the Todoist server.
- An LLM with tool calling (llama.cpp: chat template / `--jinja`, the default in
  current builds).
