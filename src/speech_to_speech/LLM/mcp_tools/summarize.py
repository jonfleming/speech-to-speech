"""Turn MCP tool results into short, speakable text for the model.

Voice replies should mention a handful of items, not dump a JSON page into the
context. Results are cut to the top ``max_items`` entries with only their
essential fields (title, time, sender, ...), IDs and links are dropped, and the
whole thing is clipped to ``max_chars``.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

AUTH_NEEDED = (
    "{label} is not signed in yet, so nothing could be read. Tell the user that {label} needs a one-time "
    "sign-in on the server (the link is in the speech-to-speech log)."
)
_AUTH_MARKERS = (
    "accounts.google.com/o/oauth2",
    "authorization url",
    "start_google_auth",
    "please authenticate",
    "authentication required",
    "needs authorization",
)

_LIST_KEYS = ("events", "items", "tasks", "messages", "results", "threads", "emails", "data", "entries")
_TITLE_KEYS = ("summary", "title", "content", "subject", "name", "text")
_TIME_KEYS = (
    "start",
    "startTime",
    "start_time",
    "dueString",
    "due",
    "date",
    "startDate",
    "deadline",
    "time",
    "receivedAt",
    "internalDate",
)
_WHO_KEYS = ("from", "sender", "organizer", "author")
_EXTRA_KEYS = ("location", "priority", "snippet")
_COUNT_KEYS = ("totalCount", "total", "count", "resultSizeEstimate")

_URL_RE = re.compile(r"https?://\S+")
_NOISE_LINE_RE = re.compile(
    r"^\s*[-*•]?\s*(message id|thread id|event id|id|link|web link|url|html link|label ids?|labels?|"
    r"next_?page_?token|page token|etag|icaluid|calendar id|attachment id)\s*[:=]",
    re.IGNORECASE,
)
_INLINE_NOISE_RE = re.compile(
    r"\s*[|,(]?\s*\b(message id|thread id|event id|ID|Link|Web link|URL)\s*:\s*\S+\)?", re.IGNORECASE
)
_ERROR_NOISE_RE = re.compile(r"^(details|event_id|request id|trace|retry_after)\b", re.IGNORECASE)
_ENTRY_START_RE = re.compile(r"^(\d+[.)]\s|[-*•]\s|(message|event|task|email)\s*#?\d+\b)", re.IGNORECASE)


def clip(text: str, max_chars: int) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    cut = text[: max_chars - 1]
    space = cut.rfind(" ")
    if space > max_chars * 0.6:
        cut = cut[:space]
    return cut.rstrip(" ,;:-") + "…"


def _scalar(value: Any) -> str | None:
    if value is None or value == "" or value == [] or value == {}:
        return None
    if isinstance(value, dict):
        for key in ("dateTime", "date", "datetime", "string", "name", "email", "value", "displayName"):
            if value.get(key):
                return _scalar(value[key])
        return None
    if isinstance(value, list):
        parts = [p for p in (_scalar(v) for v in value[:3]) if p]
        return ", ".join(parts) or None
    if isinstance(value, bool):
        return "yes" if value else "no"
    text = _URL_RE.sub("", str(value)).strip()
    return " ".join(text.split()) or None


def _first(item: dict[str, Any], keys: tuple[str, ...]) -> tuple[str | None, str | None]:
    for key in keys:
        if key in item:
            value = _scalar(item[key])
            if value:
                return key, value
    return None, None


def _render_item(item: Any) -> str:
    if not isinstance(item, dict):
        return clip(_scalar(item) or "", 160)
    parts: list[str] = []
    _, title = _first(item, _TITLE_KEYS)
    if title:
        parts.append(clip(title, 120))
    time_key, when = _first(item, _TIME_KEYS)
    if when:
        parts.append(when if time_key in ("start", "startTime", "start_time", "date") else f"due {when}")
    _, who = _first(item, _WHO_KEYS)
    if who:
        parts.append(f"from {clip(who, 60)}")
    for key in _EXTRA_KEYS:
        value = _scalar(item.get(key))
        if value:
            parts.append(f"{key} {clip(value, 80)}")
    if not parts:
        simple = [
            f"{k} {clip(v, 60)}" for k, v in ((k, _scalar(v)) for k, v in item.items()) if v and "id" not in k.lower()
        ]
        parts = simple[:3]
    return " — ".join(parts)


def _find_list(data: Any) -> tuple[list[Any] | None, int | None]:
    if isinstance(data, list):
        return data, len(data)
    if not isinstance(data, dict):
        return None, None
    total = next((data[k] for k in _COUNT_KEYS if isinstance(data.get(k), int)), None)
    for key in _LIST_KEYS:
        if isinstance(data.get(key), list):
            return data[key], total if total is not None else len(data[key])
    for value in data.values():
        if isinstance(value, list) and value and all(isinstance(v, dict) for v in value):
            return value, total if total is not None else len(value)
    return None, total


def summarize_structured(data: Any, *, max_items: int, max_chars: int) -> str | None:
    if isinstance(data, dict) and set(data) == {"result"}:
        inner = data["result"]
        if isinstance(inner, str):
            return None
        data = inner
    items, total = _find_list(data)
    if items is not None:
        if not items:
            return "No results."
        shown = items[:max_items]
        lines = [f"{i}. {_render_item(item)}" for i, item in enumerate(shown, 1)]
        count = total if isinstance(total, int) and total >= len(items) else len(items)
        header = f"{count} found, top {len(shown)}:" if count > len(shown) else f"{count} found:"
        return clip("\n".join([header, *lines]), max_chars)
    if isinstance(data, dict):
        pairs = [f"{k}: {v}" for k, v in ((k, _scalar(v)) for k, v in data.items()) if v and "id" not in k.lower()]
        if pairs:
            return clip("; ".join(pairs[:8]), max_chars)
    return None


def summarize_text(text: str, *, max_items: int, max_chars: int) -> str:
    entries: list[list[str]] = []
    header: list[str] = []
    for raw in text.splitlines():
        if not raw.strip() or _NOISE_LINE_RE.match(raw):
            continue
        line = _INLINE_NOISE_RE.sub("", _URL_RE.sub("", raw)).rstrip(" |,;")
        stripped = line.strip()
        if not stripped:
            continue
        indented = line[:1] in (" ", "\t")
        if _ENTRY_START_RE.match(stripped) and not indented:
            entries.append([stripped])
        elif entries and indented:
            entries[-1].append(stripped)
        elif entries:
            entries.append([stripped])
        else:
            header.append(stripped)
    if not entries:
        return clip(" ".join(header), max_chars)
    shown = entries[:max_items]
    lines = [" ".join(header[:1])] if header else []
    if len(entries) > len(shown):
        lines.append(f"(showing top {len(shown)} of {len(entries)})")
    lines += ["; ".join(entry) for entry in shown]
    return clip("\n".join(line for line in lines if line), max_chars)


def looks_like_auth_prompt(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in _AUTH_MARKERS)


def result_text(result: Any) -> str:
    parts: list[str] = []
    for block in getattr(result, "content", None) or []:
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
    return "\n".join(parts)


def summarize_result(result: Any, *, label: str, max_items: int = 5, max_chars: int = 900) -> str:
    """Summarize an MCP ``CallToolResult`` (or anything with the same fields)."""
    text = result_text(result)
    if looks_like_auth_prompt(text):
        logger.warning("MCP server %s needs sign-in. Complete this in a browser on the server:\n%s", label, text)
        return AUTH_NEEDED.format(label=label)
    if getattr(result, "is_error", False):
        lines = [ln.strip() for ln in _URL_RE.sub("", text).splitlines() if ln.strip()]
        lines = [ln for ln in lines if not _ERROR_NOISE_RE.match(ln)]
        return f"{label} returned an error: {clip(' '.join(lines) or 'unknown error', 240)}"
    structured = getattr(result, "structured_content", None)
    if structured is not None:
        summary = summarize_structured(structured, max_items=max_items, max_chars=max_chars)
        if summary:
            return summary
    stripped = text.strip()
    if stripped[:1] in ("{", "["):
        try:
            summary = summarize_structured(json.loads(stripped), max_items=max_items, max_chars=max_chars)
        except json.JSONDecodeError:
            summary = None
        if summary:
            return summary
    if not stripped:
        return "Done, but the service returned no details."
    return summarize_text(stripped, max_items=max_items, max_chars=max_chars)
