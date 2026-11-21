"""Spoken confirmation gate for MCP write tools.

When the model calls a write tool (add event, add task), nothing is executed.
The call is parked here per conversation, and the model gets a read-back to
speak. The next user utterance decides: an affirmative runs the parked call,
anything else drops it.
"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any

_AFFIRMATIVE = re.compile(
    r"^(yes|yeah|yep|yup|ya|yah|sure|ok|okay|confirm(ed)?|correct|right|affirmative|absolutely|definitely|"
    r"do it|go ahead|go for it|please do|sounds good|that's right|that is right|that's correct|"
    r"perfect|great|add it|save it|book it|make it so)\b",
    re.IGNORECASE,
)
_NEGATIVE = re.compile(
    r"\b(no|nope|nah|don't|do not|cancel|stop|wait|hold on|never ?mind|not|wrong|change|instead|actually)\b",
    re.IGNORECASE,
)
_FILLER = re.compile(r"^(uh+|um+|oh|well|hmm+|so|and)[\s,.!]+", re.IGNORECASE)


def _normalize(text: str) -> str:
    text = text.strip().strip("\"'").replace("’", "'")
    previous = None
    while previous != text:
        previous = text
        text = _FILLER.sub("", text).strip()
    return text


def is_affirmative(text: str | None) -> bool:
    """True only for a clear yes; any negation or correction means no."""
    if not text:
        return False
    normalized = _normalize(text)
    if not normalized or len(normalized.split()) > 8:
        return False
    if _NEGATIVE.search(normalized):
        return False
    return bool(_AFFIRMATIVE.match(normalized))


def is_negative(text: str | None) -> bool:
    if not text:
        return False
    normalized = _normalize(text)
    return bool(normalized) and len(normalized.split()) <= 6 and bool(_NEGATIVE.search(normalized))


@dataclass
class PendingAction:
    tool_name: str
    arguments: dict[str, Any]
    description: str
    created_at: float = field(default_factory=time.monotonic)
    # Assistant items already in history when the call was parked. The gate only
    # fires once a newer assistant reply (the read-back) has been committed.
    assistant_ids_before: frozenset[str] = frozenset()


class PendingActionStore:
    """Thread-safe, one parked write per conversation."""

    def __init__(self, ttl_s: float = 180.0) -> None:
        self.ttl_s = ttl_s
        self._lock = threading.Lock()
        self._pending: dict[str, PendingAction] = {}

    def stage(self, key: str, action: PendingAction) -> None:
        with self._lock:
            self._pending[key] = action

    def peek(self, key: str) -> PendingAction | None:
        with self._lock:
            action = self._pending.get(key)
            if action is not None and time.monotonic() - action.created_at > self.ttl_s:
                del self._pending[key]
                return None
            return action

    def pop(self, key: str) -> PendingAction | None:
        with self._lock:
            action = self._pending.pop(key, None)
        if action is not None and time.monotonic() - action.created_at > self.ttl_s:
            return None
        return action

    def clear(self) -> None:
        with self._lock:
            self._pending.clear()


class _SafeDict(dict):
    def __missing__(self, key: str) -> str:
        return ""


def describe_action(template: str | None, label: str, arguments: dict[str, Any]) -> str:
    """Build the read-back, e.g. 'add "Dentist" on 2026-11-21 15:00'."""
    flat: dict[str, Any] = {}
    for key, value in arguments.items():
        if isinstance(value, (str, int, float)):
            flat[key] = value
    if template:
        try:
            return " ".join(template.format_map(_SafeDict(flat)).split())
        except (ValueError, IndexError):
            pass
    shown = ", ".join(f"{k} {v}" for k, v in list(flat.items())[:4])
    return f"{label}: {shown}" if shown else label
