from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from speech_to_speech.pipeline.messages import GenerateResponseRequest

HEADER_CALLBACK = "X-S2S-Callback-Url"
HEADER_SESSION = "X-S2S-Session-Id"
HEADER_TURN = "X-S2S-Turn-Id"
HEADER_REVISION = "X-S2S-Turn-Revision"
HEADER_RESPONSE = "X-S2S-Response-Key"


class MemoryFollowupRequest(BaseModel):
    """Spoken follow-up posted by obsidian-memory after retrieval finishes."""

    session_id: str
    text: str
    turn_id: str | None = None
    turn_revision: int | None = None
    origin_response_key: str | None = None


class PendingMemoryFollowup(BaseModel):
    text: str
    turn_id: str | None = None
    turn_revision: int | None = None
    origin_response_key: str | None = None


class MemoryFollowupResult(BaseModel):
    accepted: bool
    status: str
    queued: bool = False
    events: list[Any] = Field(default_factory=list)


def memory_followup_headers(request: GenerateResponseRequest, callback_url: str | None) -> dict[str, str]:
    """Correlation headers so the memory proxy can POST a second response back."""
    if not request.session_id:
        return {}
    headers = {HEADER_SESSION: request.session_id}
    if callback_url:
        headers[HEADER_CALLBACK] = callback_url
    if request.turn_id:
        headers[HEADER_TURN] = request.turn_id
    if request.turn_revision is not None:
        headers[HEADER_REVISION] = str(request.turn_revision)
    if request.response_key:
        headers[HEADER_RESPONSE] = request.response_key
    return headers
