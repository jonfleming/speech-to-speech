#!/usr/bin/env python3
"""
Receive microphone audio over WebRTC, transcribe it, and send text back on data channel.

This is intentionally minimal for STT accuracy testing:
- No LLM response generation
- No tool calls
- Just speech -> text

Usage
-----
1. Stop any server already listening on the same port.
2. Install deps if needed: uv sync --extra webrtc --extra faster-whisper
3. Run: python src/transcribe_webrtc_audio.py
4. Connect your WebRTC caller to this endpoint.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from contextlib import asynccontextmanager

import av
import numpy as np
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.mediastreams import MediaStreamError
from fastapi import FastAPI, Request, Response
from faster_whisper import WhisperModel
import uvicorn

from speech_to_speech.api.openai_realtime.webrtc_session import (
    _strip_non_sha256_fingerprints,
    rtc_configuration_from_env,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("transcribe-webrtc")

PIPELINE_RATE = 16000
pcs: set[RTCPeerConnection] = set()


class SegmentTranscriber:
    """Small energy-based segmenter + faster-whisper transcription."""

    def __init__(
        self,
        model: WhisperModel,
        *,
        language: str | None,
        energy_threshold: float,
        end_silence_ms: int,
        min_speech_ms: int,
        max_segment_s: float,
    ) -> None:
        self.model = model
        self.language = language
        self.energy_threshold = energy_threshold
        self.end_silence_samples = int((end_silence_ms / 1000.0) * PIPELINE_RATE)
        self.min_speech_samples = int((min_speech_ms / 1000.0) * PIPELINE_RATE)
        self.max_segment_samples = int(max_segment_s * PIPELINE_RATE)

        self._speech: list[np.ndarray] = []
        self._speech_samples = 0
        self._trailing_silence_samples = 0
        self._in_speech = False

    def feed(self, chunk: np.ndarray) -> tuple[bool, np.ndarray | None]:
        """
        Feed one mono int16 chunk.

        Returns (speech_started, completed_segment_or_none).
        """
        if chunk.size == 0:
            return (False, None)

        rms = float(np.sqrt(np.mean((chunk.astype(np.float32) / 32768.0) ** 2)))
        is_voiced = rms >= self.energy_threshold

        speech_started = False
        completed: np.ndarray | None = None

        if is_voiced:
            if not self._in_speech:
                self._in_speech = True
                speech_started = True
                self._speech = []
                self._speech_samples = 0
                self._trailing_silence_samples = 0
            self._speech.append(chunk)
            self._speech_samples += int(chunk.size)
            self._trailing_silence_samples = 0
        elif self._in_speech:
            self._speech.append(chunk)
            self._speech_samples += int(chunk.size)
            self._trailing_silence_samples += int(chunk.size)

            if self._trailing_silence_samples >= self.end_silence_samples:
                completed = self._finalize_segment()

        if self._in_speech and self._speech_samples >= self.max_segment_samples:
            completed = self._finalize_segment()

        return (speech_started, completed)

    def flush(self) -> np.ndarray | None:
        if not self._in_speech:
            return None
        return self._finalize_segment()

    def _finalize_segment(self) -> np.ndarray | None:
        joined = np.concatenate(self._speech) if self._speech else np.empty(0, dtype=np.int16)
        if self._trailing_silence_samples > 0 and joined.size > self._trailing_silence_samples:
            joined = joined[: joined.size - self._trailing_silence_samples]

        self._speech = []
        self._speech_samples = 0
        self._trailing_silence_samples = 0
        self._in_speech = False

        if joined.size < self.min_speech_samples:
            return None
        return joined

    def transcribe_segment(self, samples_i16: np.ndarray) -> str:
        pcm_f32 = samples_i16.astype(np.float32) / 32768.0
        kwargs: dict[str, object] = {
            "beam_size": 1,
            "vad_filter": False,
            "condition_on_previous_text": False,
            "without_timestamps": True,
        }
        if self.language:
            kwargs["language"] = self.language

        segments, _info = self.model.transcribe(pcm_f32, **kwargs)
        text = " ".join(seg.text.strip() for seg in segments if seg.text.strip()).strip()
        return text


def _event_json(event: dict) -> str:
    return json.dumps(event, separators=(",", ":"), ensure_ascii=True)


def _safe_channel_send(channel, payload: str) -> None:
    try:
        channel.send(payload)
    except Exception as e:  # noqa: BLE001
        log.debug("Failed to send datachannel payload: %s", e)


def _send_event(channels: set, event: dict) -> None:
    payload = _event_json(event)
    dead = []
    for channel in channels:
        if getattr(channel, "readyState", None) == "open":
            _safe_channel_send(channel, payload)
        else:
            dead.append(channel)
    for channel in dead:
        channels.discard(channel)


async def _emit_transcription(channels: set, transcriber: SegmentTranscriber, segment: np.ndarray) -> None:
    _send_event(channels, {"type": "input_audio_buffer.speech_stopped"})
    text = await asyncio.to_thread(transcriber.transcribe_segment, segment)
    if not text:
        return
    log.info("Transcription: %s", text)
    _send_event(
        channels,
        {
            "type": "conversation.item.input_audio_transcription.completed",
            "transcript": text,
        },
    )


async def _capture_and_transcribe(track, channels: set, transcriber: SegmentTranscriber) -> None:
    resampler = av.AudioResampler(format="s16", layout="mono", rate=PIPELINE_RATE)

    try:
        while True:
            frame = await track.recv()
            for out in resampler.resample(frame):
                chunk = out.to_ndarray().reshape(-1).astype(np.int16, copy=False)
                speech_started, completed = transcriber.feed(chunk)
                if speech_started:
                    _send_event(channels, {"type": "input_audio_buffer.speech_started"})
                if completed is not None:
                    await _emit_transcription(channels, transcriber, completed)
    except MediaStreamError:
        pass
    finally:
        remaining = transcriber.flush()
        if remaining is not None:
            await _emit_transcription(channels, transcriber, remaining)


def _format_model_label(model_name: str, model_path: str | None) -> str:
    return model_path if model_path else model_name


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    yield
    log.info("Shutting down; closing %d peer connection(s)", len(pcs))
    for pc in pcs.copy():
        await pc.close()


def build_app(transcriber_factory):
    app = FastAPI(lifespan=_lifespan)

    @app.post("/v1/realtime/calls")
    async def handle_offer(request: Request) -> Response:
        raw = await request.body()
        body = raw.decode("utf-8", errors="replace")
        log.info("SDP offer received (%d bytes)", len(body))

        config = rtc_configuration_from_env()
        pc = RTCPeerConnection(configuration=config) if config is not None else RTCPeerConnection()
        pcs.add(pc)

        channels = set()
        capture_tasks: set[asyncio.Task] = set()

        @pc.on("datachannel")
        def on_datachannel(channel):
            channels.add(channel)
            log.info("DataChannel opened: %s", channel.label)
            if getattr(channel, "readyState", None) == "open":
                _safe_channel_send(channel, _event_json({"type": "session.created"}))

            @channel.on("open")
            def on_open():
                _safe_channel_send(channel, _event_json({"type": "session.created"}))

            @channel.on("close")
            def on_close():
                channels.discard(channel)

        @pc.on("track")
        async def on_track(track):
            if track.kind != "audio":
                return
            log.info("Audio track received; STT is active")
            transcriber = transcriber_factory()
            task = asyncio.create_task(_capture_and_transcribe(track, channels, transcriber))
            capture_tasks.add(task)
            task.add_done_callback(capture_tasks.discard)

            @track.on("ended")
            async def on_ended():
                log.info("Track ended")

        @pc.on("connectionstatechange")
        async def on_state():
            log.info("Connection state -> %s", pc.connectionState)
            if pc.connectionState in ("failed", "closed", "disconnected"):
                await pc.close()
                pcs.discard(pc)

        try:
            await pc.setRemoteDescription(RTCSessionDescription(sdp=body, type="offer"))
            answer = await pc.createAnswer()
            await pc.setLocalDescription(answer)
        except Exception as e:  # noqa: BLE001
            log.exception("Failed to process SDP offer: %s", e)
            await pc.close()
            pcs.discard(pc)
            return Response(status_code=400, media_type="text/plain", content="Invalid SDP offer")

        answer_sdp = _strip_non_sha256_fingerprints(pc.localDescription.sdp)
        if answer_sdp != pc.localDescription.sdp:
            log.info("Stripped non-sha-256 DTLS fingerprint lines from SDP answer")

        return Response(status_code=201, media_type="application/sdp", content=answer_sdp)

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WebRTC STT-only server")
    parser.add_argument("--port", type=int, default=8765, help="HTTP port (default: 8765)")
    parser.add_argument("--model", default="base", help="faster-whisper model name (default: base)")
    parser.add_argument("--model-path", default=None, help="Optional local faster-whisper model path")
    parser.add_argument("--device", default="auto", help="Model device (default: auto)")
    parser.add_argument("--compute-type", default="auto", help="faster-whisper compute type (default: auto)")
    parser.add_argument("--language", default=None, help="Optional fixed language code, e.g. en")
    parser.add_argument(
        "--energy-threshold",
        type=float,
        default=0.012,
        help="Speech energy threshold (default: 0.012)",
    )
    parser.add_argument(
        "--end-silence-ms",
        type=int,
        default=700,
        help="Silence needed to end a segment in ms (default: 700)",
    )
    parser.add_argument(
        "--min-speech-ms",
        type=int,
        default=250,
        help="Minimum speech length in ms (default: 250)",
    )
    parser.add_argument(
        "--max-segment-s",
        type=float,
        default=20.0,
        help="Maximum segment length in seconds (default: 20)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_id = _format_model_label(args.model, args.model_path)
    log.info("Loading faster-whisper model: %s", model_id)

    model = WhisperModel(
        args.model_path or args.model,
        device=args.device,
        compute_type=args.compute_type,
    )

    def transcriber_factory() -> SegmentTranscriber:
        return SegmentTranscriber(
            model,
            language=args.language,
            energy_threshold=args.energy_threshold,
            end_silence_ms=args.end_silence_ms,
            min_speech_ms=args.min_speech_ms,
            max_segment_s=args.max_segment_s,
        )

    app = build_app(transcriber_factory)

    print()
    print(f"  STT model   : {model_id}")
    print(f"  Listening on: http://0.0.0.0:{args.port}/v1/realtime/calls")
    print()
    print("  This endpoint returns only transcription events over the data channel.")
    print("  No LLM responses are generated.")
    print()

    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
