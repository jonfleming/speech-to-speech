#!/usr/bin/env python3
"""Minimal WebRTC mic client: send audio and print transcription text only."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
from fractions import Fraction
import json
from queue import Empty
from queue import Queue
from typing import Any

import av
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.mediastreams import MediaStreamError, MediaStreamTrack
import httpx
import numpy as np


@dataclass
class Args:
    scheme: str
    host: str
    port: int
    endpoint: str
    sample_rate: int
    block_size: int
    input_device: int | None
    print_json: bool


def parse_args() -> Args:
    parser = argparse.ArgumentParser(description="Stream mic over WebRTC and print returned transcriptions")
    parser.add_argument("--scheme", choices=("http", "https"), default="http")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--endpoint",
        default="/v1/realtime/calls",
        help="HTTP endpoint for SDP offer exchange (default: /v1/realtime/calls)",
    )
    parser.add_argument("--sample-rate", type=int, default=48000, help="Mic sample rate (default: 48000)")
    parser.add_argument("--block-size", type=int, default=960, help="Frames per callback (default: 960, 20ms@48k)")
    parser.add_argument("--input-device", type=int, default=None)
    parser.add_argument("--print-json", action="store_true")
    ns = parser.parse_args()
    return Args(
        scheme=ns.scheme,
        host=ns.host,
        port=ns.port,
        endpoint=ns.endpoint,
        sample_rate=ns.sample_rate,
        block_size=ns.block_size,
        input_device=ns.input_device,
        print_json=ns.print_json,
    )


class MicrophoneAudioTrack(MediaStreamTrack):
    kind = "audio"

    def __init__(self, sample_rate: int, block_size: int, input_device: int | None) -> None:
        super().__init__()
        import sounddevice as sd

        self._sample_rate = sample_rate
        self._queue: Queue[bytes | None] = Queue(maxsize=256)
        self._pts = 0
        self._stopped = False
        self._stream = sd.RawInputStream(
            samplerate=sample_rate,
            dtype="int16",
            channels=1,
            callback=self._callback,
            blocksize=block_size,
            device=input_device,
        )
        self._stream.start()

    def _callback(self, indata, _frames, _time_info, status) -> None:
        if status:
            print(f"Mic status: {status}", flush=True)
        try:
            self._queue.put_nowait(bytes(indata))
        except Exception:
            pass

    async def recv(self) -> av.AudioFrame:
        if self.readyState != "live" or self._stopped:
            raise RuntimeError("audio track is not live")

        chunk: bytes | None = None
        while chunk is None:
            if self._stopped:
                raise MediaStreamError
            try:
                chunk = await asyncio.to_thread(self._queue.get, True, 0.2)
            except Empty:
                continue

        if not chunk:
            raise MediaStreamError

        samples = np.frombuffer(chunk, dtype=np.int16)
        frame = av.AudioFrame.from_ndarray(samples[np.newaxis, :], format="s16", layout="mono")
        frame.sample_rate = self._sample_rate
        frame.pts = self._pts
        frame.time_base = Fraction(1, self._sample_rate)
        self._pts += samples.shape[0]
        return frame

    async def stop_track(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        try:
            self._queue.put_nowait(None)
        except Exception:
            pass
        try:
            self._stream.stop()
            self._stream.close()
        finally:
            self.stop()


def _offer_url(args: Args) -> str:
    endpoint = args.endpoint if args.endpoint.startswith("/") else f"/{args.endpoint}"
    return f"{args.scheme}://{args.host}:{args.port}{endpoint}"


def _parse_event(raw: str | bytes) -> dict[str, Any] | None:
    payload = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw
    try:
        event = json.loads(payload)
    except json.JSONDecodeError:
        return None
    if isinstance(event, dict):
        return event
    return None


async def run(args: Args) -> None:
    pc = RTCPeerConnection()
    mic = MicrophoneAudioTrack(
        sample_rate=args.sample_rate,
        block_size=args.block_size,
        input_device=args.input_device,
    )
    pc.addTrack(mic)

    data_channel = pc.createDataChannel("oai-events")
    connected = asyncio.Event()
    stopped = asyncio.Event()

    @data_channel.on("open")
    def on_open() -> None:
        print("Connected. Speak into your mic. Press Ctrl-C to stop.", flush=True)
        connected.set()

    @data_channel.on("message")
    def on_message(message) -> None:
        event = _parse_event(message)
        if event is None:
            if args.print_json:
                print(f"RAW: {message}", flush=True)
            return

        if args.print_json:
            print(json.dumps(event, ensure_ascii=True), flush=True)

        event_type = event.get("type")
        if event_type == "conversation.item.input_audio_transcription.completed":
            text = str(event.get("transcript", "")).strip()
            if text:
                print(f"USER: {text}", flush=True)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange() -> None:
        state = pc.connectionState
        print(f"Connection state: {state}", flush=True)
        if state in ("failed", "closed", "disconnected"):
            stopped.set()

    try:
        offer = await pc.createOffer()
        await pc.setLocalDescription(offer)

        url = _offer_url(args)
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(url, content=pc.localDescription.sdp, headers={"Content-Type": "application/sdp"})

        if resp.status_code != 201:
            raise RuntimeError(
                f"Offer rejected by {url}: HTTP {resp.status_code} {resp.text[:300]}"
            )

        await pc.setRemoteDescription(RTCSessionDescription(sdp=resp.text, type="answer"))
        await connected.wait()
        await stopped.wait()
    finally:
        await mic.stop_track()
        await pc.close()


def main() -> None:
    args = parse_args()
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        print("\nStopping.", flush=True)


if __name__ == "__main__":
    main()
