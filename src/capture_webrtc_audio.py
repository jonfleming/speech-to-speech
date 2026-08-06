#!/usr/bin/env python3
"""
Capture the ESP32 microphone audio over WebRTC and save it to a WAV file.

Steps
-----
1.  Stop the speech-to-speech server (it occupies port 8765).
2.  Install deps if needed:  uv sync
3.  Run:  python capture_webrtc_audio.py
4.  Power-cycle the ESP32 — it will connect here instead of the real server.
5.  Speak for 10-20 seconds, then press Ctrl-C.
6.  Open recording.wav and listen to check microphone quality.

If the recording sounds clear → the mic is fine, the issue is server-side VAD tuning.
If the recording sounds like noise/distortion → we need more gain or different I2S settings.
"""
import argparse
import asyncio
import logging
import sys
import wave
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, Request, Response
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.mediastreams import MediaStreamError
import uvicorn
from speech_to_speech.api.openai_realtime.webrtc_session import (
    _strip_non_sha256_fingerprints,
    rtc_configuration_from_env,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("capture")

pcs: set = set()
OUTPUT_PATH = "recording.wav"


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    yield
    log.info("Shutting down — closing %d peer connection(s)", len(pcs))
    for pc in list(pcs):
        await pc.close()


app = FastAPI(lifespan=_lifespan)


async def _capture_track_to_wav(track, output: str) -> None:
    writer: wave.Wave_write | None = None
    frame_count = 0
    total_samples_per_channel = 0
    sample_rate = 0
    channels = 0
    prev_pts: int | None = None
    try:
        while True:
            frame = await track.recv()
            ndarray = frame.to_ndarray()

            # aiortc/av commonly exposes audio as (channels, samples).
            if ndarray.ndim == 2:
                if ndarray.shape[0] <= 8:
                    interleaved = ndarray.T
                else:
                    interleaved = ndarray
            else:
                interleaved = ndarray.reshape(-1, 1)

            if interleaved.dtype != np.int16:
                if np.issubdtype(interleaved.dtype, np.floating):
                    interleaved = np.clip(interleaved, -1.0, 1.0)
                    interleaved = (interleaved * 32767.0).astype(np.int16)
                else:
                    interleaved = np.clip(interleaved, -32768, 32767).astype(np.int16)

            if writer is None:
                sample_rate = int(frame.sample_rate or 48000)
                channels = int(interleaved.shape[1])
                writer = wave.open(output, "wb")
                writer.setnchannels(channels)
                writer.setsampwidth(2)
                writer.setframerate(sample_rate)
                log.info(
                    "Audio capture format: sample_rate=%d Hz, channels=%d, time_base=%s",
                    sample_rate,
                    channels,
                    frame.time_base,
                )

            samples_per_channel = int(interleaved.shape[0])
            writer.writeframes(interleaved.tobytes())
            frame_count += 1
            total_samples_per_channel += samples_per_channel

            if frame_count % 100 == 0:
                packet_ms = (samples_per_channel / sample_rate) * 1000.0 if sample_rate else 0.0
                pts_delta_ms = None
                if prev_pts is not None and frame.pts is not None and frame.time_base is not None:
                    pts_delta_ms = (frame.pts - prev_pts) * float(frame.time_base) * 1000.0
                if frame.pts is not None:
                    prev_pts = frame.pts
                captured_s = total_samples_per_channel / sample_rate if sample_rate else 0.0
                if pts_delta_ms is None:
                    log.info(
                        "Audio frame #%d: samples/ch=%d (~%.2f ms), captured=%.2f s",
                        frame_count,
                        samples_per_channel,
                        packet_ms,
                        captured_s,
                    )
                else:
                    log.info(
                        "Audio frame #%d: samples/ch=%d (~%.2f ms), pts_delta=%.2f ms, captured=%.2f s",
                        frame_count,
                        samples_per_channel,
                        packet_ms,
                        pts_delta_ms,
                        captured_s,
                    )

            if frame.pts is not None:
                prev_pts = frame.pts
    except MediaStreamError:
        pass
    finally:
        if writer is not None:
            writer.close()
            duration_s = total_samples_per_channel / sample_rate if sample_rate else 0.0
            log.info("Track ended — file saved to %s (%.2f s)", output, duration_s)


@app.post("/v1/realtime/calls")
async def handle_offer(request: Request) -> Response:
    output = OUTPUT_PATH
    raw = await request.body()
    body = raw.decode("utf-8", errors="replace")
    log.info("SDP offer received (%d bytes)", len(body))

    config = rtc_configuration_from_env()
    pc = RTCPeerConnection(configuration=config) if config is not None else RTCPeerConnection()
    pcs.add(pc)

    @pc.on("datachannel")
    def on_datachannel(channel):
        log.info("DataChannel opened: %s (ignoring — capture mode)", channel.label)

    @pc.on("track")
    async def on_track(track):
        if track.kind == "audio":
            log.info("Audio track received — recording to %s", output)
            asyncio.create_task(_capture_track_to_wav(track, output))

        @track.on("ended")
        async def on_ended():
            log.info("Track ended event received")

    @pc.on("connectionstatechange")
    async def on_state():
        log.info("Connection state → %s", pc.connectionState)
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

    return Response(
        status_code=201,
        media_type="application/sdp",
        content=answer_sdp,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture ESP32 WebRTC mic to WAV")
    parser.add_argument("--port",   type=int, default=8765,
                        help="HTTP port to listen on (default: 8765, same as speech-to-speech server)")
    parser.add_argument("--output", default="recording.wav",
                        help="Output file (default: recording.wav; .ogg and .mp4 also supported)")
    args = parser.parse_args()

    global OUTPUT_PATH
    OUTPUT_PATH = args.output

    print()
    print(f"  Output file : {args.output}")
    print(f"  Listening on: http://0.0.0.0:{args.port}/v1/realtime/calls")
    print()
    print("  1. Make sure the speech-to-speech server is stopped.")
    print("  2. Power-cycle the ESP32 — it will connect here.")
    print("  3. Speak for a few seconds, then Ctrl-C.")
    print()

    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
