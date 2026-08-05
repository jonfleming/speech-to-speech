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
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaRecorder
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


@app.post("/v1/realtime/calls")
async def handle_offer(request: Request) -> Response:
    output = OUTPUT_PATH
    raw = await request.body()
    body = raw.decode("utf-8", errors="replace")
    log.info("SDP offer received (%d bytes)", len(body))

    config = rtc_configuration_from_env()
    pc = RTCPeerConnection(configuration=config) if config is not None else RTCPeerConnection()
    pcs.add(pc)
    recorder = MediaRecorder(output)

    @pc.on("datachannel")
    def on_datachannel(channel):
        log.info("DataChannel opened: %s (ignoring — capture mode)", channel.label)

    @pc.on("track")
    async def on_track(track):
        if track.kind == "audio":
            log.info("Audio track received — recording to %s", output)
            recorder.addTrack(track)
            await recorder.start()

        @track.on("ended")
        async def on_ended():
            await recorder.stop()
            log.info("Track ended — file saved to %s", output)

    @pc.on("connectionstatechange")
    async def on_state():
        log.info("Connection state → %s", pc.connectionState)
        if pc.connectionState in ("failed", "closed", "disconnected"):
            try:
                await recorder.stop()
            except Exception:
                pass
            await pc.close()
            pcs.discard(pc)

    try:
        await pc.setRemoteDescription(RTCSessionDescription(sdp=body, type="offer"))
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
    except Exception as e:  # noqa: BLE001
        log.exception("Failed to process SDP offer: %s", e)
        try:
            await recorder.stop()
        except Exception:
            pass
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
