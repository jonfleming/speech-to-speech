"""Tests for the WebRTC transport.

Three layers:

- Pure-unit tests for the shared ``append_pcm`` path, the ``PcmResampler``
  (stereo downmix, statefulness), and the paced ``PipelineAudioTrack``.
- Dispatch tests driving ``_dispatch_client_event`` with a fake transport,
  covering the transport-gated events (append rejected over WebRTC,
  output_audio_buffer.clear flushing server-side audio).
- One loopback integration test: a real aiortc peer performs the SDP
  handshake against the uvicorn-served app (POST /v1/realtime/calls),
  exchanges events over the 'oai-events' data channel, streams mic audio
  into the pipeline input queue, and receives paced audio from output_queue.

The whole module is skipped when the ``webrtc`` extra (aiortc) isn't installed.
"""

import asyncio
import json
import re
import time
from queue import Empty, Queue
from threading import Event as ThreadingEvent

import numpy as np
import pytest

aiortc = pytest.importorskip("aiortc")
av = pytest.importorskip("av")

import httpx  # noqa: E402  (ships with the openai dependency)
from aioice.ice import Connection  # noqa: E402
from aiortc import RTCPeerConnection, RTCSessionDescription  # noqa: E402
from aiortc.mediastreams import AudioStreamTrack, MediaStreamError  # noqa: E402
from starlette.testclient import TestClient  # noqa: E402

import speech_to_speech.api.openai_realtime.websocket_router as router_module  # noqa: E402
from speech_to_speech.api.openai_realtime.pipeline_unit import PipelineUnit  # noqa: E402
from speech_to_speech.api.openai_realtime.service import CHUNK_SIZE_BYTES, RealtimeService  # noqa: E402
from speech_to_speech.api.openai_realtime.transports import SessionTransport  # noqa: E402
from speech_to_speech.api.openai_realtime.webrtc_session import (  # noqa: E402
    WEBRTC_FRAME_SAMPLES,
    WEBRTC_SAMPLE_RATE,
    PcmResampler,
    PipelineAudioTrack,
    WebRTCSession,
    _prioritize_ice_candidates,
    _strip_non_sha256_fingerprints,
)
from speech_to_speech.pipeline.cancel_scope import CancelScope  # noqa: E402
from speech_to_speech.pipeline.events import (  # noqa: E402
    AssistantOutputEvent,
    AssistantResponseDoneEvent,
    ResponseFailedEvent,
    SpeechStartedEvent,
    TokenUsageEvent,
)
from speech_to_speech.pipeline.messages import (  # noqa: E402
    AUDIO_RESPONSE_DONE,
    AssistantTextPart,
    AssistantToolCallPart,
    AudioOutput,
)

from .test_openai_client import _ServerEnv  # noqa: E402

PIPELINE_SAMPLE_RATE = 16_000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_unit() -> PipelineUnit:
    text_prompt_queue: Queue = Queue()
    should_listen = ThreadingEvent()
    should_listen.set()
    service = RealtimeService(text_prompt_queue=text_prompt_queue, should_listen=should_listen)
    return PipelineUnit(
        index=0,
        service=service,
        cancel_scope=CancelScope(),
        should_listen=should_listen,
        response_playing=ThreadingEvent(),
        input_queue=Queue(),
        output_queue=Queue(),
        text_output_queue=Queue(),
        text_prompt_queue=text_prompt_queue,
        handlers=[],
    )


class _FakeTransport(SessionTransport):
    kind = "webrtc"

    def __init__(self):
        self.sent: list[dict] = []
        self.discards = 0

    async def send_events(self, events):
        self.sent.extend(e.model_dump() for e in events)

    async def send_audio_chunk(
        self,
        service,
        session_id,
        pcm,
        response_key=None,
    ):
        raise AssertionError("dispatch tests never send audio")

    def discard_pending_audio(self):
        self.discards += 1

    async def close(self):
        pass


# ---------------------------------------------------------------------------
# append_pcm (shared inbound path)
# ---------------------------------------------------------------------------


class TestAppendPcm:
    def test_chunks_and_remainder_carry_across_calls(self):
        unit = _make_unit()
        conn_id = unit.service.register()

        # 700 samples at pipeline rate: one 512-sample chunk + 188 remainder.
        chunks = unit.service.append_pcm(conn_id, b"\x01\x00" * 700, PIPELINE_SAMPLE_RATE)
        assert [len(c) for c in chunks] == [CHUNK_SIZE_BYTES]

        # 324 more completes the second chunk exactly (188 + 324 = 512).
        chunks = unit.service.append_pcm(conn_id, b"\x01\x00" * 324, PIPELINE_SAMPLE_RATE)
        assert [len(c) for c in chunks] == [CHUNK_SIZE_BYTES]
        assert unit.service._state(conn_id).audio_remainder == b""

    def test_sets_commit_bookkeeping(self):
        unit = _make_unit()
        conn_id = unit.service.register()

        assert unit.service.handle_audio_commit(conn_id) is not None  # empty buffer errors

        unit.service.append_pcm(conn_id, b"\x01\x00" * 512, PIPELINE_SAMPLE_RATE)
        assert unit.service.handle_audio_commit(conn_id) is None


# ---------------------------------------------------------------------------
# PcmResampler
# ---------------------------------------------------------------------------


class TestPcmResampler:
    def test_stereo_48k_downmixes_to_mono_16k(self):
        resampler = PcmResampler(PIPELINE_SAMPLE_RATE)
        n = WEBRTC_FRAME_SAMPLES
        stereo = np.zeros((2, n), dtype=np.int16)
        stereo[0, :] = 1000
        stereo[1, :] = 3000
        frame = av.AudioFrame.from_ndarray(stereo, format="s16p", layout="stereo")
        frame.sample_rate = WEBRTC_SAMPLE_RATE
        frame.pts = 0

        total = bytearray(resampler.resample_frame(frame))
        # Push several frames so filter delay flushes through.
        for i in range(1, 10):
            f = av.AudioFrame.from_ndarray(stereo, format="s16p", layout="stereo")
            f.sample_rate = WEBRTC_SAMPLE_RATE
            f.pts = i * n
            total += resampler.resample_frame(f)

        samples = np.frombuffer(bytes(total), dtype=np.int16)
        # 10 frames of 20 ms at 48 kHz → ~200 ms at 16 kHz = ~3200 samples
        # (minus filter delay). A plane-concatenating flatten bug would give
        # double that; a channel-dropping bug would average to 1000 or 3000.
        assert 2800 <= samples.shape[0] <= 3200
        steady_state = samples[samples.shape[0] // 2 :]
        assert abs(int(np.mean(steady_state)) - 2000) <= 10  # downmix average

    def test_stateful_across_pcm_chunks(self):
        resampler = PcmResampler(WEBRTC_SAMPLE_RATE)
        total = bytearray()
        for _ in range(10):
            total += resampler.resample_pcm(b"\x01\x00" * 512, PIPELINE_SAMPLE_RATE)
        samples = np.frombuffer(bytes(total), dtype=np.int16)
        # 5120 samples at 16 kHz → ~15360 at 48 kHz, minus filter delay.
        assert 15000 <= samples.shape[0] <= 15360


# ---------------------------------------------------------------------------
# Answer SDP sanitization (single sha-256 fingerprint for embedded clients)
# ---------------------------------------------------------------------------


class TestFingerprintSanitization:
    def test_strips_non_sha256_fingerprints_from_every_msection(self):
        sdp = (
            "v=0\r\n"
            "o=- 1 1 IN IP4 0.0.0.0\r\n"
            "s=-\r\n"
            "t=0 0\r\n"
            "m=audio 9 UDP/TLS/RTP/SAVPF 111\r\n"
            "a=mid:audio\r\n"
            "a=fingerprint:sha-256 AA:BB\r\n"
            "a=fingerprint:sha-384 CC:DD\r\n"
            "a=fingerprint:sha-512 EE:FF\r\n"
            "a=setup:active\r\n"
            "m=application 9 UDP/DTLS/SCTP webrtc-datachannel\r\n"
            "a=mid:datachannel\r\n"
            "a=fingerprint:sha-256 AA:BB\r\n"
            "a=fingerprint:sha-384 CC:DD\r\n"
            "a=fingerprint:sha-512 EE:FF\r\n"
            "a=setup:active\r\n"
        )
        cleaned = _strip_non_sha256_fingerprints(sdp)
        # One sha-256 fingerprint per m-section, none of the other algorithms.
        assert cleaned.count("a=fingerprint:") == 2
        assert "sha-384" not in cleaned and "sha-512" not in cleaned
        assert cleaned.count("a=fingerprint:sha-256 AA:BB") == 2
        # Non-fingerprint lines survive untouched, and no blank lines are left
        # behind where the removed lines were.
        assert "m=audio 9 UDP/TLS/RTP/SAVPF 111\r\n" in cleaned
        assert "a=setup:active\r\n" in cleaned
        assert "\r\n\r\n" not in cleaned

    def test_keeps_single_sha256_fingerprint_untouched(self):
        sdp = "v=0\r\nt=0 0\r\nm=audio 9 UDP/TLS/RTP/SAVPF 111\r\na=fingerprint:sha-256 AA:BB\r\n"
        assert _strip_non_sha256_fingerprints(sdp) == sdp

    def test_leaves_sdp_without_fingerprints_untouched(self):
        sdp = "v=0\r\nt=0 0\r\nm=application 9 UDP/DTLS/SCTP webrtc-datachannel\r\na=setup:active\r\n"
        assert _strip_non_sha256_fingerprints(sdp) == sdp


# ---------------------------------------------------------------------------
# ICE host-candidate restriction (SPEECH_TO_SPEECH_ICE_ADDRESSES)
# ---------------------------------------------------------------------------


class TestIceAddressFilter:
    def test_restricts_host_addresses_to_configured_networks(self, monkeypatch):
        import aioice.ice as aioice_ice

        import speech_to_speech.api.openai_realtime.webrtc_session as webrtc_session_module

        monkeypatch.setenv(webrtc_session_module.ICE_ADDRESSES_ENV, "192.168.0.112, 10.0.0.0/8")
        monkeypatch.setattr(webrtc_session_module, "_address_filter_installed", False)
        monkeypatch.setattr(webrtc_session_module, "_address_filter_networks", None)
        monkeypatch.setattr(
            aioice_ice,
            "get_host_addresses",
            lambda use_ipv4, use_ipv6: [
                "192.168.0.112",
                "192.168.1.9",
                "100.120.84.114",
                "172.31.80.1",
                "169.254.70.139",
                "10.0.0.5",
            ],
        )

        webrtc_session_module.install_ice_address_filter()

        assert aioice_ice.get_host_addresses(True, False) == ["192.168.0.112", "10.0.0.5"]

    def test_unset_env_leaves_aioice_untouched(self, monkeypatch):
        import aioice.ice as aioice_ice

        import speech_to_speech.api.openai_realtime.webrtc_session as webrtc_session_module

        original = aioice_ice.get_host_addresses
        monkeypatch.delenv(webrtc_session_module.ICE_ADDRESSES_ENV, raising=False)
        monkeypatch.setattr(webrtc_session_module, "_address_filter_installed", False)

        webrtc_session_module.install_ice_address_filter()

        assert aioice_ice.get_host_addresses is original

    def test_invalid_entry_fails_open(self, monkeypatch):
        import aioice.ice as aioice_ice

        import speech_to_speech.api.openai_realtime.webrtc_session as webrtc_session_module

        original = aioice_ice.get_host_addresses
        monkeypatch.setenv(webrtc_session_module.ICE_ADDRESSES_ENV, "not-an-ip")
        monkeypatch.setattr(webrtc_session_module, "_address_filter_installed", False)

        webrtc_session_module.install_ice_address_filter()

        assert aioice_ice.get_host_addresses is original

    def test_quoted_env_values_are_tolerated(self, monkeypatch):
        import aioice.ice as aioice_ice

        import speech_to_speech.api.openai_realtime.webrtc_session as webrtc_session_module

        monkeypatch.setenv(
            webrtc_session_module.ICE_ADDRESSES_ENV,
            "'192.168.0.112' \"10.0.0.0/8\"",
        )
        monkeypatch.setattr(webrtc_session_module, "_address_filter_installed", False)
        monkeypatch.setattr(webrtc_session_module, "_address_filter_networks", None)
        monkeypatch.setattr(
            aioice_ice,
            "get_host_addresses",
            lambda use_ipv4, use_ipv6: ["192.168.0.112", "10.0.0.5", "100.120.84.114"],
        )

        webrtc_session_module.install_ice_address_filter()

        assert aioice_ice.get_host_addresses(True, False) == ["192.168.0.112", "10.0.0.5"]


# ---------------------------------------------------------------------------
# Answer SDP candidate order (IPv4 host first for LAN DTLS)
# ---------------------------------------------------------------------------


class TestCandidatePrioritization:
    def test_puts_ipv4_host_then_srflx_then_relay_and_rewrites_c_line(self):
        sdp = (
            "v=0\r\n"
            "m=audio 9 UDP/TLS/RTP/SAVPF 111\r\n"
            "c=IN IP6 fd7a:115c:a1e0::1\r\n"
            "a=candidate:host6 1 udp 2130706431 fd7a:115c:a1e0::1 9 typ host\r\n"
            "a=candidate:host4 1 udp 2130706431 192.168.0.112 9 typ host\r\n"
            "a=candidate:srflx 1 udp 1694498815 50.47.198.12 9 typ srflx raddr 192.168.0.112 rport 9\r\n"
            "a=candidate:relay 1 udp 16777215 89.117.23.155 9 typ relay raddr 192.168.0.112 rport 9\r\n"
            "a=end-of-candidates\r\n"
        )
        rewritten = _prioritize_ice_candidates(sdp)
        audio = rewritten.split("m=audio", 1)[1]
        types = re.findall(r" typ (\S+)", audio)
        assert types == ["host", "srflx", "relay", "host"]
        assert "c=IN IP4 192.168.0.112" in rewritten
        assert "c=IN IP6 fd7a:115c:a1e0::1" not in rewritten

    def test_reorders_every_msection_independently(self):
        sdp = (
            "v=0\r\n"
            "m=audio 9 UDP/TLS/RTP/SAVPF 111\r\n"
            "a=candidate:h 1 udp 1 192.168.0.112 9 typ host\r\n"
            "a=candidate:r 1 udp 1 89.117.23.155 9 typ relay\r\n"
            "m=application 9 UDP/DTLS/SCTP webrtc-datachannel\r\n"
            "a=candidate:h 1 udp 1 192.168.0.112 9 typ host\r\n"
            "a=candidate:r 1 udp 1 89.117.23.155 9 typ relay\r\n"
        )
        rewritten = _prioritize_ice_candidates(sdp)
        sections = rewritten.split("m=")[1:]
        for section in sections:
            types = re.findall(r" typ (\S+)", section)
            assert types == ["host", "relay"]

    def test_leaves_sdp_without_candidates_untouched(self):
        sdp = "v=0\r\nt=0 0\r\nm=audio 9 UDP/TLS/RTP/SAVPF 111\r\na=setup:active\r\n"
        assert _prioritize_ice_candidates(sdp) == sdp


class TestTurnDataIndicationPatch:
    def test_registers_data_attribute_and_is_idempotent(self):
        from aioice import stun

        import speech_to_speech.api.openai_realtime.webrtc_session as webrtc_session_module

        webrtc_session_module.install_turn_data_indication_support()
        webrtc_session_module.install_turn_data_indication_support()
        assert "DATA" in stun.ATTRIBUTES_BY_NAME
        assert 0x0013 in stun.ATTRIBUTES_BY_TYPE

        message = stun.Message(message_method=stun.Method.DATA, message_class=stun.Class.INDICATION)
        message.attributes["XOR-PEER-ADDRESS"] = ("89.117.23.155", 63878)
        message.attributes["DATA"] = b"ice-check"
        parsed = stun.parse_message(bytes(message))
        assert parsed.message_method == stun.Method.DATA
        assert parsed.attributes["DATA"] == b"ice-check"
        assert parsed.attributes["XOR-PEER-ADDRESS"] == ("89.117.23.155", 63878)


class TestStunCompanionServers:
    def test_adds_stun_url_for_each_turn_url(self):
        import speech_to_speech.api.openai_realtime.webrtc_session as webrtc_session_module

        servers = webrtc_session_module._ice_servers_with_stun_companions(
            [{"urls": "turn:turn.techion.net:3478", "username": "u", "credential": "c"}]
        )
        urls = []
        for server in servers:
            value = server.urls
            urls.extend([value] if isinstance(value, str) else list(value))
        assert "turn:turn.techion.net:3478" in urls
        assert "stun:turn.techion.net:3478" in urls

    def test_does_not_duplicate_existing_stun(self):
        import speech_to_speech.api.openai_realtime.webrtc_session as webrtc_session_module

        servers = webrtc_session_module._ice_servers_with_stun_companions(
            [
                {"urls": "stun:turn.techion.net:3478"},
                {"urls": "turn:turn.techion.net:3478", "username": "u", "credential": "c"},
            ]
        )
        urls = []
        for server in servers:
            value = server.urls
            urls.extend([value] if isinstance(value, str) else list(value))
        assert urls.count("stun:turn.techion.net:3478") == 1


class TestParseIceServerEntries:
    def test_drops_turn_entry_without_credentials(self):
        import speech_to_speech.api.openai_realtime.webrtc_session as webrtc_session_module

        entries = webrtc_session_module._parse_ice_server_entries(
            '[{"urls": "turn:turn.techion.net"},'
            '{"urls": "turn:turn.techion.net", "username": "realtime", "credential": "secret"}]'
        )
        assert len(entries) == 1
        assert entries[0]["username"] == "realtime"

    def test_strips_wrapping_quotes_from_windows_set(self):
        import speech_to_speech.api.openai_realtime.webrtc_session as webrtc_session_module

        entries = webrtc_session_module._parse_ice_server_entries(
            '\'[{"urls": "stun:stun.example.com:3478"}]\''
        )
        assert entries == [{"urls": "stun:stun.example.com:3478"}]


# ---------------------------------------------------------------------------
# PipelineAudioTrack
# ---------------------------------------------------------------------------


class TestPipelineAudioTrack:
    async def test_recv_returns_written_audio_then_silence(self):
        track = PipelineAudioTrack()
        payload = (np.ones(WEBRTC_FRAME_SAMPLES, dtype=np.int16) * 5).tobytes()
        track.write(payload)

        frame = await track.recv()
        assert frame.sample_rate == WEBRTC_SAMPLE_RATE
        assert np.all(frame.to_ndarray() == 5)

        frame = await track.recv()  # buffer now empty → silence
        assert np.all(frame.to_ndarray() == 0)
        track.stop()

    async def test_recv_paces_to_wall_clock(self):
        track = PipelineAudioTrack()
        track.write(b"\x00" * WEBRTC_FRAME_SAMPLES * 2 * 10)  # 10 frames buffered

        start = time.monotonic()
        for _ in range(5):
            await track.recv()
        elapsed = time.monotonic() - start
        # 5 frames of 20 ms: first is immediate, the rest paced → ≥ ~80 ms.
        # Without pacing this loop completes in microseconds.
        assert elapsed >= 0.06
        track.stop()

    async def test_clear_drops_unplayed_audio(self):
        track = PipelineAudioTrack()
        track.write((np.ones(WEBRTC_FRAME_SAMPLES * 4, dtype=np.int16) * 7).tobytes())
        assert track.buffered_bytes > 0

        track.clear()
        assert track.buffered_bytes == 0
        frame = await track.recv()
        assert np.all(frame.to_ndarray() == 0)
        track.stop()

    async def test_recv_after_stop_raises(self):
        track = PipelineAudioTrack()
        track.stop()
        with pytest.raises(MediaStreamError):
            await track.recv()


# ---------------------------------------------------------------------------
# Client-event dispatch over the data channel
# ---------------------------------------------------------------------------


class TestWebRTCDispatch:
    async def test_append_rejected_over_webrtc(self):
        unit = _make_unit()
        conn_id = unit.service.register()
        transport = _FakeTransport()

        await router_module._dispatch_client_event(
            unit,
            conn_id,
            {"type": "input_audio_buffer.append", "audio": "AAAA"},
            transport,
            transport_kind="webrtc",
        )

        assert len(transport.sent) == 1
        assert transport.sent[0]["type"] == "error"
        assert transport.sent[0]["error"]["type"] == "invalid_event_for_transport"
        assert unit.input_queue.qsize() == 0

    async def test_output_audio_buffer_clear_flushes_audio(self):
        unit = _make_unit()
        conn_id = unit.service.register()
        transport = _FakeTransport()

        text_event = AssistantOutputEvent(text="before audio", response_key="response_1")
        tool_event = AssistantOutputEvent(
            tools=[{"type": "function_call", "call_id": "call_1", "name": "lookup", "arguments": "{}"}],
            response_key="response_1",
        )
        failed_event = ResponseFailedEvent(message="provider failed", response_key="response_1")
        usage_event = TokenUsageEvent(input_tokens=3, output_tokens=2, response_key="response_1")
        done_event = AssistantResponseDoneEvent(response_key="response_1")
        unit.output_queue.put(text_event)
        unit.output_queue.put(b"\x01\x00" * 512)
        unit.output_queue.put(tool_event)
        unit.output_queue.put(failed_event)
        unit.output_queue.put(usage_event)
        unit.output_queue.put(done_event)
        unit.output_queue.put(AUDIO_RESPONSE_DONE)

        await router_module._dispatch_client_event(
            unit,
            conn_id,
            {"type": "output_audio_buffer.clear"},
            transport,
            transport_kind="webrtc",
        )

        assert transport.sent == []  # no error
        assert transport.discards == 1
        # Only audio is flushed; ordered response state and the terminal survive.
        assert [unit.output_queue.get_nowait() for _ in range(6)] == [
            text_event,
            tool_event,
            failed_event,
            usage_event,
            done_event,
            AUDIO_RESPONSE_DONE,
        ]
        with pytest.raises(Empty):
            unit.output_queue.get_nowait()

    async def test_output_audio_buffer_clear_rejected_over_websocket(self):
        unit = _make_unit()
        conn_id = unit.service.register()
        transport = _FakeTransport()
        transport.kind = "websocket"

        await router_module._dispatch_client_event(
            unit,
            conn_id,
            {"type": "output_audio_buffer.clear"},
            transport,
            transport_kind="websocket",
        )

        assert len(transport.sent) == 1
        assert transport.sent[0]["type"] == "error"
        assert transport.sent[0]["error"]["type"] == "invalid_event_for_transport"

    async def test_response_cancel_discards_transport_audio(self):
        unit = _make_unit()
        conn_id = unit.service.register()
        transport = _FakeTransport()

        await router_module._dispatch_client_event(
            unit,
            conn_id,
            {"type": "response.cancel"},
            transport,
            transport_kind="webrtc",
        )

        assert transport.discards == 1


# ---------------------------------------------------------------------------
# Send-loop barge-in against transport-buffered audio
# ---------------------------------------------------------------------------


class TestBargeInAfterResponseDone:
    """Speech starting after a response finished must still flush audio the
    transport buffered but has not played yet: finish_response() runs when the
    done-sentinel is observed, not when playback completes, so fast TTS can
    leave seconds of unplayed audio in the WebRTC track with in_response
    already cleared."""

    def test_speech_start_flushes_buffered_transport_audio(self):
        unit = _make_unit()
        stop_event = ThreadingEvent()
        app = router_module.create_app(pool=[unit], stop_event=stop_event)
        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime") as ws:
                ws.receive_json()  # session.created
                # No response is active or pending. Swap in a spy transport so
                # the send loop's discard call is observable.
                spy = _FakeTransport()
                assert unit.session is not None
                unit.session.transport = spy
                generation_before = unit.cancel_scope.generation

                unit.text_output_queue.put(SpeechStartedEvent())

                deadline = time.monotonic() + 2.0
                while time.monotonic() < deadline and spy.discards == 0:
                    time.sleep(0.02)
                assert spy.discards == 1
                # Nothing to cancel: no response was active.
                assert unit.cancel_scope.generation == generation_before
        stop_event.set()


# ---------------------------------------------------------------------------
# Loopback integration: real aiortc peer against the served app
# ---------------------------------------------------------------------------


@pytest.fixture
def server_env():
    env = _ServerEnv()
    env.start()
    yield env
    env.stop()


class _DataChannelInbox:
    """Collects data-channel messages and lets tests await specific types."""

    def __init__(self, dc):
        self.events: list[dict] = []
        self._new = asyncio.Event()

        @dc.on("message")
        def on_message(msg):
            self.events.append(json.loads(msg))
            self._new.set()

    async def wait_for(self, event_type: str, timeout: float = 5.0) -> dict:
        deadline = time.monotonic() + timeout
        while True:
            for event in self.events:
                if event["type"] == event_type:
                    return event
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise AssertionError(
                    f"No '{event_type}' event within {timeout}s; got {[e['type'] for e in self.events]}"
                )
            self._new.clear()
            try:
                await asyncio.wait_for(self._new.wait(), timeout=remaining)
            except asyncio.TimeoutError:
                pass


class TestWebRTCLoopback:
    async def test_cancelled_close_keeps_teardown_running(self, monkeypatch):
        server_pc = RTCPeerConnection()
        teardown_started = asyncio.Event()
        release_teardown = asyncio.Event()
        closed_calls = []

        async def _on_client_event(_raw):
            pass

        async def _on_open():
            pass

        session = WebRTCSession(
            server_pc,
            on_client_event=_on_client_event,
            on_audio=lambda _pcm: None,
            on_open=_on_open,
            on_closed=lambda: closed_calls.append(None),
        )
        session.setup()
        close_peer_connection = session._close_peer_connection

        async def _pause_peer_close():
            teardown_started.set()
            await release_teardown.wait()
            await close_peer_connection()

        monkeypatch.setattr(session, "_close_peer_connection", _pause_peer_close)
        first_close = asyncio.create_task(session.close())
        second_close = None
        try:
            await asyncio.wait_for(teardown_started.wait(), timeout=1.0)
            first_close.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first_close

            second_close = asyncio.create_task(session.close())
            await asyncio.sleep(0)
            assert not second_close.done()

            release_teardown.set()
            await second_close
            assert server_pc.connectionState == "closed"
            assert len(closed_calls) == 1
        finally:
            release_teardown.set()
            close_tasks = [task for task in (first_close, second_close) if task is not None]
            await asyncio.gather(*close_tasks, return_exceptions=True)
            await session.close()
            await server_pc.close()

    async def test_close_awaits_pending_ice_checks(self, monkeypatch):
        client_pc = RTCPeerConnection()
        server_pc = RTCPeerConnection()
        first_close = None
        second_close = None
        release_sweep = asyncio.Event()
        closed_calls = []
        close_server_pc = server_pc.close

        def _pending_ice_checks():
            return {
                task
                for task in asyncio.all_tasks()
                if getattr(task.get_coro(), "cr_code", None) is Connection.check_start.__code__
            }

        connect_code = getattr(RTCPeerConnection, "_RTCPeerConnection__connect").__code__

        def _pending_server_connects():
            tasks = set()
            for task in asyncio.all_tasks():
                coro = task.get_coro()
                frame = getattr(coro, "cr_frame", None)
                if (
                    getattr(coro, "cr_code", None) is connect_code
                    and frame is not None
                    and frame.f_locals.get("self") is server_pc
                ):
                    tasks.add(task)
            return tasks

        async def _on_client_event(_raw):
            pass

        async def _on_open():
            pass

        session = WebRTCSession(
            server_pc,
            on_client_event=_on_client_event,
            on_audio=lambda _pcm: None,
            on_open=_on_open,
            on_closed=lambda: closed_calls.append(None),
        )
        session.setup()
        try:
            client_pc.createDataChannel("oai-events")
            client_pc.addTrack(AudioStreamTrack())
            offer = await client_pc.createOffer()
            await client_pc.setLocalDescription(offer)
            offer_sdp = client_pc.localDescription.sdp
            partial_offer_sdp = offer_sdp.replace("a=end-of-candidates\r\n", "")
            assert partial_offer_sdp != offer_sdp
            await session.negotiate(partial_offer_sdp)

            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline:
                if _pending_ice_checks():
                    break
                await asyncio.sleep(0.01)
            else:
                raise AssertionError("aioice did not start a connectivity check")
            assert _pending_server_connects()

            sweep_started = asyncio.Event()
            cancel_ice_checks = session._cancel_ice_checks
            sweep_count = 0

            async def _pause_first_sweep():
                nonlocal sweep_count
                sweep_count += 1
                if sweep_count == 1:
                    sweep_started.set()
                    await release_sweep.wait()
                await cancel_ice_checks()

            async def _fail_peer_close():
                raise RuntimeError("peer close failed")

            monkeypatch.setattr(session, "_cancel_ice_checks", _pause_first_sweep)
            monkeypatch.setattr(server_pc, "close", _fail_peer_close)
            first_close = asyncio.create_task(session.close())
            await asyncio.wait_for(sweep_started.wait(), timeout=1.0)
            second_close = asyncio.create_task(session.close())
            await asyncio.sleep(0)
            assert not second_close.done()

            release_sweep.set()
            await asyncio.gather(first_close, second_close)
            assert not _pending_ice_checks()
            assert not _pending_server_connects()
            assert len(closed_calls) == 1
        finally:
            release_sweep.set()
            close_tasks = [task for task in (first_close, second_close) if task is not None]
            if close_tasks:
                await asyncio.gather(*close_tasks, return_exceptions=True)
            monkeypatch.setattr(server_pc, "close", close_server_pc)
            await session.close()
            await server_pc.close()
            await client_pc.close()

    async def test_handshake_events_and_multi_output_audio_roundtrip(self, server_env):
        pc = RTCPeerConnection()
        try:
            dc = pc.createDataChannel("oai-events")
            inbox = _DataChannelInbox(dc)
            pc.addTrack(AudioStreamTrack())  # silent mic track

            received_frames: list = []
            track_ready = asyncio.Event()

            @pc.on("track")
            def on_track(track):
                async def _consume():
                    while True:
                        try:
                            frame = await track.recv()
                        except MediaStreamError:
                            return
                        received_frames.append(frame)
                        track_ready.set()

                asyncio.ensure_future(_consume())

            offer = await pc.createOffer()
            await pc.setLocalDescription(offer)

            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"http://127.0.0.1:{server_env.port}/v1/realtime/calls",
                    content=pc.localDescription.sdp,
                    headers={"Content-Type": "application/sdp"},
                    timeout=10.0,
                )
            assert resp.status_code == 201
            assert resp.headers["content-type"].startswith("application/sdp")
            assert resp.headers["location"].startswith("/v1/realtime/calls/")

            await pc.setRemoteDescription(RTCSessionDescription(sdp=resp.text, type="answer"))

            # session.created arrives once the data channel opens.
            created = await inbox.wait_for("session.created", timeout=10.0)
            assert "session" in created

            # Client events over the data channel reach the shared dispatch:
            # append must be rejected as transport-invalid.
            dc.send(json.dumps({"type": "input_audio_buffer.append", "audio": "AAAA"}))
            error = await inbox.wait_for("error")
            assert error["error"]["type"] == "invalid_event_for_transport"

            # Inbound mic audio lands on input_queue as 512-sample chunks.
            def _wait_for_input_chunk(timeout: float = 10.0):
                return server_env.input_queue.get(timeout=timeout)

            chunk, _cfg = await asyncio.get_running_loop().run_in_executor(None, _wait_for_input_chunk)
            assert len(chunk) == CHUNK_SIZE_BYTES

            # Two assistant messages separated by a tool call keep distinct
            # output identities over a real WebRTC media/data-channel pair.
            response_key = "webrtc_response_1"
            server_env.output_queue.put(
                AssistantOutputEvent(
                    response_key=response_key,
                    parts=[AssistantTextPart(text="before")],
                )
            )
            server_env.output_queue.put(
                AudioOutput(
                    audio=np.ones(2048, dtype=np.int16).tobytes(),
                    response_key=response_key,
                )
            )
            server_env.output_queue.put(
                AssistantOutputEvent(
                    response_key=response_key,
                    parts=[
                        AssistantToolCallPart(
                            tool={
                                "type": "function_call",
                                "call_id": "call_1",
                                "name": "tool",
                                "arguments": "{}",
                            }
                        )
                    ],
                )
            )
            server_env.output_queue.put(
                AssistantOutputEvent(
                    response_key=response_key,
                    parts=[AssistantTextPart(text="after")],
                )
            )
            server_env.output_queue.put(
                AudioOutput(
                    audio=np.ones(2048, dtype=np.int16).tobytes(),
                    response_key=response_key,
                )
            )
            server_env.output_queue.put(AssistantResponseDoneEvent(response_key=response_key))
            server_env.output_queue.put(AudioOutput(audio=AUDIO_RESPONSE_DONE, response_key=response_key))

            await inbox.wait_for("response.created", timeout=10.0)
            done = await inbox.wait_for("response.done", timeout=10.0)
            assert done["response"]["status"] == "completed"
            transcript_deltas = [
                event for event in inbox.events if event["type"] == "response.output_audio_transcript.delta"
            ]
            audio_done = [event for event in inbox.events if event["type"] == "response.output_audio.done"]
            assert [event["output_index"] for event in transcript_deltas] == [0, 2]
            assert [event["output_index"] for event in audio_done] == [0, 2]
            assert [item["type"] for item in done["response"]["output"]] == [
                "message",
                "function_call",
                "message",
            ]
            for event in audio_done:
                assert done["response"]["output"][event["output_index"]]["id"] == event["item_id"]

            await asyncio.wait_for(track_ready.wait(), timeout=10.0)
            assert received_frames[0].sample_rate == WEBRTC_SAMPLE_RATE

            # Hanging up: closing the data channel signals the server (an
            # SCTP reset, unlike a bare pc.close() which the server only
            # notices via ICE consent timeouts). The release path enqueues
            # SESSION_END and marks the session as draining.
            dc.close()
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                session = server_env.unit.session
                if session is None or session.released_at is not None:
                    break
                await asyncio.sleep(0.05)
            else:
                raise AssertionError("WebRTC disconnect did not start the unit release")
        finally:
            await pc.close()

    async def test_rejects_wrong_content_type(self, server_env):
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"http://127.0.0.1:{server_env.port}/v1/realtime/calls",
                json={"sdp": "nope"},
                timeout=10.0,
            )
        assert resp.status_code == 415

    async def test_new_offer_preempts_existing_webrtc_session_when_pool_size_is_one(self, server_env):
        pc1 = RTCPeerConnection()
        pc2 = RTCPeerConnection()
        try:
            pc1.createDataChannel("oai-events")
            pc1.addTrack(AudioStreamTrack())
            offer1 = await pc1.createOffer()
            await pc1.setLocalDescription(offer1)

            pc2.createDataChannel("oai-events")
            pc2.addTrack(AudioStreamTrack())
            offer2 = await pc2.createOffer()
            await pc2.setLocalDescription(offer2)

            async with httpx.AsyncClient() as client:
                first = await client.post(
                    f"http://127.0.0.1:{server_env.port}/v1/realtime/calls",
                    content=pc1.localDescription.sdp,
                    headers={"Content-Type": "application/sdp"},
                    timeout=10.0,
                )
                assert first.status_code == 201

                second = await client.post(
                    f"http://127.0.0.1:{server_env.port}/v1/realtime/calls",
                    content=pc2.localDescription.sdp,
                    headers={"Content-Type": "application/sdp"},
                    timeout=10.0,
                )
            assert second.status_code == 201
        finally:
            await pc1.close()
            await pc2.close()

    async def test_delete_location_hangs_up(self, server_env):
        """DELETE on the Location URL advertised by the 201 releases the unit;
        an unknown call id answers 404."""
        pc = RTCPeerConnection()
        try:
            pc.createDataChannel("oai-events")
            pc.addTrack(AudioStreamTrack())
            offer = await pc.createOffer()
            await pc.setLocalDescription(offer)

            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"http://127.0.0.1:{server_env.port}/v1/realtime/calls",
                    content=pc.localDescription.sdp,
                    headers={"Content-Type": "application/sdp"},
                    timeout=10.0,
                )
                assert resp.status_code == 201
                location = resp.headers["location"]

                missing = await client.delete(
                    f"http://127.0.0.1:{server_env.port}/v1/realtime/calls/no-such-call",
                    timeout=10.0,
                )
                assert missing.status_code == 404

                hangup = await client.delete(f"http://127.0.0.1:{server_env.port}{location}", timeout=10.0)
                assert hangup.status_code == 200
        finally:
            await pc.close()
        await _wait_for_release(server_env)

    async def test_invalid_offer_releases_unit(self, server_env):
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"http://127.0.0.1:{server_env.port}/v1/realtime/calls",
                content="not an sdp",
                headers={"Content-Type": "application/sdp"},
                timeout=10.0,
            )
        assert resp.status_code == 400
        await _wait_for_release(server_env)

    async def test_setup_failure_releases_unit(self, server_env, monkeypatch):
        """A failure between claiming the unit and negotiate() (e.g. peer
        connection construction) must release the unit, not leak it."""
        import speech_to_speech.api.openai_realtime.websocket_router as router_module

        def _boom():
            raise RuntimeError("boom")

        # The calls endpoint uses the router's module-level binding (imported
        # eagerly at load), so that's the name to patch.
        monkeypatch.setattr(router_module, "rtc_configuration_from_env", _boom)

        pc = RTCPeerConnection()
        try:
            pc.createDataChannel("oai-events")
            pc.addTrack(AudioStreamTrack())
            offer = await pc.createOffer()
            await pc.setLocalDescription(offer)

            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"http://127.0.0.1:{server_env.port}/v1/realtime/calls",
                    content=pc.localDescription.sdp,
                    headers={"Content-Type": "application/sdp"},
                    timeout=10.0,
                )
            assert resp.status_code == 500
        finally:
            await pc.close()
        await _wait_for_release(server_env)


async def _wait_for_release(server_env, timeout: float = 5.0) -> None:
    """Assert the unit was released (or is draining) after a failed call."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        session = server_env.unit.session
        if session is None or session.released_at is not None:
            return
        await asyncio.sleep(0.05)
    raise AssertionError("failed WebRTC call left the pipeline unit claimed")
