"""WebRTC transport for the OpenAI Realtime API emulation.

Requires the ``webrtc`` extra (aiortc). Audio travels over RTP media tracks
(Opus at 48 kHz, resampled to/from the 16 kHz pipeline rate); all JSON events
use the same protocol as the WebSocket transport, carried on the
``oai-events`` data channel.

``WebRTCSession`` subclasses ``SessionTransport`` from ``transports``,
so the per-unit send loop in ``websocket_router`` drives it
exactly like a WebSocket session: it stays the sole consumer of the pipeline
output queues, and this module only turns delivered PCM into paced RTP frames.
"""

from __future__ import annotations

import asyncio
import ipaddress
import json
import logging
import os
import re
import time
from collections.abc import Awaitable
from fractions import Fraction
from typing import TYPE_CHECKING, Callable, Optional

import av
import numpy as np
from aioice.ice import ICE_FAILED
from aiortc import RTCConfiguration, RTCIceServer, RTCPeerConnection, RTCSessionDescription
from aiortc.mediastreams import MediaStreamError, MediaStreamTrack

from speech_to_speech.api.openai_realtime.service import PIPELINE_SAMPLE_RATE
from speech_to_speech.api.openai_realtime.transports import SessionTransport
from speech_to_speech.pipeline.transcript_logging import log_exception, transcript_for_log

if TYPE_CHECKING:
    from speech_to_speech.api.openai_realtime.service import RealtimeService, ServerEvent

logger = logging.getLogger(__name__)

WEBRTC_SAMPLE_RATE = 48_000
AUDIO_PTIME = 0.02  # 20 ms frames
WEBRTC_FRAME_SAMPLES = int(WEBRTC_SAMPLE_RATE * AUDIO_PTIME)
DATA_CHANNEL_LABEL = "oai-events"
ICE_SERVERS_ENV = "SPEECH_TO_SPEECH_ICE_SERVERS"
ICE_ADDRESSES_ENV = "SPEECH_TO_SPEECH_ICE_ADDRESSES"
ICE_GATHERING_TIMEOUT_S = 5.0
# How long a negotiated session may sit without the peer connection reaching
# "connected" before we release its pipeline unit. Without this, a client that
# receives the SDP answer and never completes ICE would hold the unit forever.
CONNECT_TIMEOUT_S = 30.0

#: Allowed networks for ICE host candidates, parsed from SPEECH_TO_SPEECH_ICE_ADDRESSES.
_address_filter_networks: Optional[list[ipaddress.IPv4Network | ipaddress.IPv6Network]] = None
_address_filter_installed = False
_turn_indication_patch_installed = False

_CANDIDATE_LINE = re.compile(
    r"^a=candidate:\S+\s+\d+\s+\S+\s+\d+\s+(\S+)\s+\d+\s+typ\s+(\S+)",
)
# Lower is tried first by embedded ICE (list order, not RFC 5245). IPv4 host
# is first so a same-LAN ESP32 nominates host-host and DTLS stays off TURN.
# IPv6 host is last so ``c=`` is not an unreachable IPv6 address.
_CANDIDATE_TYPE_RANK = {"host": 0, "srflx": 1, "relay": 2, "prflx": 3}


def _address_allowed(address: str, networks: list[ipaddress.IPv4Network | ipaddress.IPv6Network]) -> bool:
    try:
        ip = ipaddress.ip_address(address)
    except ValueError:
        return False
    return any(ip in network for network in networks)


def install_ice_address_filter() -> None:
    """Restrict aioice's ICE host candidates to SPEECH_TO_SPEECH_ICE_ADDRESSES.

    aioice gathers a host candidate on every network interface — Tailscale,
    WSL/Hyper-V vNICs, APIPA link-local, ... — and probes each one, stalling
    ICE for roughly 20 s per unreachable pair (and logging noisy bind failures
    on addresses that cannot be bound at all). The env var holds a
    comma/space-separated list of IPs or CIDRs, e.g. ``192.168.0.112`` or
    ``192.168.0.0/24``; host candidates outside it are never gathered, so the
    answer SDP only advertises usable addresses.

    aioice exposes no configuration knob for this, so we wrap its
    ``get_host_addresses`` enumeration. The patch is installed once and is a
    no-op when the env var is unset. If aioice isn't installed or ever drops
    that hook, we log a warning and keep the default all-interface behavior
    rather than crash.
    """
    global _address_filter_installed, _address_filter_networks

    raw = os.environ.get(ICE_ADDRESSES_ENV)
    if not raw:
        return

    networks: list[ipaddress.IPv4Network | ipaddress.IPv6Network] = []
    for token in raw.replace(",", " ").split():
        # Windows `set` stores surrounding quotes literally (cmd does not
        # strip them the way bash does), so tolerate '192.168.0.112' / "…".
        token = token.strip().strip("'\"")
        if not token:
            continue
        try:
            networks.append(ipaddress.ip_network(token, strict=False))
        except ValueError:
            logger.error(f"Ignoring invalid {ICE_ADDRESSES_ENV} entry {token!r} — keeping all interfaces")
            return
    if not networks:
        return

    if not _address_filter_installed:
        try:
            from aioice import ice as aioice_ice
        except (ImportError, AttributeError):
            logger.warning("Cannot restrict ICE host candidates: aioice is not installed")
            return
        original = aioice_ice.get_host_addresses

        def _filtered(use_ipv4: bool, use_ipv6: bool) -> list[str]:
            candidates = original(use_ipv4, use_ipv6)
            allowed = _address_filter_networks
            if not allowed:
                return candidates
            return [a for a in candidates if _address_allowed(a, allowed)]

        aioice_ice.get_host_addresses = _filtered
        _address_filter_installed = True

    _address_filter_networks = networks
    logger.info(f"Restricting ICE host candidates to {', '.join(str(n) for n in networks)}")


def install_turn_data_indication_support() -> None:
    """Make aioice interoperate with Send/Data Indication TURN clients.

    aioice only relays via ChannelData after CHANNEL_BIND. Embedded stacks
    (ESP32 libpeer) use RFC 5766 Send Indications instead, and coturn often
    answers CHANNEL_BIND to another allocation on the same server with
    ``403 Forbidden IP`` (hairpin). Without this patch the ESP32's checks
    arrive as Data Indications and are dropped, and aioice's own relay sends
    die as unretrieved task exceptions.

    Registers the STUN DATA attribute, unwraps inbound Data Indications, and
    falls back to CreatePermission + Send Indication when CHANNEL_BIND is
    forbidden. Installed once; no-op if aioice isn't importable.
    """
    global _turn_indication_patch_installed
    if _turn_indication_patch_installed:
        return
    try:
        from aioice import stun, turn
    except ImportError:
        logger.warning("Cannot patch aioice TURN Data Indications: aioice is not installed")
        return

    data_attr = (0x0013, "DATA", stun.pack_bytes, stun.unpack_bytes)
    stun.ATTRIBUTES_BY_NAME.setdefault("DATA", data_attr)
    stun.ATTRIBUTES_BY_TYPE.setdefault(0x0013, data_attr)

    original_datagram = turn.TurnClientMixin.datagram_received
    original_send_data = turn.TurnClientMixin.send_data

    def _send_indication(client, data: bytes, addr: tuple[str, int]) -> None:
        message = stun.Message(message_method=stun.Method.SEND, message_class=stun.Class.INDICATION)
        message.attributes["XOR-PEER-ADDRESS"] = addr
        message.attributes["DATA"] = data
        client.send_stun(message, client.server)

    def datagram_received(self, data, addr) -> None:
        payload = bytes(data)
        if len(payload) >= 20 and not turn.is_channel_data(payload):
            try:
                message = stun.parse_message(payload)
            except ValueError:
                message = None
            if (
                message is not None
                and message.message_method == stun.Method.DATA
                and message.message_class == stun.Class.INDICATION
                and "DATA" in message.attributes
                and "XOR-PEER-ADDRESS" in message.attributes
                and self.receiver is not None
            ):
                self.receiver.datagram_received(
                    message.attributes["DATA"],
                    message.attributes["XOR-PEER-ADDRESS"],
                )
                return
        original_datagram(self, data, addr)

    async def send_data(self, data: bytes, addr: tuple[str, int]) -> None:
        fallback = getattr(self, "_send_indication_peers", None)
        if fallback is None:
            fallback = set()
            self._send_indication_peers = fallback
        if addr in fallback:
            _send_indication(self, data, addr)
            return
        relayed = getattr(self, "relayed_address", None)
        if relayed is not None and addr[0] == relayed[0]:
            logger.warning(
                "Skipping TURN hairpin to %s (coturn 403 Forbidden IP for the TURN "
                "server's own address). Set allowed-peer-ip to that address on coturn.",
                addr,
            )
            return
        try:
            await original_send_data(self, data, addr)
            return
        except stun.TransactionFailed as exc:
            code = 0
            if exc.response is not None and "ERROR-CODE" in exc.response.attributes:
                code = exc.response.attributes["ERROR-CODE"][0]
            if code != 403:
                raise
        waiters = self.peer_connect_waiters.pop(addr, [])
        for waiter in waiters:
            if not waiter.done():
                waiter.set_result(None)
        try:
            request = stun.Message(
                message_method=stun.Method.CREATE_PERMISSION,
                message_class=stun.Class.REQUEST,
            )
            request.attributes["XOR-PEER-ADDRESS"] = addr
            await self.request_with_retry(request)
        except stun.TransactionFailed as exc:
            logger.warning(
                "TURN CHANNEL_BIND 403 for %s and CreatePermission failed (%s); "
                "cannot relay to this peer",
                addr,
                exc,
            )
            return
        logger.info("TURN CHANNEL_BIND 403 for %s; falling back to Send Indication", addr)
        fallback.add(addr)
        _send_indication(self, data, addr)

    turn.TurnClientMixin.datagram_received = datagram_received
    turn.TurnClientMixin.send_data = send_data
    _turn_indication_patch_installed = True
    logger.info("Installed aioice TURN Data Indication / Send Indication fallback")


def rtc_configuration_from_env() -> Optional[RTCConfiguration]:
    """Build an RTCConfiguration from the SPEECH_TO_SPEECH_ICE_SERVERS env var.

    The variable holds a JSON list of RTCIceServer kwargs, e.g.
    ``[{"urls": "stun:stun.example.com:3478"},
       {"urls": "turn:turn.example.com", "username": "u", "credential": "c"}]``.
    Returns None (aiortc defaults) when unset or invalid.

    Also applies SPEECH_TO_SPEECH_ICE_ADDRESSES via :func:`install_ice_address_filter`
    and the TURN Send/Data Indication fallback via
    :func:`install_turn_data_indication_support`.
    """
    install_ice_address_filter()
    install_turn_data_indication_support()
    raw = os.environ.get(ICE_SERVERS_ENV)
    if not raw:
        return None
    try:
        entries = _parse_ice_server_entries(raw)
        servers = _ice_servers_with_stun_companions(entries)
    except (json.JSONDecodeError, TypeError, ValueError) as e:
        logger.error(f"Ignoring invalid {ICE_SERVERS_ENV}: {e}")
        return None
    return RTCConfiguration(iceServers=servers)


def _parse_ice_server_entries(raw: str) -> list[dict]:
    """Parse SPEECH_TO_SPEECH_ICE_SERVERS JSON, tolerating Windows ``set`` quoting.

    Drops ``turn:`` / ``turns:`` entries that have no username/credential —
    they only produce 401 Allocate attempts and never a relay candidate.
    """
    text = raw.strip().strip("'\"")
    entries = json.loads(text)
    if not isinstance(entries, list):
        raise TypeError("ICE servers JSON must be a list")
    kept: list[dict] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise TypeError("each ICE server entry must be an object")
        urls = _as_url_list(entry.get("urls"))
        is_turn = any(url.split("?")[0].startswith(("turn:", "turns:")) for url in urls)
        has_creds = bool(entry.get("username") and entry.get("credential"))
        if is_turn and not has_creds:
            logger.warning(
                "Ignoring TURN iceServer %s with no username/credential (coturn will 401)",
                urls,
            )
            continue
        kept.append(entry)
    if not kept:
        raise ValueError("no usable ICE servers after dropping unauthenticated TURN entries")
    return kept


def _as_url_list(urls: object) -> list[str]:
    if urls is None:
        return []
    if isinstance(urls, str):
        return [urls]
    return [str(url) for url in urls]


def _ice_servers_with_stun_companions(entries: list[dict]) -> list[RTCIceServer]:
    """Ensure each TURN URL also has a STUN URL so ICE gathers a public srflx.

    aiortc only produces ``typ srflx`` from STUN. A turn-only iceServers list
    advertises host+relay and no server-reflexive address, so a remote ESP32
    cannot CreatePermission the server's public NAT mapping — the only peer
    IP that works when coturn forbids hairpin to the TURN server itself.
    """
    servers = [RTCIceServer(**entry) for entry in entries]
    existing = {url.split("?")[0] for entry in entries for url in _as_url_list(entry.get("urls"))}
    for entry in entries:
        for url in _as_url_list(entry.get("urls")):
            base = url.split("?")[0]
            if not base.startswith("turn:"):
                continue
            stun_url = "stun:" + base[5:]
            if stun_url in existing:
                continue
            existing.add(stun_url)
            servers.append(RTCIceServer(urls=stun_url))
            logger.info("Adding STUN companion %s so ICE advertises a public srflx candidate", stun_url)
    return servers


# aiortc >= 1.12 advertises a fingerprint for every supported digest algorithm
# (sha-256, sha-384, sha-512) on each m-section — see RTCCertificate
# .getFingerprints() and the per-media fingerprint loop in aiortc's sdp.py.
# RFC 8122 permits several and browsers tolerate them, but many embedded
# WebRTC stacks (e.g. ESP32) require exactly one `a=fingerprint:sha-256` per
# m-section and reject the answer, aborting the DTLS handshake. The fingerprint
# is only an advertisement of the (unchanged) DTLS certificate, so pruning the
# non-sha-256 lines is safe: the peer hashes the cert with sha-256 and matches
# the remaining line.
# The `(?:\r?\n|$)` alternative consumes the whole line: with MULTILINE `$`
# alone would leave the trailing `\n` behind, littering the SDP with blank lines.
_NON_SHA256_FINGERPRINT = re.compile(r"^a=fingerprint:(?!sha-256\b)\S+\s+\S+\s*(?:\r?\n|$)", re.MULTILINE)


def _strip_non_sha256_fingerprints(sdp: str) -> str:
    """Return *sdp* with every DTLS fingerprint line but the sha-256 one removed."""
    return _NON_SHA256_FINGERPRINT.sub("", sdp)


def _candidate_type_rank(line: str) -> tuple[int, int]:
    match = _CANDIDATE_LINE.match(line.rstrip("\r\n"))
    if match is None:
        return (5, 1)
    ip, typ = match.group(1), match.group(2)
    ipv6 = 1 if ":" in ip else 0
    if typ == "host" and ipv6:
        return (4, 1)
    return (_CANDIDATE_TYPE_RANK.get(typ, 5), ipv6)


def _connection_line_for_candidate(line: str) -> str | None:
    match = _CANDIDATE_LINE.match(line.rstrip("\r\n"))
    if match is None:
        return None
    ip = match.group(1)
    version = "IP6" if ":" in ip else "IP4"
    return f"c=IN {version} {ip}"


def _prioritize_ice_candidates(sdp: str) -> str:
    """Put IPv4 host, then srflx, then relay, then IPv6 host in each m-section.

    Embedded ICE (ESP32 libpeer) nominates pairs in list order, not RFC 5245
    priority. Same-LAN clients must try host-host first — nominating TURN
    first succeeds ICE then fails DTLS (mbedtls CONN_EOF). Restrict host
    gathering with ``SPEECH_TO_SPEECH_ICE_ADDRESSES`` so the advertised host
    is the LAN NIC, not Tailscale/Docker. srflx stays ahead of relay for
    remote TURN clients that cannot hairpin to coturn's own IP.

    Also rewrites each m-section's ``c=`` line to the first candidate after
    the reorder so the default connection address is IPv4 rather than an
    unreachable IPv6 host.
    """
    newline = "\r\n" if "\r\n" in sdp else "\n"
    # Keep the terminator so a missing final newline stays missing.
    ends_with_newline = sdp.endswith("\r\n") or sdp.endswith("\n")
    raw_lines = sdp.splitlines()
    output: list[str] = []
    section: list[str] = []

    def _flush(lines: list[str]) -> None:
        candidates = [line for line in lines if line.startswith("a=candidate:")]
        if not candidates:
            output.extend(lines)
            return
        ranked = sorted(candidates, key=_candidate_type_rank)
        connection = _connection_line_for_candidate(ranked[0])
        used = 0
        for line in lines:
            if line.startswith("c=") and connection is not None:
                output.append(connection)
            elif line.startswith("a=candidate:"):
                output.append(ranked[used])
                used += 1
            else:
                output.append(line)

    for line in raw_lines:
        if line.startswith("m=") and section:
            _flush(section)
            section = [line]
        else:
            section.append(line)
    if section:
        _flush(section)

    rewritten = newline.join(output)
    if ends_with_newline:
        rewritten += newline
    return rewritten


class PcmResampler:
    """Stateful mono/s16 resampler around av.AudioResampler.

    One instance per direction per session: the libswresample filter state
    carries across calls, so 20 ms frames resample without the boundary
    artifacts a stateless per-chunk resample would introduce. Also downmixes
    multi-channel input (browser Opus is typically stereo) to mono.
    """

    def __init__(self, target_rate: int) -> None:
        self._resampler = av.AudioResampler(format="s16", layout="mono", rate=target_rate)
        self._pts = 0

    def resample_frame(self, frame: av.AudioFrame) -> bytes:
        out = bytearray()
        for resampled in self._resampler.resample(frame):
            out += resampled.to_ndarray().tobytes()
        return bytes(out)

    def resample_pcm(self, pcm: bytes, src_rate: int) -> bytes:
        samples = np.frombuffer(pcm, dtype=np.int16)
        frame = av.AudioFrame.from_ndarray(samples[np.newaxis, :], format="s16", layout="mono")
        frame.sample_rate = src_rate
        frame.pts = self._pts
        frame.time_base = Fraction(1, src_rate)
        self._pts += samples.shape[0]
        return self.resample_frame(frame)


class PipelineAudioTrack(MediaStreamTrack):
    """Outbound audio track: paced 20 ms 48 kHz frames from a PCM buffer.

    The send loop pushes generated audio in via ``write()`` (faster than
    real time); ``recv()`` paces delivery against the wall clock like
    aiortc's built-in AudioStreamTrack, emitting silence when the buffer is
    empty so the RTP stream stays continuous. ``clear()`` drops unplayed
    audio — this is the server-side equivalent of the client's speaker
    buffer, so barge-in must flush it for interruption to be audible.
    """

    kind = "audio"

    def __init__(self) -> None:
        super().__init__()
        self._buffer = bytearray()
        self._start: Optional[float] = None
        self._timestamp = 0

    def write(self, pcm: bytes) -> None:
        self._buffer.extend(pcm)

    def clear(self) -> None:
        del self._buffer[:]

    @property
    def buffered_bytes(self) -> int:
        return len(self._buffer)

    async def recv(self) -> av.AudioFrame:
        if self.readyState != "live":
            raise MediaStreamError

        if self._start is None:
            self._start = time.time()
            self._timestamp = 0
        else:
            self._timestamp += WEBRTC_FRAME_SAMPLES
            wait = self._start + (self._timestamp / WEBRTC_SAMPLE_RATE) - time.time()
            if wait > 0:
                await asyncio.sleep(wait)

        needed = WEBRTC_FRAME_SAMPLES * 2  # bytes of s16 mono
        payload = bytes(self._buffer[:needed])
        del self._buffer[: len(payload)]
        if len(payload) < needed:
            payload += b"\x00" * (needed - len(payload))

        samples = np.frombuffer(payload, dtype=np.int16)
        frame = av.AudioFrame.from_ndarray(samples[np.newaxis, :], format="s16", layout="mono")
        frame.sample_rate = WEBRTC_SAMPLE_RATE
        frame.pts = self._timestamp
        frame.time_base = Fraction(1, WEBRTC_SAMPLE_RATE)
        return frame


class WebRTCSession(SessionTransport):
    """One WebRTC peer connection, used as the SessionState transport.

    All pipeline integration arrives through callbacks supplied by the route
    handler (which owns the PipelineUnit): parsed client events, inbound PCM,
    channel-open, and close. This module never touches queues or services
    directly except through the SessionTransport methods the send loop calls.
    """

    kind = "webrtc"

    def __init__(
        self,
        pc: RTCPeerConnection,
        *,
        on_client_event: Callable[[dict], Awaitable[None]],
        on_audio: Callable[[bytes], None],
        on_open: Callable[[], Awaitable[None]],
        on_closed: Callable[[], None],
    ) -> None:
        self._pc = pc
        self._on_client_event = on_client_event
        self._on_audio = on_audio
        self._on_open = on_open
        self._on_closed = on_closed
        self._dc = None
        self._closed = False
        self._close_task: asyncio.Task[None] | None = None
        self._track = PipelineAudioTrack()
        self._out_resampler = PcmResampler(WEBRTC_SAMPLE_RATE)
        self._in_resampler = PcmResampler(PIPELINE_SAMPLE_RATE)
        # Data-channel messages funnel through one queue + consumer task so
        # client events apply in arrival order; dispatching each message in
        # its own task could reorder e.g. session.update vs response.create.
        self._dc_messages: asyncio.Queue[str] = asyncio.Queue()
        self._tasks: list[asyncio.Task] = []

    # ── Lifecycle ─────────────────────────────────

    def setup(self) -> None:
        """Wire aiortc event callbacks. Call before negotiate()."""
        self._pc.addTrack(self._track)

        @self._pc.on("datachannel")
        def on_datachannel(dc) -> None:
            if dc.label != DATA_CHANNEL_LABEL:
                logger.warning(f"[WebRTC] Ignoring unexpected data channel: {dc.label}")
                return
            self._dc = dc
            self._spawn(self._consume_dc_messages())
            logger.info(f"[WebRTC] Data channel '{DATA_CHANNEL_LABEL}' received")

            # aiortc may deliver the channel already open, in which case the
            # "open" event never fires.
            if dc.readyState == "open":
                self._spawn(self._on_open())
            else:

                @dc.on("open")
                def on_dc_open() -> None:
                    self._spawn(self._on_open())

            @dc.on("message")
            def on_message(msg) -> None:
                if isinstance(msg, str):
                    self._dc_messages.put_nowait(msg)
                else:
                    logger.warning("[WebRTC] Ignoring binary data-channel message")

            @dc.on("close")
            def on_dc_close() -> None:
                logger.info("[WebRTC] Data channel closed")
                self._spawn(self.close())

        @self._pc.on("track")
        def on_track(track) -> None:
            if track.kind == "audio":
                logger.info("[WebRTC] Inbound audio track received")
                self._spawn(self._consume_inbound_audio(track))

        @self._pc.on("connectionstatechange")
        async def on_connection_state_change() -> None:
            state = self._pc.connectionState
            logger.info(f"[WebRTC] Connection state: {state}")
            if state in ("failed", "closed"):
                await self.close()

    async def negotiate(self, offer_sdp: str) -> str:
        """Apply the client's SDP offer and return the SDP answer.

        Waits for ICE gathering so the answer carries the server's candidates
        — there is no trickle-ICE channel in the HTTP handshake.
        """
        await self._pc.setRemoteDescription(RTCSessionDescription(sdp=offer_sdp, type="offer"))
        answer = await self._pc.createAnswer()
        await self._pc.setLocalDescription(answer)

        if self._pc.iceGatheringState != "complete":
            done: asyncio.Event = asyncio.Event()

            @self._pc.on("icegatheringstatechange")
            def on_ice_change() -> None:
                if self._pc.iceGatheringState == "complete":
                    done.set()

            if self._pc.iceGatheringState == "complete":  # raced to completion
                done.set()
            try:
                await asyncio.wait_for(done.wait(), timeout=ICE_GATHERING_TIMEOUT_S)
            except asyncio.TimeoutError:
                logger.warning("[WebRTC] ICE gathering timed out, returning partial SDP")

        self._spawn(self._connect_watchdog())
        sdp = self._pc.localDescription.sdp
        answer_sdp = _prioritize_ice_candidates(_strip_non_sha256_fingerprints(sdp))
        if answer_sdp != sdp:
            logger.info("[WebRTC] Sanitized SDP answer (sha-256 fingerprint, relay-first candidates)")
        return answer_sdp

    async def close(self) -> None:
        if self._close_task is None:
            self._closed = True
            self._close_task = asyncio.create_task(self._run_close())
        await asyncio.shield(self._close_task)

    # ── SessionTransport interface ────────────────

    async def send_events(self, events: list[ServerEvent]) -> None:
        dc = self._dc
        if dc is None or dc.readyState != "open":
            return
        for event in events:
            try:
                dc.send(json.dumps(event.model_dump()))
            except Exception as e:  # noqa: BLE001
                logger.error(f"[WebRTC] Data channel send error: {e}")

    async def send_audio_chunk(
        self,
        service: RealtimeService,
        session_id: str,
        pcm: bytes,
        response_key: str | None = None,
    ) -> None:
        # Bookkeeping events (response.created on the implicit VAD path) go
        # over the data channel; the audio itself goes on the media track.
        _resp_id, _item_id, _output_index, events = service.begin_audio_output(
            session_id,
            response_key,
        )
        if events:
            await self.send_events(events)
        self._track.write(self._out_resampler.resample_pcm(pcm, PIPELINE_SAMPLE_RATE))

    def discard_pending_audio(self) -> None:
        self._track.clear()

    # ── Internals ─────────────────────────────────

    def _spawn(self, coro: Awaitable[None]) -> None:
        task = asyncio.ensure_future(coro)
        self._tasks.append(task)

    async def _run_close(self) -> None:
        # close() often runs as a _spawn()ed task (dc close handler). The
        # separate cleanup task lets us cancel session work without cancelling
        # teardown, even when the close() caller itself is cancelled.
        for task in self._tasks:
            task.cancel()
        self._track.stop()
        try:
            await self._close_peer_connection()
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[WebRTC] Error closing peer connection: {e}")
        self._on_closed()
        logger.info("[WebRTC] Session closed")

    async def _close_peer_connection(self) -> None:
        try:
            await self._cancel_ice_checks()
        finally:
            try:
                await self._pc.close()
            finally:
                try:
                    await self._close_ice_connections()
                finally:
                    await self._cancel_ice_checks()
                    await self._await_ice_connect_tasks()

    def _ice_transports(self) -> set:
        ice_transports = {transceiver.receiver.transport.transport for transceiver in self._pc.getTransceivers()}
        if self._pc.sctp is not None:
            ice_transports.add(self._pc.sctp.transport.transport)
        return ice_transports

    async def _close_ice_connections(self) -> None:
        connections = {
            ice_transport._connection
            for ice_transport in self._ice_transports()
            if not ice_transport._connection._closed
        }
        if connections:
            results = await asyncio.gather(
                *(connection.close() for connection in connections),
                return_exceptions=True,
            )
            for result in results:
                if isinstance(result, BaseException):
                    raise result

    async def _cancel_ice_checks(self) -> None:
        """Cancel and await aioice connectivity checks owned by this peer."""
        tasks = set()
        for ice_transport in self._ice_transports():
            await ice_transport.addRemoteCandidate(None)
            connection = ice_transport._connection
            if connection._check_list and not connection._check_list_done and connection._check_list_state.empty():
                connection._check_list_state.put_nowait(ICE_FAILED)
            for pair in connection._check_list:
                if pair.task is None:
                    if pair.state in (pair.State.FROZEN, pair.State.WAITING):
                        pair.state = pair.State.FAILED
                else:
                    pair.task.cancel()
                    tasks.add(pair.task)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _await_ice_connect_tasks(self) -> None:
        start_events = [
            event
            for ice_transport in self._ice_transports()
            if (event := getattr(ice_transport, "_RTCIceTransport__start")) is not None
        ]
        if start_events:
            await asyncio.gather(*(event.wait() for event in start_events))
            await asyncio.sleep(0)

    async def _connect_watchdog(self) -> None:
        await asyncio.sleep(CONNECT_TIMEOUT_S)
        if not self._closed and self._pc.connectionState != "connected":
            logger.warning(
                f"[WebRTC] Peer not connected after {CONNECT_TIMEOUT_S:.0f}s "
                f"(state: {self._pc.connectionState}); releasing session"
            )
            await self.close()

    async def _consume_dc_messages(self) -> None:
        while not self._closed:
            msg = await self._dc_messages.get()
            try:
                raw = json.loads(msg)
            except json.JSONDecodeError:
                logger.error("[WebRTC] Invalid JSON on data channel: %s", transcript_for_log(msg))
                continue
            if not isinstance(raw, dict):
                logger.error("[WebRTC] Non-object event on data channel: %s", transcript_for_log(msg))
                continue
            try:
                await self._on_client_event(raw)
            except Exception as exc:  # noqa: BLE001
                log_exception(logger, "[WebRTC] Error handling client event", exc)

    async def _consume_inbound_audio(self, track) -> None:
        while not self._closed:
            try:
                frame = await track.recv()
            except MediaStreamError:
                logger.info("[WebRTC] Inbound audio track ended")
                break
            pcm = self._in_resampler.resample_frame(frame)
            if pcm:
                self._on_audio(pcm)
