from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
from queue import Queue, Empty
import signal
import socket
import sounddevice as sd
import sys
import threading
import time
from transformers import HfArgumentParser

_log_lock = threading.Lock()

def log(message):
    # Print timestamped log messages for better debugging and monitoring.
    with _log_lock:
        print(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [ListenAndPlay] {message}",
            flush=True,
        )
@dataclass
class ListenAndPlayArguments:
    send_rate: int = field(default=16000, metadata={"help": "In Hz. Default is 16000."})
    recv_rate: int = field(default=16000, metadata={"help": "In Hz. Default is 16000."})
    listen_play_chunk_size: int = field(
        default=1024,
        metadata={"help": "The size of data chunks (in bytes). Default is 1024."},
    )
    host: str = field(
        default="localhost",
        metadata={
            "help": "The hostname or IP address for listening and playing. Default is 'localhost'."
        },
    )
    send_port: int = field(
        default=12345,
        metadata={"help": "The network port for sending data. Default is 12345."},
    )
    recv_port: int = field(
        default=12346,
        metadata={"help": "The network port for receiving data. Default is 12346."},
    )



def listen_and_play(
    send_rate=16000,
    recv_rate=16000,
    listen_play_chunk_size=1024,
    host="localhost",
    send_port=12345,
    recv_port=12346,
    playback_mute_hold_ms=250,
):
    stop_event = threading.Event()
    
    send_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        send_socket.connect((host, send_port))
    except Exception as e:
        log(f"Failed to connect send socket: {e}")
        return

    recv_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        recv_socket.connect((host, recv_port))
    except Exception as e:
        log(f"Failed to connect recv socket: {e}")
        send_socket.close()
        return

    log("Connected to server. Press Ctrl+C or Enter to stop.")

    recv_queue = Queue()
    send_queue = Queue()

    # Pre-generate a static dither buffer (±1 LSB, -96 dB)
    dither_bytes = np.random.randint(
        -1, 2, size=listen_play_chunk_size, dtype=np.int16
    ).tobytes()

    mute_hold_seconds = max(0, playback_mute_hold_ms) / 1000.0
    mute_deadline = {"t": 0.0}
    playback_state = {"active": False}
    log("[Playback] Cleared")

    def should_mute_mic():
        return time.monotonic() < mute_deadline["t"]

    def mark_playback_active():
        mute_deadline["t"] = time.monotonic() + mute_hold_seconds
        if not playback_state["active"]:
            playback_state["active"] = True
            log("[Playback] Set")

    def maybe_clear_playback_active():
        if playback_state["active"] and not should_mute_mic():
            playback_state["active"] = False
            log("[Playback] Cleared")

    def callback_recv(outdata, frames, callback_time, status):
        try:
            if status:
                log(f"[Playback] Stream status: {status}")
            try:
                data = recv_queue.get_nowait()
            except Empty:
                data = None

            if data is None:
                outdata[:] = dither_bytes
                maybe_clear_playback_active()
                return

            outdata[: len(data)] = data
            outdata[len(data) :] = b"\x00" * (len(outdata) - len(data))
            mark_playback_active()
        except Exception as e:
            log(f"[Playback] Callback error: {e}")
            outdata[:] = dither_bytes
            maybe_clear_playback_active()

    def callback_send(indata, frames, callback_time, status):
        try:
            if status:
                log(f"[Capture] Stream status: {status}")
            # Mute mic during playback to prevent feedback.
            if should_mute_mic():
                return
            send_queue.put(bytes(indata))
        except Exception as e:
            log(f"[Capture] Callback error: {e}")

    def send(stop_event, send_queue):
        sent_chunks = 0
        sent_bytes = 0
        last_log_time = time.monotonic()
        while not stop_event.is_set():
            try:
                data = send_queue.get(timeout=0.1)
                try:
                    send_socket.sendall(data)
                    sent_chunks += 1
                    sent_bytes += len(data)
                    now = time.monotonic()
                    if now - last_log_time >= 1.0:
                        log(
                            f"[Send] chunks={sent_chunks} bytes={sent_bytes} queue={send_queue.qsize()}"
                        )
                        sent_chunks = 0
                        sent_bytes = 0
                        last_log_time = now
                except (socket.error, BrokenPipeError) as e:
                    log(f"[Send] Socket error, exiting send thread: {e}")
                    break
            except Empty:
                continue
        log("[Send] Thread exiting")

    def recv(stop_event, recv_queue):
        recv_chunks = 0
        recv_bytes = 0
        last_log_time = time.monotonic()

        def receive_full_chunk(conn, chunk_size):
            data = b""
            conn.settimeout(0.5)  # Make socket non-blocking during timeout wait
            while not stop_event.is_set():
                try:
                    packet = conn.recv(chunk_size - len(data))
                except socket.timeout:
                    continue
                if not packet:
                    return None
                data += packet
                if len(data) == chunk_size:
                    return data
            return None

        while not stop_event.is_set():
            data = receive_full_chunk(recv_socket, listen_play_chunk_size * 2)
            if data is None:
                log("[Recv] No more data from server, exiting recv thread")
                break
            recv_queue.put(data)
            recv_chunks += 1
            recv_bytes += len(data)
            now = time.monotonic()
            if now - last_log_time >= 1.0:
                log(
                    f"[Recv] chunks={recv_chunks} bytes={recv_bytes} queue={recv_queue.qsize()}"
                )
                recv_chunks = 0
                recv_bytes = 0
                last_log_time = now
        log("[Recv] Thread exiting")

    send_stream = sd.RawInputStream(
        samplerate=send_rate,
        channels=1,
        dtype="int16",
        blocksize=listen_play_chunk_size,
        callback=callback_send,
    )
    recv_stream = sd.RawOutputStream(
        samplerate=recv_rate,
        channels=1,
        dtype="int16",
        blocksize=listen_play_chunk_size,
        callback=callback_recv,
    )

    send_thread = threading.Thread(target=send, args=(stop_event, send_queue), daemon=True)
    recv_thread = threading.Thread(target=recv, args=(stop_event, recv_queue), daemon=True)

    try:
        with send_stream, recv_stream:
            send_thread.start()
            recv_thread.start()

            # Wait for stop event (Ctrl+C or timeout loop)
            last_heartbeat = time.monotonic()
            while not stop_event.is_set():
                now = time.monotonic()
                if now - last_heartbeat >= 2.0:
                    log(
                        "[Heartbeat] "
                        f"send_alive={send_thread.is_alive()} "
                        f"recv_alive={recv_thread.is_alive()} "
                        f"muted={should_mute_mic()} "
                        f"send_q={send_queue.qsize()} recv_q={recv_queue.qsize()}"
                    )
                    last_heartbeat = now
                time.sleep(0.1)

    except KeyboardInterrupt:
        log("Received interrupt signal.")

    finally:
        log("Shutting down...")
        stop_event.set()

        # Kill any blocking socket operations
        try:
            recv_socket.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass  # Already closed or invalid

        send_thread.join(timeout=2.0)
        recv_thread.join(timeout=2.0)

        send_socket.close()
        recv_socket.close()
        log("All done.")


def _signal_handler(signum, frame):
    """Handle Ctrl+C gracefully."""
    stop_event.set()

def wait_for_enter(stop_event):
    try:
        input()
    except EOFError:
        pass  # Non-interactive
    stop_event.set()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Listen and play audio over network")
    parser.add_argument("--send-rate", type=int, default=16000, help="Input sample rate (Hz)")
    parser.add_argument("--recv-rate", type=int, default=16000, help="Output sample rate (Hz)")
    parser.add_argument("--chunk-size", type=int, default=1024, help="Audio chunk size in bytes")
    parser.add_argument("--host", type=str, default="localhost", help="Server hostname/IP")
    parser.add_argument("--send-port", type=int, default=12345, help="Send port")
    parser.add_argument("--recv-port", type=int, default=12346, help="Receive port")
    parser.add_argument(
        "--playback-mute-hold-ms",
        type=int,
        default=250,
        help="Keep mic muted for this many milliseconds after playback chunk",
    )

    args = parser.parse_args()

    # Override `stop_event` with global for signal handler
    stop_event = threading.Event()
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    threading.Thread(target=wait_for_enter, args=(stop_event,), daemon=True).start()
    listen_and_play(
        send_rate=args.send_rate,
        recv_rate=args.recv_rate,
        listen_play_chunk_size=args.chunk_size,
        host=args.host,
        send_port=args.send_port,
        recv_port=args.recv_port,
        playback_mute_hold_ms=args.playback_mute_hold_ms,
    )
