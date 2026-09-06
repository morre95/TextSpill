#!/usr/bin/env python3
"""textspill-asr — keeps Qwen3-ASR warm and answers transcription requests.

Loading Qwen3-ASR takes seconds; a dictation must not. This daemon therefore
loads the model once, at service start, and then serves one line of JSON per
request over a Unix domain socket:

    -> {"audio_path": "/run/user/1000/textspill/recording.wav"}
    <- {"text": "kan du gå igenom den här implementationen", "language": "Swedish"}
    <- {"error": "no such file: ..."}

Language is detected automatically, so Swedish and English can be mixed between
dictations without touching a setting.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import socket
import struct
import sys
import time
import wave
from pathlib import Path

# The Transformers-native checkpoint. `Qwen/Qwen3-ASR-1.7B-hf` is the larger,
# more accurate sibling and is a drop-in replacement here.
DEFAULT_MODEL = "Qwen/Qwen3-ASR-0.6B-hf"

# A request is one short JSON object; anything larger is a broken peer.
MAX_REQUEST_BYTES = 64 * 1024

DEFAULT_MAX_NEW_TOKENS = 512
CLIENT_TIMEOUT_SECONDS = 30
SOCKET_BACKLOG = 8

log = logging.getLogger("textspill-asr")


# --------------------------------------------------------------------------
# Paths — mirrors src/paths.rs so both halves agree without a shared config.
# --------------------------------------------------------------------------


def runtime_dir() -> Path:
    base = os.environ.get("XDG_RUNTIME_DIR")
    if not base or not Path(base).is_dir():
        base = f"/run/user/{os.getuid()}"
        if not Path(base).is_dir():
            base = f"/tmp/textspill-{os.getuid()}"
    directory = Path(base) / "textspill"
    directory.mkdir(parents=True, exist_ok=True)
    directory.chmod(0o700)
    return directory


def config_dir() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME") or (Path.home() / ".config")
    return Path(base) / "textspill"


def load_context(explicit: Path | None) -> str | None:
    """Reads the vocabulary that biases recognition towards technical terms.

    Looked up in order: `--context`, `$XDG_CONFIG_HOME/textspill/context.txt`,
    then the `context.txt` shipped next to this file. Editing the config copy
    survives upgrades and needs only a service restart.
    """
    candidates = [
        explicit,
        config_dir() / "context.txt",
        Path(__file__).resolve().parent / "context.txt",
    ]
    for path in candidates:
        if path is None or not path.is_file():
            continue
        terms = [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        if not terms:
            continue
        log.info("loaded %d context terms from %s", len(terms), path)
        return "Vocabulary: " + ", ".join(terms) + "."
    log.info("no context file found; transcribing without vocabulary bias")
    return None


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------


class Transcriber:
    """Owns the model for the lifetime of the process."""

    def __init__(self, model_id: str, context: str | None, max_new_tokens: int) -> None:
        import torch
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        self._torch = torch
        self._context = context
        self._max_new_tokens = max_new_tokens

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        log.info("loading %s on %s (%s)", model_id, device, dtype)

        started = time.monotonic()
        self._processor = AutoProcessor.from_pretrained(model_id)
        self._model = AutoModelForMultimodalLM.from_pretrained(
            model_id, dtype=dtype, device_map=device
        )
        self._model.eval()
        log.info("model ready in %.1fs", time.monotonic() - started)

    def transcribe(self, audio_path: Path) -> dict[str, str]:
        """Returns `{"text": ..., "language": ...}` for one WAV file."""
        torch = self._torch
        request = {"audio": str(audio_path)}
        if self._context:
            request["prompt"] = self._context
        # `language` is left unset: that is what enables automatic language
        # identification, so Swedish and English need no manual switch.

        inputs = self._processor.apply_transcription_request(**request).to(
            self._model.device, self._model.dtype
        )
        with torch.inference_mode():
            output_ids = self._model.generate(
                **inputs, max_new_tokens=self._max_new_tokens
            )
        generated = output_ids[:, inputs["input_ids"].shape[1] :]
        parsed = self._processor.decode(generated, return_format="parsed")[0]

        return {
            "text": (parsed.get("transcription") or "").strip(),
            "language": parsed.get("language") or "",
        }

    def warm_up(self) -> None:
        """Runs one throwaway inference so the first real dictation is fast.

        The first `generate` call allocates workspaces and compiles CUDA
        kernels; doing it at service start moves that cost off the hotkey path.
        """
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "warmup.wav"
            _write_quiet_wav(path, seconds=1.0)
            started = time.monotonic()
            try:
                self.transcribe(path)
            except Exception:
                log.exception("warm-up inference failed; serving anyway")
                return
        log.info("warm-up inference took %.2fs", time.monotonic() - started)


def _write_quiet_wav(path: Path, seconds: float, rate: int = 16000) -> None:
    """Writes near-silent 16 kHz mono PCM, matching what `pw-record` produces."""
    frames = int(rate * seconds)
    samples = struct.pack("<%dh" % frames, *((i % 7) - 3 for i in range(frames)))
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(samples)


# --------------------------------------------------------------------------
# Socket server
# --------------------------------------------------------------------------


def read_line(conn: socket.socket) -> bytes:
    """Reads one newline-terminated request, bounded by MAX_REQUEST_BYTES."""
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = conn.recv(4096)
        if not chunk:
            break
        chunks.append(chunk)
        total += len(chunk)
        if b"\n" in chunk:
            break
        if total > MAX_REQUEST_BYTES:
            raise ValueError("request exceeded %d bytes" % MAX_REQUEST_BYTES)
    return b"".join(chunks).split(b"\n", 1)[0]


def handle_request(transcriber: Transcriber, payload: bytes) -> dict[str, str]:
    try:
        request = json.loads(payload)
    except json.JSONDecodeError as e:
        return {"error": f"invalid JSON request: {e}"}
    if not isinstance(request, dict):
        return {"error": "request must be a JSON object"}

    raw_path = request.get("audio_path")
    if not isinstance(raw_path, str) or not raw_path:
        return {"error": "request is missing `audio_path`"}

    audio_path = Path(raw_path)
    if not audio_path.is_absolute():
        return {"error": f"audio_path must be absolute: {raw_path}"}
    if not audio_path.is_file():
        return {"error": f"no such audio file: {raw_path}"}

    started = time.monotonic()
    try:
        result = transcriber.transcribe(audio_path)
    except Exception as e:  # noqa: BLE001 - any model failure is a client error
        log.exception("transcription failed")
        return {"error": f"failed to transcribe audio: {e}"}

    log.info(
        "transcribed %s in %.2fs [%s] %d chars",
        audio_path.name,
        time.monotonic() - started,
        result["language"] or "unknown",
        len(result["text"]),
    )
    return result


def _install_shutdown_handler() -> None:
    """Turns systemd's SIGTERM into the same clean exit as Ctrl-C.

    Without this, the default SIGTERM disposition skips the cleanup that unlinks
    the socket, leaving a stale entry behind after `systemctl --user stop`.
    """

    def raise_interrupt(_signum: int, _frame: object) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, raise_interrupt)


def serve(socket_path: Path, transcriber: Transcriber) -> None:
    # A socket left behind by a killed daemon would make bind() fail.
    if socket_path.exists():
        log.warning("removing stale socket %s", socket_path)
        socket_path.unlink()

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    # Create the socket private, then confirm it: the runtime dir is already
    # 0700, but the socket itself must never be world-writable either.
    old_umask = os.umask(0o177)
    try:
        server.bind(str(socket_path))
    finally:
        os.umask(old_umask)
    socket_path.chmod(0o600)
    server.listen(SOCKET_BACKLOG)
    log.info("listening on %s", socket_path)

    try:
        while True:
            conn, _ = server.accept()
            with conn:
                conn.settimeout(CLIENT_TIMEOUT_SECONDS)
                try:
                    response = handle_request(transcriber, read_line(conn))
                except (OSError, ValueError) as e:
                    log.warning("dropping client: %s", e)
                    continue
                try:
                    conn.sendall(
                        json.dumps(response, ensure_ascii=False).encode("utf-8") + b"\n"
                    )
                except OSError as e:
                    log.warning("client went away before the reply: %s", e)
    except KeyboardInterrupt:
        log.info("interrupted")
    finally:
        server.close()
        socket_path.unlink(missing_ok=True)
        log.info("socket removed")


# --------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.environ.get("TEXTSPILL_ASR_MODEL", DEFAULT_MODEL))
    parser.add_argument("--socket", type=Path, default=None, help="defaults to $XDG_RUNTIME_DIR/textspill/asr.sock")
    parser.add_argument("--context", type=Path, default=None, help="vocabulary file, one term per line")
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--no-warm-up", action="store_true", help="skip the startup inference")
    parser.add_argument("--transcribe", type=Path, default=None, help="transcribe one file, print JSON and exit")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )

    transcriber = Transcriber(args.model, load_context(args.context), args.max_new_tokens)

    if args.transcribe is not None:
        result = handle_request(transcriber, json.dumps({"audio_path": str(args.transcribe.resolve())}).encode())
        print(json.dumps(result, ensure_ascii=False))
        return 0 if "error" not in result else 1

    if not args.no_warm_up:
        transcriber.warm_up()

    _install_shutdown_handler()
    serve(args.socket or (runtime_dir() / "asr.sock"), transcriber)
    return 0


if __name__ == "__main__":
    sys.exit(main())
