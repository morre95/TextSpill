#!/usr/bin/env python3
"""Tests for daemon.py's protocol, path handling and vocabulary loading.

The model is replaced by a stub, so this runs in milliseconds and needs neither
a GPU nor the virtualenv:

    python3 asr/test_daemon.py
"""

from __future__ import annotations

import importlib.util
import json
import socket
import sys
import threading
import time
import unittest
import wave
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "textspill_asr_daemon", Path(__file__).resolve().parent / "daemon.py"
)
daemon = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(daemon)


class StubTranscriber:
    """Stands in for Qwen3-ASR."""

    def __init__(self, behaviour: str = "ok") -> None:
        self.behaviour = behaviour
        self.calls: list[Path] = []

    def transcribe(self, audio_path: Path) -> dict[str, str]:
        self.calls.append(audio_path)
        if self.behaviour == "boom":
            raise RuntimeError("CUDA out of memory")
        if self.behaviour == "empty":
            return {"text": "", "language": "Swedish"}
        return {"text": "kan du gå igenom den här implementationen", "language": "Swedish"}


class ContextTest(unittest.TestCase):
    def test_bundled_vocabulary_is_loaded(self):
        context = daemon.load_context(None)
        self.assertTrue(context.startswith("Vocabulary: "))
        self.assertIn("TextSpill", context)
        self.assertIn("Hyprland", context)
        self.assertTrue(context.endswith("."))

    def test_comments_and_blank_lines_are_ignored(self):
        path = Path(self.enterContext(_tempdir())) / "context.txt"
        path.write_text("# a comment\n\n  Rust  \n\n# another\nCargo\n", encoding="utf-8")
        self.assertEqual(daemon.load_context(path), "Vocabulary: Rust, Cargo.")

    def test_an_empty_file_falls_through_to_the_bundled_list(self):
        path = Path(self.enterContext(_tempdir())) / "context.txt"
        path.write_text("# nothing but comments\n", encoding="utf-8")
        self.assertIn("TextSpill", daemon.load_context(path))


class RequestValidationTest(unittest.TestCase):
    """`audio_path` arrives over a socket, so it is validated before use."""

    def setUp(self):
        self.stub = StubTranscriber()

    def assert_error(self, payload: bytes, fragment: str):
        response = daemon.handle_request(self.stub, payload)
        self.assertIn(fragment, response.get("error", ""))
        self.assertNotIn("text", response)
        self.assertEqual(self.stub.calls, [], "the model must not be reached")

    def test_malformed_json(self):
        self.assert_error(b"not json at all", "invalid JSON")

    def test_json_that_is_not_an_object(self):
        self.assert_error(b'["audio.wav"]', "must be a JSON object")

    def test_missing_audio_path(self):
        self.assert_error(b"{}", "missing `audio_path`")

    def test_audio_path_of_the_wrong_type(self):
        self.assert_error(b'{"audio_path": 42}', "missing `audio_path`")

    def test_relative_audio_path_is_refused(self):
        self.assert_error(b'{"audio_path": "recording.wav"}', "must be absolute")

    def test_missing_file(self):
        self.assert_error(b'{"audio_path": "/nonexistent/recording.wav"}', "no such audio file")

    def test_directory_is_not_a_file(self):
        self.assert_error(b'{"audio_path": "/etc"}', "no such audio file")

    def test_a_valid_request_reaches_the_model(self):
        wav = _quiet_wav(self)
        response = daemon.handle_request(self.stub, _request(wav))
        self.assertEqual(response["language"], "Swedish")
        self.assertIn("implementationen", response["text"])
        self.assertEqual(self.stub.calls, [wav])

    def test_a_model_failure_becomes_an_error_response(self):
        wav = _quiet_wav(self)
        response = daemon.handle_request(StubTranscriber("boom"), _request(wav))
        self.assertIn("failed to transcribe audio", response["error"])
        self.assertIn("CUDA out of memory", response["error"])


class WarmUpAudioTest(unittest.TestCase):
    def test_warm_up_audio_matches_what_pw_record_produces(self):
        path = Path(self.enterContext(_tempdir())) / "warmup.wav"
        daemon._write_quiet_wav(path, seconds=0.25)
        with wave.open(str(path)) as handle:
            self.assertEqual(handle.getnchannels(), 1)
            self.assertEqual(handle.getframerate(), 16000)
            self.assertEqual(handle.getsampwidth(), 2)
            self.assertEqual(handle.getnframes(), 4000)


class SocketTest(unittest.TestCase):
    """Exercises the real accept loop against a stub model."""

    def setUp(self):
        self.stub = StubTranscriber()
        self.socket_path = Path(self.enterContext(_tempdir())) / "asr.sock"
        thread = threading.Thread(
            target=daemon.serve, args=(self.socket_path, self.stub), daemon=True
        )
        thread.start()
        deadline = time.monotonic() + 5
        while not self.socket_path.exists():
            if time.monotonic() > deadline:
                self.fail("daemon did not create its socket")
            time.sleep(0.01)

    def ask(self, payload: bytes) -> dict:
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.settimeout(5)
        client.connect(str(self.socket_path))
        client.sendall(payload)
        buffer = b""
        while not buffer.endswith(b"\n"):
            chunk = client.recv(4096)
            if not chunk:
                break
            buffer += chunk
        client.close()
        return json.loads(buffer)

    def test_the_socket_is_private_to_the_user(self):
        self.assertEqual(self.socket_path.stat().st_mode & 0o777, 0o600)

    def test_newline_delimited_round_trip(self):
        wav = _quiet_wav(self)
        self.assertIn("implementationen", self.ask(_request(wav))["text"])

    def test_non_ascii_survives_the_round_trip(self):
        wav = _quiet_wav(self)
        self.assertIn("igenom den här", self.ask(_request(wav))["text"])

    def test_unknown_request_fields_are_tolerated(self):
        wav = _quiet_wav(self)
        payload = json.dumps({"audio_path": str(wav), "profile": "coding"}).encode() + b"\n"
        self.assertIn("text", self.ask(payload))

    def test_the_daemon_keeps_serving_after_a_bad_request(self):
        wav = _quiet_wav(self)
        self.assertIn("error", self.ask(b'{"audio_path": "/nope"}\n'))
        self.assertIn("text", self.ask(_request(wav)), "one bad client must not kill the daemon")


def _request(wav: Path) -> bytes:
    return json.dumps({"audio_path": str(wav)}).encode() + b"\n"


def _quiet_wav(case: unittest.TestCase) -> Path:
    path = Path(case.enterContext(_tempdir())) / "recording.wav"
    daemon._write_quiet_wav(path, seconds=0.25)
    return path


def _tempdir():
    import tempfile

    return tempfile.TemporaryDirectory()


if __name__ == "__main__":
    unittest.main(verbosity=2)
