"""Full CLI integration against a loopback Deepgram peer. Debug build required."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import unittest
import test_live
from test_live import ROOT, BINARY


class DeepgramLifecycle(unittest.TestCase):
    run_cli = test_live.LiveLifecycle.run_cli
    wait_for = test_live.LiveLifecycle.wait_for
    tools = test_live.LiveLifecycle.tools
    tearDown = test_live.LiveLifecycle.tearDown

    def setUp(self):
        test_live.LiveLifecycle.setUp(self)
        self.env.update(TEXTSPILL_BACKEND="deepgram", DEEPGRAM_API_KEY="fake-key")
        self.env.pop("TEXTSPILL_DEEPGRAM_LANGUAGE", None)
        self.env.pop("_TEXTSPILL_TEST_DEEPGRAM_ENDPOINT", None)

    def start_server(self, **settings):
        self.env.update(settings)
        self.server = subprocess.Popen([sys.executable, str(ROOT / "tests/fake_deepgram.py")],
                                       env=self.env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.wait_for(lambda: (self.root / "endpoint").exists())
        self.env["_TEXTSPILL_TEST_DEEPGRAM_ENDPOINT"] = (self.root / "endpoint").read_text()

    def requests(self):
        path = self.root / "deepgram.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    def failure(self, command, timeout=10):
        result = subprocess.run([str(BINARY), command], env=self.env, capture_output=True, text=True, timeout=timeout)
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("fake-key", result.stderr)
        return result.stderr

    def test_stream_live_final_tail_duplicate_and_frozen_settings(self):
        self.start_server()
        self.run_cli("start")
        self.wait_for(lambda: (self.root / "typed").exists())
        self.assertEqual((self.root / "typed").read_text(), "live svenska")
        time.sleep(.3)
        self.env.update(TEXTSPILL_BACKEND="local", TEXTSPILL_DEEPGRAM_LANGUAGE="en", TEXTSPILL_LIVE="0")
        self.run_cli("stop")
        self.assertEqual((self.root / "typed").read_text(), "live svenska sista orden")
        self.assertEqual((self.root / "clipboard").read_text(), "live svenska sista orden")
        self.assertEqual(self.run_cli("status"), "idle")
        self.assertFalse((self.runtime / "recording.wav").exists())
        requests = self.requests()
        ws = next(r for r in requests if "ws" in r)
        self.assertIn("language=sv", ws["ws"])
        self.assertIn("sample_rate=16000", ws["ws"])
        self.assertTrue(ws["auth"])
        self.assertTrue(all(r["raw_pcm"] and r["audio"] <= 3200 for r in requests if "audio" in r))
        self.assertEqual(sum(r.get("type") == "CloseStream" for r in requests), 1)

    def test_classic_uses_http_and_sanitizes_text(self):
        self.start_server()
        self.env.update(TEXTSPILL_LIVE="0", TEXTSPILL_DEEPGRAM_LANGUAGE="en")
        self.run_cli("start")
        time.sleep(.4)
        self.run_cli("stop")
        self.assertEqual((self.root / "typed").read_text(), "hela texten")
        request = self.requests()[0]
        self.assertTrue(request["wav"] and request["auth"])
        self.assertIn("language=en", request["http"])

    def test_missing_key_preserves_previous_audio_and_readonly_commands_work(self):
        self.env.pop("DEEPGRAM_API_KEY")
        wav = self.runtime / "recording.wav"
        wav.write_bytes(b"previous recording")
        self.assertIn("API", self.failure("start"))
        self.assertEqual(wav.read_bytes(), b"previous recording")
        self.assertEqual(self.run_cli("status"), "idle")
        self.assertEqual(self.run_cli("preview", "--json"), "null")
        self.run_cli("cancel")

    def test_private_key_file_works_without_environment_export(self):
        self.start_server()
        self.env.pop("DEEPGRAM_API_KEY")
        folder = Path(self.env["XDG_CONFIG_HOME"]) / "textspill"
        folder.mkdir(parents=True)
        key = folder / "deepgram-api-key"
        key.write_text("fake-key\n")
        key.chmod(0o600)
        self.env["TEXTSPILL_LIVE"] = "0"
        self.run_cli("start")
        time.sleep(.4)
        self.run_cli("stop")
        self.assertTrue(self.requests()[0]["auth"])

    def test_cancel_discards_late_results_and_next_session_is_isolated(self):
        self.start_server(TEST_DG_DELAY="1")
        self.run_cli("start")
        self.wait_for(lambda: sum(r.get("audio", 0) for r in self.requests()) >= 16000)
        self.run_cli("cancel")
        time.sleep(1.2)
        self.assertFalse((self.root / "typed").exists())
        self.run_cli("start")
        self.wait_for(lambda: (self.root / "typed").exists())
        self.run_cli("stop")
        self.assertEqual((self.root / "typed").read_text(), "live svenska sista orden")

    def test_network_drop_preserves_audio_without_http_fallback(self):
        self.start_server(TEST_DG_DROP="1")
        self.run_cli("start")
        self.wait_for(lambda: json.loads((self.runtime / "stream.json").read_text())["phase"] == "error")
        self.failure("stop")
        self.assertTrue((self.runtime / "recording.wav").exists())
        self.assertTrue(json.loads((self.runtime / "preview.json").read_text())["error"])
        self.assertFalse(any("http" in r for r in self.requests()))

    def test_worker_crash_does_not_replay_committed_text(self):
        self.start_server()
        self.run_cli("start")
        self.wait_for(lambda: (self.root / "typed").exists())
        worker = json.loads((self.runtime / "stream.json").read_text())["worker_pid"]
        os.kill(worker, signal.SIGKILL)
        self.assertIn("worker exited", self.failure("stop"))
        self.assertEqual((self.root / "typed").read_text(), "live svenska")
        self.assertTrue((self.runtime / "recording.wav").exists())

    def test_failed_paste_is_never_replayed_at_stop(self):
        self.start_server(TEST_PASTE_FAIL="1")
        self.run_cli("start")
        self.wait_for(lambda: json.loads((self.runtime / "stream.json").read_text())["phase"] == "error")
        self.failure("stop")
        self.assertTrue(json.loads((self.runtime / "preview.json").read_text())["delivery_pending"])
        self.assertEqual(sum(name == "ydotool" for name, _ in self.tools()), 1)

    def test_stop_wait_does_not_block_status_or_cancel(self):
        self.start_server(TEST_DG_HANG_CLOSE="1")
        self.run_cli("start")
        time.sleep(.4)
        stop = subprocess.Popen([str(BINARY), "stop"], env=self.env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            self.wait_for(lambda: any(r.get("type") == "CloseStream" for r in self.requests()))
            self.assertEqual(self.run_cli("status"), "transcribing")
            self.run_cli("cancel")
            self.assertEqual(stop.wait(timeout=3), 0)
        finally:
            if stop.poll() is None:
                stop.kill()
            stop.communicate()

    def test_configure_preserves_language_and_unknown_fields(self):
        folder = Path(self.env["XDG_CONFIG_HOME"]) / "textspill"
        folder.mkdir(parents=True)
        path = folder / "config.json"
        path.write_text(json.dumps({"backend": "local", "deepgram_language": "en", "custom": 1}))
        self.run_cli("configure", "--backend", "deepgram")
        self.assertEqual(json.loads(path.read_text()), {"backend": "deepgram", "deepgram_language": "en", "custom": 1})
        self.assertEqual(path.stat().st_mode & 0o777, 0o600)

    def test_silence_finishes_without_pasting(self):
        self.start_server(TEST_DG_SILENCE="1")
        self.run_cli("start")
        time.sleep(.7)
        self.run_cli("stop")
        self.assertFalse((self.root / "typed").exists())
        self.assertFalse((self.runtime / "recording.wav").exists())

    def test_stream_authentication_failure_keeps_audio(self):
        self.start_server(TEST_DG_STATUS="401")
        self.run_cli("start")
        self.wait_for(lambda: json.loads((self.runtime / "stream.json").read_text())["phase"] == "error")
        self.assertIn("401", self.failure("stop"))
        self.assertTrue((self.runtime / "recording.wav").exists())

    def test_malformed_final_reply_keeps_already_pasted_text(self):
        self.start_server(TEST_DG_BAD_JSON="1")
        self.run_cli("start")
        self.wait_for(lambda: (self.root / "typed").exists())
        self.assertIn("JSON", self.failure("stop"))
        self.assertEqual((self.root / "typed").read_text(), "live svenska")
        self.assertTrue((self.runtime / "recording.wav").exists())

    def test_keepalive_when_capture_produces_no_more_samples(self):
        self.start_server(TEST_CAPTURE_PAUSE="1")
        self.run_cli("start")
        self.wait_for(lambda: any(r.get("type") == "KeepAlive" for r in self.requests()))
        self.run_cli("stop")

    def test_stop_timeout_preserves_recovery_files(self):
        self.start_server(TEST_DG_HANG_CLOSE="1")
        self.run_cli("start")
        time.sleep(.4)
        started = time.monotonic()
        self.assertIn("30s", self.failure("stop", timeout=35))
        self.assertLess(time.monotonic() - started, 33)
        self.assertTrue((self.runtime / "recording.wav").exists())
        self.assertTrue(json.loads((self.runtime / "preview.json").read_text())["error"])
