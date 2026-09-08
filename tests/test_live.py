"""Real client/worker/socket lifecycle without microphone, GPU or desktop input.

Run after cargo build: python3 -m unittest discover -s tests -v
"""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / "target/debug/textspill"

RECORDER = '''#!/usr/bin/env python3
import ctypes, os, signal, sys, time, wave
ctypes.CDLL(None).prctl(15, b"pw-record", 0, 0, 0)
running = True
def stop(*_):
    global running
    running = False
signal.signal(signal.SIGINT, stop)
signal.signal(signal.SIGTERM, stop)
with wave.open(sys.argv[-1], "wb") as audio:
    audio.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
    while running:
        audio.writeframes(b"\\x01\\x00" * 1600)
        time.sleep(.1)
'''

TOOL = '''#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
name = Path(sys.argv[0]).name
root = Path(os.environ["TEST_ROOT"])
with (root / "tools.jsonl").open("a") as log:
    log.write(json.dumps([name, sys.argv[1:]]) + "\\n")
if name == "wl-copy":
    (root / "clipboard").write_bytes(sys.stdin.buffer.read())
elif name == "wl-paste":
    sys.stdout.buffer.write((root / "clipboard").read_bytes())
elif name == "ydotool":
    if os.environ.get("TEST_PASTE_FAIL"):
        sys.exit(1)
    with (root / "typed").open("ab") as typed:
        typed.write((root / "clipboard").read_bytes())
elif name == "notify-send":
    print("123")
'''

SERVER = '''
import importlib.util, json, os, sys, time, wave
from pathlib import Path
spec = importlib.util.spec_from_file_location("daemon", sys.argv[1])
d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)
root = Path(os.environ["TEST_ROOT"])
class FakeModel:
    def transcribe(self, path, *, preview=False):
        with wave.open(str(path)) as wav:
            frames = wav.getnframes()
        with (root / "requests.jsonl").open("a") as log:
            log.write(json.dumps({"preview":preview,"frames":frames,"path":str(path)}) + "\\n")
        if preview:
            time.sleep(float(os.environ.get("TEST_PREVIEW_DELAY", "0")))
            if os.environ.get("TEST_PREVIEW_FAIL"):
                raise RuntimeError("preview unavailable")
        return {"text": "live svenska" if preview and path.name != "preview-final.wav" else "färdig svensk text", "language":"Swedish"}
d._install_shutdown_handler()
d.serve(Path(sys.argv[2]), FakeModel())
'''


class LiveLifecycle(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="ts-live-")
        self.root = Path(self.tmp.name)
        self.runtime = self.root / "textspill"
        self.runtime.mkdir(mode=0o700)
        bindir = self.root / "bin"
        bindir.mkdir()
        for name, content in [("pw-record", RECORDER)] + [
            (name, TOOL) for name in ["notify-send", "wl-copy", "wl-paste", "ydotool"]
        ]:
            path = bindir / name
            path.write_text(content)
            path.chmod(0o700)
        self.env = dict(os.environ, XDG_RUNTIME_DIR=str(self.root),
                        PATH=str(bindir) + os.pathsep + os.environ["PATH"],
                        TEST_ROOT=str(self.root), TEXTSPILL_LIVE="1",
                        TEXTSPILL_PASTE_BACKEND="ydotool")
        self.env.pop("HYPRLAND_INSTANCE_SIGNATURE", None)
        self.server = None

    def start_server(self, **settings):
        self.env.update(settings)
        self.server = subprocess.Popen([sys.executable, "-c", SERVER,
            str(ROOT / "asr/daemon.py"), str(self.runtime / "asr.sock")],
            env=self.env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.wait_for(lambda: (self.runtime / "asr.sock").exists())

    def run_cli(self, *args):
        return subprocess.run([str(BINARY), *args], env=self.env,
            capture_output=True, text=True, timeout=10, check=True).stdout.strip()

    def wait_for(self, condition, timeout=8):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if condition():
                return
            time.sleep(.05)
        self.fail("condition not reached within timeout")

    def requests(self):
        path = self.root / "requests.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    def tools(self):
        path = self.root / "tools.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    def tearDown(self):
        try:
            self.run_cli("cancel")
        finally:
            if self.server:
                self.server.terminate()
                self.server.wait(timeout=5)
            # Workers notice cancellation / socket closure without any PID kills.
            self.wait_for(lambda: not list(self.runtime.glob("preview-*.wav")))
            self.tmp.cleanup()

    def test_live_types_before_stop_and_final_only_appends_tail(self):
        self.start_server()
        self.run_cli("toggle")
        self.wait_for(lambda: (self.runtime / "preview.json").exists())
        self.wait_for(lambda: json.loads(self.run_cli("preview", "--json"))["consumed_bytes"] > 0)
        self.assertEqual(self.run_cli("status"), "recording")
        preview = json.loads(self.run_cli("preview", "--json"))
        self.assertEqual(preview["text"], "live svenska")
        self.assertFalse(preview["provisional"])
        self.wait_for(lambda: (self.root / "typed").exists())
        self.assertEqual((self.root / "typed").read_text(), "live svenska")
        time.sleep(.3)
        self.run_cli("toggle")
        self.assertEqual(self.run_cli("status"), "idle")
        self.assertEqual(self.run_cli("preview", "--json"), "null")
        self.assertEqual((self.root / "typed").read_text(), "live svenska färdig svensk text")
        self.assertEqual((self.root / "clipboard").read_text(), (self.root / "typed").read_text())
        self.assertEqual(sum(name == "ydotool" for name, _ in self.tools()), 2)
        self.assertLess(self.requests()[-1]["frames"], self.requests()[0]["frames"])
        self.assertFalse((self.runtime / "recording.wav").exists())

    def test_cancel_during_inference_discards_delayed_reply(self):
        self.start_server(TEST_PREVIEW_DELAY="1.5")
        self.run_cli("start")
        self.wait_for(lambda: any(r["preview"] for r in self.requests()))
        self.run_cli("cancel")
        count = len(self.tools())
        self.wait_for(lambda: not list(self.runtime.glob("preview-*.wav")))
        self.assertEqual(self.run_cli("preview", "--json"), "null")
        self.assertEqual(len(self.tools()), count, "late reply must not show a notification")
        self.assertFalse((self.root / "clipboard").exists())

    def test_preview_failure_keeps_recording_and_final_path(self):
        self.start_server(TEST_PREVIEW_FAIL="1")
        self.run_cli("start")
        self.wait_for(lambda: (self.runtime / "preview.json").exists())
        self.assertIn("preview unavailable", json.loads(self.run_cli("preview", "--json"))["error"])
        self.assertEqual(self.run_cli("status"), "recording")
        # Disable injected model failure for stop's snapshot inference.
        # This server fails every preview call, so final safely fails and
        # retains the audio instead of pasting a partial/duplicate transcript.
        result = subprocess.run([str(BINARY), "stop"], env=self.env, capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue((self.runtime / "recording.wav").exists())
        self.assertFalse((self.root / "typed").exists())

    def test_failed_paste_is_not_automatically_replayed_at_stop(self):
        self.start_server()
        self.env['TEST_PASTE_FAIL'] = '1'
        self.run_cli('start')
        self.wait_for(lambda: (self.runtime / 'preview.json').exists())
        self.wait_for(lambda: json.loads(self.run_cli('preview', '--json'))['error'] is not None)
        calls = sum(name == 'ydotool' for name, _ in self.tools())
        result = subprocess.run([str(BINARY), 'stop'], env=self.env, capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(sum(name == 'ydotool' for name, _ in self.tools()), calls)
        self.assertTrue((self.runtime / 'recording.wav').exists())

    def test_new_recording_rejects_previous_sessions_delayed_reply(self):
        self.start_server(TEST_PREVIEW_DELAY="1.5")
        self.run_cli("start")
        self.wait_for(lambda: any(r["preview"] for r in self.requests()))
        old_token = (self.runtime / "live-session").read_text()
        self.run_cli("cancel")
        # Start the next capture without a worker. An old reply must never
        # fill its preview or notification, even though state is recording.
        self.env["TEXTSPILL_LIVE"] = "0"
        self.run_cli("start")
        self.wait_for(lambda: not (self.runtime / f"preview-{old_token}.wav").exists())
        self.assertEqual(self.run_cli("status"), "recording")
        self.assertFalse((self.runtime / "preview.json").exists())
        self.assertFalse(any("🎙 Live · preliminary" in args for _, args in self.tools()))

    def test_opt_out_keeps_offline_flow(self):
        self.start_server()
        self.env["TEXTSPILL_LIVE"] = "0"
        self.run_cli("start")
        self.assertFalse((self.runtime / "live-session").exists())
        time.sleep(.7)
        self.run_cli("stop")
        self.assertEqual(len(self.requests()), 1)
        self.assertFalse(self.requests()[0]["preview"])


if __name__ == "__main__":
    unittest.main()
