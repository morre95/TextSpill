"""Installer branches in a temporary checkout with fake build/service commands."""
import json
import os
from pathlib import Path
import shutil
import shlex
import subprocess
import tempfile
import unittest
from test_live import ROOT, BINARY


class Installer(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="ts-install-")
        self.root = Path(self.tmp.name)
        self.repo = self.root / "repo"
        self.repo.mkdir()
        shutil.copy(ROOT / "install.sh", self.repo / "install.sh")
        (self.repo / "target/release").mkdir(parents=True)
        shutil.copy(BINARY, self.repo / "target/release/textspill")
        (self.repo / "asr").mkdir()
        (self.repo / "asr/daemon.py").touch()
        (self.repo / "systemd").mkdir()
        shutil.copy(ROOT / "systemd/textspill-asr.service", self.repo / "systemd")
        bin_dir = self.root / "tools"
        bin_dir.mkdir()
        for name in ["cargo", "systemctl"]:
            tool = bin_dir / name
            tool.write_text('#!/bin/sh\nprintf "%s\\n" "$0 $*" >> "$TEST_INSTALL_LOG"\n')
            tool.chmod(0o755)
        self.env = dict(os.environ, HOME=str(self.root / "home"), XDG_CONFIG_HOME=str(self.root / "config"),
                        XDG_DATA_HOME=str(self.root / "data"), TEST_INSTALL_LOG=str(self.root / "commands"),
                        PATH=str(bin_dir) + os.pathsep + os.environ["PATH"])
        self.env.pop("TEXTSPILL_BACKEND", None)
        self.config_dir = self.root / "config/textspill"

    def tearDown(self):
        self.tmp.cleanup()

    def install(self, *args):
        return subprocess.run(["bash", str(self.repo / "install.sh"), *args], env=self.env,
                              stdin=subprocess.DEVNULL, capture_output=True, text=True, check=True)

    def commands(self):
        path = self.root / "commands"
        return path.read_text() if path.exists() else ""

    def test_deepgram_skips_local_dependencies_and_preserves_settings_on_rerun(self):
        self.config_dir.mkdir(parents=True)
        (self.config_dir / "config.json").write_text('{"backend":"deepgram","deepgram_language":"en"}')
        key = self.config_dir / "deepgram-api-key"
        key.write_text("unchanged-key")
        key.chmod(0o600)
        self.install("--backend", "deepgram")
        self.install()
        self.assertFalse((self.root / "data/textspill/venv").exists())
        self.assertNotIn("systemctl", self.commands())
        self.assertEqual(key.read_text(), "unchanged-key")
        self.assertEqual(json.loads((self.config_dir / "config.json").read_text())["deepgram_language"], "en")

    def test_switch_to_deepgram_disables_existing_model_service_without_deleting_it(self):
        units = self.root / "config/systemd/user"
        units.mkdir(parents=True)
        unit = units / "textspill-asr.service"
        unit.write_text("old service")
        self.install("--backend", "deepgram")
        self.assertIn("--user disable --now textspill-asr.service", self.commands())
        self.assertEqual(unit.read_text(), "old service")

    def test_local_default_uses_existing_venv_and_installs_service(self):
        share = Path(self.env["XDG_DATA_HOME"]) / "textspill"
        self.prepare_venv(share)
        self.install()
        self.assertIn("pip install --upgrade torch", self.commands())
        self.assertIn("--user daemon-reload", self.commands())
        self.assert_service_paths(share)
        self.assertEqual(json.loads((self.config_dir / "config.json").read_text())["backend"], "local")

    def prepare_venv(self, share):
        venv = share / "venv/bin"
        venv.mkdir(parents=True)
        for name in ["python", "pip"]:
            tool = venv / name
            tool.write_text('#!/bin/sh\nprintf "%s\\n" "$0 $*" >> "$TEST_INSTALL_LOG"\n')
            tool.chmod(0o755)

    def assert_service_paths(self, share):
        unit = self.root / "config/systemd/user/textspill-asr.service"
        command = next(line.removeprefix("ExecStart=") for line in unit.read_text().splitlines()
                       if line.startswith("ExecStart="))
        self.assertTrue(command.startswith(":/usr/bin/env "))
        # ':' disables environment expansion; percent specifiers still expand.
        args = [arg.replace("%%", "%") for arg in shlex.split(command)[1:]]
        self.assertEqual(args, [str(share / "venv/bin/python"), str(share / "daemon.py")])
        self.assertTrue(all(Path(arg).exists() for arg in args))
        self.assertEqual(unit.stat().st_mode & 0o777, 0o644)

    def test_local_service_handles_spaces_and_systemd_metacharacters(self):
        self.env["XDG_DATA_HOME"] = str(self.root / 'data with % and $ and "quotes"')
        share = Path(self.env["XDG_DATA_HOME"]) / "textspill"
        self.prepare_venv(share)
        self.install("--backend", "local")
        self.assert_service_paths(share)
        if shutil.which("systemd-analyze"):
            unit = self.root / "config/systemd/user/textspill-asr.service"
            result = subprocess.run(["systemd-analyze", "verify", "--man=no", str(unit)],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_local_service_uses_default_data_path_when_xdg_is_unset(self):
        self.env.pop("XDG_DATA_HOME")
        share = Path(self.env["HOME"]) / ".local/share/textspill"
        self.prepare_venv(share)
        self.install("--backend", "local")
        self.assert_service_paths(share)

    def test_invalid_backend_fails_before_build(self):
        result = subprocess.run(["bash", str(self.repo / "install.sh"), "--backend", "bad"],
                                env=self.env, capture_output=True)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(self.commands(), "")
