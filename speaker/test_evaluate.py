"""Evaluation safeguards and metrics; no model downloads or microphone needed."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch

import evaluate as ev


class EvaluationTests(unittest.TestCase):
    def test_embeddings_are_normalized_and_invalid_embeddings_rejected(self):
        np.testing.assert_allclose(ev.unit(np.array([3, 4])), [.6, .8])
        for vector in [[], [0, 0], [np.nan, 1], [np.inf, 1]]:
            with self.subTest(vector=vector), self.assertRaises(ValueError):
                ev.unit(np.array(vector))

    def test_malformed_audio_is_not_treated_as_a_voice(self):
        for pcm in [[], [[1, 0]], [np.nan], [np.inf], [1.1]]:
            with self.subTest(pcm=pcm), self.assertRaises(ValueError):
                ev.validate_pcm(np.array(pcm))

    def test_silence_and_short_audio_have_no_identity_score(self):
        # No session is needed: these inputs must be rejected before inference.
        cam = ev.CamPlusPlus.__new__(ev.CamPlusPlus)
        self.assertIsNone(cam.score(np.zeros(ev.RATE), np.ones(192)))
        self.assertIsNone(cam.score(np.ones(100), np.ones(192)))

    def test_wrong_model_contents_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model.onnx"
            model.write_bytes(b"not the pinned model")
            with self.assertRaisesRegex(ValueError, "unexpected model"):
                ev.check_hash(model, ev.CAM_SHA256)

    def test_si_sdr_is_gain_invariant_and_detects_interference(self):
        time = np.arange(16000) / 16000
        target = np.sin(2 * np.pi * 220 * time)
        other = np.sin(2 * np.pi * 440 * time)
        self.assertGreater(ev.si_sdr(target, target * .1), 100)
        self.assertAlmostEqual(ev.si_sdr(target, target + other), 0, places=5)
        self.assertIsNone(ev.si_sdr(np.zeros(16000), other))
        with self.assertRaises(ValueError):
            ev.si_sdr(target, other[:10])

    def test_audio_reader_rejects_stereo_and_wrong_rate(self):
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / "audio.wav"
            for samples, rate in [(np.zeros((100, 2)), 16000), (np.zeros(100), 8000)]:
                sf.write(audio, samples, rate)
                with self.assertRaisesRegex(ValueError, "mono 16 kHz"):
                    ev.read_audio(audio)

    def test_manifest_rejects_same_file_for_enrollment_and_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sf.write(root / "voice.wav", np.zeros(3 * ev.RATE), ev.RATE)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"enrollment": "voice.wav", "cases": [
                {"name": "self", "audio": "voice.wav", "target_present": True}
            ]}))
            with self.assertRaisesRegex(ValueError, "must differ"):
                ev.load_manifest(manifest)

    def test_requested_threads_are_restored_after_model_import(self):
        class Cam:
            def __init__(self, *args, **kwargs):
                pass

            def embed(self, pcm):
                return np.array([1., 0.])

            def score(self, pcm, profile):
                return .8

        class Separator:
            def __init__(self, *args):
                # Reproduce Silero's import side effect.
                torch.set_num_threads(1)

            def extract(self, pcm, enrollment):
                if torch.get_num_threads() != 4:
                    raise AssertionError("thread limit was reset by model import")
                return pcm

        from argparse import Namespace
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sf.write(root / "enroll.wav", np.ones(3 * ev.RATE) * .1, ev.RATE)
            sf.write(root / "test.wav", np.ones(2 * ev.RATE) * .1, ev.RATE)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"enrollment": "enroll.wav", "cases": [
                {"name": "target", "audio": "test.wav", "target_present": True}
            ]}))
            args = Namespace(manifest=manifest, threads=4, window_seconds=4,
                             repeats=1, cam=root / "unused", wesep=root / "unused")
            with patch.object(ev, "CamPlusPlus", Cam), patch.object(ev, "Separator", Separator), \
                    patch.object(torch, "set_num_interop_threads"), patch("builtins.print"):
                report = ev.evaluate(args)
            self.assertFalse(report["production_ready"])
            self.assertEqual(report["results"][0]["audio_seconds"], 2)


if __name__ == "__main__":
    unittest.main()
