"""Personal speaker threshold calibration tests; no models or microphone needed."""
from pathlib import Path
import tempfile
import unittest

import calibrate
import numpy as np
import soundfile as sf


def row(name: str, target: bool, score: float | None) -> dict:
    return {"name": name, "target_present": target, "extracted_cosine": score}


class CalibrationTests(unittest.TestCase):
    def test_perfectly_separated_calibration_selects_a_working_midpoint(self):
        rows = [
            row("target-a", True, .8), row("target-b", True, .7),
            row("other-a", False, .2), row("other-b", False, .1),
        ]
        threshold, result = calibrate.choose_threshold(rows)
        self.assertGreater(threshold, .2)
        self.assertLess(threshold, .7)
        self.assertEqual(result["false_acceptances"], 0)
        self.assertEqual(result["false_rejections"], 0)

    def test_missing_score_is_rejected_not_silently_omitted(self):
        result = calibrate.metrics(
            [row("silent-target", True, None), row("silence", False, None)], .5
        )
        self.assertEqual(result["false_rejections"], 1)
        self.assertEqual(result["false_acceptances"], 0)

    def test_calibration_and_held_out_metrics_are_kept_separate(self):
        report = {"results": [
            row("cal-target", True, .8), row("cal-other", False, .1),
            row("test-target", True, .4), row("test-other", False, .6),
        ]}
        manifest = {"cases": [
            {"name": "cal-target", "split": "calibration", "condition": "target"},
            {"name": "cal-other", "split": "calibration", "condition": "other"},
            {"name": "test-target", "split": "test", "condition": "target"},
            {"name": "test-other", "split": "test", "condition": "other"},
        ]}
        result = calibrate.add_calibration(report, manifest)["personal_calibration"]
        self.assertTrue(result["ready"])
        self.assertEqual(result["calibration"]["false_acceptances"], 0)
        self.assertEqual(result["held_out_test"]["false_acceptances"], 1)
        self.assertEqual(result["held_out_test"]["false_rejections"], 1)

    def test_dataset_is_private_and_existing_manifest_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "personal"
            manifest = calibrate.init_dataset(dataset)
            self.assertEqual(manifest.stat().st_mode & 0o777, 0o600)
            self.assertEqual(dataset.stat().st_mode & 0o777, 0o700)
            with self.assertRaisesRegex(ValueError, "already exists"):
                calibrate.init_dataset(dataset)

    def test_case_names_cannot_escape_the_audio_directory(self):
        for name in ["../voice", "/tmp/voice", "Voice", "", "a.b"]:
            with self.subTest(name=name), self.assertRaises(ValueError):
                calibrate.validate_name(name)

    def test_valid_recording_is_accepted_when_sigint_status_is_nonzero(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recording.wav"
            sf.write(path, np.ones(18 * 16000) * .01, 16000, subtype="PCM_16")
            _, duration = calibrate.validate_recording(
                path, -2, str(path), requested_seconds=20, minimum_seconds=15
            )
            self.assertEqual(duration, 18)

    def test_short_recording_reports_process_status_and_diagnostic(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recording.wav"
            sf.write(path, np.ones(2 * 16000) * .01, 16000, subtype="PCM_16")
            with self.assertRaisesRegex(ValueError, "status 1: device disconnected"):
                calibrate.validate_recording(
                    path, 1, "device disconnected", requested_seconds=20,
                    minimum_seconds=15,
                )

    def test_doctor_repairs_missing_references_and_keeps_a_backup(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "personal"
            manifest_path = calibrate.init_dataset(dataset)
            _, manifest = calibrate.load_manifest(dataset)
            manifest["cases"] = [
                {"name": "missing", "audio": "audio/missing.wav", "target_present": True},
                {"name": "present", "audio": "audio/present.wav", "target_present": False},
            ]
            sf.write(dataset / "audio/present.wav", np.ones(16000) * .01, 16000)
            calibrate.private_json(manifest_path, manifest)
            missing, invalid = calibrate.inspect_case_audio(manifest_path, manifest)
            self.assertEqual(missing, ["missing"])
            self.assertEqual(invalid, [])
            backup = calibrate.repair_missing_cases(manifest_path, manifest, missing)
            self.assertTrue(backup.is_file())
            _, repaired = calibrate.load_manifest(dataset)
            self.assertEqual([case["name"] for case in repaired["cases"]], ["present"])


if __name__ == "__main__":
    unittest.main()
