#!/usr/bin/env python3
"""Offline CAM++ / WeSep feasibility evaluation; never records or delivers text.

The manifest contains an enrollment WAV and labelled evaluation WAVs. All audio
must be mono 16 kHz. Scores are cosine similarities, not probabilities. This
tool deliberately does not choose a production verification threshold.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import resource
import statistics
import time
import warnings

import numpy as np
import soundfile as sf
import torch
import torchaudio.compliance.kaldi as kaldi

RATE = 16000
CAM_SHA256 = "b50810498b5bcf5773d086f6993d344476bd0c88b566a41e8d801aaf8461efad"
WESEP_SHA256 = "3d0502171eab31b7cf25835f35d1969b415bb95f2ac52e2c5e2a743ebd8f90e5"
CONFIG_SHA256 = "3bd99d6029a837f9db4a6e57e2cd7bf2f9af226dc3be95ae9862d34af628530d"


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def check_hash(path: Path, expected: str) -> None:
    if digest(path) != expected:
        raise ValueError(f"unexpected model contents: {path}")


def read_audio(path: Path) -> np.ndarray:
    pcm, rate = sf.read(path, dtype="float32", always_2d=True)
    if rate != RATE or pcm.shape[1] != 1:
        raise ValueError(f"{path}: requires mono 16 kHz audio")
    return validate_pcm(pcm[:, 0])


def validate_pcm(pcm: np.ndarray) -> np.ndarray:
    pcm = np.asarray(pcm, dtype=np.float32)
    if pcm.ndim != 1 or not len(pcm) or not np.isfinite(pcm).all():
        raise ValueError("audio must be nonempty, finite, mono samples")
    if np.max(np.abs(pcm)) > 1.0001:
        raise ValueError("audio exceeds the normalized PCM range")
    return pcm


def unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(vector.astype(np.float64))
    if not len(vector) or not np.isfinite(vector).all() or not np.isfinite(norm) or norm < 1e-8:
        raise ValueError("invalid or zero speaker embedding")
    return (vector / norm).astype(np.float32)


class CamPlusPlus:
    """Pinned WeSpeaker VoxCeleb CAM++, with upstream infer_onnx preprocessing."""

    def __init__(self, model: Path, threads: int = 1):
        import onnxruntime as ort

        check_hash(model, CAM_SHA256)
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(model), sess_options=options, providers=["CPUExecutionProvider"]
        )

    def embed(self, pcm: np.ndarray) -> np.ndarray:
        pcm = validate_pcm(pcm)
        if len(pcm) < RATE:
            raise ValueError("CAM++ evaluation requires at least one second")
        if float(np.sqrt(np.mean(pcm.astype(np.float64) ** 2))) < 1e-6:
            raise ValueError("cannot identify a speaker in silence")
        waveform = torch.from_numpy(pcm.copy()).unsqueeze(0) * (1 << 15)
        # These parameters must stay identical for enrollment and verification.
        features = kaldi.fbank(
            waveform, num_mel_bins=80, frame_length=25, frame_shift=10,
            dither=0.0, sample_frequency=RATE, window_type="hamming",
            use_energy=False,
        )
        features -= features.mean(dim=0)
        embedding = self.session.run(
            ["embs"], {"feats": features.unsqueeze(0).numpy()}
        )[0]
        return unit(embedding)

    def score(self, pcm: np.ndarray, profile: np.ndarray) -> float | None:
        if len(pcm) < RATE or np.sqrt(np.mean(pcm.astype(np.float64) ** 2)) < 1e-6:
            return None
        return float(np.clip(np.dot(self.embed(pcm), profile), -1, 1))


class Separator:
    def __init__(self, model_dir: Path):
        check_hash(model_dir / "avg_model.pt", WESEP_SHA256)
        check_hash(model_dir / "config.yaml", CONFIG_SHA256)
        from wesep import load_model_local

        # This pinned checkpoint is hash-verified immediately above. Suppress
        # PyTorch's generic warning from the upstream loader; it is actionable
        # for arbitrary pickle inputs, not for this exact verified file.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"You are using `torch\.load` with `weights_only=False`.*",
                category=FutureWarning,
            )
            self.model = load_model_local(str(model_dir))
        self.model.set_device("cpu")
        # Peak normalization would amplify residual speech when the target is
        # absent and can divide by zero for silent output. Keep native levels.
        self.model.set_output_norm(False)

    def extract(self, pcm: np.ndarray, enrollment: np.ndarray) -> np.ndarray:
        with torch.inference_mode():
            output = self.model.extract_speech_from_pcm(
                torch.from_numpy(pcm.copy()).unsqueeze(0), RATE,
                torch.from_numpy(enrollment.copy()).unsqueeze(0), RATE,
            )
        if output is None:
            raise ValueError("separator returned no waveform")
        result = output.detach().cpu().numpy().reshape(-1)
        if result.shape != pcm.shape or not np.isfinite(result).all():
            raise ValueError("separator returned invalid waveform or duration")
        return np.clip(result, -1, 1).astype(np.float32)


def si_sdr(reference: np.ndarray, estimate: np.ndarray) -> float | None:
    if reference.shape != estimate.shape:
        raise ValueError("clean reference and estimate must have equal length")
    reference = reference.astype(np.float64) - reference.mean()
    estimate = estimate.astype(np.float64) - estimate.mean()
    energy = np.dot(reference, reference)
    if energy < 1e-12:
        return None
    projection = reference * (np.dot(reference, estimate) / energy)
    return float(10 * np.log10(
        (np.dot(projection, projection) + 1e-12)
        / (np.sum((estimate - projection) ** 2) + 1e-12)
    ))


def load_manifest(path: Path) -> tuple[np.ndarray, list[dict]]:
    data = json.loads(path.read_text())
    enrollment_path = (path.parent / data["enrollment"]).resolve()
    enrollment = read_audio(enrollment_path)
    if len(enrollment) < 3 * RATE:
        raise ValueError("evaluation enrollment must contain at least 3 seconds")
    cases = data["cases"]
    if not isinstance(cases, list) or not cases:
        raise ValueError("manifest requires at least one case")
    names = set()
    for case in cases:
        if not isinstance(case["name"], str) or not case["name"] or case["name"] in names:
            raise ValueError("case names must be nonempty and unique")
        names.add(case["name"])
        if type(case["target_present"]) is not bool:
            raise ValueError("target_present must be a boolean")
        case["path"] = (path.parent / case["audio"]).resolve()
        if case["path"] == enrollment_path:
            raise ValueError("evaluation audio must differ from enrollment")
        case["pcm"] = read_audio(case["path"])
        if "clean_target" in case:
            case["reference"] = read_audio((path.parent / case["clean_target"]).resolve())
            if case["reference"].shape != case["pcm"].shape:
                raise ValueError("clean target must match mixture length")
    return enrollment, cases


def evaluate(args: argparse.Namespace) -> dict:
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    enrollment, cases = load_manifest(args.manifest)
    started = time.perf_counter()
    cam = CamPlusPlus(args.cam, threads=1)
    separator = Separator(args.wesep)
    # silero_vad (imported by WeSep) resets PyTorch's thread count at import.
    # Apply the requested limit after all model imports, not just before them.
    torch.set_num_threads(args.threads)
    load_seconds = time.perf_counter() - started
    # Multiple enrollment sections, with duration-weighted pooling avoided: each
    # section has an equal vote. This is experimental, not production enrollment.
    section_count = max(1, len(enrollment) // (3 * RATE))
    profile = unit(np.mean([cam.embed(p) for p in np.array_split(enrollment, section_count)], axis=0))
    warm_audio = cases[0]["pcm"][:4 * RATE]
    cam.score(warm_audio, profile)
    separator.extract(warm_audio, enrollment)
    rows = []
    window = int(args.window_seconds * RATE)
    for case in cases:
        pcm = case["pcm"]
        # Keep every sample. Very short tails are measured but cannot be
        # identified by CAM++; score() returns None instead of a fabricated ID.
        for offset in range(0, len(pcm), window):
            part = pcm[offset:offset + window]
            if len(part) < 512:
                raise ValueError("tail shorter than separator STFT window; adjust fixture length")
            raw_score = cam.score(part, profile)
            extraction_times, verification_times = [], []
            for _ in range(args.repeats):
                started = time.perf_counter()
                output = separator.extract(part, enrollment)
                extraction_times.append(time.perf_counter() - started)
                started = time.perf_counter()
                score = cam.score(output, profile)
                verification_times.append(time.perf_counter() - started)
            duration = len(part) / RATE
            row = {
                "name": case["name"], "language": case.get("language", "unknown"),
                "target_present": case["target_present"],
                "audio_sha256": digest(case["path"]),
                "start_seconds": offset / RATE, "audio_seconds": duration,
                "raw_cosine": raw_score, "extracted_cosine": score,
                "extraction_seconds": extraction_times,
                "verification_seconds": verification_times,
                "median_rtf": statistics.median([
                    (a + b) / duration for a, b in zip(extraction_times, verification_times)
                ]),
                "output_rms": float(np.sqrt(np.mean(output.astype(np.float64) ** 2))),
            }
            if "reference" in case:
                reference = case["reference"][offset:offset + len(part)]
                row["input_si_sdr_db"] = si_sdr(reference, part)
                row["output_si_sdr_db"] = si_sdr(reference, output)
            rows.append(row)
            if not getattr(args, "quiet", False):
                print(json.dumps(row, allow_nan=False), flush=True)
    manifest_data = json.loads(args.manifest.read_text())
    return {
        "schema_version": 1, "purpose": "feasibility_only",
        "threads": torch.get_num_threads(), "window_seconds": args.window_seconds,
        "repeats": args.repeats, "model_load_seconds": load_seconds,
        "peak_process_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "models": {"cam_sha256": CAM_SHA256, "wesep_sha256": WESEP_SHA256,
                   "config_sha256": CONFIG_SHA256},
        "packages": {p: importlib.metadata.version(p) for p in
                     ["torch", "torchaudio", "onnxruntime", "numpy", "soundfile"]},
        "enrollment_sha256": digest((args.manifest.parent / manifest_data["enrollment"]).resolve()),
        "enrollment_seconds": len(enrollment) / RATE,
        "fixture_provenance": manifest_data.get("provenance", {}),
        "fixture_limitations": manifest_data.get("limitations", []),
        "all_windows_faster_than_realtime": all(r["median_rtf"] < 1 for r in rows),
        "production_ready": False,
        "limitations": [
            "Non-overlapping windows: live overlap processing would add work.",
            "No production threshold or calibrated error-rate estimate.",
            "No ASR, word error rate, microphone capture, or text delivery.",
            "Peak RSS includes model loading and the Python runtime.",
        ],
        "results": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cam", type=Path, required=True)
    parser.add_argument("--wesep", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--window-seconds", type=float, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.threads < 1 or args.repeats < 1 or not math.isfinite(args.window_seconds) or not 1 <= args.window_seconds <= 15:
        parser.error("positive threads/repeats and a window of 1–15 seconds are required")
    report = evaluate(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Do not accidentally expose evaluation metadata when using personal audio.
    with os.fdopen(os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w") as handle:
        os.fchmod(handle.fileno(), 0o600)
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return 0 if report["all_windows_faster_than_realtime"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
