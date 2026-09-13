#!/usr/bin/env python3
"""Fetch pinned public models/examples and construct reproducible smoke fixtures.

No microphone input. Audio stays in the ignored evaluation directory. These
short example clips are NOT a calibration corpus or a substitute for testing
15–30 second enrollment with real Swedish multi-speaker recordings.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import urllib.request

import numpy as np
from scipy.signal import resample_poly
import soundfile as sf

DEMO_REVISION = "c3212546b3d42328c059562b7a94508dd3833094"
MODEL_REVISION = "b77ec86561b647a08717d4c27c9e0c971d694252"
CAM_SHA256 = "b50810498b5bcf5773d086f6993d344476bd0c88b566a41e8d801aaf8461efad"
WESEP_ARCHIVE_SHA256 = "a129b8247e47a2a2fe7407768c69a55c0f97433ba1016c943e9afa1ad2414ffb"


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def fetch(url: str, path: Path, expected: str | None = None) -> None:
    if path.is_file() and expected is not None and sha256(path) == expected:
        return
    temporary = path.with_suffix(path.suffix + ".download")
    try:
        with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output)
        if expected is not None and sha256(temporary) != expected:
            raise ValueError(f"checksum mismatch: {path.name}")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def prepare(directory: Path) -> None:
    models = directory / "models"
    fixtures = directory / "fixtures"
    original = directory / "source-audio"
    for folder in [models / "wesep", fixtures, original]:
        folder.mkdir(parents=True, exist_ok=True)
    fetch(
        "https://huggingface.co/Wespeaker/wespeaker-voxceleb-campplus/resolve/main/voxceleb_CAM%2B%2B.onnx",
        models / "voxceleb_CAM++.onnx", CAM_SHA256,
    )
    archive = models / "bsrnn_ecapa_vox1.tar.gz"
    fetch(
        f"https://www.modelscope.cn/datasets/wenet/wesep_pretrained_models/resolve/{MODEL_REVISION}/bsrnn_ecapa_vox1.tar.gz",
        archive, WESEP_ARCHIVE_SHA256,
    )
    # Extract only the two expected regular files, never arbitrary archive paths.
    with tarfile.open(archive) as package:
        for name in ["avg_model.pt", "config.yaml"]:
            member = package.getmember("./" + name)
            if not member.isfile():
                raise ValueError(f"expected regular model file: {name}")
            with package.extractfile(member) as source, (models / "wesep" / name).open("wb") as out:
                shutil.copyfileobj(source, out)
    source_hashes = {}
    for name in ["enroll_1.wav", "enroll_2.wav", "enroll2_zh.wav", "mixture.wav"]:
        path = original / name
        fetch(
            f"https://huggingface.co/spaces/wenet-e2e/wesep-tse-2speaker-demo/resolve/{DEMO_REVISION}/examples/{name}",
            path,
        )
        source_hashes[name] = sha256(path)
    target, rate = sf.read(original / "enroll_1.wav", dtype="float32")
    other, other_rate = sf.read(original / "enroll_2.wav", dtype="float32")
    third, third_rate = sf.read(original / "enroll2_zh.wav", dtype="float32")
    mixture, mixture_rate = sf.read(original / "mixture.wav", dtype="float32")
    if (rate, other_rate, third_rate, mixture_rate) != (16000, 16000, 44100, 16000):
        raise ValueError("unexpected upstream sample rates")
    if any(x.ndim != 1 for x in [target, other, third, mixture]):
        raise ValueError("unexpected upstream channel count")
    if len(target) < 80000 or len(other) < 32000 or len(mixture) < 64000:
        raise ValueError("upstream fixture is too short")
    # No sample overlap between the target's enrollment and evaluation audio.
    sf.write(fixtures / "enroll.wav", target[:48000], 16000, subtype="PCM_16")
    target = target[48000:80000]
    other = other[:32000]
    third = resample_poly(third, 160, 441)[:32000]
    if len(third) != 32000:
        raise ValueError("third speaker fixture is too short")
    target_rms = np.sqrt(np.mean(target ** 2))
    other_equal = other / np.sqrt(np.mean(other ** 2)) * target_rms
    third_equal = third / np.sqrt(np.mean(third ** 2)) * target_rms
    # All fixtures have known target presence. Only constructed mixtures have
    # a sample-aligned clean target suitable for SI-SDR.
    examples = [
        ("target", target, True, target, "en"),
        ("other", other, False, None, "en"),
        ("overlap", (target + other) * .5, True, target * .5, "en"),
        ("louder_other", target * .25 + other * .75, True, target * .25, "en"),
        ("silence", np.zeros(32000), False, None, "none"),
        ("official_mixture", mixture[:64000], True, None, "en"),
        ("three_speakers", (target + other_equal + third_equal) / 3, True, target / 3, "en+zh"),
        ("two_others", (other_equal + third_equal) / 2, False, None, "en+zh"),
        ("speaker_turn", np.r_[target[:16000], other_equal[16000:]], True,
         np.r_[target[:16000], np.zeros(16000)], "en"),
    ]
    cases = []
    for name, pcm, present, clean, language in examples:
        sf.write(fixtures / f"{name}.wav", pcm, 16000, subtype="PCM_16")
        row = {"name": name, "audio": f"{name}.wav", "target_present": present, "language": language}
        if clean is not None:
            sf.write(fixtures / f"{name}-clean.wav", clean, 16000, subtype="PCM_16")
            row["clean_target"] = f"{name}-clean.wav"
        cases.append(row)
    (fixtures / "manifest.json").write_text(json.dumps({
        "enrollment": "enroll.wav", "cases": cases,
        "provenance": {"demo_revision": DEMO_REVISION, "source_sha256": source_hashes},
        "limitations": ["3-second enrollment; 2–4 second evaluation clips",
                        "same-recording split for enrollment and positive clean test",
                        "English/Chinese only; not Swedish; no threshold calibration"],
    }, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    prepare(parser.parse_args().directory)
