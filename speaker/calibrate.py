#!/usr/bin/env python3
"""Record a private speaker dataset and calibrate a CAM++ acceptance threshold."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import time

import numpy as np
import soundfile as sf

import evaluate as ev


MIN_ENROLLMENT_SECONDS = 15
MIN_CASE_SECONDS = 1
DEFAULT_PERSONAL_WINDOW_SECONDS = 6
SAFE_NAME = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}")


def private_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)
    path.chmod(0o600)


def manifest_path(directory: Path) -> Path:
    return directory.resolve() / "manifest.json"


def load_manifest(directory: Path) -> tuple[Path, dict]:
    path = manifest_path(directory)
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError as error:
        raise ValueError(f"no dataset at {directory}; run the init command first") from error
    if not isinstance(data, dict) or not isinstance(data.get("cases"), list):
        raise ValueError(f"invalid personal speaker manifest: {path}")
    return path, data


def inspect_case_audio(path: Path, data: dict) -> tuple[list[str], list[tuple[str, str]]]:
    missing = []
    invalid = []
    for case in data["cases"]:
        audio = (path.parent / case["audio"]).resolve()
        if not audio.is_file():
            missing.append(case["name"])
            continue
        try:
            ev.read_audio(audio)
        except (OSError, RuntimeError, ValueError) as error:
            invalid.append((case["name"], str(error)))
    return missing, invalid


def repair_missing_cases(path: Path, data: dict, missing: list[str]) -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    backup = path.with_name(f"manifest.backup-{stamp}.json")
    private_json(backup, data)
    missing_set = set(missing)
    data["cases"] = [case for case in data["cases"] if case["name"] not in missing_set]
    private_json(path, data)
    return backup


def init_dataset(directory: Path) -> Path:
    directory = directory.resolve()
    path = manifest_path(directory)
    if path.exists():
        raise ValueError(f"dataset already exists: {path}")
    directory.mkdir(parents=True, exist_ok=True)
    directory.chmod(0o700)
    audio = directory / "audio"
    audio.mkdir(mode=0o700, exist_ok=True)
    private_json(path, {
        "schema_version": 1,
        "enrollment": "audio/enrollment.wav",
        "cases": [],
        "provenance": {"kind": "personal microphone calibration"},
        "limitations": [
            "Personal calibration data; results do not generalize to other users or devices."
        ],
    })
    return path


def validate_name(name: str) -> None:
    if not SAFE_NAME.fullmatch(name):
        raise ValueError("names must match [a-z0-9][a-z0-9_-]{0,63}")


def save_audio(path: Path, pcm: np.ndarray) -> None:
    temporary = path.with_name(f".{path.stem}-{os.getpid()}.wav")
    try:
        sf.write(temporary, pcm, ev.RATE, subtype="PCM_16")
        temporary.chmod(0o600)
        os.replace(temporary, path)
        path.chmod(0o600)
    finally:
        temporary.unlink(missing_ok=True)


def import_audio(source: Path, destination: Path, minimum_seconds: int) -> float:
    pcm = ev.read_audio(source.resolve())
    duration = len(pcm) / ev.RATE
    if duration < minimum_seconds:
        raise ValueError(
            f"{source} is {duration:.1f}s; at least {minimum_seconds}s is required"
        )
    save_audio(destination, pcm)
    return duration


def validate_recording(
    path: Path,
    returncode: int,
    stderr: str,
    requested_seconds: float,
    minimum_seconds: int,
) -> tuple[np.ndarray, float]:
    detail = stderr.strip() or "no diagnostic output"
    try:
        pcm = ev.read_audio(path)
    except (OSError, RuntimeError, ValueError) as error:
        raise ValueError(f"pw-record failed with status {returncode}: {detail}") from error
    duration = len(pcm) / ev.RATE
    required = max(minimum_seconds, requested_seconds * 0.9)
    if duration < required:
        raise ValueError(
            f"pw-record produced only {duration:.1f}s (wanted {requested_seconds:g}s); "
            f"status {returncode}: {detail}"
        )
    # pw-record commonly reports SIGINT as a nonzero process status even though
    # SIGINT is how its documented recording lifecycle finalises the WAV. The
    # format and duration checks above are the authoritative success criteria.
    return pcm, duration


def record_audio(destination: Path, seconds: float, minimum_seconds: int) -> float:
    recorder = shutil.which("pw-record")
    if recorder is None:
        raise ValueError("pw-record is required to capture calibration audio")
    input(f"Press Enter when ready to record {seconds:g} seconds: ")
    temporary = destination.with_name(f".{destination.stem}-{os.getpid()}.wav")
    command = [
        recorder, "--rate", str(ev.RATE), "--channels", "1", "--format", "s16",
        "--", str(temporary),
    ]
    process = subprocess.Popen(
        command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE, text=True,
    )
    print("Recording...", flush=True)
    try:
        time.sleep(seconds)
        process.send_signal(signal.SIGINT)
        try:
            _, stderr = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            _, stderr = process.communicate()
            raise ValueError("pw-record did not stop cleanly")
        _, duration = validate_recording(
            temporary, process.returncode, stderr, seconds, minimum_seconds
        )
        temporary.chmod(0o600)
        os.replace(temporary, destination)
        destination.chmod(0o600)
        print(f"Saved {duration:.1f}s to {destination}")
        return duration
    except BaseException:
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        raise
    finally:
        temporary.unlink(missing_ok=True)


def enrollment(args: argparse.Namespace) -> None:
    path, data = load_manifest(args.directory)
    destination = path.parent / data["enrollment"]
    if destination.exists() and not args.replace:
        raise ValueError(f"enrollment already exists: {destination}; pass --replace to replace it")
    destination.parent.mkdir(mode=0o700, exist_ok=True)
    if args.from_wav:
        duration = import_audio(args.from_wav, destination, MIN_ENROLLMENT_SECONDS)
        print(f"Imported {duration:.1f}s to {destination}")
    else:
        record_audio(destination, args.seconds, MIN_ENROLLMENT_SECONDS)


def add_case(
    path: Path,
    data: dict,
    name: str,
    split: str,
    kind: str,
    language: str,
    seconds: float,
    source: Path | None = None,
) -> None:
    validate_name(name)
    if any(case.get("name") == name for case in data["cases"]):
        raise ValueError(f"case already exists: {name}")
    destination = path.parent / "audio" / f"{name}.wav"
    if destination.exists():
        raise ValueError(f"audio already exists but is not in the manifest: {destination}")
    if source:
        duration = import_audio(source, destination, MIN_CASE_SECONDS)
        print(f"Imported {duration:.1f}s to {destination}")
    else:
        record_audio(destination, seconds, MIN_CASE_SECONDS)
    data["cases"].append({
        "name": name,
        "audio": str(destination.relative_to(path.parent)),
        "target_present": kind in {"target", "mixed"},
        "language": language,
        "split": split,
        "condition": kind,
    })
    private_json(path, data)


def capture(args: argparse.Namespace) -> None:
    path, data = load_manifest(args.directory)
    prefix = args.prefix or f"{args.split}-{args.kind}"
    validate_name(prefix)
    instructions = {
        "target": "Only the enrolled target speaker should talk.",
        "other": "The enrolled target must stay silent while another person talks.",
        "mixed": "The enrolled target and at least one other person should overlap.",
    }
    print(instructions[args.kind])
    existing = {case["name"] for case in data["cases"]}
    number = 1
    for _ in range(args.count):
        while f"{prefix}-{number:03d}" in existing:
            number += 1
        name = f"{prefix}-{number:03d}"
        print(f"\n{name} ({args.kind}, {args.split})")
        add_case(path, data, name, args.split, args.kind, args.language, args.seconds)
        existing.add(name)
        number += 1


def add_imported_case(args: argparse.Namespace) -> None:
    path, data = load_manifest(args.directory)
    add_case(
        path, data, args.name, args.split, args.kind, args.language,
        seconds=0, source=args.from_wav,
    )


def metrics(rows: list[dict], threshold: float) -> dict:
    positives = [row for row in rows if row["target_present"]]
    negatives = [row for row in rows if not row["target_present"]]

    def accepted(row: dict) -> bool:
        score = row.get("extracted_cosine")
        return score is not None and score >= threshold

    false_rejections = sum(not accepted(row) for row in positives)
    false_acceptances = sum(accepted(row) for row in negatives)
    return {
        "windows": len(rows),
        "target_windows": len(positives),
        "nontarget_windows": len(negatives),
        "false_acceptances": false_acceptances,
        "false_rejections": false_rejections,
        "false_acceptance_rate": false_acceptances / len(negatives) if negatives else None,
        "false_rejection_rate": false_rejections / len(positives) if positives else None,
    }


def choose_threshold(rows: list[dict]) -> tuple[float, dict]:
    if not any(row["target_present"] for row in rows):
        raise ValueError("calibration split has no target-present windows")
    if not any(not row["target_present"] for row in rows):
        raise ValueError("calibration split has no target-absent windows")
    scores = sorted({
        float(row["extracted_cosine"])
        for row in rows if row.get("extracted_cosine") is not None
    })
    if not scores:
        raise ValueError("calibration split has no speaker scores")
    candidates = [math.nextafter(scores[0], -math.inf)]
    candidates.extend((left + right) / 2 for left, right in zip(scores, scores[1:]))
    candidates.append(math.nextafter(scores[-1], math.inf))

    ranked = []
    for threshold in candidates:
        result = metrics(rows, threshold)
        far = result["false_acceptance_rate"]
        frr = result["false_rejection_rate"]
        ranked.append((max(far, frr), far + frr, -threshold, threshold, result))
    _, _, _, threshold, result = min(ranked)
    return threshold, result


def add_calibration(report: dict, manifest: dict) -> dict:
    metadata = {case["name"]: case for case in manifest["cases"]}
    for row in report["results"]:
        case = metadata[row["name"]]
        row["split"] = case.get("split", "unspecified")
        row["condition"] = case.get("condition", "unspecified")
    calibration_rows = [row for row in report["results"] if row["split"] == "calibration"]
    test_rows = [row for row in report["results"] if row["split"] == "test"]
    try:
        threshold, calibration_metrics = choose_threshold(calibration_rows)
    except ValueError as error:
        report["personal_calibration"] = {"ready": False, "error": str(error)}
        return report
    report["personal_calibration"] = {
        "ready": True,
        "method": "minimize the worse of FAR and FRR on calibration windows",
        "selected_threshold": threshold,
        "calibration": calibration_metrics,
        "held_out_test": metrics(test_rows, threshold) if test_rows else None,
        "warning": (
            "This is a personal experimental threshold, not a probability or a production guarantee."
        ),
    }
    return report


def run_evaluation(args: argparse.Namespace) -> int:
    path, manifest = load_manifest(args.directory)
    enrollment_file = path.parent / manifest["enrollment"]
    if not enrollment_file.is_file():
        raise ValueError("enrollment is missing; run the enroll command first")
    if not manifest["cases"]:
        raise ValueError("dataset has no cases; run capture or add first")
    missing, invalid = inspect_case_audio(path, manifest)
    if missing:
        names = ", ".join(missing[:5])
        remainder = f" and {len(missing) - 5} more" if len(missing) > 5 else ""
        raise ValueError(
            f"{len(missing)} manifest case(s) have no audio: {names}{remainder}; "
            f"run: bash speaker/calibrate.sh doctor {args.directory} --remove-missing"
        )
    if invalid:
        name, error = invalid[0]
        raise ValueError(
            f"{len(invalid)} case audio file(s) are invalid; first is {name}: {error}"
        )
    namespace = argparse.Namespace(
        manifest=path,
        cam=args.cam,
        wesep=args.wesep,
        threads=args.threads,
        window_seconds=args.window_seconds,
        repeats=args.repeats,
        quiet=True,
    )
    report = add_calibration(ev.evaluate(namespace), manifest)
    output = path.parent / "report.json"
    private_json(output, report)
    calibration = report["personal_calibration"]
    print(f"\nReport: {output}")
    if not calibration["ready"]:
        print(f"Calibration incomplete: {calibration['error']}")
        return 0
    print(f"Selected cosine threshold: {calibration['selected_threshold']:.6f}")
    print_metrics("Calibration", calibration["calibration"])
    if calibration["held_out_test"]:
        print_metrics("Held-out test", calibration["held_out_test"])
    else:
        print("Held-out test: no test cases")
    return 0 if report["all_windows_faster_than_realtime"] else 2


def print_metrics(label: str, result: dict) -> None:
    far = result["false_acceptance_rate"]
    frr = result["false_rejection_rate"]
    far_text = "n/a" if far is None else f"{far:.1%}"
    frr_text = "n/a" if frr is None else f"{frr:.1%}"
    print(
        f"{label}: FAR {far_text} ({result['false_acceptances']} false accepts), "
        f"FRR {frr_text} ({result['false_rejections']} false rejects), "
        f"{result['windows']} windows"
    )


def show_dataset(directory: Path) -> None:
    path, data = load_manifest(directory)
    enrollment_file = path.parent / data["enrollment"]
    if enrollment_file.exists():
        seconds = len(ev.read_audio(enrollment_file)) / ev.RATE
        print(f"Enrollment: {seconds:.1f}s ({enrollment_file})")
    else:
        print("Enrollment: missing")
    counts = Counter(
        (case.get("split", "unspecified"), "target" if case["target_present"] else "nontarget")
        for case in data["cases"]
    )
    print(f"Cases: {len(data['cases'])}")
    for (split, kind), count in sorted(counts.items()):
        print(f"  {split:<12} {kind:<9} {count}")
    missing, invalid = inspect_case_audio(path, data)
    if missing:
        print(f"Warning: {len(missing)} case(s) reference missing audio files")
    if invalid:
        print(f"Warning: {len(invalid)} case(s) contain invalid audio files")
    print(f"Manifest: {path}")


def doctor_dataset(args: argparse.Namespace) -> None:
    path, data = load_manifest(args.directory)
    missing, invalid = inspect_case_audio(path, data)
    if not missing and not invalid:
        print(f"Dataset is consistent: {len(data['cases'])} case audio files are valid")
        return
    if missing:
        print(f"Missing audio ({len(missing)}):")
        for name in missing:
            print(f"  {name}")
    if invalid:
        print(f"Invalid audio ({len(invalid)}):")
        for name, error in invalid:
            print(f"  {name}: {error}")
    if args.remove_missing and missing:
        backup = repair_missing_cases(path, data, missing)
        print(f"Removed {len(missing)} missing references from the manifest")
        print(f"Backup: {backup}")
    elif missing:
        print("No changes made. Pass --remove-missing to repair the manifest.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cam", type=Path, required=True, help=argparse.SUPPRESS)
    parser.add_argument("--wesep", type=Path, required=True, help=argparse.SUPPRESS)
    subparsers = parser.add_subparsers(dest="command", required=True)

    init = subparsers.add_parser("init", help="create a private dataset")
    init.add_argument("directory", type=Path)

    enroll = subparsers.add_parser("enroll", help="record or import the reference voice")
    enroll.add_argument("directory", type=Path)
    enroll.add_argument("--seconds", type=float, default=20)
    enroll.add_argument("--from-wav", type=Path)
    enroll.add_argument("--replace", action="store_true")

    capture_parser = subparsers.add_parser("capture", help="record one or more labelled cases")
    capture_parser.add_argument("directory", type=Path)
    capture_parser.add_argument("--split", choices=["calibration", "test"], required=True)
    capture_parser.add_argument("--kind", choices=["target", "other", "mixed"], required=True)
    capture_parser.add_argument("--count", type=int, default=1)
    capture_parser.add_argument("--seconds", type=float, default=5)
    capture_parser.add_argument("--language", default="sv")
    capture_parser.add_argument("--prefix")

    add = subparsers.add_parser("add", help="import an existing labelled mono 16 kHz WAV")
    add.add_argument("directory", type=Path)
    add.add_argument("--name", required=True)
    add.add_argument("--split", choices=["calibration", "test"], required=True)
    add.add_argument("--kind", choices=["target", "other", "mixed"], required=True)
    add.add_argument("--language", default="sv")
    add.add_argument("--from-wav", type=Path, required=True)

    run = subparsers.add_parser("run", help="evaluate and select a personal threshold")
    run.add_argument("directory", type=Path)
    run.add_argument("--threads", type=int, default=4)
    # Captured cases default to five seconds. A six-second evaluation window
    # keeps each case whole instead of manufacturing an unscorable <1s tail.
    run.add_argument(
        "--window-seconds", type=float, default=DEFAULT_PERSONAL_WINDOW_SECONDS
    )
    run.add_argument("--repeats", type=int, default=3)

    show = subparsers.add_parser("show", help="summarize the dataset")
    show.add_argument("directory", type=Path)

    doctor = subparsers.add_parser("doctor", help="find and optionally remove broken case references")
    doctor.add_argument("directory", type=Path)
    doctor.add_argument(
        "--remove-missing", action="store_true",
        help="back up the manifest and remove cases whose WAV file is missing",
    )
    return parser


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if hasattr(args, "seconds") and (not math.isfinite(args.seconds) or args.seconds <= 0):
        parser.error("seconds must be positive")
    if getattr(args, "command", None) == "enroll" and not args.from_wav \
            and args.seconds < MIN_ENROLLMENT_SECONDS:
        parser.error(f"enrollment recording must be at least {MIN_ENROLLMENT_SECONDS} seconds")
    if hasattr(args, "count") and args.count < 1:
        parser.error("count must be positive")
    if getattr(args, "command", None) == "run":
        if args.threads < 1 or args.repeats < 1:
            parser.error("threads and repeats must be positive")
        if not math.isfinite(args.window_seconds) or not 1 <= args.window_seconds <= 15:
            parser.error("window-seconds must be between 1 and 15")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args, parser)
    if args.command == "init":
        print(f"Created {init_dataset(args.directory)}")
    elif args.command == "enroll":
        enrollment(args)
    elif args.command == "capture":
        capture(args)
    elif args.command == "add":
        add_imported_case(args)
    elif args.command == "run":
        return run_evaluation(args)
    elif args.command == "show":
        show_dataset(args.directory)
    elif args.command == "doctor":
        doctor_dataset(args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"calibrate: {error}", file=os.sys.stderr)
        raise SystemExit(1)
