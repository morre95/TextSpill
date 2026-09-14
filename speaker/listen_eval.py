#!/usr/bin/env python3
"""Generate and play before/after audio from the speaker evaluation fixtures."""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import soundfile as sf
import torch

import evaluate as ev


def select_cases(cases: list[dict], requested: list[str], use_all: bool) -> list[dict]:
    by_name = {case["name"]: case for case in cases}
    if use_all:
        return cases
    if not requested:
        return []
    unknown = [name for name in requested if name not in by_name]
    if unknown:
        available = ", ".join(by_name)
        raise ValueError(f"unknown case(s): {', '.join(unknown)}; available: {available}")
    # Preserve the user's order while avoiding accidental duplicate playback.
    return [by_name[name] for name in dict.fromkeys(requested)]


def extract_windows(
    separator: ev.Separator,
    pcm: np.ndarray,
    enrollment: np.ndarray,
    window_samples: int,
) -> np.ndarray:
    output = []
    for offset in range(0, len(pcm), window_samples):
        part = pcm[offset : offset + window_samples]
        if len(part) < 512:
            raise ValueError("tail shorter than separator STFT window; adjust window size")
        output.append(separator.extract(part, enrollment))
    return np.concatenate(output)


def private_wav(path: Path, pcm: np.ndarray) -> None:
    sf.write(path, pcm, ev.RATE, subtype="PCM_16")
    path.chmod(0o600)


def generate_pairs(
    manifest: Path,
    model: Path,
    output_dir: Path,
    requested: list[str],
    use_all: bool,
    threads: int,
    window_seconds: float,
) -> list[tuple[str, Path, Path]]:
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    enrollment, cases = ev.load_manifest(manifest)
    chosen = select_cases(cases, requested, use_all)
    if not chosen:
        return []

    separator = ev.Separator(model)
    # Importing WeSep brings in Silero VAD, which changes this global setting.
    torch.set_num_threads(threads)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir.chmod(0o700)
    window_samples = int(window_seconds * ev.RATE)
    pairs = []
    for case in chosen:
        name = case["name"]
        before = output_dir / f"{name}-before.wav"
        after = output_dir / f"{name}-after.wav"
        filtered = extract_windows(separator, case["pcm"], enrollment, window_samples)
        private_wav(before, case["pcm"])
        private_wav(after, filtered)
        pairs.append((name, before, after))
        print(f"generated {name}: {before.name}, {after.name}", flush=True)
    return pairs


def find_player(requested: str | None) -> str:
    candidates = [requested] if requested else ["pw-play", "paplay", "aplay", "ffplay"]
    for candidate in candidates:
        if candidate and (path := shutil.which(candidate)):
            return path
    names = requested or "pw-play, paplay, aplay or ffplay"
    raise ValueError(f"no audio player found ({names})")


def play(player: str, path: Path) -> None:
    command = [player, str(path)]
    if Path(player).name == "ffplay":
        command[1:1] = ["-nodisp", "-autoexit", "-loglevel", "error"]
    subprocess.run(command, check=True)


def play_pairs(pairs: list[tuple[str, Path, Path]], player: str) -> None:
    for name, before, after in pairs:
        input(f"\n{name}: tryck Enter för FÖRE (original/blandning) ")
        play(player, before)
        input(f"{name}: tryck Enter för EFTER (utplockad målröst) ")
        play(player, after)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and interactively play evaluation audio before and after WeSep extraction."
    )
    parser.add_argument("cases", nargs="*", help="case names to process")
    parser.add_argument("--all", action="store_true", help="process every manifest case")
    parser.add_argument("--list", action="store_true", help="list cases without loading models")
    parser.add_argument("--generate-only", action="store_true", help="write WAV files without playback")
    parser.add_argument("--player", help="audio player executable (default: auto-detect)")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--window-seconds", type=float, default=4)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--wesep", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.all and args.cases:
        parser.error("use case names or --all, not both")
    if args.threads < 1 or not math.isfinite(args.window_seconds) or not 1 <= args.window_seconds <= 15:
        parser.error("threads must be positive and window-seconds must be between 1 and 15")
    return args


def main() -> int:
    args = parse_args()
    if args.list or (not args.cases and not args.all):
        _, cases = ev.load_manifest(args.manifest)
        print("Available cases:")
        for case in cases:
            presence = "target present" if case["target_present"] else "target absent"
            print(f"  {case['name']:<18} {presence}")
        if not args.list:
            print("\nChoose one, for example: bash speaker/listen_eval.sh overlap")
        return 0

    pairs = generate_pairs(
        args.manifest,
        args.wesep,
        args.output_dir,
        args.cases,
        args.all,
        args.threads,
        args.window_seconds,
    )
    if args.generate_only:
        print(f"WAV files saved in {args.output_dir}")
        return 0
    if not sys.stdin.isatty():
        raise ValueError("interactive playback needs a terminal; use --generate-only instead")
    play_pairs(pairs, find_player(args.player))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"listen_eval: {error}", file=sys.stderr)
        raise SystemExit(1)
