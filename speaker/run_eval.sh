#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
evaluation="$root/target/speaker-eval"
export PYTHONPATH="$evaluation/src/wesep:$evaluation/src/wespeaker${PYTHONPATH:+:$PYTHONPATH}"
exec "$evaluation/venv/bin/python" "$root/speaker/evaluate.py" \
    --manifest "$evaluation/fixtures/manifest.json" \
    --cam "$evaluation/models/voxceleb_CAM++.onnx" \
    --wesep "$evaluation/models/wesep" \
    --output "$evaluation/report.json" "$@"
