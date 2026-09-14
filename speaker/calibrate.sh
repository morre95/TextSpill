#!/usr/bin/env bash
# Record a private speaker dataset and calibrate CAM++ / WeSep against it.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
evaluation="$root/target/speaker-eval"
python="$evaluation/venv/bin/python"

if [[ ! -x "$python" ]]; then
    echo "Speaker evaluation is not installed. Run: bash speaker/setup_eval.sh" >&2
    exit 1
fi

export PYTHONPATH="$evaluation/src/wesep:$evaluation/src/wespeaker${PYTHONPATH:+:$PYTHONPATH}"
exec "$python" "$root/speaker/calibrate.py" \
    --cam "$evaluation/models/voxceleb_CAM++.onnx" \
    --wesep "$evaluation/models/wesep" "$@"
