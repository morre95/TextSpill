#!/usr/bin/env bash
# Generate and compare original and speaker-extracted evaluation audio.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
evaluation="$root/target/speaker-eval"
python="$evaluation/venv/bin/python"

if [[ ! -x "$python" || ! -d "$evaluation/models/wesep" ]]; then
    echo "Speaker evaluation is not installed. Run: bash speaker/setup_eval.sh" >&2
    exit 1
fi

export PYTHONPATH="$evaluation/src/wesep:$evaluation/src/wespeaker${PYTHONPATH:+:$PYTHONPATH}"
exec "$python" "$root/speaker/listen_eval.py" \
    --manifest "$evaluation/fixtures/manifest.json" \
    --wesep "$evaluation/models/wesep" \
    --output-dir "$evaluation/listening" "$@"
