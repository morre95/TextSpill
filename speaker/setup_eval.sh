#!/usr/bin/env bash
# Isolated evaluation only: no TextSpill installation, service or user config.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
evaluation="$root/target/speaker-eval"
python="${TEXTSPILL_PYTHON:-python3.12}"
mkdir -p "$evaluation/src"
if [[ ! -x "$evaluation/venv/bin/python" ]]; then
    "$python" -m venv "$evaluation/venv"
fi
"$evaluation/venv/bin/pip" install torch==2.5.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cpu
"$evaluation/venv/bin/pip" install -r "$root/speaker/requirements-eval.txt"

checkout() {
    local name="$1" revision="$2"
    local destination="$evaluation/src/$name"
    if [[ ! -d "$destination/.git" ]]; then
        git init -q "$destination"
        git -C "$destination" remote add origin "https://github.com/wenet-e2e/$name.git"
    fi
    if ! git -C "$destination" cat-file -e "$revision^{commit}" 2>/dev/null; then
        git -C "$destination" fetch --depth 1 origin "$revision"
    fi
    # Refuse to overwrite local edits to a cached source checkout.
    if [[ -n "$(git -C "$destination" status --porcelain)" ]]; then
        echo "Uncommitted changes in $destination; refusing to change it" >&2
        exit 1
    fi
    git -C "$destination" checkout --detach "$revision"
}
checkout wesep 99eca54b60300d39b9353d93cf285a14bba37854
# Matches the extraction checkpoint without importing unrelated modern ASR models.
checkout wespeaker 820acb41d3ea1ffe2c465375189d7578f4553996
"$evaluation/venv/bin/python" "$root/speaker/prepare_eval.py" --directory "$evaluation"
echo "Ready. Run: bash speaker/run_eval.sh --threads 4"
