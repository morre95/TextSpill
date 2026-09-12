#!/usr/bin/env bash
# Installs TextSpill with either local Qwen3-ASR or Deepgram.
# Safe to re-run; it upgrades an existing installation in place.
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
share="${XDG_DATA_HOME:-$HOME/.local/share}/textspill"
bindir="$HOME/.local/bin"
units="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
python="${TEXTSPILL_PYTHON:-python3.12}"
backend=""
while (($#)); do
    case "$1" in
        --backend)
            [[ $# -ge 2 ]] || { echo '--backend requires local or deepgram' >&2; exit 2; }
            backend="$2"
            shift 2
            ;;
        --help|-h)
            echo 'Usage: ./install.sh [--backend local|deepgram]'
            exit 0
            ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done
if [[ -n "$backend" && "$backend" != local && "$backend" != deepgram ]]; then
    echo '--backend must be local or deepgram' >&2
    exit 2
fi

say() { printf '\033[1;34m==>\033[0m %s\n' "$1"; }

say "Building the textspill binary"
cargo build --release --manifest-path "$repo/Cargo.toml"
mkdir -p "$bindir"
install -m755 "$repo/target/release/textspill" "$bindir/textspill"

if [[ -z "$backend" ]]; then
    backend="$("$bindir/textspill" configure)"
    if [[ -t 0 ]]; then
        printf '\nTranscription backend:\n  1) Local Qwen3-ASR (requires RAM/VRAM and several GB of downloads)\n  2) Deepgram (internet and a paid API account; audio is sent to Deepgram)\n'
        read -r -p "Choose 1/local or 2/deepgram [$backend]: " choice
        case "$choice" in
            1|local) backend=local ;;
            2|deepgram) backend=deepgram ;;
            '') ;;
            *) echo 'Invalid backend choice' >&2; exit 2 ;;
        esac
    fi
fi

if [[ "$backend" == local ]]; then
say "Linking the ASR daemon into $share"
mkdir -p "$share"
# Symlinks, not copies: editing the repo takes effect on the next service restart.
ln -sfn "$repo/asr/daemon.py" "$share/daemon.py"

if [ ! -x "$share/venv/bin/python" ]; then
    say "Creating the Python environment with $python"
    command -v "$python" >/dev/null || {
        echo "$python not found; set TEXTSPILL_PYTHON to an interpreter torch supports" >&2
        exit 1
    }
    "$python" -m venv "$share/venv"
fi

say "Installing Python dependencies (this downloads PyTorch, ~2-3 GB)"
"$share/venv/bin/pip" install --upgrade pip
"$share/venv/bin/pip" install --upgrade \
    "torch" \
    "transformers>=5.13.0" \
    "accelerate" \
    "librosa" \
    "soundfile"

say "Installing the user service"
mkdir -p "$units"
install -m644 "$repo/systemd/textspill-asr.service" "$units/textspill-asr.service"
systemctl --user daemon-reload
else
    say "Using Deepgram: audio is sent to Deepgram and billed to your API account"
    # Stop the old model to release RAM/VRAM, but retain downloads for switching back.
    if [[ -f "$units/textspill-asr.service" ]]; then
        systemctl --user disable --now textspill-asr.service
    fi
fi

"$bindir/textspill" configure --backend "$backend" >/dev/null

if [[ "$backend" == local ]]; then
cat <<'LOCAL'

Start the local ASR daemon (first run downloads the model, ~1.5 GB):
  systemctl --user enable --now textspill-asr.service
  journalctl --user -u textspill-asr.service -f
LOCAL
else
cat <<'DEEPGRAM'

Configure your Deepgram API key separately before recording:
  mkdir -p "${XDG_CONFIG_HOME:-$HOME/.config}/textspill"
  (umask 077; touch "${XDG_CONFIG_HOME:-$HOME/.config}/textspill/deepgram-api-key")
  chmod 600 "${XDG_CONFIG_HOME:-$HOME/.config}/textspill/deepgram-api-key"
  $EDITOR "${XDG_CONFIG_HOME:-$HOME/.config}/textspill/deepgram-api-key"

Put only the API key in that file. DEEPGRAM_API_KEY overrides the file, but a
terminal export alone may not reach Omarchy hotkeys. No ASR service is needed.
Swedish is the default. Set deepgram_language to "en" in config.json for English.
DEEPGRAM
fi

cat <<'NEXT'

Done. Remaining steps:

  1. Enable the paste backend:
       sudo pacman -S ydotool
       sudo usermod -aG input "$USER"     # log out and back in
       systemctl --user enable --now ydotool.service

  2. Bind the hotkey. On Omarchy, in ~/.config/hypr/bindings.lua:
       o.bind("CTRL + SHIFT + INSERT", "Dictate (live)", "env TEXTSPILL_LIVE=1 textspill toggle")
       o.bind("SUPER + PERIOD", "Dictate (toggle)", "env TEXTSPILL_LIVE=0 textspill toggle")
     then `hyprctl reload`. (SUPER+SPACE is the Omarchy menu; unbind it first
     with `hl.unbind("SUPER + SPACE")` if you want that key instead.)

  3. Make sure ~/.local/bin is on your PATH.

NEXT
